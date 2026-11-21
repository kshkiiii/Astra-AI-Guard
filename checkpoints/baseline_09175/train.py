"""Фиксированное групповое разбиение, обучение на train и отчёт на validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

from model import GuardModel, load_config
from text_processing import prepare_input, text_variants


def read_rows(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Набор пуст или содержит повторяющиеся id")
    return rows


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def grouping_text(row: dict, text_options: dict) -> str:
    """Текст для проверки пересечений; преобразования не являются ML-признаками."""
    prepared = prepare_input(row["user_prompt"], row["context"], text_options)
    pieces = []
    for segment in prepared.segments:
        text = text_variants(segment.text, text_options)[-1]
        text = re.sub(r"https?://\S+|[\w.+-]+@[\w.-]+|\b\d+(?:\.\d+){3}\b", " address ", text)
        text = re.sub(r"\b\w+(?:[-_]\w+)+\b", " resource ", text)
        text = re.sub(r"\b\d+\b", " number ", text)
        text = re.sub(r"добрый день|здравствуйте|привет|слушай|вопрос|срочно|пожалуйста|"
                      r"заранее спасибо|очень нужно|спасибо|quick question|urgent|hello|hey|"
                      r"\bhi\b|please|thanks|asap|\bpls\b", " ", text)
        text = re.sub(r"[^\w]+", " ", text).strip()
        pieces.append(segment.source + " " + text)
    return " ".join(pieces)


def similarity_groups(rows: list[dict], text_options: dict, threshold: float) -> list[int]:
    """Связные компоненты похожих текстов; роль и разметка не участвуют.

    Матрица O(N²) рассчитана на текущий набор из 500 примеров. При расширении
    данных потребуется приближённый поиск соседей вместо плотной матрицы.
    """
    if not 0 < threshold <= 1:
        raise ValueError("similarity_threshold должен быть в (0, 1]")
    texts = [grouping_text(row, text_options) for row in rows]
    features = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1).fit_transform(texts)
    similarities = (features @ features.T).toarray()
    parents = list(range(len(rows)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, j in zip(*np.where(np.triu(similarities >= threshold, k=1))):
        parents[root(int(j))] = root(int(i))
    # Точное совпадение объединяется даже при погрешности cosine около 1.0.
    exact = {}
    for i, text in enumerate(texts):
        if text in exact:
            parents[root(i)] = root(exact[text])
        exact[text] = i
    return [root(i) for i in range(len(rows))]


def get_splits(rows: list[dict], config: dict, rebuild: bool = False) -> tuple[dict, dict]:
    options = config["data"]
    source_hash = hashlib.sha256(Path(options["source"]).read_bytes()).hexdigest()
    split_options = {k: options[k] for k in ("random_state", "validation_fraction", "test_fraction", "similarity_threshold")}
    directory = Path(options["split_dir"])
    manifest_path = directory / "manifest.json"
    by_id = {r["id"]: r for r in rows}
    if manifest_path.exists() and not rebuild:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["source_sha256"] != source_hash or manifest["split_options"] != split_options
                or manifest.get("split_version") != 2):
            raise ValueError("Источник или параметры разбиения изменились. Используйте другую split_dir или явно --rebuild-splits.")
        ids = [i for part in manifest["ids"].values() for i in part]
        if len(ids) != len(set(ids)) or set(ids) != set(by_id):
            raise ValueError("Повреждён manifest: пересечение или пропуск примеров")
        splits = {name: [by_id[i] for i in ids] for name, ids in manifest["ids"].items()}
    else:
        test_fraction, val_fraction = options["test_fraction"], options["validation_fraction"]
        if not 0 < test_fraction < 1 or not 0 < val_fraction < 1 or test_fraction + val_fraction >= 1:
            raise ValueError("Нужны положительные доли validation/test с суммой < 1")
        groups = np.array(similarity_groups(rows, config["text"], options["similarity_threshold"]))
        labels = np.array([r["risk_category"] for r in rows])
        indices = np.arange(len(rows))
        # Число фолдов задаёт приблизительную долю; группы никогда не разрываются.
        outer = StratifiedGroupKFold(n_splits=max(2, round(1 / test_fraction)), shuffle=True,
                                     random_state=options["random_state"])
        development, test = next(outer.split(indices, labels, groups))
        inner_fraction = val_fraction / (1 - test_fraction)
        inner = StratifiedGroupKFold(n_splits=max(2, round(1 / inner_fraction)), shuffle=True,
                                     random_state=options["random_state"] + 1)
        train_rel, val_rel = next(inner.split(development, labels[development], groups[development]))
        parts = {"train": development[train_rel], "validation": development[val_rel], "test": test}
        group_sets = {name: set(groups[idx]) for name, idx in parts.items()}
        assert not (group_sets["train"] & group_sets["validation"] or group_sets["train"] & group_sets["test"]
                    or group_sets["validation"] & group_sets["test"])
        splits = {name: [rows[i] for i in idx] for name, idx in parts.items()}
        manifest = {"source_sha256": source_hash, "split_options": split_options, "split_version": 2,
                    "group_count": len(set(groups)), "grouping": "char TF-IDF connected components",
                    "ids": {name: [r["id"] for r in part] for name, part in splits.items()},
                    "groups": {r["id"]: int(g) for r, g in zip(rows, groups)}}
        write_json(manifest_path, manifest)
    for name, part in splits.items():
        if not part or len({r["is_dangerous"] for r in part}) < 2:
            raise ValueError(f"В {name} недостаточно опасных/легитимных примеров для оценки")
        write_jsonl(directory / f"{name}.jsonl", part)
    return splits, manifest


def add_training_data(splits: dict, config: dict) -> tuple[list[dict], list[float], dict]:
    """Присоединение только к train. Совпадение одной версии исключает семью.

    Для поиска пересечений читаются тексты публичных частей, но не их метки.
    Проверка близости эвристическая и не гарантирует семантическую дедупликацию.
    """
    from model import THREAT_CATEGORIES
    originals = splits["train"]
    additions, source_info = [], []
    for source in config["data"].get("extra_sources", []):
        path = Path(source)
        rows = read_rows(path)
        source_info.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows)})
        additions.extend(rows)
    if not additions:
        return list(originals), [1.] * len(originals), {"accepted": 0, "sources": source_info, "rejected": []}
    if len({r["id"] for r in additions}) != len(additions):
        raise ValueError("Повторяющиеся id в добавочных источниках")
    original_ids = {r["id"] for part in splits.values() for r in part}
    if original_ids & {r["id"] for r in additions}:
        raise ValueError("ID дополнительных данных совпадают с публичным набором")
    allowed_categories = THREAT_CATEGORIES | {"none", "sensitive_action", "tool_abuse", "unauthorized_data_access"}
    for row in additions:
        prepare_input(row["user_prompt"], row["context"], config["text"])
        provenance = row.get("provenance", {})
        if not all(provenance.get(k) for k in ("family_id", "dataset", "license", "revision")):
            raise ValueError(f"У {row['id']} нет полного происхождения")
        if (row["risk_category"] not in allowed_categories or row["risk_level"] not in {"none", "low", "medium", "high", "critical"}
                or row["expected_decision"] not in {"allow", "confirm", "block"}
                or row["is_dangerous"] != int(row["expected_decision"] == "block")):
            raise ValueError(f"Некорректная разметка: {row['id']}")
    threshold = config["data"].get("extra_similarity_threshold", .8)
    weight = config["data"].get("extra_sample_weight", .5)
    languages = config["data"].get("extra_languages")
    if languages is not None and (not isinstance(languages, list) or not languages
                                   or not set(languages) <= {"ru", "en", "mix"}):
        raise ValueError("extra_languages должен быть непустым списком ru/en/mix")
    if not 0 < threshold <= 1 or not np.isfinite(weight) or weight <= 0:
        raise ValueError("Нужны 0 < extra_similarity_threshold <= 1 и положительный extra_sample_weight")
    reference = splits["validation"] + splits["test"] + originals
    protected_count = len(splits["validation"]) + len(splits["test"])
    reference_texts = [grouping_text(r, config["text"]) for r in reference]
    extra_texts = [grouping_text(r, config["text"]) for r in additions]
    # Этот векторизатор используется только для исключения пересечений. Модель
    # по-прежнему обучает свой отдельный TF-IDF только на допущенном train.
    vectors = TfidfVectorizer(analyzer="char", ngram_range=(3, 5)).fit_transform(reference_texts + extra_texts)
    reference_vectors = vectors[:len(reference)]
    extra_vectors = vectors[len(reference):]
    families = {}
    exact_reference = {text: i for i, text in reversed(list(enumerate(reference_texts)))}
    for offset in range(0, len(additions), 256):
        similarities = (extra_vectors[offset:offset + 256] @ reference_vectors.T).toarray()
        for local, values in enumerate(similarities):
            index = offset + local
            match = exact_reference.get(extra_texts[index], int(values.argmax()))
            if extra_texts[index] in exact_reference or values[match] >= threshold:
                family = additions[index]["provenance"]["family_id"]
                reason = "heldout_overlap" if match < protected_count else "original_train_duplicate"
                families.setdefault(family, {"reason": reason, "reference_id": reference[match]["id"],
                                             "cosine_similarity": round(float(values[match]), 6)})
    accepted, rejected, seen = [], [], {}
    for row in additions:
        family = row["provenance"]["family_id"]
        if family in families:
            rejected.append({"id": row["id"], "family_id": family, **families[family]})
            continue
        # Фильтр применяется после проверки всех языковых версий на утечки.
        if languages is not None and row.get("lang") not in languages:
            rejected.append({"id": row["id"], "family_id": family, "reason": "language_filter"})
            continue
        # При дедупликации сохраняются разные доверенные роли: их метки могут
        # различаться у одинакового запроса и проверяются отдельно по политике.
        key = (prepare_input(row["user_prompt"], row["context"], config["text"]).text, row["context"]["user_role"])
        labels = (row["expected_decision"], row["risk_category"], row["risk_level"])
        if key in seen:
            if seen[key] != labels:
                raise ValueError(f"Противоречащая разметка дубликата: {row['id']}")
            rejected.append({"id": row["id"], "family_id": family, "reason": "extra_duplicate"})
            continue
        seen[key] = labels
        accepted.append(row)
    audit = {"sources": source_info, "accepted": len(accepted), "rejected_count": len(rejected),
             "sample_weight": weight, "similarity_threshold": threshold,
             "accepted_by_dataset": dict(Counter(r["provenance"]["dataset"] for r in accepted)),
             "accepted_by_language": dict(Counter(r["lang"] for r in accepted)),
             "accepted_by_decision": dict(Counter(r["expected_decision"] for r in accepted)),
             "rejected": rejected}
    return [*originals, *accepted], [1.] * len(originals) + [weight] * len(accepted), audit


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--rebuild-splits", action="store_true", help="Явно пересоздать разбиение (меняет локальный тест)")
    args = parser.parse_args()
    config = load_config(args.config)
    rows = read_rows(config["data"]["source"])
    splits, manifest = get_splits(rows, config, args.rebuild_splits)
    for name, part in splits.items():
        print(f"{name}: {len(part)}; решения: {dict(Counter(r['expected_decision'] for r in part))}")
    print(f"Семейств похожих текстов: {manifest['group_count']}")
    training_rows, sample_weights, extra_audit = add_training_data(splits, config)
    print(f"Добавлено внешних примеров: {extra_audit['accepted']}; всего train: {len(training_rows)}")
    model = GuardModel(config).fit(training_rows, sample_weight=sample_weights)
    model.metadata.update({"source_sha256": manifest["source_sha256"],
                           "extra_sources": extra_audit["sources"], "extra_accepted": extra_audit["accepted"],
                           "extra_sample_weight": config["data"].get("extra_sample_weight", .5),
                           "validation_ids": manifest["ids"]["validation"], "test_ids": manifest["ids"]["test"]})
    model.save(config["paths"]["model"])
    print(f"Модель сохранена: {config['paths']['model']}")
    # Тест не запускается автоматически: он остаётся для финальной самопроверки.
    from check import evaluate_rows
    from model import Guard
    from policy import SecurityPolicy
    report_dir = Path(config["paths"]["reports"]) / "validation"
    write_json(report_dir / "additional_data_audit.json", extra_audit)
    report = evaluate_rows(splits["validation"], Guard(model, SecurityPolicy(config["paths"]["policy"])),
                           config, report_dir)
    write_json(report_dir / "training_metadata.json", {"model": model.metadata, "config": config,
               "split_sizes": {k: len(v) for k, v in splits.items()}, "effective_train_size": len(training_rows)})
    print(json.dumps({k: report[k] for k in ("score", "macro_f1_decisions", "strict_block_recall_high_critical",
                                            "hard_negative_block_rate", "latency_p95_ms")}, indent=2))


if __name__ == "__main__":
    main()
