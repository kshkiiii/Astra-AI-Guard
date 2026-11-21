"""Проверка одного запроса или JSONL-набора; метрики считает исходный evaluator."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

from model import Guard, GuardModel, load_config
from policy import SecurityPolicy


def load_evaluator(path: str):
    spec = importlib.util.spec_from_file_location("organizer_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_threshold(rows: list[dict], predictions: list[dict], guard: Guard, evaluator) -> dict:
    """Полный перебор различных вариантов решений, без повторного инференса.

    Поиск разрешён только на неизменной validation текущей модели. Результат
    выбирается по Score при strict block recall high/critical >= 0.85.
    """
    ids = {r["id"] for r in rows}
    if ids != set(guard.model.metadata.get("validation_ids", [])) or ids & set(guard.model.metadata["train_ids"]):
        raise ValueError("Подбор разрешён только на validation текущей модели, без пересечения с train")
    if len(ids) != len(rows) or len(predictions) != len(rows) or {p["id"] for p in predictions} != ids:
        raise ValueError("Предсказания должны точно соответствовать строкам validation")
    scores = sorted({p["threshold_inputs"]["analysis"]["attack_probability"]
                     for p in predictions if "threshold_inputs" in p})
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
        raise ValueError("Некорректная вероятность модели")
    boundaries = sorted({0.0, 1.0, *scores})
    # Между соседними вероятностями решения постоянны. Середины интервалов
    # покрывают все варианты и не ставят порог вплотную к одному примеру.
    thresholds = {guard.threshold, *(p for p in scores if 0 < p < 1)}
    thresholds.update((a + b) / 2 for a, b in zip(boundaries, boundaries[1:]) if a < b)
    candidates = []
    for threshold in sorted(t for t in thresholds if 0 < t < 1):
        replay = []
        for prediction in predictions:
            if "threshold_inputs" in prediction:
                saved = prediction["threshold_inputs"]
                result = guard.decide(saved["analysis"], saved["policy_result"], threshold)
                result.update(id=prediction["id"], latency_ms=prediction["latency_ms"])
            else:
                # Невалидный вход остаётся ошибкой при любом пороге.
                result = dict(prediction)
            replay.append(result)
        report = evaluator.score(rows, replay)
        candidates.append({"threshold": threshold, **{k: report[k] for k in (
            "score", "balanced_cost_score", "strict_block_recall_high_critical", "eligible_for_prizes",
            "hard_negative_block_rate", "unnecessary_confirmation_rate", "macro_f1_decisions")}})
    feasible = [c for c in candidates if c["eligible_for_prizes"]]
    def rank(candidate):
        return (candidate["score"], -candidate["hard_negative_block_rate"],
                -candidate["unnecessary_confirmation_rate"], -abs(candidate["threshold"] - guard.threshold))
    return {"selection_split": "validation", "current_threshold": guard.threshold,
            "minimum_high_critical_recall": .85, "applied": False,
            "recommended": max(feasible, key=rank) if feasible else None,
            "best_unconstrained": max(candidates, key=rank), "candidates": candidates,
            "timing_note": "Офлайн-перебор использует задержки исходного прогона, без повторного замера."}


def evaluate_rows(rows: list[dict], guard: Guard, config: dict, output_dir: Path,
                  tune_threshold: bool = False) -> dict:
    from train import write_json, write_jsonl
    evaluator = load_evaluator(config["paths"]["evaluator"])
    predictions, errors, details = [], [], []
    for row in rows:
        started = time.perf_counter()
        try:
            prediction = guard.check(row["user_prompt"], row["context"], diagnostics=True)
        except (ValueError, TypeError, KeyError) as exc:
            prediction = {"_error": type(exc).__name__, "decision": "invalid",
                          "latency_ms": round((time.perf_counter() - started) * 1000, 3)}
        prediction["id"] = row["id"]
        predictions.append(prediction)
        details.append({"id": row["id"], "expected_decision": row["expected_decision"],
                        "expected_category": row["risk_category"], "is_dangerous": row["is_dangerous"],
                        "expected_risk_level": row["risk_level"], "prediction": prediction})
        if prediction["decision"] != row["expected_decision"] or prediction.get("risk_category") != row["risk_category"]:
            # Без исходных текстов/секретов: строки для разбора ищутся по id.
            errors.append({"id": row["id"], "expected_decision": row["expected_decision"],
                           "expected_category": row["risk_category"], "prediction": prediction})
    report = evaluator.score(rows, predictions)
    report.update({"measurement_mode": "offline_unverified_timings",
                   "cpu_bonus_requires_organizer_hardware_verification": True,
                   "overlap_with_training": len({r["id"] for r in rows} & set(guard.model.metadata["train_ids"])),
                   "attack_threshold": guard.threshold, "policy_version": guard.policy.version})
    if tune_threshold:
        search = select_threshold(rows, predictions, guard, evaluator)
        write_json(output_dir / "threshold_search.json", search)
        report["threshold_search"] = {k: v for k, v in search.items() if k != "candidates"}
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_jsonl(output_dir / "errors.jsonl", errors)
    write_jsonl(output_dir / "details.jsonl", details)
    write_json(output_dir / "report.json", report)
    return report


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Один запрос")
    source.add_argument("--input", type=Path, help="JSON: user_prompt и context")
    source.add_argument("--data", type=Path, help="Размеченный JSONL-набор")
    source.add_argument("--split", choices=("train", "validation", "test"), help="Часть фиксированного разбиения; по умолчанию validation")
    parser.add_argument("--role", default="developer", choices=("developer", "sre", "admin"))
    parser.add_argument("--threshold", type=float, help="Переопределить порог без переобучения")
    parser.add_argument("--model", type=Path, help="Проверить другой файл весов, сохранив текущий конфиг и порог")
    parser.add_argument("--tune-threshold", action="store_true", help="Подобрать порог только на validation (не применять автоматически)")
    parser.add_argument("--policy", type=Path, help="Другая YAML-политика без переобучения")
    parser.add_argument("--output-dir", type=Path, help="Куда сохранить отчёт и предсказания")
    args = parser.parse_args()
    if args.tune_threshold and (args.text is not None or args.input or args.data or args.split in ("train", "test")):
        parser.error("--tune-threshold используется только с --split validation (или без --split)")
    config = load_config(args.config)
    model_path = args.model or Path(config["paths"]["model"])
    if not model_path.exists():
        parser.error("Сначала обучите модель: python train.py")
    model = GuardModel.load(model_path)
    threshold = config["model"]["attack_threshold"] if args.threshold is None else args.threshold
    guard = Guard(model, SecurityPolicy(args.policy or config["paths"]["policy"]), threshold)
    if args.text is not None or args.input:
        if args.input:
            request = json.loads(args.input.read_text(encoding="utf-8-sig"))
        else:
            request = {"user_prompt": args.text, "context": {"target_agent": "it_ops_agent", "user_role": args.role,
                                                           "history": [], "attachments": []}}
        result = guard.check(request["user_prompt"], request["context"])
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        from train import read_rows
        part = args.split or "validation"
        path = args.data or Path(config["data"]["split_dir"]) / f"{part}.jsonl"
        output_dir = args.output_dir or Path(config["paths"]["reports"]) / (args.data.stem if args.data else part)
        report = evaluate_rows(read_rows(path), guard, config, output_dir, tune_threshold=args.tune_threshold)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Отчёт, предсказания и ошибки: {output_dir}")
        if args.tune_threshold:
            recommended = report["threshold_search"]["recommended"]
            if recommended:
                print(f"Рекомендуемый порог на validation: {recommended['threshold']:.12g} (не применён).")
                print(f"При нём Score={recommended['score']:.4f}; high/critical={recommended['strict_block_recall_high_critical']:.2%}.")
            else:
                print("На validation не найден порог, достигающий 85% блокировок high/critical.")
        recall = report["strict_block_recall_high_critical"]
        print("\n" + "=" * 64)
        print(f"  Общий Score:                  {report['score']:.4f} / 1.0000")
        print(f"  Блокировки high/critical:     {recall:.2%}")
        print("=" * 64)


if __name__ == "__main__":
    main()
