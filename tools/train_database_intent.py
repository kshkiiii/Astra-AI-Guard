import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from threadpoolctl import threadpool_limits

from intent import SemanticSecurityModel
from tools.train_intent import features, read, verify_splits

TEMPLATES = {
    "train": ["Сотри базу данных {name} ({env})", "Дропни БД {name} в {env}",
              "Удали database {name} ({env})", "DROP DATABASE {name}; -- {env}",
              "Delete the database {name} in {env}", "Remove database {name} ({env})",
              "DROP TABLE IF EXISTS {name}; -- {env}", "Уничтожь таблицу {name} в {env}"],
    "validation": ["Уничтожь базу данных {name} ({env})", "Please drop database {name} in {env}",
                   "Удали БД {name} в окружении {env}", "DROP SCHEMA IF EXISTS {name}; -- {env}"],
    "test": ["Удали базу данных {name} ({env})", "Срочно сотри БД {name} в {env}",
             "Now delete database {name} ({env})", "DROP DATABASE IF EXISTS \"{name}\"; -- {env}"],
}


def authored(split):
    names = {"train": ("ledger", "catalog", "metrics", "inventory"),
             "validation": ("billing", "reporting"), "test": ("users", "archive")}[split]
    rows = [dict(text=template.format(name=name, env=env), operation="delete_resource", mode="execute", split=split,
                 family_id=f"database-{split}-{i}", provenance="authored synthetic database deletion")
            for i, template in enumerate(TEMPLATES[split]) for name in names for env in ("dev", "production")]
    negative_templates = {
        "train": ["{env}", "-- {env}", "# {env}", "Окружение: {env}"],
        "validation": ["-- environment: {env}", "Текущая среда {env}"],
        "test": ["/* environment: {env} */", "Только имя окружения: {env}"],
    }
    rows += [dict(text=template.format(env=env), operation="none", mode="informational", split=split,
                  family_id=f"database-negative-{split}-{i}", provenance="authored nonexecuting environment metadata")
             for i, template in enumerate(negative_templates[split]) for env in ("dev", "production", "staging", "test")]
    return rows


def operation_metrics(head, matrix, rows):
    probabilities = head.predict_proba(matrix)
    labels = head.classes_[probabilities.argmax(axis=1)]
    gold = np.asarray([r["operation"] for r in rows])
    confidence = probabilities.max(axis=1)
    return dict(n=len(rows), correct=int((labels == gold).sum()),
                recognized=int(((labels == gold) & (confidence >= .35)).sum()),
                min_confidence=float(confidence.min()))


def fit_delete(prior, matrix, rows, strength):
    index = list(prior.classes_).index("delete_resource")
    initial = np.r_[prior.coef_[index], prior.intercept_[index]]
    other = [i for i in range(len(prior.classes_)) if i != index]
    offset = logsumexp(matrix @ prior.coef_[other].T + prior.intercept_[other], axis=1)


    target = expit(matrix @ prior.coef_[index] + prior.intercept_[index] - offset)
    authored_rows = np.asarray([bool(r.get("provenance")) for r in rows])
    labels = np.asarray([r["operation"] == "delete_resource" for r in rows], dtype=float)
    target[authored_rows] = labels[authored_rows]
    weight = np.where(authored_rows, 3., 2.)

    def objective(params):
        logits = matrix @ params[:-1] + params[-1] - offset
        delta = params - initial
        loss = np.sum(weight * (np.logaddexp(0., logits) - target * logits)) + .5 * strength * (delta @ delta)
        residual = weight * (expit(logits) - target)
        gradient = np.r_[np.asarray(matrix.T @ residual).ravel(), residual.sum()] + strength * delta
        return loss, gradient

    result = minimize(objective, initial, method="L-BFGS-B", jac=True, options={"maxiter": 350})
    if not result.success:
        raise RuntimeError(result.message)
    head = copy.deepcopy(prior)
    head.coef_[index] = result.x[:-1]
    head.intercept_[index] = result.x[-1]
    return head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", type=Path, default=ROOT / "old/additional_datasets/security_v3")
    parser.add_argument("--data", type=Path, default=ROOT / "training/database-intent")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/database-intent-training.json")
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.source.resolve():
        raise ValueError("Write a new candidate; preserve the active model")
    model = SemanticSecurityModel.load(args.source, ROOT / "artifacts/encoders/multilingual-e5-small")
    allowed = set(model.metadata["semantic_train_ids"])
    parts = {split: authored(split) for split in TEMPLATES}
    replay = {}
    for split in ("train", "validation"):
        replay[split] = [{key: row[key] for key in ("text", "operation", "mode", "family_id", "split")}
                         for row in read(args.replay / f"{split}.jsonl") if split != "train" or row["id"] in allowed]
    verify_splits({split: parts[split] + replay.get(split, []) for split in parts})
    args.data.mkdir(parents=True, exist_ok=True)
    for name, rows in {**parts, **{f"replay_{k}": v for k, v in replay.items()}}.items():
        (args.data / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with threadpool_limits(limits=1):
        train = replay["train"] + parts["train"]
        train_x = features(model, train)
        valid_x = features(model, parts["validation"])
        replay_x = features(model, replay["validation"])
        prior = model.operation_head
        before_valid = operation_metrics(prior, valid_x, parts["validation"])
        before_replay = operation_metrics(prior, replay_x, replay["validation"])
        print(json.dumps(dict(before_validation=before_valid, before_regression=before_replay)), flush=True)
        baseline_labels = prior.predict(replay_x)
        gold = np.asarray([r["operation"] for r in replay["validation"]])
        candidates = []
        for strength in (.01, .05, .2, 1.):
            head = fit_delete(prior, train_x, train, strength)
            regression = operation_metrics(head, replay_x, replay["validation"])
            validation = operation_metrics(head, valid_x, parts["validation"])
            introduced = int(((baseline_labels == gold) & (head.predict(replay_x) != gold)).sum())
            record = dict(prior_strength=strength, validation=validation, regression=regression, introduced_errors=introduced)
            print(json.dumps(record), flush=True)
            candidates.append((head, record))
        eligible = [(h, r) for h, r in candidates if not r["introduced_errors"]
                    and r["validation"]["recognized"] > before_valid["recognized"]
                    and r["regression"]["recognized"] >= before_replay["recognized"]]
        if not eligible:
            raise RuntimeError("No safe validation improvement; active model unchanged")
        head, selected = max(eligible, key=lambda pair: (pair[1]["validation"]["recognized"], pair[1]["prior_strength"]))
        test_x = features(model, parts["test"])
        report = dict(train_n=len(train), before=dict(validation=before_valid, regression=before_replay,
                                                      test=operation_metrics(prior, test_x, parts["test"])),
                      after=dict(validation=selected["validation"], regression=selected["regression"],
                                 test=operation_metrics(head, test_x, parts["test"])), selected=selected,
                      candidates=[r for _, r in candidates],
                      source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                      data_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.data.glob("*.jsonl"))},
                      changed="Only the delete_resource row in the existing operation_head; all other weights frozen",
                      note="Synthetic family-disjoint data; original replay overlap exclusions preserved; test not used for selection")
        model.operation_head = head
        model.metadata["database_intent_training"] = report
        model.vectorizer = SemanticSecurityModel.load(args.source, ROOT / "artifacts/encoders/multilingual-e5-small").vectorizer
        model.clear_cache()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"format": model.FORMAT, "model": model}, args.output, compress=3)
        report.update(output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      source_bytes=args.source.stat().st_size, output_bytes=args.output.stat().st_size)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
