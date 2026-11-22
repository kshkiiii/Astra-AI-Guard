"""Check requests or an archived test split with the current backend."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

from threadpoolctl import threadpool_limits

from model import Guard, load_config
from policy import SecurityPolicy


def evaluate_split(guard, split):
    root = Path(__file__).resolve().parent
    evaluator_path = root / "old/public_dataset/astra_public_v2/evaluate.py"
    data_path = root / "old/artifacts/splits" / f"{split}.jsonl"
    if not evaluator_path.is_file() or not data_path.is_file():
        raise FileNotFoundError("Для --split нужны сохранённые данные и evaluator в old/.")
    spec = importlib.util.spec_from_file_location("organizer_evaluator", evaluator_path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    rows = evaluator.load_jsonl(data_path)
    predictions = []
    with threadpool_limits(1):
        for row in rows:
            started = time.perf_counter()
            try:
                result = guard.check(row["user_prompt"], row["context"])
            except Exception as exc:
                result = {"_error": type(exc).__name__}
            result.update(id=row["id"], latency_ms=(time.perf_counter()-started)*1000)
            predictions.append(result)
    report = evaluator.score(rows, predictions)
    report.update(measurement_mode="offline_unverified_timings",
                  correct_decisions=sum(p.get("decision") == r["expected_decision"] for r,p in zip(rows,predictions)))
    destination = root / "old/reports/cli" / split
    destination.mkdir(parents=True, exist_ok=True)
    (destination/"predictions.jsonl").write_text("".join(json.dumps(p,ensure_ascii=False)+"\n" for p in predictions),encoding="utf-8")
    (destination/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print(f"Решения: {report['correct_decisions']}/{len(rows)}; Score: {report['score']:.4f}")
    print(f"Отчёт: {destination}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Текст запроса")
    source.add_argument("--input", type=Path, help="JSON с user_prompt и context")
    source.add_argument("--split", type=lambda value: value.rstrip("/\\"), choices=("train", "validation", "test"),
                        help="Проверить сохранённое разбиение из old/")
    parser.add_argument("--role", choices=("developer", "sre", "admin"), default="developer")
    parser.add_argument("--policy", type=Path, help="Другая YAML-политика")
    args = parser.parse_args()
    config = load_config(args.config)
    guard = Guard.from_config(config, policy=SecurityPolicy(args.policy) if args.policy else None)
    if args.split:
        evaluate_split(guard, args.split)
        return
    if args.input:
        request = json.loads(args.input.read_text(encoding="utf-8-sig"))
    else:
        request = dict(user_prompt=args.text, context=dict(target_agent="it_ops_agent", user_role=args.role,
            history=[], attachments=[]))
    result = guard.check(request["user_prompt"], request["context"])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
