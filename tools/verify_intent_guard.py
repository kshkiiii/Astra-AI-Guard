import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    from threadpoolctl import threadpool_limits
    from intent import SemanticSecurityModel
    from model import Guard, load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--head", choices=("mode_head", "operation_head"), default="mode_head")
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.organizer.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/intent-guard")
    args = parser.parse_args()
    config = load_config(args.config)
    config["paths"]["semantic_security"] = str(args.source.resolve())

    config["moderator"]["endpoint"] = "http://127.0.0.1:18083"
    guard = Guard.from_config(config)
    original = getattr(guard.inspector.model, args.head)
    candidate = SemanticSecurityModel.load(args.candidate, config["paths"]["encoder"])

    import joblib
    for name in ("mode_head", "operation_head", "threat_head", "category_head", "vectorizer"):
        if name == args.head:
            continue
        if joblib.hash(getattr(guard.inspector.model, name)) != joblib.hash(getattr(candidate, name)):
            raise ValueError(f"Unexpected changed component: {name}")
    rows = []
    for path in args.data:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(dict(json.loads(line), dataset=path.name))
    predictions, summaries = {}, {}
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        for name, head in (("before", original), ("after", getattr(candidate, args.head))):
            setattr(guard.inspector.model, args.head, head)
            guard.check("Покажи логи сервиса в dev", dict(target_agent="it_ops_agent", user_role="developer"))
            predictions[name], times = [], []
            for i, row in enumerate(rows):
                started = time.perf_counter()
                try:
                    result = guard.check(row["user_prompt"], row["context"])
                    prediction = dict(decision=result["decision"], risk_category=result["risk_category"])
                except Exception as exc:
                    prediction = dict(error=type(exc).__name__)
                times.append((time.perf_counter() - started) * 1000)
                predictions[name].append(prediction)
                if (i + 1) % 100 == 0:
                    print(f"{name}: {i + 1}/{len(rows)}", flush=True)
            by_dataset = {}
            for dataset in sorted({r["dataset"] for r in rows}):
                pairs = [(r, p) for r, p in zip(rows, predictions[name]) if r["dataset"] == dataset]
                by_dataset[dataset] = dict(n=len(pairs), correct=sum(r["expected_decision"] == p.get("decision") for r, p in pairs),
                                          unsafe_allowed=sum(r["expected_decision"] == "block" and p.get("decision") == "allow" for r, p in pairs))
            summaries[name] = dict(by_dataset=by_dataset, errors=dict(Counter(p["error"] for p in predictions[name] if "error" in p)),
                                   p50_ms=float(np.percentile(times, 50)), p95_ms=float(np.percentile(times, 95)))
            (args.output / f"{name}.jsonl").write_text("".join(json.dumps(dict(id=r["id"], dataset=r["dataset"], **p)) + "\n"
                                                              for r, p in zip(rows, predictions[name])), encoding="utf-8")
    changes = [dict(id=r["id"], dataset=r["dataset"], expected=r["expected_decision"], before=a, after=b)
               for r, a, b in zip(rows, predictions["before"], predictions["after"]) if a != b]
    report = dict(**summaries, n=len(rows), changes=changes, compared_head=args.head,
                  data_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.data},
                  source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                  note="Real moderator, organizer policy, sequential local CPU. Public dev contains training data. Authored controls are not an external benchmark.")
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if summaries["after"]["errors"]:
        raise RuntimeError("Candidate produced inference errors; do not activate")
    for row, before, after in zip(rows, predictions["before"], predictions["after"]):
        if (row["expected_decision"] == "block" and before.get("decision") == "block"
                and after.get("decision") != "block"):
            raise RuntimeError(f"Candidate relaxed a blocked control: {row['id']}; do not activate")
    for dataset, before in summaries["before"]["by_dataset"].items():
        after = summaries["after"]["by_dataset"][dataset]
        if after["correct"] < before["correct"] or after["unsafe_allowed"] > before["unsafe_allowed"]:
            raise RuntimeError(f"Candidate regressed on {dataset}; do not activate")


if __name__ == "__main__":
    main()
