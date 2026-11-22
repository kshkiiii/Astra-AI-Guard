import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18090/v1/check")
    parser.add_argument("--data", type=Path, default=Path("evaluation/dev.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("organizer_evaluator", Path(__file__).resolve().parents[1]/"evaluation/evaluate.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    rows = evaluator.load_jsonl(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    warmups = [evaluator.call_endpoint(args.endpoint, row) for row in rows[:5]]
    if any(p.get("_error") for p in warmups):
        raise RuntimeError("Warmup failed")
    predictions = []
    started = time.perf_counter()
    for row in rows:
        predictions.append(evaluator.call_endpoint(args.endpoint, row))
        if len(predictions) % 100 == 0:
            print(f"{len(predictions)}/{len(rows)}", flush=True)
    report = evaluator.score(rows, predictions)
    report.update(measurement_mode="endpoint", warmup_requests=5, sequential=True,
                  elapsed_seconds=time.perf_counter()-started,
                  cpu_bonus_requires_organizer_hardware_verification=True,
                  quality_note="Public dev includes training rows and is not independent quality evidence.")
    root = Path(__file__).resolve().parents[1]
    files = [*root.glob("*.py"), *root.glob("*.yaml"), *(root/"api").glob("*.py"), args.data]
    report["source_sha256"] = {str(p.relative_to(root) if p.is_relative_to(root) else p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (args.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (args.output/"predictions.jsonl").write_text("".join(json.dumps(p,ensure_ascii=False)+"\n" for p in predictions),encoding="utf-8")
    errors = [dict(id=row["id"], prompt=row["user_prompt"], expected=row["expected_decision"], predicted=p.get("decision"), error=p.get("_error")) for row,p in zip(rows,predictions) if row["expected_decision"] != p.get("decision")]
    (args.output/"errors.json").write_text(json.dumps(errors,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "source_sha256"},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
