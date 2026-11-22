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
from scipy.sparse import vstack
from scipy.special import expit
from threadpoolctl import threadpool_limits

from intent import SemanticSecurityModel
from text_processing import normalize


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_splits(parts):
    for split, rows in parts.items():
        if not rows or any(r.get("split") != split or r.get("mode") not in {"execute", "informational"} for r in rows):
            raise ValueError(f"Invalid labeled split: {split}")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if {r["family_id"] for r in parts[a]} & {r["family_id"] for r in parts[b]}:
            raise ValueError(f"Overlapping families: {a}/{b}")
        if {normalize(r["text"]) for r in parts[a]} & {normalize(r["text"]) for r in parts[b]}:
            raise ValueError(f"Overlapping inputs: {a}/{b}")


def features(model, rows):
    batches = []
    for start in range(0, len(rows), 32):
        batch = rows[start:start + 32]
        batches.append(model.vectorizer.transform([normalize(r["text"]) for r in batch]))
        model.clear_cache()
        if start % 256 == 0:
            print(f"encoded {start}/{len(rows)}", flush=True)
    return vstack(batches, format="csr")


def metrics(head, matrix, rows):
    predictions = head.predict(matrix)
    gold = np.asarray([r["mode"] for r in rows])
    return dict(n=len(rows), correct=int((predictions == gold).sum()),
                accuracy=float((predictions == gold).mean()),
                execution_as_information=int(((gold == "execute") & (predictions == "informational")).sum()),
                information_as_execution=int(((gold == "informational") & (predictions == "execute")).sum()))


def fit_head(prior, matrix, rows, strength):
    if set(prior.classes_) != {"execute", "informational"} or prior.coef_.shape[0] != 1:
        raise ValueError("Expected the existing binary mode head")
    head = copy.deepcopy(prior)
    initial = np.r_[prior.coef_[0], prior.intercept_]
    target = np.asarray([r["mode"] == prior.classes_[1] for r in rows], dtype=float)
    weight = np.asarray([3. if r.get("provenance") else 1. for r in rows])

    def objective(params):
        logits = matrix @ params[:-1] + params[-1]
        delta = params - initial
        loss = np.sum(weight * (np.logaddexp(0., logits) - target * logits)) + .5 * strength * (delta @ delta)
        residual = weight * (expit(logits) - target)
        gradient = np.r_[np.asarray(matrix.T @ residual).ravel(), residual.sum()] + strength * delta
        return loss, gradient

    result = minimize(objective, initial, method="L-BFGS-B", jac=True,
                      options={"maxiter": 350, "gtol": 1e-5})
    if not result.success:
        raise RuntimeError(f"Mode training failed: {result.message}")
    head.coef_ = result.x[None, :-1]
    head.intercept_ = result.x[-1:]
    return head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "training/intent")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/intent-training.json")
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.source.resolve():
        raise ValueError("Write a candidate first; do not overwrite active weights")
    parts = {split: read(args.data / f"{split}.jsonl") for split in ("train", "validation", "test")}
    replay = {split: read(args.data / f"replay_{split}.jsonl") for split in ("train", "validation")}
    verify_splits({split: parts[split] + replay.get(split, []) for split in parts})
    model = SemanticSecurityModel.load(args.source, ROOT / "artifacts/encoders/multilingual-e5-small")
    train = replay["train"] + parts["train"]
    with threadpool_limits(limits=1):
        train_x = features(model, train)
        valid_x = features(model, parts["validation"])
        replay_x = features(model, replay["validation"])
        prior = model.mode_head
        before_valid = metrics(prior, valid_x, parts["validation"])
        before_replay = metrics(prior, replay_x, replay["validation"])
        candidates = []
        for strength in (1., 4., 16.):
            head = fit_head(prior, train_x, train, strength)
            validation = metrics(head, valid_x, parts["validation"])
            regression = metrics(head, replay_x, replay["validation"])
            record = dict(prior_strength=strength, validation=validation, regression=regression)
            print(json.dumps(record), flush=True)
            candidates.append((head, record))
        eligible = [(head, record) for head, record in candidates
                    if record["regression"]["correct"] >= before_replay["correct"]
                    and record["regression"]["execution_as_information"] <= before_replay["execution_as_information"]
                    and record["validation"]["correct"] > before_valid["correct"]
                    and record["validation"]["execution_as_information"] <= before_valid["execution_as_information"]]
        if not eligible:
            raise RuntimeError("No candidate improved validation without legacy regression; active weights preserved")
        head, selected = max(eligible, key=lambda pair: (pair[1]["validation"]["correct"], pair[1]["prior_strength"]))
        test_x = features(model, parts["test"])
        report = dict(before=dict(validation=before_valid, regression=before_replay,
                                  test=metrics(prior, test_x, parts["test"])),
                      after=dict(validation=selected["validation"], regression=selected["regression"],
                                 test=metrics(head, test_x, parts["test"])),
                      selected=selected, candidates=[record for _, record in candidates],
                      train_n=len(train), data_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.data.glob("*.jsonl"))},
                      source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                      changed="mode_head weights only; encoder, vocabulary, operations and threat heads frozen",
                      note="Authored synthetic data; families disjoint; test not used for selection. Not an independent benchmark.")
        model.mode_head = head
        model.metadata["intent_training"] = report


        model.vectorizer = SemanticSecurityModel.load(args.source, ROOT / "artifacts/encoders/multilingual-e5-small").vectorizer
        model.clear_cache()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"format": model.FORMAT, "model": model}, args.output, compress=3)
        report.update(output_bytes=args.output.stat().st_size, source_bytes=args.source.stat().st_size,
                      output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
