import json
from pathlib import Path
import unittest

from threadpoolctl import threadpool_limits

from intent import SemanticSecurityModel
from model import load_config
from tools.train_intent import verify_splits

ROOT = Path(__file__).resolve().parents[1]


class IntentTrainingTests(unittest.TestCase):
    def test_training_validation_and_test_have_disjoint_families_and_inputs(self):
        for name in ("intent", "database-intent"):
            with self.subTest(dataset=name):
                directory = ROOT / "training" / name
                parts = {}
                for split in ("train", "validation", "test"):
                    paths = [directory / f"{split}.jsonl"]
                    if split != "test":
                        paths.append(directory / f"replay_{split}.jsonl")
                    parts[split] = [json.loads(line) for path in paths for line in path.read_text(encoding="utf-8").splitlines()]
                    self.assertTrue(parts[split])
                    self.assertTrue(all(row["split"] == split for row in parts[split]))
                verify_splits(parts)

    def test_learned_head_understands_heldout_discussion_and_execution(self):
        config = load_config(ROOT / "config.yaml")
        model = SemanticSecurityModel.load(config["paths"]["semantic_security"], config["paths"]["encoder"])
        rows = [json.loads(line) for line in (ROOT / "training/intent/test.jsonl").read_text(encoding="utf-8").splitlines()]
        try:
            with threadpool_limits(limits=1):
                predictions = model.predict([row["text"] for row in rows])
            actual = ["execute" if p["execute_probability"] >= .5 else "informational" for p in predictions]
            self.assertGreaterEqual(sum(p == r["mode"] for p, r in zip(actual, rows)) / len(rows), .9)
            self.assertFalse(any(p == "informational" and r["mode"] == "execute" for p, r in zip(actual, rows)))
        finally:
            model.clear_cache()

    def test_learned_operation_recognizes_database_deletion(self):
        config = load_config(ROOT / "config.yaml")
        model = SemanticSecurityModel.load(config["paths"]["semantic_security"], config["paths"]["encoder"])
        try:
            with threadpool_limits(limits=1):
                predictions = model.predict(["удали базу данных users (dev)", 'DROP DATABASE IF EXISTS "users"; -- dev'])
            for prediction in predictions:
                self.assertEqual(prediction["operation"], "delete_resource")
                self.assertGreaterEqual(prediction["confidence"], config["guard"]["operation_threshold"])
            with threadpool_limits(limits=1):
                metadata = model.predict(["dev", "-- dev"])
            self.assertTrue(all(p["operation"] == "none" for p in metadata))
        finally:
            model.clear_cache()


if __name__ == "__main__":
    unittest.main()
