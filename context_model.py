from pathlib import Path
import json
import re
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

LABELS = ("information", "execute", "review", "threat")


def request_frame(text):
    from request_framing import QUOTES
    text = QUOTES.sub(" [reference] ", text)
    return re.sub(r"(?m)^\s*(?:user|assistant|system):.*$", " [reference] ", text)


class ContextModel:
    def __init__(self, directory, threads=4):
        path = Path(directory)
        self.metadata = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path / self.metadata["onnx"]), sess_options=options, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256, stride=48)
        pad = "[PAD]" if self.tokenizer.token_to_id("[PAD]") is not None else "<pad>"
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id(pad), pad_token=pad)

    def predict(self, texts):
        results = []
        for text in texts:
            from inference_budget import check_budget
            check_budget()
            encoding = self.tokenizer.encode(text)
            scores = []
            for window in [encoding, *encoding.overflowing]:
                ids = np.array([window.ids], dtype=np.int64)
                mask = np.array([window.attention_mask], dtype=np.int64)
                logits = self.session.run(None, dict(input_ids=ids, attention_mask=mask))[0][0].astype(float)
                probs = np.exp(logits-logits.max())
                scores.append(probs/probs.sum())
                check_budget()

            probs = np.asarray(scores)
            labels = self.metadata["labels"]
            result = dict.fromkeys(LABELS, 0.)
            result.update(zip(labels, probs.max(axis=0).tolist()))
            safe_columns = [labels.index(label) for label in ("information", "review") if label in labels]
            result["safe_probability"] = float(probs[:, safe_columns].sum(axis=1).min())
            result["windows"] = len(scores)
            result["label"] = labels[int(np.argmax(probs.max(axis=0)))]
            results.append(result)
        return results

    def probability(self, texts):
        return np.asarray([max(p["information"], p["review"]) for p in self.predict(texts)])

    def clear_cache(self):
        pass
