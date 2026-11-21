"""Локальный замороженный E5: ONNX на CPU, без сетевых вызовов при работе."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


@lru_cache(maxsize=2)
def _runtime(directory: str, threads: int, max_length: int):
    import onnxruntime as ort
    from tokenizers import Tokenizer
    path = Path(directory)
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path / "model_qint8_avx512_vnni.onnx"),
                                   sess_options=options, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=max_length, stride=64)
    tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
    return session, tokenizer


class LocalE5Transformer(TransformerMixin, BaseEstimator):
    def __init__(self, directory: str, threads: int = 4, max_length: int = 512, batch_size: int = 1):
        self.directory = directory
        self.threads = threads
        self.max_length = max_length
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_cache"}

    def transform(self, X):
        if self.batch_size != 1:
            raise ValueError("Для воспроизводимых INT8-признаков требуется batch_size=1")
        texts = list(X)
        cache = self.__dict__.setdefault("_cache", {})
        missing = list(dict.fromkeys(text for text in texts if text not in cache))
        if missing:
            session, tokenizer = _runtime(str(Path(self.directory).resolve()), self.threads, self.max_length)
            inputs = {value.name for value in session.get_inputs()}
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start:start + self.batch_size]
                originals = tokenizer.encode_batch(["query: " + text for text in batch])
                windows, owners = [], []
                for owner, encoding in enumerate(originals):
                    for window in [encoding, *encoding.overflowing]:
                        windows.append(window)
                        owners.append(owner)
                # Обрабатываем overflow-окна тоже, а не теряем хвост при 512 токенах.
                pooled = []
                for offset in range(0, len(windows), self.batch_size):
                    part = windows[offset:offset + self.batch_size]
                    length = max(len(e.ids) for e in part)
                    ids = np.full((len(part), length), 1, dtype=np.int64)
                    mask = np.zeros_like(ids)
                    for i, encoding in enumerate(part):
                        ids[i, :len(encoding.ids)] = encoding.ids
                        mask[i, :len(encoding.attention_mask)] = encoding.attention_mask
                    feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
                    hidden = session.run(None, {k: v for k, v in feed.items() if k in inputs})[0]
                    averaged = (hidden * mask[..., None]).sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
                    pooled.extend(averaged)
                vectors = np.asarray(pooled, dtype=np.float32)
                for owner, text in enumerate(batch):
                    vector = vectors[np.asarray(owners) == owner].mean(axis=0)
                    cache[text] = vector / max(float(np.linalg.norm(vector)), 1e-12)
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        return np.asarray([cache[text] for text in texts], dtype=np.float32)
