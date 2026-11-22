
from __future__ import annotations

import copy
from pathlib import Path

import joblib
import numpy as np
import yaml

THREAT_CATEGORIES = frozenset({
    "jailbreak", "prompt_injection", "indirect_injection", "system_prompt_extraction",
    "system_prompt_override", "privilege_escalation", "data_exfiltration",
    "harmful_content", "fraud_social_engineering", "abuse",
})
FORMAT_VERSION = 1


def load_config(path: str | Path = "config.yaml") -> dict:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or not all(k in config for k in ("text", "model", "guard", "moderator", "paths")):
        raise ValueError("В конфигурации нужны text, model, guard, moderator, paths")
    config = copy.deepcopy(config)
    for key, value in config["paths"].items():
        config["paths"][key] = str((path.parent / value).resolve())
    for key in ("binary", "model"):
        config["moderator"][key] = str((path.parent / config["moderator"][key]).resolve())
    threshold = config["model"]["attack_threshold"]
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise ValueError("attack_threshold должен быть между 0 и 1")
    return config


class GuardModel:


    def analyze(self, chunks: tuple[str, ...]) -> dict:
        features = self.vectorizer.transform(chunks)
        scores = self.attack_head.predict_proba(features)[:, list(self.attack_head.classes_).index(1)]
        binary_scores = scores
        mode = self.config["model"].get("score_mode", "binary")
        if mode == "binary_risk":
            risk_probabilities = self.risk_head.predict_proba(features)
            if "none" not in self.risk_head.classes_:
                raise ValueError("Для binary_risk нужен класс риска none")
            none_index = list(self.risk_head.classes_).index("none")
            scores = np.sqrt(binary_scores * np.clip(1.0 - risk_probabilities[:, none_index], 0., 1.))
        elif mode != "binary":
            raise ValueError(f"Неизвестный score_mode: {mode}")
        index = int(np.argmax(scores))
        probabilities = self.category_head.predict_proba(features[index:index + 1])[0]

        candidates = [(float(p), str(c)) for p, c in zip(probabilities, self.category_head.classes_)
                      if c in THREAT_CATEGORIES]
        category = max(candidates)[1] if candidates else "prompt_injection"
        if chunks[index].startswith("[attachment]"):
            category = "indirect_injection"
        risk = str(self.risk_head.predict(features[index:index + 1])[0])
        result = {"attack_probability": float(scores[index]), "risk_category": category,
                  "risk_level": risk, "evidence_source": chunks[index].split("\n", 1)[0]}
        if mode == "binary_risk":
            result["binary_attack_probability"] = float(binary_scores[index])
            result["nonzero_risk_probability"] = float(1.0 - risk_probabilities[index, none_index])
        return result


    def clear_cache(self):
        def clear(transformer):
            if transformer.__class__.__name__ == "LocalE5Transformer":
                transformer.__dict__.pop("_cache", None)
            for _, child in getattr(transformer, "transformer_list", []):
                clear(child)
        clear(self.vectorizer)
        for member in getattr(self, "members", []):
            member.clear_cache()


    @staticmethod
    def load(path: str | Path, encoder: str | Path | None = None) -> "GuardModel":
        artifact = joblib.load(path)
        if artifact.get("format_version") != FORMAT_VERSION or not isinstance(artifact.get("model"), GuardModel):
            raise ValueError("Неподдерживаемый формат классификатора")
        model = artifact["model"]
        if getattr(model, "members", None):
            raise ValueError("Артефакт не относится к текущему backend")
        if encoder is None:
            encoder = Path(__file__).resolve().parent / "artifacts/encoders/multilingual-e5-small"
        def rebind(transformer):
            if transformer.__class__.__name__ == "LocalE5Transformer":
                transformer.directory = str(Path(encoder).resolve())
            for _, child in getattr(transformer, "transformer_list", []):
                rebind(child)
        rebind(model.vectorizer)
        model.clear_cache()
        return model


class Guard:
    @classmethod
    def from_config(cls, config, policy=None):
        if config["guard"].get("backend") != "context":
            raise ValueError("Текущая версия поддерживает только backend context")
        from context_guard import ContextGuard
        return ContextGuard(config, policy=policy)
