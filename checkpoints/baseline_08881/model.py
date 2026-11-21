"""Обучаемый TF-IDF baseline и единая точка проверки без HTTP-зависимостей."""
from __future__ import annotations

import copy
import importlib.metadata
import time
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from policy import SecurityPolicy
from text_processing import prepare_input


# Это таксономия семантической головы, а не матрица прав. Решение по каждой
# категории определяется действующим always_block из YAML во время проверки.
THREAT_CATEGORIES = frozenset({
    "jailbreak", "prompt_injection", "indirect_injection", "system_prompt_extraction",
    "system_prompt_override", "privilege_escalation", "data_exfiltration",
    "harmful_content", "fraud_social_engineering", "abuse",
})
FORMAT_VERSION = 1


def load_config(path: str | Path = "config.yaml") -> dict:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or not all(k in config for k in ("data", "text", "model", "paths")):
        raise ValueError("В конфигурации нужны data, text, model, paths")
    config = copy.deepcopy(config)
    for section, keys in (("data", ("source", "split_dir")),
                          ("paths", ("model", "policy", "evaluator", "reports"))):
        for key in keys:
            config[section][key] = str((path.parent / config[section][key]).resolve())
    config["data"]["extra_sources"] = [str((path.parent / p).resolve())
                                        for p in config["data"].get("extra_sources", [])]
    threshold = config["model"]["attack_threshold"]
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise ValueError("attack_threshold должен быть между 0 и 1")
    return config


class GuardModel:
    """Векторизатор общий; головы опасности, категории и уровня риска отдельные."""
    def __init__(self, config: dict):
        self.config = copy.deepcopy(config)
        params = config["model"]
        common = {"min_df": params["min_df"], "sublinear_tf": params["sublinear_tf"],
                  "dtype": np.float32, "lowercase": False}
        self.vectorizer = FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=tuple(params["word_ngram_range"]),
                                     max_features=params["word_max_features"], **common)),
            ("chars", TfidfVectorizer(analyzer="char", ngram_range=tuple(params["char_ngram_range"]),
                                     max_features=params["char_max_features"], **common)),
        ])
        classifier = {"C": params["C"], "class_weight": params["class_weight"],
                      "max_iter": params["max_iter"], "random_state": config["data"]["random_state"]}
        self.attack_head = LogisticRegression(**classifier)
        self.category_head = LogisticRegression(**classifier)
        self.risk_head = LogisticRegression(**classifier)
        self.metadata = {}

    def fit(self, rows: list[dict], sample_weight=None) -> "GuardModel":
        texts = [prepare_input(r["user_prompt"], r["context"], self.config["text"]).text for r in rows]
        # fit_transform вызывается исключительно на train, включая словарь и IDF.
        features = self.vectorizer.fit_transform(texts)
        categories = [r["risk_category"] for r in rows]
        attacks = [int(c in THREAT_CATEGORIES) for c in categories]
        if len(set(attacks)) < 2:
            raise ValueError("В train нужны и атаки, и легитимные запросы")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if (sample_weight.shape != (len(rows),) or not np.all(np.isfinite(sample_weight))
                    or not np.all(sample_weight > 0)):
                raise ValueError("Для каждой строки нужен конечный положительный sample_weight")
        self.attack_head.fit(features, attacks, sample_weight=sample_weight)
        self.category_head.fit(features, categories, sample_weight=sample_weight)
        self.risk_head.fit(features, [r["risk_level"] for r in rows], sample_weight=sample_weight)
        self.metadata.update({"train_ids": sorted(r["id"] for r in rows),
                              "packages": {p: importlib.metadata.version(p)
                                           for p in ("scikit-learn", "numpy", "scipy", "joblib", "PyYAML")}})
        return self

    def analyze(self, chunks: tuple[str, ...]) -> dict:
        features = self.vectorizer.transform(chunks)
        scores = self.attack_head.predict_proba(features)[:, list(self.attack_head.classes_).index(1)]
        index = int(np.argmax(scores))
        probabilities = self.category_head.predict_proba(features[index:index + 1])[0]
        # Категория угрозы выбирается среди угроз, а не none/sensitive_action.
        candidates = [(float(p), str(c)) for p, c in zip(probabilities, self.category_head.classes_)
                      if c in THREAT_CATEGORIES]
        category = max(candidates)[1] if candidates else "prompt_injection"
        if chunks[index].startswith("[attachment]"):
            category = "indirect_injection"
        risk = str(self.risk_head.predict(features[index:index + 1])[0])
        return {"attack_probability": float(scores[index]), "risk_category": category,
                "risk_level": risk, "evidence_source": chunks[index].split("\n", 1)[0]}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        joblib.dump({"format_version": FORMAT_VERSION, "model": self}, temporary, compress=3)
        temporary.replace(path)

    @staticmethod
    def load(path: str | Path) -> "GuardModel":
        # joblib/pickle загружаем только из собственного доверенного артефакта.
        artifact = joblib.load(path)
        if artifact.get("format_version") != FORMAT_VERSION or not isinstance(artifact.get("model"), GuardModel):
            raise ValueError("Неподдерживаемый формат модели; переобучите через train.py")
        return artifact["model"]


class Guard:
    def __init__(self, model: GuardModel, policy: SecurityPolicy, attack_threshold: float | None = None):
        self.model, self.policy = model, policy
        self.threshold = model.config["model"]["attack_threshold"] if attack_threshold is None else attack_threshold
        if not 0 < self.threshold < 1:
            raise ValueError("Порог должен быть между 0 и 1")

    def decide(self, analysis: dict, policy_result: dict, threshold: float | None = None) -> dict:
        """Общее правило для инференса и перебора порогов по сохранённым оценкам."""
        threshold = self.threshold if threshold is None else threshold
        if not 0 < threshold < 1:
            raise ValueError("Порог должен быть между 0 и 1")
        category = analysis["risk_category"]
        if analysis["attack_probability"] >= threshold and category in self.policy.rules["always_block"]:
            risk = analysis["risk_level"]
            if risk in ("none", "low"):
                risk = "high"
            result = {"decision": "block", "risk_category": category, "risk_level": risk,
                      "explanation": f"Модель выявила {category}; категория запрещена always_block в YAML-политике."}
        else:
            result = dict(policy_result)
        return result

    def check(self, user_prompt: str, context: dict, diagnostics: bool = False) -> dict:
        started = time.perf_counter()
        prepared = prepare_input(user_prompt, context, self.model.config["text"])
        analysis = self.model.analyze(prepared.chunks)
        policy_result = self.policy.evaluate(user_prompt, context, self.model.config["text"])
        result = self.decide(analysis, policy_result)
        result.update({"latency_ms": round((time.perf_counter() - started) * 1000, 3),
                       "policy_version": self.policy.version,
                       "attack_probability": analysis["attack_probability"]})
        if diagnostics:
            # Без текста и секретов; этого достаточно для точного повторения
            # решений при любом пороге, включая независимые блокировки политикой.
            result["threshold_inputs"] = {"analysis": analysis, "policy_result": policy_result}
        return result
