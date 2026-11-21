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
    if "encoder" in config["paths"]:
        config["paths"]["encoder"] = str((path.parent / config["paths"]["encoder"]).resolve())
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
        weights = {"words": params.get("word_weight", 1.0), "chars": params.get("char_weight", 1.0)}
        if any(not isinstance(w, (int, float)) or not np.isfinite(w) or w < 0 for w in weights.values()) or not any(weights.values()):
            raise ValueError("Веса признаков должны быть конечными, неотрицательными и не все нулевыми")
        analyzer = params.get("char_analyzer", "char")
        if analyzer not in ("char", "char_wb"):
            raise ValueError("char_analyzer должен быть char или char_wb")
        if params.get("score_mode", "binary") not in ("binary", "binary_risk"):
            raise ValueError("score_mode должен быть binary или binary_risk")
        self.vectorizer = FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=tuple(params["word_ngram_range"]),
                                     max_features=params["word_max_features"], **common)),
            ("chars", TfidfVectorizer(analyzer=analyzer, ngram_range=tuple(params["char_ngram_range"]),
                                     max_features=params["char_max_features"], **common)),
        ], transformer_weights=weights)
        feature_mode = params.get("feature_mode", "tfidf")
        if feature_mode not in ("tfidf", "semantic", "hybrid", "ensemble"):
            raise ValueError("feature_mode должен быть tfidf, semantic, hybrid или ensemble")
        if feature_mode in ("semantic", "hybrid"):
            from semantic_features import LocalE5Transformer
            encoder_path = config["paths"].get("encoder", str(Path(__file__).parent / "artifacts/encoders/multilingual-e5-small"))
            # Динамическая INT8-квантизация зависит от состава batch. Один
            # текст на batch сохраняет одинаковые признаки при fit и инференсе.
            encoder = LocalE5Transformer(encoder_path, threads=params.get("encoder_threads", 4), batch_size=1)
            if feature_mode == "semantic":
                self.vectorizer = encoder
            else:
                self.vectorizer = FeatureUnion([("tfidf", self.vectorizer), ("semantic", encoder)],
                    transformer_weights={"tfidf": 1., "semantic": params.get("semantic_weight", 3.)})
        classifier = {"C": params["C"], "class_weight": params["class_weight"],
                      "max_iter": params["max_iter"], "random_state": config["data"]["random_state"]}
        self.attack_head = LogisticRegression(**classifier)
        self.category_head = LogisticRegression(**classifier)
        self.risk_head = LogisticRegression(**classifier)
        self.metadata = {}
        if feature_mode == "ensemble":
            fusion = params.get("fusion", "mean")
            if fusion not in ("mean", "rescue"):
                raise ValueError("fusion должен быть mean или rescue")
            self.members = []
            for member_mode, c in (("tfidf", params.get("base_C", 3.)), ("semantic", params.get("semantic_C", 300.))):
                member_config = copy.deepcopy(config)
                member_config["model"].update(feature_mode=member_mode, C=c, score_mode="binary_risk")
                self.members.append(GuardModel(member_config))

    def fit(self, rows: list[dict], sample_weight=None) -> "GuardModel":
        if any(row.get("provenance", {}).get("use") == "evaluation_only"
               or "evaluation_only" in row.get("tags", []) for row in rows):
            raise ValueError("Контрольные evaluation_only примеры запрещены для обучения")
        if self.config["model"].get("feature_mode") == "ensemble":
            for member in self.members:
                member.fit(rows, sample_weight=sample_weight)
            self.metadata = copy.deepcopy(self.members[0].metadata)
            self.metadata["member_parameters"] = [member.config["model"] for member in self.members]
            return self
        texts = [prepare_input(r["user_prompt"], r["context"], self.config["text"]).text for r in rows]
        # fit_transform вызывается исключительно на train, включая словарь и IDF.
        if self.config["model"].get("vocabulary_from_primary_only", False):
            primary = [text for text, row in zip(texts, rows) if not row.get("provenance")]
            if not primary:
                raise ValueError("Для vocabulary_from_primary_only нужны исходные обучающие примеры")
            self.vectorizer.fit(primary)
            features = self.vectorizer.transform(texts)
        else:
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
        if self.config["model"].get("feature_mode", "tfidf") in ("semantic", "hybrid"):
            self.metadata["packages"].update({p: importlib.metadata.version(p) for p in ("onnxruntime", "tokenizers")})
        return self

    def analyze(self, chunks: tuple[str, ...]) -> dict:
        if self.config["model"].get("feature_mode") == "ensemble":
            analyses = [member.analyze(chunks) for member in self.members]
            base, semantic = [a["attack_probability"] for a in analyses]
            params = self.config["model"]
            if params.get("fusion", "mean") == "rescue":
                base_threshold = params.get("base_threshold", .47)
                semantic_threshold = params.get("semantic_threshold", .75)
                if not 0 < base_threshold < 1 or not 0 < semantic_threshold < 1:
                    raise ValueError("Внутренние пороги ensemble должны быть между 0 и 1")
                score = min(1., .5 * max(base / base_threshold, semantic / semantic_threshold))
                # Прежний block сохраняет категорию; семантическая ветка
                # дополняет его, когда базовая оценка не достигает порога.
                winner = 0 if base >= base_threshold else 1
            else:
                weight = params.get("semantic_mix", .75)
                if not 0 <= weight <= 1:
                    raise ValueError("semantic_mix должен быть в [0, 1]")
                score = (1-weight)*base + weight*semantic
                winner = 0 if base >= semantic else 1
            result = dict(analyses[winner])
            result.update(attack_probability=float(score), tfidf_score=float(base), semantic_score=float(semantic))
            return result
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
        # Категория угрозы выбирается среди угроз, а не none/sensitive_action.
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
        model = artifact["model"]
        # Переносимый bundle: при наличии энкодера рядом с весами используем
        # локальную копию, а не абсолютный путь машины, где модель обучалась.
        local_encoder = Path(path).resolve().parent / "encoders/multilingual-e5-small"
        if (local_encoder / "model_qint8_avx512_vnni.onnx").exists():
            def rebind(transformer):
                if transformer.__class__.__name__ == "LocalE5Transformer":
                    transformer.directory = str(local_encoder)
                for _, child in getattr(transformer, "transformer_list", []):
                    rebind(child)
            def bind_member(member):
                rebind(member.vectorizer)
                for child in getattr(member, "members", []):
                    bind_member(child)
            bind_member(model)
        return model


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
