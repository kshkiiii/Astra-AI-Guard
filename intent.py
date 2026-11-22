
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np

from semantic_features import LocalE5Transformer
from text_processing import normalize

TOOLS = frozenset({"read_logs", "restart_service", "delete_resource", "run_shell", "write_repository"})
FORBIDDEN = frozenset({"disable_audit_or_wipe_logs", "pipe_remote_script_to_shell",
                       "add_ssh_keys_or_sudoers_via_chat", "send_secrets_or_data_to_external_hosts"})
OPERATIONS = TOOLS | FORBIDDEN | {"secret_values", "secret_names", "none"}


@dataclass(frozen=True)
class ActionIntent:
    operation: str
    object: str | None
    environment: str
    secrets: str
    mode: str
    source: str
    confidence: float

    def __post_init__(self):
        if (self.operation not in OPERATIONS or self.environment not in {"prod", "nonprod", "unknown"}
                or self.secrets not in {"none", "names", "values"} or self.mode not in {"execute", "informational"}
                or self.source not in {"current_user", "history_user", "history_assistant", "attachment"}
                or not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1):
            raise ValueError("Некорректная структура намерения")

    def to_dict(self):
        return asdict(self)


class SemanticSecurityModel:


    FORMAT = "semantic-security-v3"


    def predict(self, texts):
        features = self.vectorizer.transform([normalize(text) for text in texts])
        operations = self.operation_head.predict_proba(features)
        modes = self.mode_head.predict_proba(features)
        threats = self.threat_head.predict_proba(features)[:, list(self.threat_head.classes_).index(1)]
        categories = self.category_head.predict(features)
        output = []
        for i in range(len(texts)):
            op = int(operations[i].argmax())
            output.append({"operation": str(self.operation_head.classes_[op]), "confidence": float(operations[i, op]),
                           "execute_probability": float(modes[i, list(self.mode_head.classes_).index("execute")]),
                           "attack_probability": float(threats[i]), "risk_category": str(categories[i])})
        return output

    def clear_cache(self):
        for _, transformer in self.vectorizer.transformer_list:
            if isinstance(transformer, LocalE5Transformer):
                transformer.__dict__.pop("_cache", None)


    @classmethod
    def load(cls, path, encoder=None):
        artifact = joblib.load(path)
        if artifact.get("format") != cls.FORMAT or not isinstance(artifact.get("model"), cls):
            raise ValueError("Неизвестный формат семантической модели")
        model = artifact["model"]
        if encoder:
            for _, transformer in model.vectorizer.transformer_list:
                if isinstance(transformer, LocalE5Transformer):
                    transformer.directory = str(encoder)
        model.clear_cache()
        return model
