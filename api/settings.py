from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    config_path: Path = ROOT / "config.yaml"
    audit_path: Path = ROOT / "api/runtime/audit.sqlite3"
    cors_origins: tuple[str, ...] = ()
    max_body_bytes: int = 1_048_576
    confirmation_ttl_seconds: int = 600
    max_pending: int = 256
    demo_enabled: bool = True

    @classmethod
    def from_env(cls) -> Settings:
        def path(name: str, default: Path) -> Path:
            return (ROOT / os.environ.get(name, str(default))).resolve()

        return cls(
            config_path=path("GUARD_CONFIG", cls.config_path),
            audit_path=path("GUARD_AUDIT_PATH", cls.audit_path),
            cors_origins=tuple(s.strip() for s in os.getenv("GUARD_CORS_ORIGINS", "").split(",") if s.strip()),
            demo_enabled=os.getenv("GUARD_DEMO_ENABLED", "1") == "1",
        )
