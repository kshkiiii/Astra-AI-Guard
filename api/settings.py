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
    admin_token: str = ""
    admin_user: str = "operator"
    session_ttl_seconds: int = 86400
    session_max_chars: int = 100_000
    cookie_secure: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        def path(name: str, default: Path) -> Path:
            return (ROOT / os.environ.get(name, str(default))).resolve()

        return cls(
            config_path=path("GUARD_CONFIG", cls.config_path),
            audit_path=path("GUARD_AUDIT_PATH", cls.audit_path),
            cors_origins=tuple(s.strip() for s in os.getenv("GUARD_CORS_ORIGINS", "").split(",") if s.strip()),
            demo_enabled=os.getenv("GUARD_DEMO_ENABLED", "1") == "1",
            admin_token=os.getenv("GUARD_ADMIN_TOKEN", ""),
            admin_user=os.getenv("GUARD_ADMIN_USER", "operator"),
            session_ttl_seconds=max(60, min(604800, int(os.getenv("GUARD_SESSION_TTL", "86400")))),
            cookie_secure=os.getenv("GUARD_COOKIE_SECURE", "0") == "1",
        )
