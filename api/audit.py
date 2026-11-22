from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path


class AuditLog:


    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event TEXT NOT NULL,
                    user_role TEXT NOT NULL,
                    target_agent TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    risk_category TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    policy_version TEXT NOT NULL
                )
            """)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=.1)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def append(self, entry: dict) -> None:
        with self._connect() as connection:
            connection.execute("""
                INSERT INTO events (request_id, timestamp, event, user_role, target_agent,
                    decision, risk_category, risk_level, latency_ms, policy_version)
                VALUES (:request_id, :timestamp, :event, :user_role, :target_agent,
                    :decision, :risk_category, :risk_level, :latency_ms, :policy_version)
            """, entry)

    def recent(self, limit: int = 50, before: int | None = None, decision: str | None = None,
               user_role=None, category=None, since=None, until=None) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("""
                SELECT * FROM events WHERE (? IS NULL OR id < ?)
                    AND (? IS NULL OR decision = ?)
                    AND (? IS NULL OR user_role = ?)
                    AND (? IS NULL OR risk_category = ?)
                    AND (? IS NULL OR timestamp >= ?)
                    AND (? IS NULL OR timestamp <= ?)
                ORDER BY id DESC LIMIT ?
            """, (before, before, decision, decision, user_role, user_role, category, category,
                    since, since, until, until, limit)).fetchall()
        return [dict(row) for row in rows]
