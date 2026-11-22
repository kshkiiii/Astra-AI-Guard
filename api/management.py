from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import yaml
from fastapi import Cookie, Header, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from api.schemas import Attachment, CheckRequest, Message
from policy import SecurityPolicy


class PolicyEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    yaml: str = Field(max_length=64000)
    revision: str = Field(max_length=64)


class SessionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(default="Новый диалог", min_length=1, max_length=80)
    role: Literal["developer", "sre", "admin"] = "developer"


class SessionTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_prompt: str = Field(min_length=1, max_length=200000)
    attachments: list[Attachment] = Field(default_factory=list, max_length=100)
    send: bool = False
    revision: int = Field(ge=0)


def masked(text):
    text = re.sub(r"(?im)(\b(?:password|passwd|token|api[_-]?key|secret|пароль|токен)\b\s*[:=]\s*)[^\s,;]+", r"\1[скрыто]", text)
    text = re.sub(r"\b(?:sk-[A-Za-z0-9_-]{16,}|AKIA[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{20,})\b", "[скрыто]", text)
    return re.sub(r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----", "[закрытый ключ скрыт]", text, flags=re.S)


def register_management(app, service, settings):
    lock = threading.RLock()

    @contextmanager
    def db():
        conn = sqlite3.connect(service().audit.path, timeout=2)
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE IF NOT EXISTS policy_changes (id INTEGER PRIMARY KEY, timestamp TEXT, author TEXT, version INTEGER, revision TEXT, yaml TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, owner TEXT, title TEXT, role TEXT, updated REAL, revision INTEGER, messages TEXT)")
        conn.execute("DELETE FROM sessions WHERE updated < ?", (time.time()-settings.session_ttl_seconds,))
        conn.commit()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def admin(authorization):
        if not settings.admin_token or len(settings.admin_token) < 24:
            raise HTTPException(503, detail={"error": "admin_disabled"})
        supplied = authorization.removeprefix("Bearer ") if authorization.startswith("Bearer ") else ""
        if not secrets.compare_digest(supplied, settings.admin_token):
            raise HTTPException(401, detail={"error": "admin_unauthorized"})
        return settings.admin_user

    def snapshot():
        policy = service().guard.policy
        text = yaml.safe_dump(policy.rules, allow_unicode=True, sort_keys=False)
        return dict(yaml=text, rules=policy.rules, version=policy.version, revision=hashlib.sha256(text.encode()).hexdigest())

    def parse(text):
        try:
            return SecurityPolicy.from_text(text, service().guard.policy.path)
        except (ValueError, TypeError, KeyError, yaml.YAMLError, RecursionError):
            raise HTTPException(422, detail={"error": "invalid_policy"}) from None

    def owner(request, response, create=False):
        if not settings.demo_enabled:
            raise HTTPException(404)
        token = request.cookies.get("guard_owner", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{43}", token):
            if not create:
                raise HTTPException(404, detail={"error": "session_not_found"})
            token = secrets.token_urlsafe(32)
            response.set_cookie("guard_owner", token, httponly=True, samesite="strict", secure=settings.cookie_secure, max_age=settings.session_ttl_seconds)
        return hashlib.sha256(token.encode()).hexdigest()

    def mutation(request):
        if request.headers.get("X-Guard-Client") != "web":
            raise HTTPException(403, detail={"error": "client_header_required"})

    def session(conn, sid, user):
        row = conn.execute("SELECT * FROM sessions WHERE id=? AND owner=?", (sid, user)).fetchone()
        if row is None:
            raise HTTPException(404, detail={"error": "session_not_found"})
        return dict(row)

    def revoke(user, sid):
        current = service()
        with current._demo_lock:
            for token, pending in list(current._pending.items()):
                if pending.user_id == f"{user}:{sid}":
                    del current._pending[token]

    def public(row):
        row = dict(row)
        row.pop("owner", None)
        row["messages"] = json.loads(row["messages"])
        row["expires_at"] = datetime.fromtimestamp(row["updated"]+settings.session_ttl_seconds, timezone.utc).isoformat()
        return row

    @app.get("/v1/policy", tags=["administration"])
    def get_policy():
        with lock:
            return dict(snapshot(), editing_enabled=len(settings.admin_token) >= 24)

    @app.post("/v1/policy/validate", tags=["administration"])
    def validate_policy(payload: PolicyEdit, authorization: Annotated[str, Header()] = ""):
        admin(authorization)
        policy = parse(payload.yaml)
        return dict(valid=True, version=policy.version, agents=list(policy.rules["agents"]))

    @app.put("/v1/policy", tags=["administration"])
    def apply_policy(payload: PolicyEdit, authorization: Annotated[str, Header()] = ""):
        author = admin(authorization)
        policy = parse(payload.yaml)
        current = service()
        with lock, current._demo_lock:
            if not current._inference_lock.acquire(blocking=False):
                raise HTTPException(503, detail={"error": "guard_busy"})
            try:
                previous = snapshot()
                if payload.revision != previous["revision"]:
                    raise HTTPException(409, detail={"error": "policy_conflict"})
                policy.rules["version"] = max(policy.version, current.guard.policy.version+1)
                policy.version = policy.rules["version"]
                text = yaml.safe_dump(policy.rules, allow_unicode=True, sort_keys=False)
                revision = hashlib.sha256(text.encode()).hexdigest()
                path = policy.path
                original = path.read_bytes()
                temp = path.with_name(path.name+".pending")
                try:
                    temp.write_text(text, encoding="utf-8")
                    with db() as conn:
                        conn.execute("INSERT INTO policy_changes(timestamp,author,version,revision,yaml) VALUES(?,?,?,?,?)",
                                     (datetime.now(timezone.utc).isoformat(), author, policy.version, revision, text))
                        os.replace(temp, path)
                except Exception:
                    if path.read_bytes() != original:
                        temp.write_bytes(original)
                        os.replace(temp, path)
                    temp.unlink(missing_ok=True)
                    raise
                current.guard.policy = policy
                current._pending.clear()
                return dict(snapshot(), author=author)
            finally:
                current._inference_lock.release()

    @app.get("/v1/policy/changes", tags=["administration"])
    def policy_changes(authorization: Annotated[str, Header()] = ""):
        admin(authorization)
        with db() as conn:
            return dict(items=[dict(r) for r in conn.execute("SELECT id,timestamp,author,version,revision FROM policy_changes ORDER BY id DESC LIMIT 100")])

    @app.get("/v1/sessions", tags=["sessions"])
    def sessions(request: Request, response: Response):
        user = owner(request, response, True)
        with lock, db() as conn:
            rows = conn.execute("SELECT * FROM sessions WHERE owner=? ORDER BY updated DESC", (user,)).fetchall()
            return dict(items=[dict(id=r["id"], title=r["title"], role=r["role"], revision=r["revision"]) for r in rows], ttl_seconds=settings.session_ttl_seconds, max_chars=settings.session_max_chars)

    @app.post("/v1/sessions", tags=["sessions"])
    def create_session(payload: SessionCreate, request: Request, response: Response):
        mutation(request)
        user = owner(request, response, True)
        with lock, db() as conn:
            if conn.execute("SELECT count(*) FROM sessions WHERE owner=?", (user,)).fetchone()[0] >= 20 or conn.execute("SELECT count(*) FROM sessions").fetchone()[0] >= 1000:
                raise HTTPException(409, detail={"error": "session_capacity"})
            sid = str(uuid4())
            conn.execute("INSERT INTO sessions VALUES(?,?,?,?,?,?,?)", (sid, user, masked(payload.title), payload.role, time.time(), 0, "[]"))
            return public(session(conn, sid, user))

    @app.get("/v1/sessions/{sid}", tags=["sessions"])
    def read_session(sid: str, request: Request, response: Response):
        user = owner(request, response)
        with lock, db() as conn:
            return public(session(conn, sid, user))

    @app.delete("/v1/sessions/{sid}", tags=["sessions"])
    def delete_session(sid: str, request: Request, response: Response):
        mutation(request)
        user = owner(request, response)
        with lock, db() as conn:
            session(conn, sid, user)
            conn.execute("DELETE FROM sessions WHERE id=? AND owner=?", (sid, user))
            revoke(user, sid)
        return dict(deleted=True)

    @app.post("/v1/sessions/{sid}/clear", tags=["sessions"])
    def clear_session(sid: str, request: Request, response: Response):
        mutation(request)
        user = owner(request, response)
        with lock, db() as conn:
            session(conn, sid, user)
            conn.execute("UPDATE sessions SET messages='[]',revision=revision+1,updated=? WHERE id=?", (time.time(), sid))
            revoke(user, sid)
            return public(session(conn, sid, user))

    @app.post("/v1/sessions/{sid}/check", tags=["sessions"])
    def session_check(sid: str, payload: SessionTurn, request: Request, response: Response):
        mutation(request)
        user = owner(request, response)
        with lock, db() as conn:
            row = session(conn, sid, user)
            if payload.revision != row["revision"]:
                raise HTTPException(409, detail={"error": "session_conflict"})
            messages = json.loads(row["messages"])
            size = sum(len(m["content"]) for m in messages)+len(payload.user_prompt)
            size += sum(len(a.content) for a in payload.attachments)
            if size > settings.session_max_chars or len(messages) >= 100:
                raise HTTPException(413, detail={"error": "session_full"})
            checked = CheckRequest(user_prompt=payload.user_prompt, context=dict(target_agent="it_ops_agent", user_role=row["role"], user_id=f"{user}:{sid}", history=messages, attachments=[a.model_dump() for a in payload.attachments]))
            if payload.send:
                result = service().submit(checked).model_dump(mode="json")
            else:
                check = service().check(checked).model_dump(mode="json")
                result = dict(request_id=check["request_id"], check=check, status="checked")
            stored = payload.user_prompt
            if payload.attachments:
                stored += "\n[Недоверенные вложения]\n"+"\n".join(a.content for a in payload.attachments)
            messages.append(dict(role="user", content=masked(stored), decision=result["check"]["decision"], explanation=result["check"]["explanation"]))
            conn.execute("UPDATE sessions SET messages=?,revision=revision+1,updated=? WHERE id=?", (json.dumps(messages,ensure_ascii=False), time.time(), sid))
            result["session"] = public(session(conn, sid, user))
            result["history_effect"] = f"Учтено предыдущих реплик: {len(messages)-1}. Роль {row['role']} закреплена за сессией; история не расширяет полномочия."
            return result

    @app.post("/v1/sessions/{sid}/confirm/{token}", tags=["sessions"])
    def session_confirm(sid: str, token: str, request: Request, response: Response, payload: dict):
        mutation(request)
        if payload != {"confirmed": True} or type(payload.get("confirmed")) is not bool:
            raise HTTPException(422)
        user = owner(request, response)
        with lock, db() as conn:
            session(conn, sid, user)
            return service().confirm(token, f"{user}:{sid}")
