from __future__ import annotations

import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

from api.schemas import CheckRequest, CheckResponse, Confirmation, Decision, DemoRequest, DemoResponse
from api.service import GuardService
from api.settings import Settings


logger = logging.getLogger("guard.api")


class BodyLimit:
    def __init__(self, app: ASGIApp, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.limit:
                return await JSONResponse({"error": "body_too_large"}, status_code=413)(scope, receive, send)
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        consumed = False

        async def replay():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def create_app(settings: Settings | None = None, service_factory=GuardService.load) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service_factory(settings)
        try:
            yield
        finally:
            if hasattr(app.state.service,"close"):
                app.state.service.close()
            app.state.service = None

    app = FastAPI(title="ИИ-сторож API", version="1.0.0", lifespan=lifespan,
                  description="Проверка запросов и отдельный сценарий передачи локальному демо-агенту.")
    app.state.service = None

    @app.middleware("http")
    async def safe_errors(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:

            logger.error("Request failed: %s", type(exc).__name__)
            return JSONResponse({"error": "service_unavailable"}, status_code=503)

    app.add_middleware(BodyLimit, limit=settings.max_body_bytes)
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                           allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-Demo-User"])

    def service() -> GuardService:
        if app.state.service is None:
            raise HTTPException(503, detail={"error": "not_ready"})
        return app.state.service

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, exc: RequestValidationError):

        errors = [{"loc": error["loc"], "type": error["type"], "msg": error["msg"]} for error in exc.errors()]
        return JSONResponse({"error": "invalid_request", "details": errors}, status_code=422)

    @app.get("/health", tags=["service"])
    def health():
        return {"status": "ok"}

    @app.get("/ready", tags=["service"])
    def ready():
        guard = service().guard
        if hasattr(guard, "is_ready") and not guard.is_ready():
            raise HTTPException(503, detail={"error": "llm_unavailable"})
        return {"status": "ready", "policy_version": guard.policy.version,
                "backend": getattr(guard, "backend", "layered"),
                "demo_enabled": settings.demo_enabled}

    @app.post("/v1/check", response_model=CheckResponse, tags=["guard"])
    def check(payload: CheckRequest):

        return service().check(payload)

    @app.get("/v1/audit", tags=["audit"])
    def audit(limit: Annotated[int, Query(ge=1, le=200)] = 50,
              before: Annotated[int | None, Query(ge=1)] = None, decision: Decision | None = None,
              user_role: Annotated[str | None, Query(max_length=128)] = None,
              category: Annotated[str | None, Query(max_length=128)] = None,
              since: datetime | None = None, until: datetime | None = None):

        def utc(value):
            return value.replace(tzinfo=value.tzinfo or timezone.utc).astimezone(timezone.utc).isoformat() if value else None
        if since and until and utc(since) > utc(until):
            raise HTTPException(422, detail={"error": "invalid_date_range"})
        items = service().audit.recent(limit, before, decision, user_role, category, utc(since), utc(until))
        return {"items": items, "next_before": items[-1]["id"] if len(items) == limit else None}

    if settings.demo_enabled:
        def identity(role: str) -> dict:

            if role not in {"developer", "sre", "admin"}:
                raise HTTPException(401, detail={"error": "unknown_demo_user"})
            return {"user_id": f"demo-{role}", "user_role": role, "target_agent": "it_ops_agent"}

        @app.post("/v1/demo/requests", response_model=DemoResponse, tags=["demo"])
        def submit(payload: DemoRequest, demo_user: Annotated[str, Header(alias="X-Demo-User")]):

            context = identity(demo_user)
            request = CheckRequest(user_prompt=payload.user_prompt, context={
                **context, "history": payload.history, "attachments": payload.attachments})
            return service().submit(request)

        @app.post("/v1/demo/confirmations/{token}", response_model=DemoResponse, tags=["demo"])
        def confirm(token: str, payload: Confirmation,
                    demo_user: Annotated[str, Header(alias="X-Demo-User")]):

            return service().confirm(token, identity(demo_user)["user_id"])

    from api.management import register_management
    register_management(app, service, settings)

    frontend = Path(__file__).resolve().parents[1] / "frontend"

    @app.get("/", include_in_schema=False)
    def interface():
        return FileResponse(frontend / "index.html", headers={"Cache-Control": "no-store"})

    app.mount("/assets", StaticFiles(directory=frontend / "assets"), name="frontend-assets")
    return app


app = create_app()
