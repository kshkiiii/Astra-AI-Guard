from __future__ import annotations

import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import HTTPException
from threadpoolctl import threadpool_limits

from api.audit import AuditLog
from api.schemas import AgentReceipt, CheckRequest, CheckResponse, CheckResult, DemoResponse
from api.settings import Settings
from model import Guard, load_config


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DemoAgent:


    def receive(self, request_id: str, request: CheckRequest) -> AgentReceipt:
        return AgentReceipt(receipt_id=str(uuid4()), request_id=request_id,
                            target_agent=request.context.target_agent, received_at=now())


@dataclass
class Pending:
    user_id: str
    request: CheckRequest
    response: DemoResponse
    deadline: float


class GuardService:
    def __init__(self, guard: Guard, audit: AuditLog, settings: Settings, agent: DemoAgent | None = None):
        self.guard = guard
        self.audit = audit
        self.settings = settings
        self.agent = agent or DemoAgent()
        self._inference_lock = threading.Lock()
        self._demo_lock = threading.Lock()
        self._pending: dict[str, Pending] = {}
        self._executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix="guard-cpu") if getattr(guard,"deadline_seconds",None) else None

    @classmethod
    def load(cls, settings: Settings) -> GuardService:
        config = load_config(settings.config_path)
        guard = Guard.from_config(config)
        service = cls(guard, AuditLog(settings.audit_path), settings)


        service._infer(CheckRequest(user_prompt="Покажи логи сервиса в dev", context={
            "target_agent": "it_ops_agent", "user_role": "developer"}))
        return service

    @staticmethod
    def _clear_text_cache(model) -> None:
        def clear(transformer):
            if transformer.__class__.__name__ == "LocalE5Transformer":
                transformer.__dict__.pop("_cache", None)
            for _, child in getattr(transformer, "transformer_list", []):
                clear(child)

        clear(model.vectorizer)
        for member in getattr(model, "members", []):
            GuardService._clear_text_cache(member)

    def _infer(self, request: CheckRequest) -> CheckResult:


        if not self._inference_lock.acquire(blocking=False):
            raise HTTPException(503, detail={"error": "guard_busy"}, headers={"Retry-After": "1"})
        if self._executor:
            future=self._executor.submit(self._execute_infer,request)
            try:
                return future.result(timeout=self.guard.deadline_seconds)
            except FutureTimeout:
                raise HTTPException(503,detail={"error":"guard_budget_exceeded"}) from None
        return self._execute_infer(request)

    def close(self):
        if self._executor:
            self._executor.shutdown(wait=False,cancel_futures=True)

    def _execute_infer(self,request):
        try:
            with threadpool_limits(limits=1, user_api="blas"):
                from local_runtime import ModelInputTooLong, ModelUnavailable
                from inference_budget import InferenceBudgetExceeded
                try:
                    result = self.guard.check(request.user_prompt, request.context.model_dump(exclude_none=True))
                except ModelInputTooLong:
                    raise HTTPException(413, detail={"error": "llm_context_too_long"}) from None
                except ModelUnavailable:
                    raise HTTPException(503, detail={"error": "llm_unavailable"}) from None
                except InferenceBudgetExceeded:
                    raise HTTPException(503,detail={"error":"guard_budget_exceeded"}) from None
            result["intent_total"] = len(result.get("intent", []))
            result["details_truncated"] = result["intent_total"] > 32
            result["intent"] = result.get("intent", [])[:32]
            bounded = CheckResult.model_validate(result)
            if len(bounded.model_dump_json().encode("utf-8")) > 60_000:
                bounded.intent = []
                bounded.checks = []
                bounded.explanation = bounded.explanation[:2000]
                bounded.details_truncated = True
            return bounded
        finally:
            try:
                self._clear_text_cache(self.guard.model)
            finally:
                self._inference_lock.release()

    def _record(self, request_id: str, request: CheckRequest, result: CheckResult, event: str) -> None:

        agents = self.guard.policy.rules["agents"]
        agent = agents.get(request.context.target_agent, {})
        self.audit.append({
            "request_id": request_id, "timestamp": now(), "event": event,
            "user_role": request.context.user_role if request.context.user_role in agent.get("roles", {}) else "unknown",
            "target_agent": request.context.target_agent if request.context.target_agent in agents else "unknown",
            "decision": result.decision, "risk_category": result.risk_category,
            "risk_level": result.risk_level, "latency_ms": result.latency_ms,
            "policy_version": str(result.policy_version),
        })

    def check(self, request: CheckRequest, event: str = "check") -> CheckResponse:
        size = len(request.user_prompt) + sum(len(m.content) for m in request.context.history)
        size += sum(len(a.content) for a in request.context.attachments)
        if size > self.guard.model.config["text"].get("max_input_chars", 200_000):
            raise HTTPException(413, detail={"error": "input_too_long"})
        result = self._infer(request)
        request_id = str(uuid4())
        self._record(request_id, request, result, event)
        return CheckResponse(request_id=request_id, **result.model_dump())

    def _expire(self) -> None:
        expired = [token for token, pending in self._pending.items() if pending.deadline <= time.monotonic()]
        for token in expired:
            del self._pending[token]

    def submit(self, request: CheckRequest) -> DemoResponse:
        with self._demo_lock:
            self._expire()
            checked = self.check(request, "demo_submit")
            result = CheckResult.model_validate(checked.model_dump())
            response = DemoResponse(request_id=checked.request_id, check=result, status="blocked")
            if result.decision == "allow":
                response.receipt = self.agent.receive(checked.request_id, request)
                response.status = "delivered"
            elif result.decision == "confirm":
                if len(self._pending) >= self.settings.max_pending:
                    raise HTTPException(503, detail={"error": "confirmation_capacity"})
                token = secrets.token_urlsafe(32)
                ttl = self.settings.confirmation_ttl_seconds
                response.status = "pending_confirmation"
                response.confirmation_token = token
                response.expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl)).isoformat()
                self._pending[token] = Pending(request.context.user_id, request.model_copy(deep=True),
                                               response.model_copy(deep=True), time.monotonic() + ttl)
            return response

    def confirm(self, token: str, user_id: str) -> DemoResponse:
        with self._demo_lock:
            self._expire()
            pending = self._pending.get(token)
            if pending is None or not secrets.compare_digest(pending.user_id, user_id):
                raise HTTPException(404, detail={"error": "confirmation_not_found"})
            if pending.response.status == "delivered":
                return pending.response.model_copy(deep=True)


            checked = self.check(pending.request, "demo_recheck")
            result = CheckResult.model_validate(checked.model_dump())
            if result.decision == "block":
                del self._pending[token]
                return DemoResponse(request_id=pending.response.request_id, check=result, status="blocked")
            self._record(pending.response.request_id, pending.request, result, "demo_confirm")
            pending.response.receipt = self.agent.receive(pending.response.request_id, pending.request)
            pending.response.status = "delivered"
            pending.response.check = result
            pending.response.confirmation_token = None
            return pending.response.model_copy(deep=True)
