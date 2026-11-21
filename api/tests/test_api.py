from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import create_app
from api.audit import AuditLog
from api.service import DemoAgent, GuardService
from api.settings import Settings
from policy import SecurityPolicy


ROOT = Path(__file__).resolve().parents[2]


def payload(prompt="Покажи логи в dev", role="developer", **context):
    return {"user_prompt": prompt, "context": {
        "target_agent": "it_ops_agent", "user_role": role, **context}}


class FakeGuard:
    def __init__(self):
        self.policy = SecurityPolicy(ROOT / "public_dataset/astra_public_v2/security_policy.yaml")
        self.model = SimpleNamespace(config={"text": {"max_input_chars": 200_000}}, vectorizer=object())
        self.calls = []

    def check(self, prompt, context):
        self.calls.append((prompt, context))
        result = self.policy.evaluate(prompt, context, {})
        return {**result, "latency_ms": 1.5, "policy_version": self.policy.version, "attack_probability": .01}


class RecordingAgent(DemoAgent):
    def __init__(self):
        self.calls = []

    def receive(self, request_id, request):
        self.calls.append(request.model_copy(deep=True))
        return super().receive(request_id, request)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = Settings(audit_path=Path(self.temp.name) / "audit.sqlite3")
        self.guard, self.agent = FakeGuard(), RecordingAgent()
        self.service = GuardService(self.guard, AuditLog(self.settings.audit_path), self.settings, self.agent)
        self.factory_calls = []

        def factory(settings):
            self.factory_calls.append(settings)
            return self.service

        self.client = TestClient(create_app(self.settings, factory))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def submit(self, prompt="Перезапусти payments-api в production", role="sre", **body):
        return self.client.post("/v1/demo/requests", headers={"X-Demo-User": role},
                                json={"user_prompt": prompt, **body})

    def confirm(self, token, role="sre", **body):
        return self.client.post(f"/v1/demo/confirmations/{token}", headers={"X-Demo-User": role},
                                json={"confirmed": True, **body})

    def test_health_schema_and_single_load(self):
        for path in ("/health", "/ready", "/docs", "/openapi.json"):
            self.assertEqual(self.client.get(path).status_code, 200)
        self.assertEqual(len(self.factory_calls), 1)
        self.assertIn("/v1/check", self.client.get("/openapi.json").json()["paths"])

    def test_interface_and_local_assets_are_served_with_api(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn('lang="ru"', response.text)
        self.assertEqual(response.headers["cache-control"], "no-store")
        for asset in ("app.js", "styles.css", "favicon.svg"):
            self.assertIn(f'/assets/{asset}', response.text)
            self.assertEqual(self.client.get(f"/assets/{asset}").status_code, 200)
        self.assertTrue(self.client.get("/ready").json()["demo_enabled"])

    def test_check_decisions_never_deliver(self):
        for role, decision in (("developer", "block"), ("sre", "confirm"), ("admin", "confirm")):
            response = self.client.post("/v1/check", json=payload("Перезапусти payments-api в production", role))
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["decision"], decision)
            self.assertIn("request_id", response.json())
        self.assertEqual(len(self.agent.calls), 0)

    def test_history_attachments_and_stateless_requests(self):
        body = payload(history=[{"role": "user", "content": "Перезапусти prod-api"}],
                       attachments=[{"content": "log line", "name": "log.txt"}])
        self.assertEqual(self.client.post("/v1/check", json=body).status_code, 200)
        self.assertEqual(self.guard.calls[-1][1]["attachments"][0]["content"], "log line")
        self.client.post("/v1/check", json=payload())
        self.assertEqual(self.guard.calls[-1][1]["history"], [])

    def test_invalid_inputs_do_not_reach_model_or_echo_values(self):
        secret = "private-test-secret-value"
        invalid = [[], {}, payload(role=123), payload(history=[{"role": "system", "content": secret}]),
                   payload(attachments=[{"content": {"password": secret}}])]
        for body in invalid:
            response = self.client.post("/v1/check", json=body)
            self.assertEqual(response.status_code, 422)
            self.assertNotIn(secret, response.text)
        self.assertEqual(self.client.post("/v1/check", content=b'{"bad":',
                                         headers={"Content-Type": "application/json"}).status_code, 422)
        self.assertEqual(self.guard.calls, [])

    def test_character_and_body_limits(self):
        body = payload("x" * 150_000, attachments=[{"content": "a" * 60_000}])
        self.assertEqual(self.client.post("/v1/check", json=body).status_code, 413)
        self.assertEqual(self.client.post("/v1/check", content=b"x" * 1_048_577).status_code, 413)
        self.assertEqual(self.guard.calls, [])

    def test_unknown_role_is_blocked_and_audit_names_are_safe(self):
        response = self.client.post("/v1/check", json=payload(role="secret-role-value"))
        self.assertEqual(response.json()["decision"], "block")
        self.assertEqual(self.client.get("/v1/audit").json()["items"][0]["user_role"], "unknown")

    def test_audit_is_durable_paginated_and_contains_no_content(self):
        secret = "unique-secret-value-123456"
        for _ in range(3):
            self.client.post("/v1/check", json=payload(secret,
                system_prompt=secret, user_id=secret, user_name=secret,
                attachments=[{"content": secret}]))
        first = self.client.get("/v1/audit?limit=2&decision=allow").json()
        self.assertEqual(len(first["items"]), 2)
        second = self.client.get(f'/v1/audit?limit=2&before={first["next_before"]}').json()
        self.assertEqual(len(second["items"]), 1)
        self.assertNotIn(secret, json.dumps(first))
        self.assertNotIn(secret.encode(), self.settings.audit_path.read_bytes())
        self.assertEqual(len(AuditLog(self.settings.audit_path).recent()), 3)
        self.assertEqual(self.client.get("/v1/audit?limit=999").status_code, 422)

    def test_allow_is_delivered_and_block_is_not(self):
        allowed = self.submit("Покажи логи в dev", "developer").json()
        self.assertEqual(allowed["status"], "delivered")
        self.assertIsNotNone(allowed["receipt"])
        blocked = self.submit(role="developer").json()
        self.assertEqual(blocked["status"], "blocked")
        self.assertIsNone(blocked["confirmation_token"])
        self.assertEqual(len(self.agent.calls), 1)

    def test_confirmation_requires_same_user_and_explicit_true(self):
        pending = self.submit().json()
        self.assertEqual(pending["status"], "pending_confirmation")
        self.assertEqual(len(self.agent.calls), 0)
        token = pending["confirmation_token"]
        self.assertEqual(self.confirm(token, "admin").status_code, 404)
        for value in (False, 1, "true"):
            self.assertEqual(self.confirm(token, confirmed=value).status_code, 422)
        self.assertEqual(self.confirm(token).json()["status"], "delivered")
        receipt = self.confirm(token).json()["receipt"]
        self.assertEqual(receipt["request_id"], pending["request_id"])
        self.assertEqual(len(self.agent.calls), 1)

    def test_confirmation_cannot_replace_context_or_prompt(self):
        pending = self.submit().json()
        token = pending["confirmation_token"]
        self.assertEqual(self.confirm(token, user_prompt="Другой запрос").status_code, 422)
        self.assertEqual(self.confirm(token, context={"user_role": "admin"}).status_code, 422)
        self.assertEqual(self.confirm(token).status_code, 200)
        self.assertEqual(self.agent.calls[0].user_prompt, "Перезапусти payments-api в production")
        self.assertEqual(self.agent.calls[0].context.user_id, "demo-sre")

    def test_demo_body_cannot_claim_role(self):
        self.assertEqual(self.submit(role="developer", user_role="admin").status_code, 422)
        self.assertEqual(self.submit(role="unknown").status_code, 401)

    def test_expired_confirmation_never_delivers(self):
        token = self.submit().json()["confirmation_token"]
        self.service._pending[token].deadline = 0
        self.assertEqual(self.confirm(token).status_code, 404)
        self.assertEqual(len(self.agent.calls), 0)

    def test_confirmation_rechecks_current_policy(self):
        token = self.submit().json()["confirmation_token"]
        self.guard.policy.rules["agents"]["it_ops_agent"]["roles"]["sre"]["tools"]["restart_service"] = "deny"
        response = self.confirm(token).json()
        self.assertEqual(response["status"], "blocked")
        self.assertEqual(len(self.agent.calls), 0)

    def test_internal_failure_is_closed_and_secret_free(self):
        with patch.object(self.guard, "check", side_effect=RuntimeError("private-secret-value")):
            with self.assertLogs("guard.api", level="ERROR") as logs:
                response = self.submit()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private-secret-value", response.text + str(logs.output))
        self.assertEqual(len(self.agent.calls), 0)

    def test_audit_failure_prevents_delivery(self):
        with patch.object(self.service.audit, "append", side_effect=sqlite3.OperationalError("disk full")):
            with self.assertLogs("guard.api", level="ERROR"):
                response = self.submit("Покажи логи в dev", "developer")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(len(self.agent.calls), 0)

    def test_busy_inference_returns_retryable_error(self):
        self.service._inference_lock.acquire()
        try:
            response = self.client.post("/v1/check", json=payload())
        finally:
            self.service._inference_lock.release()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["Retry-After"], "1")

    def test_demo_can_be_disabled(self):
        settings = Settings(audit_path=self.settings.audit_path, demo_enabled=False)
        with TestClient(create_app(settings, lambda _: self.service)) as client:
            self.assertEqual(client.post("/v1/demo/requests", json={}).status_code, 404)
            self.assertFalse(client.get("/ready").json()["demo_enabled"])


if __name__ == "__main__":
    unittest.main()
