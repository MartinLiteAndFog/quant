"""Family control API tests that cannot place exchange orders or access DB."""
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from quant.execution.family_shadow_app import app


class ShadowAppTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_declares_no_order_execution(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["orders_enabled"], False)
        self.assertEqual(response.json()["mode"], "shadow")

    def test_family_endpoints_fail_closed_when_not_enabled(self):
        with patch.dict(os.environ, {"FAMILY_ROUTER_ENABLED": "0"}):
            paths = [
                ("GET", "/api/family/strategies"),
                ("GET", "/api/family/allocation-preview?equity=1000"),
                ("GET", "/api/family/inbox"),
                ("POST", "/api/family/bootstrap-legacy"),
            ]
            for method, path in paths:
                with self.subTest(method=method, path=path):
                    rsp = self.client.request(method, path)
                    self.assertEqual(rsp.status_code, 404)

    def test_no_admin_key_fails_closed_even_when_router_enabled(self):
        with patch.dict(
            os.environ,
            {"FAMILY_ROUTER_ENABLED": "1", "FAMILY_ADMIN_TOKEN": ""},
        ):
            response = self.client.get("/api/family/strategies")
            self.assertEqual(response.status_code, 503)
            response = self.client.get("/api/family/allocation-preview?equity=1000")
            self.assertEqual(response.status_code, 503)

    def test_invalid_registration_token_is_rejected_before_database(self):
        with patch.dict(
            os.environ,
            {"FAMILY_ROUTER_ENABLED": "1", "FAMILY_ADMIN_TOKEN": "abc"},
        ):
            response = self.client.post(
                "/api/family/strategies",
                headers={"X-Family-Admin-Token": "wrong"},
                json={"strategy_id": "parent-002", "display_name": "Parent 2"},
            )
            self.assertEqual(response.status_code, 401)

    def test_missing_strategy_credential_rejected_without_db(self):
        with patch.dict(os.environ, {"FAMILY_ROUTER_ENABLED": "1"}):
            response = self.client.post(
                "/webhook/family",
                json={"strategy_id": "parent-002", "action": "entry", "side": "buy"},
            )
            self.assertEqual(response.status_code, 401)

    def test_legacy_webhook_absent_from_shadow_only_app(self):
        response = self.client.post(
            "/webhook/tv-execute", json={"action": "entry", "side": "buy"}
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
