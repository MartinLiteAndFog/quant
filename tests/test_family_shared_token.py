"""Tests for one URL and one shared webhook secret across all strategy IDs."""
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from quant.execution.family_ingress import _check_shared_token, _validate_registration, NewStrategy
from quant.execution.family_shadow_app import app
from quant.execution.family_pilot_roster import BLAUPUNKT_PILOT_ROSTER


class SharedTokenTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_one_webhook_route_regardless_of_id(self):
        # Test actual HTTP dispatch rather than Starlette route internals.
        # Every strategy uses the same path and shared secret.
        with patch.dict(os.environ, {
            "FAMILY_ROUTER_ENABLED": "1",
            "FAMILY_WEBHOOK_TOKEN": "one-shared-secret",
        }):
            for item in BLAUPUNKT_PILOT_ROSTER:
                response = self.client.post("/webhook/family", json={
                    "strategy_id": item.strategy_id,
                    "token": "bad-secret",
                    "event_id": "unique-1",
                    "action": "entry", "side": "buy",
                })
                self.assertEqual(response.status_code, 401)
        self.assertEqual(len(BLAUPUNKT_PILOT_ROSTER), 12)
        self.assertEqual(
            len({x.strategy_id for x in BLAUPUNKT_PILOT_ROSTER}), 12,
        )

    def test_shared_secret_is_required(self):
        with patch.dict(os.environ, {
            "FAMILY_ROUTER_ENABLED": "1",
            "FAMILY_WEBHOOK_TOKEN": "one-common-token",
        }):
            _check_shared_token("one-common-token")
            self.assertEqual(
                self.client.post("/webhook/family", json={
                    "strategy_id": "blaupunkt-p02",
                    "token": "bad-token",
                    "event_id": "unique-1", "action": "entry", "side": "buy",
                }).status_code,
                401,
            )

    def test_no_webhook_secret_fails_closed(self):
        with patch.dict(os.environ, {
            "FAMILY_ROUTER_ENABLED": "1",
            "FAMILY_WEBHOOK_TOKEN": "",
        }):
            response = self.client.post("/webhook/family", json={
                "strategy_id": "blaupunkt-p03",
                "token": "some-token",
            })
            self.assertEqual(response.status_code, 503)

    def test_registration_requires_admin_token_not_strategy_token(self):
        with patch.dict(os.environ, {
            "FAMILY_ROUTER_ENABLED": "1",
            "FAMILY_ADMIN_TOKEN": "admin-secret",
            "FAMILY_WEBHOOK_TOKEN": "common-secret",
        }):
            response = self.client.post(
                "/api/family/strategies",
                headers={"X-Family-Admin-Token": "common-secret"},
                json={"strategy_id": "another-parent", "display_name": "Parent"},
            )
            self.assertEqual(response.status_code, 401)

    def test_all_children_have_registered_parent_ids(self):
        ids = {x.strategy_id for x in BLAUPUNKT_PILOT_ROSTER}
        for child in BLAUPUNKT_PILOT_ROSTER:
            if child.parent_strategy_id:
                self.assertIn(child.parent_strategy_id, ids)

    def test_child_registration_does_not_require_unique_token(self):
        model = NewStrategy(
            strategy_id="blaupunkt-p02-d01",
            display_name="Test daughter",
            parent_strategy_id="blaupunkt-p02",
        )
        _validate_registration(model)
        self.assertFalse(hasattr(model, "token"))


if __name__ == "__main__":
    unittest.main()
