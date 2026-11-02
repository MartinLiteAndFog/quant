"""PostgreSQL integration: one shared token routes a 3x3 strategy fleet.

Safe ONLY in the isolated GitHub Actions PostgreSQL test database.
Never connects to production exchange or sends orders.
"""
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from quant.execution.family_shadow_app import app


@unittest.skipUnless(os.getenv("FAMILY_TEST_DATABASE_URL"), "isolated PostgreSQL only")
class RegistryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        expected = os.environ["FAMILY_TEST_DATABASE_URL"]
        current = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
        if expected != current or "quant_family_test" not in expected:
            raise RuntimeError("refuse to initialize registry on non-test database")
        cls.client = TestClient(app)
        cls.patch = patch.dict(os.environ, {
            "FAMILY_ID": "blaupunkt",
            "FAMILY_ROUTER_ENABLED": "1",
            "FAMILY_ADMIN_TOKEN": "admin-test-secret",
            "FAMILY_WEBHOOK_TOKEN": "one-shared-test-secret",
        })
        cls.patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()

    def admin(self):
        return {"X-Family-Admin-Token": "admin-test-secret"}

    def test_01_one_bootstrap_registers_all_twelve_ids(self):
        preview = self.client.get("/api/family/pilot-roster", headers=self.admin())
        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertEqual(preview.json()["total"], 12)
        self.assertEqual(preview.json()["parents"], 3)
        self.assertTrue(preview.json()["shared_webhook_token_required"])

        response = self.client.post("/api/family/bootstrap-pilot", headers=self.admin())
        self.assertEqual(response.status_code, 201, response.text)
        body = response.json()
        self.assertEqual(body["created_strategies"], 11)
        self.assertEqual(body["shared_auth_env"], "FAMILY_WEBHOOK_TOKEN")
        self.assertNotIn("token_once", response.text)
        all_bots = self.client.get("/api/family/strategies", headers=self.admin())
        self.assertEqual(len(all_bots.json()["strategies"]), 12)

    def test_02_same_shared_secret_routes_different_bots(self):
        for id in ("blaupunkt-p02", "blaupunkt-p03-d01"):
            response = self.client.post("/webhook/family", json={
                "strategy_id": id,
                "token": "one-shared-test-secret",
                "event_id": "unique:" + id,
                "action": "entry",
                "side": "buy",
                "symbol": "SOL-USDT",
            })
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["strategy_id"], id)
            self.assertFalse(response.json()["order_sent"])
        dupe = self.client.post("/webhook/family", json={
            "strategy_id": "blaupunkt-p02",
            "token": "one-shared-test-secret",
            "event_id": "unique:blaupunkt-p02",
            "action": "entry",
            "side": "buy",
            "symbol": "SOL-USDT",
        })
        self.assertEqual(dupe.status_code, 200)
        self.assertTrue(dupe.json()["duplicate"])

    def test_03_invalid_ids_and_tokens_fail_closed(self):
        wrong = self.client.post("/webhook/family", json={
            "strategy_id": "blaupunkt-p02", "token": "wrong",
            "event_id": "x", "action": "entry", "side": "buy",
        })
        self.assertEqual(wrong.status_code, 401)
        unknown = self.client.post("/webhook/family", json={
            "strategy_id": "unknown-p999", "token": "one-shared-test-secret",
            "event_id": "x", "action": "entry", "side": "buy",
        })
        self.assertEqual(unknown.status_code, 401)

    def test_04_allocation_90pct_three_roots_and_nine_daughters(self):
        result = self.client.get(
            "/api/family/allocation-preview?equity=1000",
            headers=self.admin(),
        )
        self.assertEqual(result.status_code, 200, result.text)
        body = result.json()
        self.assertEqual(body["active_parents"], 3)
        self.assertEqual(float(body["reserve"]), 100.0)
        for root in ("blaupunkt", "blaupunkt-p02", "blaupunkt-p03"):
            self.assertEqual(float(body["family_budgets"][root]), 300.0)
            self.assertEqual(float(body["bot_budgets"][root]), 282.0)
        for id in ("blaupunkt-d01", "blaupunkt-p02-d02", "blaupunkt-p03-d03"):
            self.assertEqual(float(body["bot_budgets"][id]), 6.0)

    def test_05_monthly_preview_does_not_transfer_funds(self):
        monthly = self.client.post(
            "/api/family/monthly-preview", headers=self.admin(),
            json={
                "period": "2026-10", "equity": "1000",
                "metrics": [
                    {
                        "strategy_id": "blaupunkt",
                        "monthly_net_twr": "-0.05",
                        "max_drawdown": "0.20",
                        "closed_trades": 40,
                        "observed_days": 31,
                        "cashflows_adjusted": True,
                        "marked_to_market": True,
                        "fees_and_funding_included": True,
                    },
                    {
                        "strategy_id": "blaupunkt-d01",
                        "monthly_net_twr": "0.20",
                        "max_drawdown": "0.02",
                        "closed_trades": 40,
                        "observed_days": 31,
                        "cashflows_adjusted": True,
                        "marked_to_market": True,
                        "fees_and_funding_included": True,
                    },
                ],
            },
        )
        self.assertEqual(monthly.status_code, 200, monthly.text)
        obj = monthly.json()
        self.assertFalse(obj["orders_sent"])
        self.assertFalse(obj["persisted"])
        self.assertFalse(obj["real_capital_transfers"])
        self.assertEqual(obj["parent_family_count"], 3)
        first = next(x for x in obj["families"] if x["parent_strategy_id"] == "blaupunkt")
        self.assertGreater(float(first["bot_targets"]["blaupunkt-d01"]), 6.0)


if __name__ == "__main__":
    unittest.main()
