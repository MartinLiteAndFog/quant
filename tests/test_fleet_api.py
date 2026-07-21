"""Unit tests for fleet compounded trade % curves and registry helpers."""

from __future__ import annotations

import json
import os
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

import pandas as pd

from quant.execution.fleet_api import (
    _compounded_trade_curve,
    _normalize_account_curve,
    build_fleet_performance,
    fleet_bot_registry,
)


def _trade(exit_ts: str, pnl_pct: float, trade_id: str = "t") -> Dict[str, Any]:
    return {
        "trade_id": trade_id,
        "venue": "kucoin",
        "symbol": "SOL-USDT",
        "entry_ts": exit_ts,
        "exit_ts": exit_ts,
        "side": "long",
        "qty": 1.0,
        "entry_price": 100.0,
        "exit_price": 100.0 * (1.0 + pnl_pct / 100.0),
        "pnl_pct": pnl_pct,
        "exit_event": "tp_exit",
        "strategy": "test",
        "strategy_instance": "sol-pilot-canonical",
    }


class CompoundedTradeCurveTests(unittest.TestCase):
    def test_empty_returns_zero_stats(self) -> None:
        points, stats = _compounded_trade_curve(pd.DataFrame())
        self.assertEqual(points, [])
        self.assertEqual(stats["trade_count"], 0)
        self.assertEqual(stats["return_pct"], 0.0)

    def test_rebases_to_zero_and_compounds(self) -> None:
        # +10% then -10% of new equity => 1.1 * 0.9 = 0.99 => -1%
        df = pd.DataFrame(
            [
                _trade("2026-07-01T10:00:00Z", 10.0, "a"),
                _trade("2026-07-01T12:00:00Z", -10.0, "b"),
            ]
        )
        points, stats = _compounded_trade_curve(df)
        self.assertGreaterEqual(len(points), 3)
        self.assertEqual(points[0]["equity_pct"], 0.0)
        self.assertAlmostEqual(points[1]["equity_pct"], 10.0, places=5)
        self.assertAlmostEqual(points[-1]["equity_pct"], -1.0, places=5)
        self.assertAlmostEqual(stats["return_pct"], -1.0, places=5)
        self.assertEqual(stats["trade_count"], 2)
        self.assertEqual(stats["wins"], 1)
        self.assertEqual(stats["losses"], 1)
        self.assertAlmostEqual(stats["win_rate"], 0.5, places=5)
        self.assertAlmostEqual(stats["profit_factor"], 1.0, places=5)

    def test_max_drawdown_from_peak(self) -> None:
        df = pd.DataFrame(
            [
                _trade("2026-07-01T10:00:00Z", 20.0, "a"),
                _trade("2026-07-01T11:00:00Z", -25.0, "b"),  # 1.2 * 0.75 = 0.9
            ]
        )
        _, stats = _compounded_trade_curve(df)
        # peak 1.2, trough 0.9 => DD = 25%
        self.assertAlmostEqual(stats["max_drawdown_pct"], 25.0, places=5)
        self.assertAlmostEqual(stats["return_pct"], -10.0, places=5)


class NormalizeAccountCurveTests(unittest.TestCase):
    def test_normalizes_to_percent_from_first_point(self) -> None:
        pts = [
            {"t": 1, "equity": 100.0},
            {"t": 2, "equity": 110.0},
            {"t": 3, "equity": 90.0},
        ]
        out = _normalize_account_curve(pts)
        self.assertEqual(out[0]["equity_pct"], 0.0)
        self.assertAlmostEqual(out[1]["equity_pct"], 10.0, places=5)
        self.assertAlmostEqual(out[2]["equity_pct"], -10.0, places=5)


class FleetRegistryTests(unittest.TestCase):
    def test_default_registry_includes_pilots_and_kraken(self) -> None:
        with patch.dict(os.environ, {"FLEET_BOTS_JSON": ""}, clear=False):
            bots = fleet_bot_registry()
        ids = {b["strategy_instance"] for b in bots}
        self.assertIn("sol-pilot-canonical", ids)
        self.assertIn("sol-pilot-pc3axis", ids)
        self.assertIn("sol-pilot-countertrend", ids)
        self.assertIn("kraken_bot", ids)

    def test_env_override(self) -> None:
        payload = [
            {
                "id": "custom",
                "display_name": "Custom",
                "strategy_instance": "custom_inst",
                "venue": "kucoin",
                "health_url": "https://example.test/health",
            }
        ]
        with patch.dict(os.environ, {"FLEET_BOTS_JSON": json.dumps(payload)}, clear=False):
            bots = fleet_bot_registry()
        self.assertEqual(len(bots), 1)
        self.assertEqual(bots[0]["strategy_instance"], "custom_inst")


class BuildFleetPerformanceFilterTests(unittest.TestCase):
    def test_instance_filter_and_empty_backfill_hint(self) -> None:
        empty = pd.DataFrame()

        def _fake_load(**kwargs: Any) -> pd.DataFrame:
            return empty

        with patch("quant.execution.fleet_api._load_closed_trades_for_instance", side_effect=_fake_load), patch(
            "quant.execution.fleet_api._load_equity_snapshots", return_value=[]
        ):
            out = build_fleet_performance(hours=24.0, instance_ids=["imba-runner"])
        self.assertTrue(out["ok"])
        self.assertEqual(len(out["series"]), 1)
        self.assertEqual(out["series"][0]["id"], "imba-runner")
        self.assertTrue(out["series"][0]["needs_backfill"])
        self.assertEqual(out["series"][0]["trade_curve"], [])


if __name__ == "__main__":
    unittest.main()
