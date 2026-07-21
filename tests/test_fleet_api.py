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

        with patch("quant.execution.fleet_api._load_closed_trades_for_bot", return_value=empty), patch(
            "quant.execution.fleet_api._load_account_points_for_bot", return_value=[]
        ), patch(
            "quant.execution.fleet_api.list_fleet_bots",
            return_value={"ok": True, "bots": []},
        ):
            out = build_fleet_performance(hours=24.0, instance_ids=["imba-runner"])
        self.assertTrue(out["ok"])
        self.assertEqual(len(out["series"]), 1)
        self.assertEqual(out["series"][0]["id"], "imba-runner")
        self.assertTrue(out["series"][0]["needs_backfill"])
        self.assertEqual(out["series"][0]["trade_curve"], [])
        self.assertEqual(out["series"][0]["account_curve_abs"], [])


class LiveStitchAndAbsCurveTests(unittest.TestCase):
    def test_stitch_appends_live_equity(self) -> None:
        from quant.execution.fleet_api import _stitch_live_equity, _absolute_account_curve

        pts = [{"t": 100, "equity": 100.0}]
        out = _stitch_live_equity(pts, live_equity=120.5, now_ts=200)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[-1]["equity"], 120.5)
        self.assertEqual(out[-1]["t"], 200)
        abs_curve = _absolute_account_curve(out)
        self.assertEqual(abs_curve[-1]["equity"], 120.5)

    def test_performance_uses_equity_account_and_live_stitch(self) -> None:
        empty = pd.DataFrame()
        snaps = [{"t": 1000, "equity": 20.0, "currency": "USDT"}]
        health = {
            "ok": True,
            "bots": [
                {
                    "id": "quant-main",
                    "equity": 22.5,
                    "currency": "USDT",
                    "health": {"ok": True, "equity": 22.5},
                }
            ],
        }

        with patch("quant.execution.fleet_api._load_closed_trades_for_bot", return_value=empty), patch(
            "quant.execution.fleet_api._load_account_points_for_bot", return_value=snaps
        ), patch("quant.execution.fleet_api.list_fleet_bots", return_value=health):
            out = build_fleet_performance(hours=0, instance_ids=["quant-main"])
        s = out["series"][0]
        self.assertEqual(s["id"], "quant-main")
        self.assertGreaterEqual(len(s["account_curve_abs"]), 2)
        self.assertAlmostEqual(s["account_curve_abs"][-1]["equity"], 22.5, places=5)
        self.assertAlmostEqual(s["live_equity"], 22.5, places=5)


class MultiInstanceTradesTests(unittest.TestCase):
    def test_loads_union_of_trade_instances(self) -> None:
        from quant.execution.fleet_api import _load_closed_trades_for_bot

        df_a = pd.DataFrame(
            [
                {
                    "trade_id": "a",
                    "venue": "kucoin",
                    "symbol": "SOLUSDT",
                    "entry_ts": "2026-07-01T10:00:00Z",
                    "exit_ts": "2026-07-01T11:00:00Z",
                    "side": "long",
                    "qty": 1,
                    "entry_price": 1,
                    "exit_price": 1.1,
                    "pnl_pct": 10.0,
                    "exit_event": "tp",
                    "strategy": "x",
                    "strategy_instance": "quant",
                }
            ]
        )
        df_b = pd.DataFrame(
            [
                {
                    "trade_id": "b",
                    "venue": "kucoin",
                    "symbol": "SOLUSDT",
                    "entry_ts": "2026-07-02T10:00:00Z",
                    "exit_ts": "2026-07-02T11:00:00Z",
                    "side": "long",
                    "qty": 1,
                    "entry_price": 1,
                    "exit_price": 1.05,
                    "pnl_pct": 5.0,
                    "exit_event": "tp",
                    "strategy": "x",
                    "strategy_instance": "live_executor",
                }
            ]
        )

        def _fake_load(**kwargs: Any) -> pd.DataFrame:
            inst = kwargs.get("strategy_instance")
            if inst == "quant":
                return df_a
            if inst == "live_executor":
                return df_b
            return pd.DataFrame()

        bot = {
            "strategy_instance": "quant",
            "trade_instances": ["quant", "live_executor"],
            "venue": "kucoin",
            "symbol": "SOL-USDT",
        }
        with patch(
            "quant.execution.fleet_api._load_closed_trades_for_instance",
            side_effect=_fake_load,
        ), patch(
            "quant.execution.fleet_api._load_closed_trades_null_instance",
            return_value=pd.DataFrame(),
        ):
            out = _load_closed_trades_for_bot(bot)
        self.assertEqual(len(out), 2)
        self.assertEqual(set(out["trade_id"]), {"a", "b"})


class CapitalizationLiveEquityTests(unittest.TestCase):
    def test_prefers_health_equity_over_snapshots(self) -> None:
        from quant.execution.fleet_api import build_fleet_capitalization

        bots = {
            "ok": True,
            "bots": [
                {
                    "id": "imba-runner",
                    "display_name": "Imba Runner",
                    "strategy_instance": "sol-pilot-canonical",
                    "venue": "kucoin",
                    "status": "live",
                    "executor_ready": True,
                    "live_trading_enabled": True,
                    "dry_run": False,
                    "health": {
                        "ok": True,
                        "equity": 250.5,
                        "available": 180.0,
                        "unrealised_pnl": 1.25,
                        "currency": "USDT",
                        "equity_source": "kucoin_live",
                    },
                }
            ],
        }
        with patch("quant.execution.fleet_api.list_fleet_bots", return_value=bots):
            with patch("quant.execution.fleet_api._load_equity_snapshots", return_value=[]):
                out = build_fleet_capitalization()
        self.assertTrue(out["ok"])
        self.assertEqual(out["accounts"][0]["equity"], 250.5)
        self.assertEqual(out["accounts"][0]["available"], 180.0)
        self.assertEqual(out["accounts"][0]["equity_source"], "kucoin_live")


if __name__ == "__main__":
    unittest.main()
