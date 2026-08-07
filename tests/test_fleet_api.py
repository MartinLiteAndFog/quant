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
    _downsample_points,
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

    def test_large_jump_remains_visible_in_raw_equity_percent(self) -> None:
        out = _normalize_account_curve(
            [{"t": 1, "equity": 100.0}, {"t": 2, "equity": 200.0}]
        )
        self.assertAlmostEqual(out[-1]["equity_pct"], 100.0, places=5)


class DownsampleAccountCurveTests(unittest.TestCase):
    def test_preserves_start_of_new_flat_level(self) -> None:
        """A durable new balance must not appear only at the live endpoint."""
        points = [
            {"t": i * 60, "equity": 100.0 + (i % 7)}
            for i in range(220)
        ]
        points.extend(
            [
                {"t": 220 * 60, "equity": 68.0},
                {"t": 221 * 60, "equity": 68.0},
                {"t": 400 * 60, "equity": 68.0},
            ]
        )

        out = _downsample_points(
            points,
            max_points=40,
            value_key="equity",
            min_interval_sec=900,
        )

        first_new_level = next(p for p in out if p["equity"] == 68.0)
        self.assertEqual(first_new_level["t"], 220 * 60)
        self.assertEqual(out[-1], {"t": 400 * 60, "equity": 68.0})


class TwrAccountCurveTests(unittest.TestCase):
    def test_deposit_jump_excluded_from_returns(self) -> None:
        from quant.execution.fleet_api import _twr_account_curve

        pts = [
            {"t": 100, "equity": 100.0},
            {"t": 200, "equity": 102.0},  # +2% trading
            {"t": 300, "equity": 300.0},  # deposit (+194%) → excluded
            {"t": 400, "equity": 306.0},  # +2% trading
        ]
        out = _twr_account_curve(pts, jump_threshold_pct=10.0)
        self.assertEqual(out[0]["equity_pct"], 0.0)
        self.assertAlmostEqual(out[1]["equity_pct"], 2.0, places=5)
        self.assertAlmostEqual(out[2]["equity_pct"], 2.0, places=5)  # flat over deposit
        self.assertAlmostEqual(out[3]["equity_pct"], 4.04, places=5)  # 1.02*1.02

    def test_withdrawal_jump_excluded(self) -> None:
        from quant.execution.fleet_api import _twr_account_curve

        pts = [
            {"t": 100, "equity": 300.0},
            {"t": 200, "equity": 100.0},  # withdrawal (-66%) → excluded
            {"t": 300, "equity": 99.0},  # -1% trading
        ]
        out = _twr_account_curve(pts, jump_threshold_pct=10.0)
        self.assertAlmostEqual(out[1]["equity_pct"], 0.0, places=5)
        self.assertAlmostEqual(out[2]["equity_pct"], -1.0, places=5)


class CashflowCorrectedReturnTests(unittest.TestCase):
    def test_deposit_is_removed_without_using_jump_threshold(self) -> None:
        from quant.execution.fleet_api import _cashflow_corrected_return

        points = [{"t": 100, "equity": 100.0}, {"t": 200, "equity": 160.0}]
        flows = [{"t": 150, "reporting_amount": 50.0, "equity_after": None}]
        self.assertAlmostEqual(
            _cashflow_corrected_return(points, flows) or 0.0,
            10.0,
            places=5,
        )

    def test_withdrawal_is_removed_from_interval_return(self) -> None:
        from quant.execution.fleet_api import _cashflow_corrected_return

        points = [{"t": 100, "equity": 100.0}, {"t": 200, "equity": 72.0}]
        flows = [{"t": 150, "reporting_amount": -20.0, "equity_after": None}]
        self.assertAlmostEqual(
            _cashflow_corrected_return(points, flows) or 0.0,
            -8.0,
            places=5,
        )

    def test_event_equity_segments_true_subperiods(self) -> None:
        from quant.execution.fleet_api import _cashflow_corrected_return

        points = [{"t": 100, "equity": 100.0}, {"t": 200, "equity": 165.0}]
        flows = [{"t": 150, "reporting_amount": 50.0, "equity_after": 155.0}]
        expected = ((105.0 / 100.0) * (165.0 / 155.0) - 1.0) * 100.0
        self.assertAlmostEqual(
            _cashflow_corrected_return(points, flows) or 0.0,
            expected,
            places=5,
        )

    def test_corrected_curve_matches_scalar_end_return(self) -> None:
        from quant.execution.fleet_api import (
            _cashflow_corrected_curve,
            _cashflow_corrected_return,
        )

        points = [
            {"t": 100, "equity": 100.0},
            {"t": 200, "equity": 160.0},
            {"t": 300, "equity": 176.0},
        ]
        flows = [{"t": 150, "reporting_amount": 50.0, "equity_after": None}]
        curve = _cashflow_corrected_curve(points, flows)
        self.assertEqual(curve[0], {"t": 100, "equity_pct": 0.0})
        scalar = _cashflow_corrected_return(points, flows)
        self.assertIsNotNone(scalar)
        self.assertAlmostEqual(curve[-1]["equity_pct"], float(scalar), places=5)
        self.assertAlmostEqual(curve[1]["equity_pct"], 10.0, places=5)
        self.assertAlmostEqual(curve[2]["equity_pct"], 21.0, places=5)

    def test_corrected_curve_empty_when_insufficient_points(self) -> None:
        from quant.execution.fleet_api import _cashflow_corrected_curve

        self.assertEqual(
            _cashflow_corrected_curve([{"t": 1, "equity": 100.0}], []),
            [],
        )

    def test_portfolio_is_value_weighted_and_excludes_unavailable_bot(self) -> None:
        from quant.execution.fleet_api import _build_cashflow_return

        series = [
            {
                "id": "a",
                "strategy_instance": "a",
                "venue": "kucoin",
                "account_curve_abs": [
                    {"t": 100, "equity": 100.0},
                    {"t": 200, "equity": 160.0},
                ],
            },
            {
                "id": "b",
                "strategy_instance": "b",
                "venue": "kucoin",
                "account_curve_abs": [
                    {"t": 100, "equity": 200.0},
                    {"t": 200, "equity": 220.0},
                ],
            },
        ]
        registry = [
            {"id": "a", "strategy_instance": "a", "venue": "kucoin"},
            {"id": "b", "strategy_instance": "b", "venue": "kucoin"},
            {
                "id": "counter-sl-reverse",
                "strategy_instance": "missing",
                "venue": "kucoin",
                "cashflow_return_excluded": True,
            },
        ]
        state = {
            "coverage_start": pd.Timestamp(90, unit="s", tz="UTC"),
            "coverage_end": pd.Timestamp(200, unit="s", tz="UTC"),
            "last_success_at": pd.Timestamp(200, unit="s", tz="UTC"),
            "last_error": None,
            "source": "test",
        }

        def load(*, venue, account, since, until):
            flows = (
                [
                    {
                        "t": 150,
                        "reporting_amount": 50.0,
                        "currency": "USDT",
                    }
                ]
                if account == "a"
                else []
            )
            return flows, state

        with patch("quant.execution.fleet_api._load_cashflow_data", side_effect=load):
            metric = _build_cashflow_return(
                series,
                registry,
                since=pd.Timestamp(100, unit="s", tz="UTC"),
                now_ts=200,
            )
        self.assertTrue(metric["available"])
        self.assertAlmostEqual(metric["return_pct"], 10.0, places=5)
        self.assertEqual(metric["net_cashflow"], 50.0)
        self.assertEqual(metric["flow_count"], 1)
        self.assertEqual(metric["excluded_bot_ids"], ["counter-sl-reverse"])

    def test_common_scope_start_does_not_create_a_late_join_jump(self) -> None:
        from quant.execution.fleet_api import _build_cashflow_return

        series = [
            {
                "id": "early",
                "strategy_instance": "early",
                "venue": "kucoin",
                "account_curve_abs": [
                    {"t": 90, "equity": 100.0},
                    {"t": 190, "equity": 110.0},
                ],
            },
            {
                "id": "late",
                "strategy_instance": "late",
                "venue": "kucoin",
                "account_curve_abs": [
                    {"t": 100, "equity": 200.0},
                    {"t": 190, "equity": 220.0},
                ],
            },
        ]
        registry = [
            {"id": "early", "strategy_instance": "early", "venue": "kucoin"},
            {"id": "late", "strategy_instance": "late", "venue": "kucoin"},
        ]
        state = {
            "coverage_start": pd.Timestamp(80, unit="s", tz="UTC"),
            "coverage_end": pd.Timestamp(190, unit="s", tz="UTC"),
            "last_success_at": pd.Timestamp(190, unit="s", tz="UTC"),
            "last_error": None,
            "source": "test",
        }
        seen_since: list[pd.Timestamp] = []

        def load(*, venue, account, since, until):
            seen_since.append(since)
            flows = (
                [
                    {
                        "t": 95,
                        "reporting_amount": 50.0,
                        "currency": "USDT",
                    }
                ]
                if account == "early"
                else []
            )
            return flows, state

        with patch(
            "quant.execution.fleet_api._load_cashflow_data",
            side_effect=load,
        ):
            metric = _build_cashflow_return(
                series,
                registry,
                since=pd.Timestamp(80, unit="s", tz="UTC"),
                now_ts=190,
            )
        self.assertTrue(metric["available"])
        self.assertAlmostEqual(metric["return_pct"], 10.0, places=5)
        # The TWR common equity scope starts at t=100, but Net Flows describes
        # the user's selected range beginning at t=80.
        self.assertEqual(metric["net_cashflow"], 50.0)
        self.assertEqual(metric["flow_count"], 1)
        self.assertEqual(
            seen_since,
            [
                pd.Timestamp(80, unit="s", tz="UTC"),
                pd.Timestamp(80, unit="s", tz="UTC"),
            ],
        )

    def test_missing_sync_is_unavailable_instead_of_inferred(self) -> None:
        from quant.execution.fleet_api import _build_cashflow_return

        series = [
            {
                "id": "a",
                "strategy_instance": "a",
                "venue": "kucoin",
                "account_curve_abs": [
                    {"t": 100, "equity": 100.0},
                    {"t": 200, "equity": 200.0},
                ],
            }
        ]
        registry = [{"id": "a", "strategy_instance": "a", "venue": "kucoin"}]
        with patch(
            "quant.execution.fleet_api._load_cashflow_data",
            return_value=([], None),
        ):
            metric = _build_cashflow_return(
                series,
                registry,
                since=pd.Timestamp(100, unit="s", tz="UTC"),
                now_ts=200,
            )
        self.assertFalse(metric["available"])
        self.assertIsNone(metric["return_pct"])
        self.assertEqual(metric["reason"], "ledger_sync_unavailable")
        self.assertEqual(metric["unavailable_bot_ids"], ["a"])


class BotCorrectedPayloadTests(unittest.TestCase):
    def test_inactive_bot_has_empty_curve_but_still_loads_cashflows(self) -> None:
        from quant.execution.fleet_api import _bot_corrected_payload

        flows = [
            {
                "t": 150,
                "reporting_amount": -20.0,
                "direction": "out",
                "currency": "USDT",
                "flow_type": "TransferOut",
                "amount": -20.0,
                "status": "completed",
                "equity_after": None,
                "source_ref": "x",
            }
        ]
        state = {
            "coverage_start": pd.Timestamp(90, unit="s", tz="UTC"),
            "coverage_end": pd.Timestamp(200, unit="s", tz="UTC"),
            "last_success_at": pd.Timestamp(200, unit="s", tz="UTC"),
            "last_error": None,
            "source": "test",
        }
        with patch(
            "quant.execution.fleet_api._load_cashflow_data",
            return_value=(flows, state),
        ) as load:
            out = _bot_corrected_payload(
                abs_points=[
                    {"t": 100, "equity": 100.0},
                    {"t": 200, "equity": 80.0},
                ],
                venue="kucoin",
                account="a",
                since=pd.Timestamp(100, unit="s", tz="UTC"),
                until_ts=200,
                bot_status="down",
                bot_disabled=False,
            )
        load.assert_called_once()
        self.assertEqual(out["corrected_curve"], [])
        self.assertEqual(out["corrected_meta"]["method"], "unavailable")
        self.assertEqual(out["corrected_meta"]["reason"], "inactive")
        self.assertEqual(len(out["cashflows"]), 1)

    def test_active_without_sync_uses_jump_twr(self) -> None:
        from quant.execution.fleet_api import _bot_corrected_payload

        pts = [
            {"t": 100, "equity": 100.0},
            {"t": 200, "equity": 102.0},
            {"t": 300, "equity": 300.0},
            {"t": 400, "equity": 306.0},
        ]
        with patch(
            "quant.execution.fleet_api._load_cashflow_data",
            return_value=([], None),
        ):
            out = _bot_corrected_payload(
                abs_points=pts,
                venue="kucoin",
                account="a",
                since=pd.Timestamp(100, unit="s", tz="UTC"),
                until_ts=400,
                bot_status="live",
                bot_disabled=False,
            )
        self.assertEqual(out["corrected_meta"]["method"], "jump_twr")
        self.assertAlmostEqual(out["corrected_curve"][-1]["equity_pct"], 4.04, places=5)

    def test_active_with_ledger_uses_cashflow_curve(self) -> None:
        from quant.execution.fleet_api import _bot_corrected_payload

        pts = [
            {"t": 100, "equity": 100.0},
            {"t": 200, "equity": 160.0},
        ]
        flows = [
            {
                "t": 150,
                "reporting_amount": 50.0,
                "direction": "in",
                "currency": "USDT",
                "flow_type": "TransferIn",
                "amount": 50.0,
                "status": "completed",
                "equity_after": None,
                "source_ref": "y",
            }
        ]
        state = {
            "coverage_start": pd.Timestamp(90, unit="s", tz="UTC"),
            "coverage_end": pd.Timestamp(200, unit="s", tz="UTC"),
            "last_success_at": pd.Timestamp(200, unit="s", tz="UTC"),
            "last_error": None,
            "source": "test",
        }
        with patch(
            "quant.execution.fleet_api._load_cashflow_data",
            return_value=(flows, state),
        ):
            out = _bot_corrected_payload(
                abs_points=pts,
                venue="kucoin",
                account="a",
                since=pd.Timestamp(100, unit="s", tz="UTC"),
                until_ts=200,
                bot_status="live",
                bot_disabled=False,
            )
        self.assertEqual(out["corrected_meta"]["method"], "ledger")
        self.assertAlmostEqual(out["corrected_curve"][-1]["equity_pct"], 10.0, places=5)
        self.assertEqual(out["corrected_meta"]["flow_count"], 1)
        self.assertEqual(out["corrected_meta"]["net_cashflow"], 50.0)


class BuildFleetCorrectedCurveTests(unittest.TestCase):
    def test_performance_series_includes_corrected_fields(self) -> None:
        registry = [
            {
                "id": "a",
                "display_name": "A",
                "strategy_instance": "a",
                "venue": "kucoin",
                "symbol": "SOL-USDT",
                "color": "#fff",
            }
        ]
        acct = [
            {"t": 100, "equity": 100.0, "currency": "USDT"},
            {"t": 200, "equity": 110.0, "currency": "USDT"},
        ]
        corrected = {
            "corrected_curve": [
                {"t": 100, "equity_pct": 0.0},
                {"t": 200, "equity_pct": 10.0},
            ],
            "corrected_meta": {
                "method": "jump_twr",
                "available": True,
                "reason": None,
                "flow_count": 0,
                "net_cashflow": 0.0,
                "source": "db",
            },
            "cashflows": [],
        }
        with patch(
            "quant.execution.fleet_api.fleet_bot_registry", return_value=registry
        ), patch(
            "quant.execution.fleet_api.list_fleet_bots",
            return_value={
                "bots": [{"id": "a", "status": "live", "equity": 110.0}]
            },
        ), patch(
            "quant.execution.fleet_api._load_display_trades_for_bot",
            return_value=pd.DataFrame(),
        ), patch(
            "quant.execution.fleet_api._load_account_points_for_bot",
            return_value=acct,
        ), patch(
            "quant.execution.fleet_api._bot_corrected_payload",
            return_value=corrected,
        ), patch(
            "quant.execution.fleet_api._build_cashflow_return",
            return_value={"available": False},
        ):
            out = build_fleet_performance(hours=0)
        row = out["series"][0]
        self.assertIn("corrected_curve", row)
        self.assertIn("corrected_meta", row)
        self.assertIn("cashflows", row)
        self.assertEqual(row["corrected_meta"]["method"], "jump_twr")
        self.assertIn("corrected_curve", out["portfolio"])
        self.assertTrue(len(out["portfolio"]["corrected_curve"]) >= 1)


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
        self.assertIn("portfolio", out)
        self.assertEqual(out["portfolio"]["id"], "portfolio")

    def test_bot_performance_reset_rebases_metrics_but_keeps_live_equity(self) -> None:
        bot = {
            "id": "quant-main",
            "display_name": "Quant (KuCoin main)",
            "strategy_instance": "quant",
            "equity_account": "futures",
            "venue": "kucoin",
            "symbol": "SOL-USDT",
            "performance_start": "2026-07-28T10:00:00Z",
            "cashflow_return_excluded": True,
        }
        seen_trade_since: list[pd.Timestamp] = []
        seen_equity_since: list[pd.Timestamp] = []

        def _load_trades(_bot: Dict[str, Any], *, since=None, limit=5000):
            seen_trade_since.append(since)
            return pd.DataFrame()

        def _load_equity(_bot: Dict[str, Any], *, since=None):
            seen_equity_since.append(since)
            return [{"t": 1785232801, "equity": 43.772525}]

        with patch(
            "quant.execution.fleet_api.fleet_bot_registry",
            return_value=[bot],
        ), patch(
            "quant.execution.fleet_api.list_fleet_bots",
            return_value={
                "ok": True,
                "bots": [
                    {
                        "id": "quant-main",
                        "equity": 43.772525,
                        "currency": "USDT",
                    }
                ],
            },
        ), patch(
            "quant.execution.fleet_api._load_display_trades_for_bot",
            side_effect=_load_trades,
        ), patch(
            "quant.execution.fleet_api._load_account_points_for_bot",
            side_effect=_load_equity,
        ):
            out = build_fleet_performance(hours=0)

        reset = pd.Timestamp("2026-07-28T10:00:00Z")
        self.assertEqual(seen_trade_since, [reset])
        self.assertEqual(seen_equity_since, [reset])
        quant = out["series"][0]
        self.assertEqual(quant["live_equity"], 43.772525)
        self.assertEqual(quant["stats"]["trade_count"], 0)
        self.assertEqual(quant["stats"]["return_pct"], 0.0)
        self.assertEqual(quant["trade_curve"], [])
        self.assertEqual(quant["account_curve"][-1]["equity_pct"], 0.0)
        self.assertEqual(
            quant["performance_start"],
            "2026-07-28T10:00:00+00:00",
        )


class PortfolioAggregateTests(unittest.TestCase):
    def test_sums_forward_filled_equities(self) -> None:
        from quant.execution.fleet_api import _build_portfolio_curve

        series = [
            {
                "id": "a",
                "live_equity": 10.0,
                "account_curve_abs": [
                    {"t": 100, "equity": 10.0},
                    {"t": 200, "equity": 12.0},
                ],
            },
            {
                "id": "b",
                "live_equity": 20.0,
                "account_curve_abs": [
                    {"t": 150, "equity": 20.0},
                    {"t": 250, "equity": 22.0},
                ],
            },
        ]
        port = _build_portfolio_curve(series)
        self.assertEqual(port["id"], "portfolio")
        self.assertEqual(port["color"], "#ffffff")
        abs_curve = port["account_curve_abs"]
        self.assertGreaterEqual(len(abs_curve), 3)
        by_t = {p["t"]: p["equity"] for p in abs_curve}
        self.assertAlmostEqual(by_t[100], 10.0, places=5)
        self.assertAlmostEqual(by_t[150], 30.0, places=5)  # 10 + 20
        self.assertAlmostEqual(by_t[200], 32.0, places=5)  # 12 + 20
        self.assertAlmostEqual(by_t[250], 34.0, places=5)  # 12 + 22
        self.assertAlmostEqual(port["live_equity"], 30.0, places=5)
        # Equal-weight % of each bot's own-base return (not sum-then-rebase).
        pct_by_t = {p["t"]: p["equity_pct"] for p in port["account_curve"]}
        self.assertAlmostEqual(pct_by_t[100], 0.0, places=5)
        self.assertAlmostEqual(pct_by_t[150], 0.0, places=5)  # A 0% + B 0%
        self.assertAlmostEqual(pct_by_t[200], 10.0, places=5)  # A +20% + B 0%
        self.assertAlmostEqual(pct_by_t[250], 15.0, places=5)  # A +20% + B +10%
        self.assertEqual(port["note"], "equal_weight_raw_pct_mean_abs_sum")

    def test_portfolio_pct_ignores_late_large_account_join_spike(self) -> None:
        """Regression: pilots ~$15 then Kraken ~$360 must not rebase to +2000%+."""
        from quant.execution.fleet_api import _build_portfolio_curve

        series = [
            {
                "id": "pilot",
                "live_equity": 14.7,
                "account_curve_abs": [
                    {"t": 100, "equity": 15.0},
                    {"t": 200, "equity": 14.7},  # -2%
                    {"t": 300, "equity": 14.7},
                ],
            },
            {
                "id": "kraken",
                "live_equity": 366.3,
                "account_curve_abs": [
                    {"t": 250, "equity": 360.0},
                    {"t": 300, "equity": 366.3},  # +1.75%
                ],
            },
        ]
        port = _build_portfolio_curve(series)
        pct_by_t = {p["t"]: p["equity_pct"] for p in port["account_curve"]}
        # Old bug: (14.7+366.3)/15 - 1 ≈ +2440%. Equal-weight stays near bot %.
        self.assertAlmostEqual(pct_by_t[200], -2.0, places=5)
        self.assertAlmostEqual(pct_by_t[250], -1.0, places=5)  # (-2 + 0) / 2
        self.assertAlmostEqual(pct_by_t[300], (-2.0 + 1.75) / 2.0, places=5)
        self.assertLess(abs(pct_by_t[300]), 5.0)
        # Abs sum still honest about total capital step-up when Kraken joins.
        abs_by_t = {p["t"]: p["equity"] for p in port["account_curve_abs"]}
        self.assertAlmostEqual(abs_by_t[100], 15.0, places=5)
        self.assertAlmostEqual(abs_by_t[250], 14.7 + 360.0, places=5)

class ForwardFillGridTests(unittest.TestCase):
    def test_uniform_steps_and_holds_value(self) -> None:
        from quant.execution.fleet_api import _forward_fill_on_grid

        pts = [
            {"t": 1_000, "equity": 10.0},
            {"t": 1_250, "equity": 12.0},
        ]
        out = _forward_fill_on_grid(
            pts, value_key="equity", t0=1_000, t1=1_400, interval_sec=100
        )
        self.assertGreaterEqual(len(out), 4)
        by_t = {p["t"]: p["equity"] for p in out}
        self.assertAlmostEqual(by_t[1_000], 10.0, places=5)
        # Mid-gap holds prior value until the 1250 observation is reached.
        self.assertAlmostEqual(by_t[1_100], 10.0, places=5)
        self.assertAlmostEqual(by_t[1_300], 12.0, places=5)
        self.assertAlmostEqual(by_t[1_400], 12.0, places=5)
        # No invented history before first observation even if t0 is earlier.
        early = _forward_fill_on_grid(
            pts, value_key="equity", t0=500, t1=1_200, interval_sec=100
        )
        self.assertEqual(early[0]["t"], 1_000)

    def test_align_series_shares_clock(self) -> None:
        from quant.execution.fleet_api import _align_series_to_shared_clock

        series = [
            {
                "id": "a",
                "account_curve_abs": [
                    {"t": 1_000_000, "equity": 10.0},
                    {"t": 1_000_900, "equity": 11.0},
                ],
                "account_curve": [],
                "trade_curve": [],
            },
            {
                "id": "b",
                "account_curve_abs": [
                    {"t": 1_000_500, "equity": 20.0},
                    {"t": 1_001_200, "equity": 22.0},
                ],
                "account_curve": [],
                "trade_curve": [],
            },
        ]
        aligned, clock = _align_series_to_shared_clock(
            series, hours=1.0, now_ts=1_001_200
        )
        self.assertEqual(clock["t1"], 1_001_200)
        self.assertGreater(clock["interval_sec"], 0)
        # No invented history PAST a bot's last real observation: A stops at
        # its last snapshot instead of being dragged to the clock end.
        self.assertLessEqual(aligned[0]["account_curve_abs"][-1]["t"], 1_000_900)
        self.assertEqual(aligned[1]["account_curve_abs"][-1]["t"], 1_001_200)
        # Bot B does not invent points before its first observation.
        self.assertGreaterEqual(aligned[1]["account_curve_abs"][0]["t"], 1_000_500)


class DownsampleCurveTests(unittest.TestCase):
    def test_collapses_flat_high_frequency_spam(self) -> None:
        from quant.execution.fleet_api import _downsample_points

        # ~40s health polls of a flat 15 USDT balance over ~1h
        pts = [{"t": 1_000_000 + i * 40, "equity": 15.0} for i in range(90)]
        out = _downsample_points(pts, max_points=180, value_key="equity", min_interval_sec=900)
        self.assertLessEqual(len(out), 4)
        self.assertEqual(out[0]["equity"], 15.0)
        self.assertEqual(out[-1]["equity"], 15.0)
        self.assertGreater(out[-1]["t"], out[0]["t"])


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
        # Fresh snapshot (60s old) → live equity is stitched onto the curve.
        now_ts = int(pd.Timestamp.now("UTC").timestamp())
        snaps = [{"t": now_ts - 60, "equity": 20.0, "currency": "USDT"}]
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
        self.assertIsNotNone(s["last_snapshot_ts"])
        self.assertLessEqual(s["snapshot_age_sec"], 120)

    def test_performance_does_not_stitch_onto_stale_snapshots(self) -> None:
        """Regression: stale seed + live stitch fabricated flat-line-plus-cliff
        curves (audit 2026-07-22, +261% jumps from Nov-2025 seeds)."""
        empty = pd.DataFrame()
        now_ts = int(pd.Timestamp.now("UTC").timestamp())
        stale_ts = now_ts - 30 * 24 * 3600  # 30 days old
        snaps = [{"t": stale_ts, "equity": 15.0, "currency": "USDT"}]
        health = {
            "ok": True,
            "bots": [
                {
                    "id": "quant-main",
                    "equity": 54.2,
                    "currency": "USDT",
                    "health": {"ok": True, "equity": 54.2},
                }
            ],
        }

        with patch("quant.execution.fleet_api._load_closed_trades_for_bot", return_value=empty), patch(
            "quant.execution.fleet_api._load_account_points_for_bot", return_value=snaps
        ), patch("quant.execution.fleet_api.list_fleet_bots", return_value=health):
            out = build_fleet_performance(hours=0, instance_ids=["quant-main"])
        s = out["series"][0]
        # No fabricated cliff: curve ends on the stale snapshot, not live equity.
        self.assertAlmostEqual(s["account_curve_abs"][-1]["equity"], 15.0, places=5)
        # Honest flags instead.
        self.assertTrue(s["needs_backfill"])
        self.assertEqual(s["last_snapshot_ts"], stale_ts)
        self.assertGreater(s["snapshot_age_sec"], 3600)
        # Live equity still reported for the legend/capitalization readouts.
        self.assertAlmostEqual(s["live_equity"], 54.2, places=5)


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

    def test_kraken_synced_trade_reaches_performance_after_shared_cutoff(self) -> None:
        from quant.execution.fleet_api import build_fleet_performance

        bot = {
            "id": "kraken-legacy",
            "display_name": "Kraken Legacy",
            "strategy_instance": "kraken_bot",
            "trade_instances": ["kraken_bot", "live_executor_2"],
            "venue": "kraken",
            "symbol": "SOL-USD",
            "color": "#b07050",
        }
        trades = pd.DataFrame(
            [
                {
                    "trade_id": "kraken-position:1",
                    "venue": "kraken",
                    "symbol": "SOL-USD",
                    "entry_ts": pd.Timestamp("2026-07-22T12:00:00Z"),
                    "exit_ts": pd.Timestamp("2026-07-23T12:00:00Z"),
                    "side": "long",
                    "qty": 2.0,
                    "entry_price": 100.0,
                    "exit_price": 103.0,
                    "pnl_pct": 3.0,
                    "exit_event": "kraken_position_reverse",
                    "strategy": "kraken_tv_executor",
                    "strategy_instance": "kraken_bot",
                }
            ]
        )
        seen_since: list[pd.Timestamp] = []

        def _load(bot_arg, *, since=None, limit=5000):
            seen_since.append(since)
            return trades

        with patch.dict(
            "os.environ",
            {"FLEET_HISTORY_START": "2026-07-22"},
            clear=False,
        ), patch(
            "quant.execution.fleet_api.fleet_bot_registry",
            return_value=[bot],
        ), patch(
            "quant.execution.fleet_api.list_fleet_bots",
            return_value={"ok": True, "bots": []},
        ), patch(
            "quant.execution.fleet_api._load_display_trades_for_bot",
            side_effect=_load,
        ), patch(
            "quant.execution.fleet_api._load_account_points_for_bot",
            return_value=[],
        ):
            out = build_fleet_performance(hours=0)

        self.assertEqual(seen_since, [pd.Timestamp("2026-07-22T00:00:00Z")])
        self.assertEqual(out["since"], "2026-07-22T00:00:00+00:00")
        self.assertEqual(out["series"][0]["stats"]["trade_count"], 1)
        self.assertTrue(out["series"][0]["trade_curve"])


class ExecutionActivityTradeTests(unittest.TestCase):
    @staticmethod
    def _row(
        *,
        event_id: str,
        ts: str,
        side: str,
        reduce_only: bool,
        reason: str,
        before: int,
        after: int,
        price: float,
        instance: str = "sol-pilot-canonical",
    ) -> tuple[Any, ...]:
        return (
            event_id,
            pd.Timestamp(ts),
            "kucoin",
            "SOLUSDT",
            instance,
            side,
            1.0,
            price,
            reduce_only,
            "sent",
            f"quant:tv:{reason}:1",
            {
                "reason_code": f"tv_{reason}",
                "position_before": before,
                "position_after": after,
            },
        )

    def test_completed_activity_round_trip_becomes_display_trade(self) -> None:
        from quant.execution.fleet_api import _trades_from_execution_activity

        rows = [
            self._row(
                event_id="open",
                ts="2026-07-22T01:00:00Z",
                side="buy",
                reduce_only=False,
                reason="entry",
                before=0,
                after=1,
                price=100.0,
            ),
            self._row(
                event_id="close",
                ts="2026-07-22T02:00:00Z",
                side="sell",
                reduce_only=True,
                reason="exit",
                before=1,
                after=0,
                price=102.0,
            ),
        ]
        bot = {
            "id": "imba-runner",
            "strategy_instance": "sol-pilot-canonical",
            "venue": "kucoin",
            "symbol": "SOL-USDT",
        }
        out = _trades_from_execution_activity(rows, bot=bot)
        self.assertEqual(len(out), 1)
        trade = out.iloc[0]
        self.assertEqual(trade["trade_id"], "activity:close")
        self.assertEqual(trade["side"], "long")
        self.assertAlmostEqual(float(trade["pnl_pct"]), 2.0)
        self.assertEqual(trade["display_source"], "execution_activity")

    def test_partial_take_profit_does_not_complete_trade(self) -> None:
        from quant.execution.fleet_api import _trades_from_execution_activity

        rows = [
            self._row(
                event_id="open",
                ts="2026-07-22T01:00:00Z",
                side="sell",
                reduce_only=False,
                reason="entry",
                before=0,
                after=-1,
                price=100.0,
            ),
            self._row(
                event_id="partial",
                ts="2026-07-22T01:30:00Z",
                side="buy",
                reduce_only=True,
                reason="tp1",
                before=-1,
                after=-1,
                price=98.0,
            ),
            self._row(
                event_id="close",
                ts="2026-07-22T02:00:00Z",
                side="buy",
                reduce_only=True,
                reason="tp2",
                before=-1,
                after=0,
                price=96.0,
            ),
        ]
        bot = {
            "id": "imba-runner",
            "strategy_instance": "sol-pilot-canonical",
            "venue": "kucoin",
            "symbol": "SOL-USDT",
        }
        out = _trades_from_execution_activity(rows, bot=bot)
        self.assertEqual(list(out["trade_id"]), ["activity:close"])
        self.assertAlmostEqual(float(out.iloc[0]["pnl_pct"]), 4.0)

    def test_opposite_open_activity_closes_prior_display_leg(self) -> None:
        from quant.execution.fleet_api import _trades_from_execution_activity

        rows = [
            self._row(
                event_id="open-long",
                ts="2026-07-22T01:00:00Z",
                side="buy",
                reduce_only=False,
                reason="flip_entry",
                before=0,
                after=1,
                price=100.0,
            ),
            self._row(
                event_id="reverse-short",
                ts="2026-07-22T02:00:00Z",
                side="sell",
                reduce_only=False,
                reason="flip_entry",
                before=0,
                after=-1,
                price=102.0,
            ),
            self._row(
                event_id="reverse-long",
                ts="2026-07-22T03:00:00Z",
                side="buy",
                reduce_only=False,
                reason="flip_entry",
                before=0,
                after=1,
                price=99.0,
            ),
        ]
        bot = {
            "id": "pure-imbatp",
            "strategy_instance": "sol-pilot-pc3axis",
            "venue": "kucoin",
            "symbol": "SOL-USDT",
        }
        out = _trades_from_execution_activity(rows, bot=bot)
        self.assertEqual(
            list(out["trade_id"]),
            ["activity:reverse-short", "activity:reverse-long"],
        )
        self.assertEqual(list(out["side"]), ["long", "short"])
        self.assertEqual(set(out["exit_event"]), {"activity_reversal"})
        self.assertAlmostEqual(float(out.iloc[0]["pnl_pct"]), 2.0)
        self.assertAlmostEqual(float(out.iloc[1]["pnl_pct"]), 100.0 * (1.0 - 99.0 / 102.0))

    def test_closed_trade_wins_over_matching_activity_inference(self) -> None:
        from quant.execution.fleet_api import _load_display_trades_for_bot

        exit_ts = pd.Timestamp("2026-07-22T02:00:00Z")
        closed = pd.DataFrame(
            [
                {
                    **_trade(exit_ts.isoformat(), 1.5, "durable"),
                    "symbol": "SOLUSDT",
                }
            ]
        )
        inferred = pd.DataFrame(
            [
                {
                    **_trade((exit_ts + pd.Timedelta(seconds=2)).isoformat(), 1.4, "activity:close"),
                    "symbol": "SOL-USDT",
                    "display_source": "execution_activity",
                }
            ]
        )
        bot = {
            "id": "imba-runner",
            "strategy_instance": "sol-pilot-canonical",
            "venue": "kucoin",
            "symbol": "SOL-USDT",
        }
        with patch(
            "quant.execution.fleet_api._load_closed_trades_for_bot",
            return_value=closed,
        ), patch(
            "quant.execution.fleet_api._load_execution_activity_trades_for_bot",
            return_value=inferred,
        ):
            out = _load_display_trades_for_bot(bot)
        self.assertEqual(list(out["trade_id"]), ["durable"])

    def test_fleet_trade_view_uses_activity_fallback(self) -> None:
        from quant.execution.fleet_api import build_fleet_trades

        bot = {
            "id": "imba-runner",
            "display_name": "imba5",
            "strategy_instance": "sol-pilot-canonical",
            "venue": "kucoin",
            "symbol": "SOL-USDT",
        }
        inferred = pd.DataFrame(
            [
                {
                    **_trade("2026-07-22T02:00:00Z", 2.0, "activity:close"),
                    "strategy_instance": "sol-pilot-canonical",
                }
            ]
        )
        with patch(
            "quant.execution.fleet_api.fleet_bot_registry",
            return_value=[bot],
        ), patch(
            "quant.execution.fleet_api._load_display_trades_for_bot",
            return_value=inferred,
        ):
            out = build_fleet_trades(
                strategy_instance="sol-pilot-canonical",
                hours=0,
            )
        self.assertEqual(out["count"], 1)
        self.assertEqual(out["trades"][0]["trade_id"], "activity:close")
        self.assertEqual(out["trades"][0]["bot_id"], "imba-runner")
        self.assertEqual(out["trades"][0]["display_name"], "imba5")


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
