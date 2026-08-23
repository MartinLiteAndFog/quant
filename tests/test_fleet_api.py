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
    _kraken_position_events_frame,
    _load_display_trades_for_bot,
    _load_equity_snapshots,
    _normalize_account_curve,
    _risk_normalized_allocation_payload,
    _strategy_return_payload,
    build_fleet_performance,
    fleet_bot_registry,
)


class EquitySnapshotLoadTests(unittest.TestCase):
    def test_default_load_keeps_complete_history_without_sql_limit(self) -> None:
        executed: Dict[str, Any] = {}

        class _Cur:
            def execute(self, sql: str, params: Dict[str, Any]) -> None:
                executed["sql"] = " ".join(sql.split()).lower()
                executed["params"] = params

            def fetchall(self) -> List[Any]:
                return []

            def __enter__(self) -> "_Cur":
                return self

            def __exit__(self, *_a: Any) -> None:
                return None

        class _Conn:
            def cursor(self) -> _Cur:
                return _Cur()

            def __enter__(self) -> "_Conn":
                return self

            def __exit__(self, *_a: Any) -> None:
                return None

        with patch("quant.execution.fleet_api.get_conn", return_value=_Conn()):
            rows = _load_equity_snapshots(venue="kucoin", account="pilot")

        self.assertEqual(rows, [])
        self.assertNotIn(" limit ", f" {executed['sql']} ")
        self.assertTrue(str(executed["sql"]).endswith("order by ts asc"))
        self.assertNotIn("limit", executed["params"])

    def test_limit_selects_latest_rows_then_returns_them_chronologically(self) -> None:
        older = pd.Timestamp("2026-08-22T10:00:00Z")
        newer = pd.Timestamp("2026-08-23T10:00:00Z")
        executed: Dict[str, Any] = {}

        class _Cur:
            def execute(self, sql: str, params: Dict[str, Any]) -> None:
                executed["sql"] = " ".join(sql.split()).lower()
                executed["params"] = params

            def fetchall(self) -> List[Any]:
                return [
                    (older, 10.0, "USDT", "pilot", "writer"),
                    (newer, 11.0, "USDT", "pilot", "writer"),
                ]

            def __enter__(self) -> "_Cur":
                return self

            def __exit__(self, *_a: Any) -> None:
                return None

        class _Conn:
            def cursor(self) -> _Cur:
                return _Cur()

            def __enter__(self) -> "_Conn":
                return self

            def __exit__(self, *_a: Any) -> None:
                return None

        with patch("quant.execution.fleet_api.get_conn", return_value=_Conn()):
            rows = _load_equity_snapshots(
                venue="kucoin", account="pilot", limit=5000
            )

        sql = str(executed["sql"])
        self.assertIn("order by ts desc limit %(limit)s", sql)
        self.assertTrue(sql.endswith("order by ts asc"))
        self.assertEqual(executed["params"]["limit"], 5000)
        self.assertEqual([row["equity"] for row in rows], [10.0, 11.0])


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


class KrakenDirectTradeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bot = {
            "id": "kraken-legacy",
            "display_name": "Kraken Legacy",
            "strategy_instance": "kraken_bot",
            "venue": "kraken",
            "symbol": "SOL-USD",
        }

    def test_position_events_map_closes_and_openings(self) -> None:
        frame = _kraken_position_events_frame(
            [
                {
                    "executionUid": "close-long",
                    "updateReason": "trade",
                    "positionChange": "close",
                    "oldPosition": "0.5",
                    "newPosition": "0",
                    "oldAverageEntryPrice": "100",
                    "executionPrice": "110",
                    "executionSize": "0.5",
                    "fillTime": 1_800_000,
                },
                {
                    "executionUid": "open-short",
                    "updateReason": "trade",
                    "positionChange": "open",
                    "oldPosition": "0",
                    "newPosition": "-0.25",
                    "newAverageEntryPrice": "108",
                    "executionPrice": "108",
                    "executionSize": "0.25",
                    "fillTime": 1_900_000,
                },
            ],
            bot=self.bot,
            since=pd.Timestamp(0, unit="s", tz="UTC"),
        )

        self.assertEqual(frame["trade_id"].tolist(), ["kraken-batch:close-long"])
        self.assertAlmostEqual(float(frame.iloc[0]["pnl_pct"]), 10.0)
        self.assertEqual(frame.iloc[0]["side"], "long")

    def test_position_events_weight_split_fills_once_and_reach_latest_realization(self) -> None:
        frame = _kraken_position_events_frame(
            [
                {
                    "executionUid": "july-close",
                    "updateReason": "trade",
                    "positionChange": "close",
                    "oldPosition": "1",
                    "newPosition": "0",
                    "oldAverageEntryPrice": "100",
                    "executionPrice": "101",
                    "executionSize": "1",
                    "fee": "0.05",
                    "realizedPnL": "1",
                    "fillTime": 1_800_000,
                },
                {
                    "executionUid": "aug-part-1",
                    "updateReason": "trade",
                    "positionChange": "decrease",
                    "oldPosition": "1",
                    "newPosition": "0.6",
                    "oldAverageEntryPrice": "100",
                    "executionPrice": "110",
                    "executionSize": "0.4",
                    "fee": "0.04",
                    "realizedPnL": "4",
                    "fillTime": 2_800_000,
                },
                {
                    "executionUid": "aug-part-2",
                    "updateReason": "trade",
                    "positionChange": "close",
                    "oldPosition": "0.6",
                    "newPosition": "0",
                    "oldAverageEntryPrice": "100",
                    "executionPrice": "112",
                    "executionSize": "0.6",
                    "fee": "0.06",
                    "realizedPnL": "7.2",
                    "fillTime": 2_800_000,
                },
            ],
            bot=self.bot,
            since=pd.Timestamp(0, unit="s", tz="UTC"),
        )

        self.assertEqual(len(frame), 2)
        latest = frame.iloc[-1]
        self.assertEqual(latest["fill_count"], 2)
        self.assertAlmostEqual(float(latest["qty"]), 1.0)
        self.assertAlmostEqual(float(latest["exit_price"]), 111.2)
        self.assertAlmostEqual(float(latest["pnl_pct"]), 11.2)
        self.assertAlmostEqual(float(latest["fee"]), 0.1)
        self.assertEqual(int(latest["exit_ts"].timestamp()), 2_800)

    def test_position_events_activity_preserves_reductions_fees_and_funding(self) -> None:
        from quant.execution.fleet_api import _kraken_position_event_activity_items

        items = _kraken_position_event_activity_items(
            [
                {
                    "executionUid": "reduce-long",
                    "updateReason": "trade",
                    "positionChange": "decrease",
                    "oldPosition": "1.0",
                    "newPosition": "0.4",
                    "oldAverageEntryPrice": "100",
                    "executionPrice": "110",
                    "executionSize": "0.6",
                    "fee": "0.12",
                    "feeCurrency": "USD",
                    "realizedPnL": "6.0",
                    "fillTime": 1_900_000,
                    "accountUid": "acct-1",
                    "tradeable": "PF_SOLUSD",
                },
                {
                    "updateReason": "fundingRealisation",
                    "positionChange": "noChange",
                    "oldPosition": "0.4",
                    "newPosition": "0.4",
                    "realizedFunding": "-0.03",
                    "fundingRealizationTime": 2_000_000,
                    "timestamp": 2_000_000,
                    "accountUid": "acct-1",
                    "tradeable": "PF_SOLUSD",
                },
            ],
            bot=self.bot,
            since=pd.Timestamp(0, unit="s", tz="UTC"),
        )
        self.assertEqual([item["action"] for item in items], ["reduce", "funding"])
        self.assertEqual(items[0]["side"], "sell")
        self.assertEqual(items[0]["position_before"], 1.0)
        self.assertEqual(items[0]["position_after"], 0.4)
        self.assertEqual(items[0]["fee"], 0.12)
        self.assertEqual(items[0]["fee_currency"], "USD")
        self.assertEqual(items[1]["realized_funding"], -0.03)
        self.assertEqual(items[1]["t"], 2_000)
        self.assertEqual(items[0]["source"], "kraken_position_history")
        self.assertEqual(items[0]["entry_price"], 100.0)
        self.assertEqual(items[0]["exit_price"], 110.0)

    @patch("quant.execution.fleet_api.urlopen")
    def test_remote_position_event_proxy_uses_dedicated_read_token(self, urlopen) -> None:
        from quant.execution.fleet_api import (
            _KRAKEN_POSITION_EVENT_CACHE,
            _load_kraken_position_events_for_bot,
        )

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args: Any) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps({"events": [{"executionUid": "remote-1"}]}).encode()

        _KRAKEN_POSITION_EVENT_CACHE.clear()
        urlopen.return_value = Response()
        with patch.dict(
            os.environ,
            {
                "FLEET_KRAKEN_DIRECT_EVENTS_URL": "https://kraken.example/events",
                "FLEET_KRAKEN_READ_TOKEN": "read-only-token",
            },
            clear=True,
        ):
            rows = _load_kraken_position_events_for_bot(
                self.bot, since=None, limit=10_000
            )

        self.assertEqual(rows, [{"executionUid": "remote-1"}])
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer read-only-token")
        self.assertIn("limit=10000", request.full_url)

    @patch("quant.execution.kraken_futures.KrakenFuturesClient.get_position_events")
    def test_position_event_proxy_whitelists_and_reports_full_page(self, get_events) -> None:
        from quant.execution.fleet_api import build_kraken_position_events

        get_events.return_value = [
            {
                "executionUid": "first",
                "fillTime": 1_000,
                "executionPrice": "100",
                "accountUid": "must-not-leak",
            },
            {
                "executionUid": "last",
                "fundingRealizationTime": 3_000,
                "realizedFunding": "-0.1",
            },
        ]
        out = build_kraken_position_events(limit=10_000)

        self.assertEqual(out["count"], 2)
        self.assertEqual(out["oldest_ms"], 1_000)
        self.assertEqual(out["newest_ms"], 3_000)
        self.assertNotIn("accountUid", out["events"][0])
        self.assertEqual(get_events.call_args.kwargs["limit"], 10_000)
        self.assertTrue(get_events.call_args.kwargs["include_funding"])

    @patch("quant.execution.kraken_futures.KrakenFuturesClient.get_position_events")
    def test_position_event_proxy_can_skip_funding_for_fast_performance_reads(self, get_events) -> None:
        from quant.execution.fleet_api import build_kraken_position_events

        get_events.return_value = []
        out = build_kraken_position_events(limit=10_000, include_funding=False)

        self.assertFalse(out["include_funding"])
        self.assertFalse(get_events.call_args.kwargs["include_funding"])

    def test_activity_limit_retains_complete_bounded_kraken_ledger(self) -> None:
        from quant.execution.fleet_api import _limit_activity_items

        kraken = [
            {
                "id": f"kraken-{index}",
                "t": index,
                "source": "kraken_position_history",
            }
            for index in range(1, 6)
        ]
        other = [
            {"id": f"other-{index}", "t": 100 + index, "source": "database"}
            for index in range(1, 6)
        ]
        out = _limit_activity_items(kraken + other, cap=5)

        self.assertEqual(
            [row["id"] for row in out],
            ["kraken-5", "kraken-4", "kraken-3", "kraken-2", "kraken-1"],
        )

    @patch("quant.execution.fleet_api._load_kraken_exchange_trades_for_bot")
    @patch("quant.execution.fleet_api._load_execution_activity_trades_for_bot")
    @patch("quant.execution.fleet_api._load_closed_trades_for_bot")
    def test_display_loader_merges_and_deduplicates_exchange_truth(
        self,
        load_closed,
        load_inferred,
        load_direct,
    ) -> None:
        db = pd.DataFrame([_trade("2026-07-27T14:25:13Z", 2.0, "same")])
        db["venue"] = "kraken"
        db["strategy_instance"] = "kraken_bot"
        direct = pd.DataFrame(
            [
                {**db.iloc[0].to_dict(), "pnl_pct": 9.0},
                _trade("2026-08-12T12:00:00Z", -1.0, "new"),
            ]
        )
        direct["venue"] = "kraken"
        direct["strategy_instance"] = "kraken_bot"
        load_closed.return_value = db
        load_inferred.return_value = pd.DataFrame()
        load_direct.return_value = direct

        out = _load_display_trades_for_bot(self.bot, limit=20)

        self.assertEqual(out["trade_id"].tolist(), ["same", "new"])
        self.assertEqual(float(out.iloc[0]["pnl_pct"]), 2.0)

    @patch("quant.execution.fleet_api._load_kraken_exchange_trades_for_bot")
    @patch("quant.execution.fleet_api._load_kraken_position_events_for_bot")
    @patch("quant.execution.fleet_api._load_execution_activity_trades_for_bot")
    @patch("quant.execution.fleet_api._load_closed_trades_for_bot")
    def test_display_loader_prefers_complete_event_ledger_through_august(
        self,
        load_closed,
        load_inferred,
        load_events,
        load_legacy,
    ) -> None:
        load_closed.return_value = pd.DataFrame([_trade("2026-07-27T14:25:13Z", 1.0, "old")])
        load_inferred.return_value = pd.DataFrame()
        load_events.return_value = [
            {
                "executionUid": "latest",
                "updateReason": "trade",
                "positionChange": "close",
                "oldPosition": "1",
                "newPosition": "0",
                "oldAverageEntryPrice": "100",
                "executionPrice": "102",
                "executionSize": "1",
                "fillTime": int(pd.Timestamp("2026-08-23T08:40:08Z").timestamp() * 1000),
            }
        ]

        out = _load_display_trades_for_bot(self.bot, limit=20)

        self.assertEqual(out["trade_id"].tolist(), ["kraken-batch:latest"])
        self.assertEqual(out.iloc[-1]["exit_ts"], pd.Timestamp("2026-08-23T08:40:08Z"))
        load_legacy.assert_not_called()


class StrategyAndAllocationMetricTests(unittest.TestCase):
    def test_strategy_return_refuses_incomplete_historical_leverage(self) -> None:
        result = _strategy_return_payload(pd.DataFrame([_trade("2026-08-01T00:00:00Z", 2.0)]))
        self.assertFalse(result["strategy_meta"]["available"])
        self.assertEqual(
            result["strategy_meta"]["reason"],
            "historical_notional_leverage_or_cost_basis_incomplete",
        )

    def test_strategy_return_accepts_only_explicit_complete_net_rows(self) -> None:
        frame = pd.DataFrame(
            [
                {**_trade("2026-08-01T00:00:00Z", 2.0, "a"), "strategy_return_pct": 4.0, "strategy_return_complete": True},
                {**_trade("2026-08-02T00:00:00Z", 2.0, "b"), "strategy_return_pct": -1.0, "strategy_return_complete": True},
            ]
        )
        result = _strategy_return_payload(frame)
        self.assertTrue(result["strategy_meta"]["available"])
        self.assertAlmostEqual(result["strategy_curve"][-1]["equity_pct"], 2.96, places=5)

    def test_allocation_benchmark_uses_common_window_and_equal_risk(self) -> None:
        series = [
            {
                "id": "steady",
                "strategy_meta": {"available": True},
                "strategy_curve": [
                    {"t": 1, "equity_pct": 0.0},
                    {"t": 2, "equity_pct": 1.0},
                    {"t": 3, "equity_pct": 0.0},
                    {"t": 4, "equity_pct": 1.0},
                ],
            },
            {
                "id": "volatile",
                "strategy_meta": {"available": True},
                "strategy_curve": [
                    {"t": 1, "equity_pct": 0.0},
                    {"t": 2, "equity_pct": 4.0},
                    {"t": 3, "equity_pct": 0.0},
                    {"t": 4, "equity_pct": 4.0},
                ],
            },
        ]
        portfolio = [
            {"t": 1, "equity_pct": 0.0},
            {"t": 2, "equity_pct": 2.0},
            {"t": 3, "equity_pct": 1.0},
            {"t": 4, "equity_pct": 3.0},
        ]
        result = _risk_normalized_allocation_payload(series, portfolio)
        self.assertTrue(result["available"])
        self.assertEqual(result["common_start"], 1)
        self.assertEqual(result["common_end"], 4)
        self.assertEqual(result["included_bot_ids"], ["steady", "volatile"])
        self.assertIsNotNone(result["contribution_pct"])


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


class CorrectedReturnCurveTests(unittest.TestCase):
    def test_confirmed_small_deposit_is_not_mistaken_for_performance(self) -> None:
        from quant.execution.fleet_api import (
            _cashflow_corrected_curve,
            _twr_account_curve,
        )

        points = [
            {"t": 100, "equity": 100.0},
            {"t": 200, "equity": 107.0},
        ]
        flows = [
            {"t": 150, "reporting_amount": 5.0, "equity_after": None},
        ]

        ledger_curve = _cashflow_corrected_curve(points, flows)
        heuristic_curve = _twr_account_curve(points, jump_threshold_pct=10.0)

        self.assertAlmostEqual(ledger_curve[-1]["equity_pct"], 2.0, places=5)
        self.assertAlmostEqual(heuristic_curve[-1]["equity_pct"], 7.0, places=5)

    def test_confirmed_deposit_and_withdrawal_are_removed(self) -> None:
        from quant.execution.fleet_api import _cashflow_corrected_curve

        points = [
            {"t": 100, "equity": 100.0},
            {"t": 200, "equity": 160.0},
            {"t": 300, "equity": 132.0},
        ]
        flows = [
            {"t": 150, "reporting_amount": 50.0, "equity_after": None},
            {"t": 250, "reporting_amount": -20.0, "equity_after": None},
        ]
        curve = _cashflow_corrected_curve(points, flows)
        self.assertEqual(curve[0], {"t": 100, "equity_pct": 0.0})
        self.assertAlmostEqual(curve[1]["equity_pct"], 10.0, places=5)
        self.assertAlmostEqual(curve[2]["equity_pct"], 4.5, places=5)

    def test_stale_ledger_coverage_uses_twr_fallback(self) -> None:
        from quant.execution.fleet_api import _bot_corrected_payload

        state = {
            "coverage_start": pd.Timestamp(90, unit="s", tz="UTC"),
            "coverage_end": pd.Timestamp(200, unit="s", tz="UTC"),
            "last_success_at": pd.Timestamp(200, unit="s", tz="UTC"),
            "last_error": None,
            "source": "test",
        }
        with patch(
            "quant.execution.fleet_api._load_cashflow_data",
            return_value=([], state),
        ):
            result = _bot_corrected_payload(
                abs_points=[
                    {"t": 100, "equity": 100.0},
                    {"t": 300, "equity": 150.0},
                ],
                venue="kucoin",
                account="test",
                since=pd.Timestamp(100, unit="s", tz="UTC"),
                until_ts=300,
                bot_status="live",
                bot_disabled=False,
            )
        self.assertEqual(result["corrected_meta"]["method"], "jump_twr")
        self.assertEqual(
            result["corrected_meta"]["reason"],
            "ledger_coverage_incomplete",
        )
        self.assertAlmostEqual(
            result["corrected_curve"][-1]["equity_pct"], 0.0, places=5
        )

    def test_complete_ledger_coverage_uses_confirmed_flows(self) -> None:
        from quant.execution.fleet_api import _bot_corrected_payload

        state = {
            "coverage_start": pd.Timestamp(90, unit="s", tz="UTC"),
            "coverage_end": pd.Timestamp(300, unit="s", tz="UTC"),
            "last_success_at": pd.Timestamp(300, unit="s", tz="UTC"),
            "last_error": None,
            "source": "test",
        }
        flows = [
            {
                "t": 150,
                "reporting_amount": 50.0,
                "direction": "in",
                "currency": "USDT",
                "flow_type": "TransferIn",
                "equity_after": None,
            }
        ]
        with patch(
            "quant.execution.fleet_api._load_cashflow_data",
            return_value=(flows, state),
        ):
            result = _bot_corrected_payload(
                abs_points=[
                    {"t": 100, "equity": 100.0},
                    {"t": 300, "equity": 160.0},
                ],
                venue="kucoin",
                account="test",
                since=pd.Timestamp(100, unit="s", tz="UTC"),
                until_ts=300,
                bot_status="live",
                bot_disabled=False,
            )
        self.assertEqual(result["corrected_meta"]["method"], "ledger")
        self.assertEqual(result["corrected_meta"]["flow_count"], 1)
        self.assertAlmostEqual(
            result["corrected_curve"][-1]["equity_pct"], 10.0, places=5
        )


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
            out = build_fleet_performance(hours=0, instance_ids=["imba-runner"])
        self.assertTrue(out["ok"])
        self.assertEqual(len(out["series"]), 1)
        self.assertEqual(out["series"][0]["id"], "imba-runner")
        self.assertTrue(out["series"][0]["needs_backfill"])
        self.assertEqual(out["series"][0]["trade_curve"], [])
        self.assertEqual(out["series"][0]["account_curve_abs"], [])
        self.assertIn("portfolio", out)
        self.assertEqual(out["portfolio"]["id"], "portfolio")

    def test_performance_emits_exact_bps_and_does_not_infer_strategy_return(self) -> None:
        trades = pd.DataFrame(
            [
                _trade("2026-08-22T10:00:00Z", 1.0, "one-percent"),
            ]
        )
        with patch(
            "quant.execution.fleet_api._load_display_trades_for_bot",
            return_value=trades,
        ), patch(
            "quant.execution.fleet_api._load_account_points_for_bot",
            return_value=[],
        ), patch(
            "quant.execution.fleet_api.list_fleet_bots",
            return_value={"ok": True, "bots": []},
        ):
            out = build_fleet_performance(hours=0, instance_ids=["imba-runner"])

        row = out["series"][0]
        self.assertEqual(row["price_move_meta"]["return_bps"], 100.0)
        self.assertEqual(row["price_move_curve_bps"][-1]["equity_pct"], 100.0)
        self.assertFalse(row["strategy_meta"]["available"])
        self.assertEqual(row["strategy_curve"], [])
        self.assertEqual(out["portfolio"]["corrected_meta"]["method"], "jump_twr")
        self.assertFalse(out["portfolio"]["allocation"]["available"])


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
        # Portfolio TWR compounds the summed equity and ignores the capital
        # step from the late-arriving second account.
        pct_by_t = {p["t"]: p["equity_pct"] for p in port["account_curve"]}
        self.assertAlmostEqual(pct_by_t[100], 0.0, places=5)
        self.assertAlmostEqual(pct_by_t[150], 0.0, places=5)  # capital step excluded
        self.assertAlmostEqual(pct_by_t[200], 6.6666667, places=5)  # 30 -> 32
        self.assertAlmostEqual(pct_by_t[250], 13.3333333, places=5)  # 32 -> 34
        self.assertEqual(port["note"], "portfolio_twr_from_abs_sum")

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
        # The large late capital step is excluded; only portfolio growth remains.
        self.assertAlmostEqual(pct_by_t[200], -2.0, places=5)
        self.assertAlmostEqual(pct_by_t[250], -2.0, places=5)  # deposit excluded
        # Exact capital-weighted portfolio result: -2%, followed by
        # 381 / 374.7 - 1 = +1.681345%, compounds to -0.352282%.
        self.assertAlmostEqual(pct_by_t[300], -0.352282, places=5)
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


class FleetActivityUnifiedTests(unittest.TestCase):
    def test_item_helpers_unified_shape(self) -> None:
        from quant.execution.fleet_api import (
            _activity_item_from_event,
            _activity_item_from_fill,
        )

        bot = {
            "id": "quant-main",
            "display_name": "Quant",
            "strategy_instance": "quant",
            "color": "#9a8f6a",
        }
        ts = pd.Timestamp("2026-07-24T12:00:00Z")
        event = _activity_item_from_event(
            ts=ts,
            venue="kucoin",
            symbol="SOLUSDT",
            strategy_instance="quant",
            side="buy",
            qty=10.0,
            price=75.0,
            stage="market_fill",
            status="sent",
            event_id="e1",
            bot=bot,
        )
        self.assertEqual(event["kind"], "event")
        self.assertEqual(event["action"], "market_fill")
        self.assertEqual(event["status"], "sent")
        self.assertIsNone(event["pnl_pct"])

        fill_row = pd.Series(
            {
                "trade_id": "t1",
                "side": "long",
                "qty": 10.0,
                "exit_price": 76.0,
                "pnl_pct": 1.25,
                "entry_ts": ts,
                "exit_ts": ts,
                "exit_event": "tp_exit",
                "strategy_instance": "quant",
                "venue": "kucoin",
                "symbol": "SOLUSDT",
            }
        )
        fill = _activity_item_from_fill(row=fill_row, bot=bot)
        self.assertEqual(fill["kind"], "fill")
        self.assertEqual(fill["action"], "tp_exit")
        self.assertEqual(fill["status"], "closed")
        self.assertAlmostEqual(fill["pnl_pct"], 1.25)

    def test_build_fleet_activity_merges_events_and_fills(self) -> None:
        from quant.execution.fleet_api import build_fleet_activity

        bot = {
            "id": "quant-main",
            "display_name": "Quant",
            "strategy_instance": "quant",
            "trade_instances": ["quant", "live_executor"],
            "venue": "kucoin",
            "symbol": "SOL-USDT",
            "color": "#9a8f6a",
        }
        ts = pd.Timestamp("2026-07-24T12:00:00Z")

        class _Cur:
            def execute(self, *_a: Any, **_k: Any) -> None:
                return None

            def fetchall(self) -> List[Any]:
                return [
                    (
                        ts,
                        "kucoin",
                        "SOLUSDT",
                        "live_executor",
                        "sell",
                        5.0,
                        74.0,
                        "market_fill",
                        "sent",
                        "ev1",
                    )
                ]

            def __enter__(self) -> "_Cur":
                return self

            def __exit__(self, *_a: Any) -> None:
                return None

        class _Conn:
            def cursor(self) -> _Cur:
                return _Cur()

            def __enter__(self) -> "_Conn":
                return self

            def __exit__(self, *_a: Any) -> None:
                return None

        fills = pd.DataFrame(
            [
                {
                    "trade_id": "t1",
                    "side": "long",
                    "qty": 5.0,
                    "exit_price": 74.0,
                    "pnl_pct": -0.5,
                    "entry_ts": ts,
                    "exit_ts": ts + pd.Timedelta(minutes=1),
                    "exit_event": "sl_exit",
                    "strategy_instance": "quant",
                    "venue": "kucoin",
                    "symbol": "SOLUSDT",
                }
            ]
        )

        with patch("quant.execution.fleet_api.fleet_bot_registry", return_value=[bot]), patch(
            "quant.execution.fleet_api.get_conn", return_value=_Conn()
        ), patch(
            "quant.execution.fleet_api._load_display_trades_for_bot", return_value=fills
        ):
            out = build_fleet_activity(hours=24, limit=50)

        self.assertTrue(out["ok"])
        self.assertEqual(out["count"], 2)
        kinds = {i["kind"] for i in out["items"]}
        self.assertEqual(kinds, {"event", "fill"})
        # Alias live_executor mapped to Quant bot.
        event = next(i for i in out["items"] if i["kind"] == "event")
        self.assertEqual(event["bot_id"], "quant-main")
        self.assertEqual(event["display_name"], "Quant")
        # Legacy events list excludes fills.
        self.assertEqual(len(out["events"]), 1)
        self.assertEqual(out["events"][0]["kind"], "event")


if __name__ == "__main__":
    unittest.main()
