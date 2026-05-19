from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import quant.execution.dashboard_state as ds
from quant.execution.dashboard_state import (
    build_regime_overlay,
    load_active_levels,
    load_dashboard_strategy,
    load_fills_cache_rows,
    load_live_fill_markers,
    load_renko_bars,
    load_trade_markers,
)
from quant.regime import RegimeDecision, RegimeService, RegimeStore


class DashboardStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        os.environ["REGIME_DB_PATH"] = str(self.tmp_path / "regime.db")
        os.environ["DASHBOARD_RENKO_PARQUET"] = str(self.tmp_path / "renko.parquet")
        os.environ["DASHBOARD_RENKO_AUTO_REFRESH_ON_READ"] = "0"
        os.environ["DASHBOARD_LEVELS_JSON"] = str(self.tmp_path / "execution_state.json")
        os.environ["DASHBOARD_TRADES_PARQUET"] = str(self.tmp_path / "trades.parquet")
        ds._LAST_REFRESH_TS = None
        ds._LAST_REFRESH_ERROR = None
        ds._LAST_FILLS_REFRESH_TS = None
        ds._LAST_FILLS_REFRESH_ERROR = None

        # Seed renko parquet
        renko = pd.DataFrame(
            {
                "ts": pd.date_range("2026-02-20", periods=3, freq="h", tz="UTC"),
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.5, 100.5, 101.5],
                "close": [100.8, 101.7, 102.8],
            }
        )
        renko.to_parquet(self.tmp_path / "renko.parquet", index=False)

        # Seed execution levels
        (self.tmp_path / "execution_state.json").write_text(
            json.dumps({"sl": 99.1, "ttp": 103.2, "tp1": 104.0, "tp2": 105.5}),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {"entry_ts": "2026-02-20T00:00:00Z", "exit_ts": "2026-02-20T00:10:00Z", "side": 1, "exit_event": "tp_exit"},
                {"entry_ts": "2026-02-20T01:00:00Z", "exit_ts": "2026-02-20T01:05:00Z", "side": -1, "exit_event": "sl_exit"},
                {"entry_ts": "2026-02-20T02:00:00Z", "exit_ts": "2026-02-20T02:30:00Z", "side": 1, "exit_event": "signal_flip_exit"},
            ]
        ).to_parquet(self.tmp_path / "trades.parquet", index=False)

        store = RegimeStore()
        svc = RegimeService(store)
        svc.upsert_decision(
            RegimeDecision(
                ts="2026-02-20T00:00:00Z",
                symbol="SOL-USDT",
                gate_on=1,
                regime_state="trend",
                regime_score=0.8,
                confidence=0.7,
                reason_code="seed",
            )
        )
        svc.upsert_decision(
            RegimeDecision(
                ts="2026-02-21T00:00:00Z",
                symbol="SOL-USDT",
                gate_on=0,
                regime_state="countertrend",
                regime_score=-0.9,
                confidence=0.9,
                reason_code="flip",
            )
        )

    def tearDown(self) -> None:
        os.environ.pop("DASHBOARD_FILLS_PARQUET", None)
        os.environ.pop("DASHBOARD_EXPECTED_TRADES_JSONL", None)
        os.environ.pop("DASHBOARD_FILLS_REFRESH_COOLDOWN_SEC", None)
        os.environ.pop("DASHBOARD_FILLS_AUTO_REFRESH_ON_READ", None)
        self.tmp.cleanup()

    def test_load_renko_bars(self) -> None:
        bars = load_renko_bars(max_points=10)
        self.assertEqual(len(bars), 3)
        self.assertIn("open", bars[0])
        self.assertIn("time", bars[0])
        # ``time`` must be epoch *seconds* regardless of the underlying
        # datetime64 precision (pandas 2.x reads parquet as us-resolution by
        # default, which historically collapsed every bar timestamp to
        # ``epoch_seconds // 1000`` and pushed markers off to the chart edge).
        expected_first = int(pd.Timestamp("2026-02-20", tz="UTC").timestamp())
        self.assertEqual(int(bars[0]["time"]), expected_first)
        self.assertGreater(int(bars[0]["time"]), 1_000_000_000)

    def test_load_levels(self) -> None:
        levels = load_active_levels()
        self.assertAlmostEqual(float(levels["sl"]), 99.1, places=6)
        self.assertAlmostEqual(float(levels["tp2"]), 105.5, places=6)

    def test_regime_overlay(self) -> None:
        overlay = build_regime_overlay(symbol="SOL-USDT", hours=24 * 90)
        self.assertTrue(len(overlay["points"]) >= 2)
        self.assertTrue(len(overlay["spans"]) >= 1)
        self.assertIn("latest", overlay)
        # OFF regime should be rendered in red family downstream.
        latest = overlay["latest"]
        self.assertEqual(int(latest["gate_on"]), 0)

    def test_load_trade_markers_returns_all_trades(self) -> None:
        markers = load_trade_markers(max_points=100000)
        # 3 trades -> 3 entries + 3 exits
        self.assertEqual(len(markers), 6)

    def test_load_trade_markers_uses_real_entry_and_exit_timestamps(self) -> None:
        entry1 = pd.Timestamp("2026-04-01T10:00:00Z")
        exit1 = pd.Timestamp("2026-04-01T10:15:00Z")
        entry2 = pd.Timestamp("2026-04-02T11:00:00Z")
        exit2 = pd.Timestamp("2026-04-02T11:30:00Z")
        entry3 = pd.Timestamp("2026-04-03T12:00:00Z")
        exit3 = pd.Timestamp("2026-04-03T13:00:00Z")
        df = pd.DataFrame(
            {
                "trade_id": ["t1", "t2", "t3"],
                "venue": ["kucoin"] * 3,
                "symbol": ["SOL-USDT"] * 3,
                "entry_ts": [entry1, entry2, entry3],
                "exit_ts": [exit1, exit2, exit3],
                "side": ["long", "short", "long"],
                "qty": [1.0, 1.0, 1.0],
                "entry_price": [82.5, 84.3, 81.0],
                "exit_price": [83.0, 82.9, 81.5],
                "pnl_pct": [0.6, 1.6, 0.6],
                "exit_event": ["tp_exit", "signal_flip_exit", "tp_exit"],
            }
        )
        markers = ds.load_trade_markers(max_points=100, _trades_df=df)
        self.assertEqual(len(markers), 6)
        # Markers must carry the real per-trade timestamps, never the same
        # min-time or a fallback to the chart start.
        seen_times = [int(m["time"]) for m in markers]
        self.assertEqual(
            sorted(seen_times),
            sorted(
                int(t.timestamp())
                for t in (entry1, exit1, entry2, exit2, entry3, exit3)
            ),
        )
        self.assertEqual(len(set(seen_times)), len(seen_times))
        # Long entry should sit below the bar with an up arrow; short entry
        # above with a down arrow.
        first_long_entry = next(
            m for m in markers if int(m["time"]) == int(entry1.timestamp())
        )
        self.assertEqual(first_long_entry["position"], "belowBar")
        self.assertEqual(first_long_entry["shape"], "arrowUp")
        self.assertTrue(first_long_entry["text"].startswith("L"))

        short_entry = next(
            m for m in markers if int(m["time"]) == int(entry2.timestamp())
        )
        self.assertEqual(short_entry["position"], "aboveBar")
        self.assertEqual(short_entry["shape"], "arrowDown")
        self.assertTrue(short_entry["text"].startswith("S"))

        long_exit = next(
            m for m in markers if int(m["time"]) == int(exit1.timestamp())
        )
        self.assertEqual(long_exit["position"], "aboveBar")
        self.assertEqual(long_exit["shape"], "arrowDown")

    @patch("quant.execution.dashboard_state.list_fills")
    def test_load_live_fill_markers_parses_microsecond_trade_time(self, mock_list_fills) -> None:
        os.environ["DASHBOARD_FILLS_PARQUET"] = str(self.tmp_path / "fills_cache.parquet")
        os.environ["DASHBOARD_FILLS_REFRESH_COOLDOWN_SEC"] = "0"
        ts = pd.Timestamp("2026-02-27T17:30:00Z")
        trade_time_us = int(ts.value // 1_000)  # microseconds since epoch
        mock_list_fills.return_value = [
            {
                "tradeTime": trade_time_us,  # no createdAt -> forces tradeTime parsing
                "side": "buy",
                "size": 20,
                "price": 83.0,
            }
        ]
        markers = load_live_fill_markers(symbol="SOL-USDT", limit=10, start_ts=int(ts.timestamp()) - 60)
        self.assertEqual(len(markers), 1)
        self.assertEqual(int(markers[0]["time"]), int(ts.timestamp()))

    @patch("quant.execution.dashboard_state.list_fills")
    def test_load_live_fill_markers_respects_refresh_cooldown(self, mock_list_fills) -> None:
        os.environ["DASHBOARD_FILLS_PARQUET"] = str(self.tmp_path / "fills_cache.parquet")
        os.environ["DASHBOARD_FILLS_REFRESH_COOLDOWN_SEC"] = "999"
        ts = pd.Timestamp("2026-02-27T17:40:00Z")
        mock_list_fills.return_value = [
            {
                "createdAt": int(ts.timestamp() * 1000),
                "side": "sell",
                "size": 2,
                "price": 84.0,
            }
        ]
        m1 = load_live_fill_markers(symbol="SOL-USDT", limit=10, start_ts=int(ts.timestamp()) - 60)
        m2 = load_live_fill_markers(symbol="SOL-USDT", limit=10, start_ts=int(ts.timestamp()) - 60)
        self.assertEqual(len(m1), 1)
        self.assertEqual(len(m2), 1)
        self.assertEqual(mock_list_fills.call_count, 1)

    def test_load_fills_cache_rows_prefers_client_oid_reason_mapping(self) -> None:
        fills_path = self.tmp_path / "fills_cache.parquet"
        expected_path = self.tmp_path / "expected_trades.jsonl"
        os.environ["DASHBOARD_FILLS_PARQUET"] = str(fills_path)
        os.environ["DASHBOARD_EXPECTED_TRADES_JSONL"] = str(expected_path)

        base_ts = int(pd.Timestamp("2026-02-27T17:30:00Z").timestamp())
        pd.DataFrame(
            [
                {
                    "time": base_ts,
                    "side": "buy",
                    "size": 5.0,
                    "price": 83.0,
                    "client_oid": "manual-flatten-short-001",
                },
                {
                    "time": base_ts + 1,
                    "side": "buy",
                    "size": 5.0,
                    "price": 83.0,
                    "client_oid": "other-oid",
                },
            ]
        ).to_parquet(fills_path, index=False)

        expected_path.write_text(
            "\n".join(
                [
                    '{"ts":"2026-02-27T17:29:58Z","symbol":"SOL-USDT","side":"short","action":"exit_sl","qty":5,"client_oid":"manual-flatten-short-001","note":"event=manual_flatten_short source=test"}',
                    '{"ts":"2026-02-27T17:29:59Z","symbol":"SOL-USDT","side":"short","action":"exit_sl","qty":5,"client_oid":"different-oid","note":"event=sl_exit source=test"}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        rows = load_fills_cache_rows(max_points=10)
        self.assertEqual(len(rows), 2)
        row = next(r for r in rows if r.get("client_oid") == "manual-flatten-short-001")
        self.assertEqual(row.get("reason"), "manual_flatten_short")

    @patch("quant.execution.dashboard_state.get_live_gate_state", create=True)
    def test_load_dashboard_strategy_prefers_daily_gate_regime_label(self, mock_gate_state) -> None:
        mock_gate_state.return_value = {
            "gate_on": 1,
            "gate_countertrend_on": 1,
            "gate_trend_on": 0,
            "source": "postgres_daily_gate",
        }
        (self.tmp_path / "execution_state.json").write_text(
            json.dumps({"mode": "TP2", "strategy": "tp2"}),
            encoding="utf-8",
        )

        out = load_dashboard_strategy(symbol="SOL-USDT")

        self.assertEqual(out.get("strategy_label"), "countertrend")
        self.assertEqual(out.get("regime_state"), "countertrend")
        self.assertEqual(out.get("source"), "daily_gate")

    def test_renko_functions_accept_preloaded_df(self) -> None:
        """Renko functions should accept a pre-loaded DataFrame to avoid redundant reads."""
        df = pd.DataFrame({
            "ts": pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC"),
            "open": range(100, 110),
            "high": range(101, 111),
            "low": range(99, 109),
            "close": range(100, 110),
        })

        bars = ds.load_renko_bars(max_points=100, _df=df)
        self.assertEqual(len(bars), 10)

        health = ds.load_renko_health(_df=df)
        self.assertTrue(health["ok"])
        self.assertEqual(health["bars"], 10)

        fibo = ds.build_fibo_levels(max_points=100, lookback=3, _df=df)
        self.assertIn("long", fibo)
        self.assertGreater(len(fibo["long"]), 0)
        self.assertGreater(len(fibo["mid"]), 0)
        self.assertGreater(len(fibo["short"]), 0)
        self.assertAlmostEqual(float(fibo["latest"]["long"]), float(fibo["long"][-1]["value"]), places=6)
        self.assertAlmostEqual(float(fibo["latest"]["mid"]), float(fibo["mid"][-1]["value"]), places=6)
        self.assertAlmostEqual(float(fibo["latest"]["short"]), float(fibo["short"][-1]["value"]), places=6)

    def test_trade_functions_accept_preloaded_df(self) -> None:
        """Trade functions should accept a pre-loaded trades DataFrame."""
        df = pd.DataFrame({
            "trade_id": ["t1"],
            "venue": ["kucoin"],
            "symbol": ["SOL-USDT"],
            "entry_ts": [pd.Timestamp("2025-01-01", tz="UTC")],
            "exit_ts": [pd.Timestamp("2025-01-02", tz="UTC")],
            "side": ["long"],
            "qty": [1.0],
            "entry_price": [100.0],
            "exit_price": [105.0],
            "pnl_pct": [5.0],
            "exit_event": ["tp1"],
        })

        markers = ds.load_trade_markers(max_points=100, _trades_df=df)
        self.assertGreater(len(markers), 0)

        diary = ds.build_trading_diary(max_points=100, _trades_df=df)
        self.assertGreater(len(diary.get("entries", [])), 0)

        equity = ds.build_equity_curve(max_points=100, _trades_df=df)
        trades = equity.get("trades", [])
        self.assertGreater(len(trades), 0)
        first = trades[0]
        self.assertIn("entry_time", first)
        self.assertIn("exit_time", first)
        self.assertIn("entry_price", first)
        self.assertIn("exit_price", first)
        self.assertIn("side", first)
        self.assertEqual(
            int(first["entry_time"]),
            int(pd.Timestamp("2025-01-01", tz="UTC").timestamp()),
        )
        self.assertEqual(
            int(first["exit_time"]),
            int(pd.Timestamp("2025-01-02", tz="UTC").timestamp()),
        )
        self.assertEqual(int(first["exit_time"]), int(first["time"]))
        self.assertAlmostEqual(float(first["entry_price"]), 100.0)
        self.assertAlmostEqual(float(first["exit_price"]), 105.0)

    def test_regime_functions_accept_preloaded_rows(self) -> None:
        """Regime functions should accept pre-loaded rows to avoid duplicate queries."""
        rows = [
            {"ts": "2025-01-01T00:00:00+00:00", "gate_on": 1, "confidence": 0.8, "regime_state": "trend", "regime_score": 0.6},
            {"ts": "2025-01-01T01:00:00+00:00", "gate_on": 0, "confidence": 0.3, "regime_state": "range", "regime_score": -0.2},
        ]
        overlay = ds.build_regime_overlay(symbol="SOL-USDT", hours=168, _rows=rows)
        self.assertGreater(len(overlay["spans"]), 0)
        self.assertIsNotNone(overlay["latest"])

        scores = ds.build_regime_scores(symbol="SOL-USDT", hours=168, _rows=rows)
        self.assertGreater(len(scores["scores"]), 0)

    def test_build_trading_diary_queries_postgres_once(self) -> None:
        """build_trading_diary should only call load_closed_trades_from_postgres once."""
        call_count = 0
        original_fn = ds.load_closed_trades_from_postgres

        def counting_wrapper(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_fn(*args, **kwargs)

        with patch.object(ds, "load_closed_trades_from_postgres", side_effect=counting_wrapper):
            ds.build_trading_diary(max_points=100)

        self.assertLessEqual(call_count, 1, f"Expected at most 1 Postgres call, got {call_count}")

    def test_build_trading_diary_merges_partial_exits_into_logical_trade(self) -> None:
        df = pd.DataFrame(
            {
                "trade_id": ["a1", "a2"],
                "venue": ["kucoin", "kucoin"],
                "symbol": ["SOL-USDT", "SOL-USDT"],
                "entry_ts": [
                    pd.Timestamp("2026-03-20T10:00:00Z"),
                    pd.Timestamp("2026-03-20T10:00:00Z"),
                ],
                "exit_ts": [
                    pd.Timestamp("2026-03-20T10:05:00Z"),
                    pd.Timestamp("2026-03-20T10:08:00Z"),
                ],
                "side": ["long", "long"],
                "qty": [1.0, 3.0],
                "entry_price": [100.0, 100.0],
                "exit_price": [102.0, 104.0],
                "pnl_pct": [2.0, 4.0],
                "exit_event": ["tp1", "tp2"],
            }
        )
        diary = ds.build_trading_diary(max_points=100, _trades_df=df)
        entries = diary.get("entries", [])
        self.assertEqual(len(entries), 1)
        self.assertAlmostEqual(float(entries[0]["pnl_pct"]), 3.5, places=6)

    def test_build_trading_diary_filters_preloaded_rows_by_symbol_and_venue(self) -> None:
        df = pd.DataFrame(
            {
                "trade_id": ["k1", "x1"],
                "venue": ["kucoin", "kraken"],
                "symbol": ["SOL-USDT", "SOL-USDT"],
                "entry_ts": [pd.Timestamp("2026-03-20T10:00:00Z"), pd.Timestamp("2026-03-20T11:00:00Z")],
                "exit_ts": [pd.Timestamp("2026-03-20T10:08:00Z"), pd.Timestamp("2026-03-20T11:08:00Z")],
                "side": ["long", "long"],
                "qty": [1.0, 1.0],
                "entry_price": [100.0, 100.0],
                "exit_price": [101.0, 120.0],
                "pnl_pct": [1.0, 20.0],
                "exit_event": ["tp1", "tp1"],
            }
        )
        diary = ds.build_trading_diary(max_points=100, symbol="SOL-USDT", venue="kucoin", _trades_df=df)
        entries = diary.get("entries", [])
        self.assertEqual(len(entries), 1)
        self.assertAlmostEqual(float(entries[0]["pnl_pct"]), 1.0, places=6)

    def test_build_trading_diary_normalizes_symbol_format(self) -> None:
        df = pd.DataFrame(
            {
                "trade_id": ["k1"],
                "venue": ["kucoin"],
                "symbol": ["SOLUSDT"],
                "entry_ts": [pd.Timestamp("2026-03-20T10:00:00Z")],
                "exit_ts": [pd.Timestamp("2026-03-20T10:08:00Z")],
                "side": ["long"],
                "qty": [1.0],
                "entry_price": [100.0],
                "exit_price": [101.0],
                "pnl_pct": [1.0],
                "exit_event": ["tp_exit"],
                "strategy": ["live_executor"],
            }
        )
        diary = ds.build_trading_diary(max_points=100, symbol="SOL-USDT", venue="kucoin", _trades_df=df)
        self.assertEqual(len(diary.get("entries", [])), 1)

    def test_build_trading_diary_live_only_excludes_reconstruction(self) -> None:
        df = pd.DataFrame(
            {
                "trade_id": ["live1", "reco1"],
                "venue": ["kucoin", "kucoin"],
                "symbol": ["SOLUSDT", "SOLUSDT"],
                "entry_ts": [pd.Timestamp("2026-03-20T10:00:00Z"), pd.Timestamp("2026-03-20T11:00:00Z")],
                "exit_ts": [pd.Timestamp("2026-03-20T10:08:00Z"), pd.Timestamp("2026-03-20T11:08:00Z")],
                "side": ["long", "short"],
                "qty": [1.0, 1.0],
                "entry_price": [100.0, 100.0],
                "exit_price": [101.0, 95.0],
                "pnl_pct": [1.0, 5.0],
                "exit_event": ["tp_exit", "fills_reconstructed"],
                "strategy": ["live_executor", "dashboard_fills_reconstruction"],
            }
        )
        diary = ds.build_trading_diary(
            max_points=100,
            symbol="SOL-USDT",
            venue="kucoin",
            live_only=True,
            include_reconstructed=False,
            allow_fill_reconstruction=False,
            _trades_df=df,
        )
        entries = diary.get("entries", [])
        self.assertEqual(len(entries), 1)
        self.assertAlmostEqual(float(entries[0]["pnl_pct"]), 1.0, places=6)

    def test_build_trading_diary_live_only_keeps_distinct_same_entry_rows(self) -> None:
        df = pd.DataFrame(
            {
                "trade_id": ["t1", "t2"],
                "venue": ["kucoin", "kucoin"],
                "symbol": ["SOLUSDT", "SOLUSDT"],
                "entry_ts": [pd.Timestamp("2026-03-20T10:00:00Z"), pd.Timestamp("2026-03-20T10:00:00Z")],
                "exit_ts": [pd.Timestamp("2026-03-20T10:05:00Z"), pd.Timestamp("2026-03-20T10:08:00Z")],
                "side": ["short", "short"],
                "qty": [1.0, 1.0],
                "entry_price": [100.0, 100.0],
                "exit_price": [99.0, 101.0],
                "pnl_pct": [1.0, -1.0],
                "exit_event": ["signal_flip_exit", "signal_flip_exit"],
                "strategy": ["live_executor", "live_executor"],
            }
        )
        diary = ds.build_trading_diary(
            max_points=100,
            symbol="SOL-USDT",
            venue="kucoin",
            live_only=True,
            include_reconstructed=False,
            allow_fill_reconstruction=False,
            _trades_df=df,
        )
        entries = diary.get("entries", [])
        self.assertEqual(len(entries), 2)
        self.assertEqual([round(float(e["pnl_pct"]), 4) for e in entries], [1.0, -1.0])

    def test_reconstruct_trades_from_execution_fills_keeps_partials_in_single_trade(self) -> None:
        fills = pd.DataFrame(
            {
                "ts": [
                    pd.Timestamp("2026-03-20T10:00:00Z"),
                    pd.Timestamp("2026-03-20T10:05:00Z"),
                    pd.Timestamp("2026-03-20T10:08:00Z"),
                    pd.Timestamp("2026-03-20T10:10:00Z"),
                    pd.Timestamp("2026-03-20T10:14:00Z"),
                ],
                "seq": [1, 2, 3, 4, 5],
                "side": ["buy", "sell", "sell", "sell", "buy"],
                "qty": [1.0, 0.5, 0.5, 1.0, 1.0],
                "price": [100.0, 102.0, 104.0, 103.0, 100.0],
            }
        )
        trades = ds._reconstruct_trades_from_execution_fills_df(
            fills_df=fills,
            max_points=100,
            source="test",
        )
        self.assertEqual(len(trades), 2)
        self.assertAlmostEqual(float(trades[0]["pnl_pct"]), 3.0, places=4)
        self.assertGreater(float(trades[1]["pnl_pct"]), 0.0)

    @patch("quant.execution.dashboard_state.load_execution_fills_from_postgres")
    def test_build_trading_diary_live_only_prefers_execution_events(self, mock_load_exec_fills) -> None:
        mock_load_exec_fills.return_value = pd.DataFrame(
            {
                "ts": [
                    pd.Timestamp("2026-03-20T10:00:00Z"),
                    pd.Timestamp("2026-03-20T10:05:00Z"),
                ],
                "seq": [1, 2],
                "side": ["buy", "sell"],
                "qty": [1.0, 1.0],
                "price": [100.0, 102.0],
                "execution_stage": ["fill", "fill"],
                "status": ["fill", "fill"],
                "reduce_only": [False, True],
                "payload_json": [{}, {}],
            }
        )
        diary = ds.build_trading_diary(
            max_points=100,
            symbol="SOL-USDT",
            venue="kucoin",
            live_only=True,
            include_reconstructed=False,
            allow_file_fallback=False,
            allow_fill_reconstruction=False,
        )
        self.assertEqual(diary.get("source"), "postgres:execution_events_reconstructed")
        self.assertEqual(len(diary.get("entries", [])), 1)

    @patch("quant.execution.dashboard_state.load_closed_trades_from_postgres")
    def test_dashboard_performance_uses_logical_trades_and_neutral_bucket(self, mock_load_trades) -> None:
        mock_load_trades.return_value = pd.DataFrame(
            {
                "trade_id": ["a1", "a2", "b1", "c1"],
                "venue": ["kucoin", "kucoin", "kucoin", "kucoin"],
                "symbol": ["SOL-USDT", "SOL-USDT", "SOL-USDT", "SOL-USDT"],
                "entry_ts": [
                    pd.Timestamp("2026-03-20T10:00:00Z"),
                    pd.Timestamp("2026-03-20T10:00:00Z"),
                    pd.Timestamp("2026-03-20T11:00:00Z"),
                    pd.Timestamp("2026-03-20T12:00:00Z"),
                ],
                "exit_ts": [
                    pd.Timestamp("2026-03-20T10:05:00Z"),
                    pd.Timestamp("2026-03-20T10:08:00Z"),
                    pd.Timestamp("2026-03-20T11:07:00Z"),
                    pd.Timestamp("2026-03-20T12:09:00Z"),
                ],
                "side": ["long", "long", "short", "long"],
                "qty": [1.0, 3.0, 2.0, 1.0],
                "entry_price": [100.0, 100.0, 105.0, 110.0],
                "exit_price": [102.0, 104.0, 104.0, 110.0],
                "pnl_pct": [2.0, 4.0, -1.0, 0.0],
                "exit_event": ["tp1", "tp2", "sl_exit", "be_exit"],
                "strategy": ["live_executor", "live_executor", "live_executor", "live_executor"],
            }
        )

        perf = ds.build_dashboard_performance(symbol="SOL-USDT", venue="kucoin", max_points=100)
        self.assertEqual(int(perf["trade_count"]), 3)
        self.assertEqual(int(perf["winning_trade_count"]), 1)
        self.assertEqual(int(perf["losing_trade_count"]), 1)
        self.assertAlmostEqual(float(perf["average_gain"]), (3.5 - 1.0 + 0.0) / 3.0, places=4)


if __name__ == "__main__":
    unittest.main()
