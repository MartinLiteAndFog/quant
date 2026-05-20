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

    def test_load_trade_markers_returns_entry_arrow_only_for_priceless_rows(self) -> None:
        # Seeded trades.parquet rows carry side + entry/exit_ts but no
        # entry_price / exit_price / pnl_pct. New contract: one arrow
        # marker per trade at the entry timestamp, no pnl-text companion,
        # no exit marker.
        markers = load_trade_markers(max_points=100000)
        self.assertEqual(len(markers), 3)
        for m in markers:
            self.assertIn(m["shape"], ("arrowUp", "arrowDown"))
            self.assertEqual(m["text"], "")
        seeded_entry_ts = {
            int(pd.Timestamp(s, tz="UTC").timestamp())
            for s in (
                "2026-02-20T00:00:00Z",
                "2026-02-20T01:00:00Z",
                "2026-02-20T02:00:00Z",
            )
        }
        self.assertEqual({int(m["time"]) for m in markers}, seeded_entry_ts)

    def test_load_trade_markers_emits_entry_arrow_and_pnl_text(self) -> None:
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
                "pnl_pct": [0.6, -1.6, 0.6],
                "exit_event": ["tp_exit", "signal_flip_exit", "tp_exit"],
            }
        )
        markers = ds.load_trade_markers(max_points=100, _trades_df=df)

        # 3 trades -> 3 arrows + 3 pnl-text companions = 6 markers, all
        # anchored to entry timestamps. No exit markers at all.
        self.assertEqual(len(markers), 6)
        entry_ts_set = {int(t.timestamp()) for t in (entry1, entry2, entry3)}
        exit_ts_set = {int(t.timestamp()) for t in (exit1, exit2, exit3)}
        seen_times = {int(m["time"]) for m in markers}
        self.assertEqual(seen_times, entry_ts_set)
        self.assertFalse(seen_times & exit_ts_set)

        long_entry_at = int(entry1.timestamp())
        long_markers = [m for m in markers if int(m["time"]) == long_entry_at]
        self.assertEqual(len(long_markers), 2)
        long_arrow = next(m for m in long_markers if m["shape"] == "arrowUp")
        self.assertEqual(long_arrow["position"], "belowBar")
        self.assertEqual(long_arrow["color"], "#22c55e")
        self.assertEqual(long_arrow["text"], "")
        self.assertEqual(int(long_arrow["size"]), 2)
        long_text = next(m for m in long_markers if m["shape"] != "arrowUp")
        # Winning long -> green text companion at the entry timestamp.
        self.assertEqual(long_text["position"], "belowBar")
        self.assertEqual(long_text["color"], "#22c55e")
        self.assertTrue(long_text["text"].endswith("%"))
        self.assertTrue(long_text["text"].startswith("+"))

        short_entry_at = int(entry2.timestamp())
        short_markers = [m for m in markers if int(m["time"]) == short_entry_at]
        self.assertEqual(len(short_markers), 2)
        short_arrow = next(m for m in short_markers if m["shape"] == "arrowDown")
        self.assertEqual(short_arrow["position"], "aboveBar")
        self.assertEqual(short_arrow["color"], "#ef4444")
        self.assertEqual(short_arrow["text"], "")
        self.assertEqual(int(short_arrow["size"]), 2)
        short_text = next(m for m in short_markers if m["shape"] != "arrowDown")
        # Losing short -> red text companion (negative pnl).
        self.assertEqual(short_text["position"], "aboveBar")
        self.assertEqual(short_text["color"], "#ef4444")
        self.assertTrue(short_text["text"].startswith("-"))
        self.assertTrue(short_text["text"].endswith("%"))

    def test_load_trade_markers_pairs_pnl_text_with_same_trade(self) -> None:
        entry1 = pd.Timestamp("2026-04-01T10:00:00Z")
        entry2 = pd.Timestamp("2026-04-01T10:01:00Z")
        df = pd.DataFrame(
            {
                "trade_id": ["first_trade", "second_trade"],
                "venue": ["kucoin"] * 2,
                "symbol": ["SOL-USDT"] * 2,
                "entry_ts": [entry1, entry2],
                "exit_ts": [
                    entry1 + pd.Timedelta(minutes=5),
                    entry2 + pd.Timedelta(minutes=5),
                ],
                "side": ["long", "short"],
                "qty": [1.0, 1.0],
                "entry_price": [100.0, 100.0],
                "exit_price": [101.0, 103.0],
                "pnl_pct": [1.0, -3.0],
                "exit_event": ["tp_exit", "sl_exit"],
            }
        )

        markers = ds.load_trade_markers(max_points=100, _trades_df=df)

        self.assertEqual(len(markers), 4)
        for arrow, label, trade_id, expected_text in zip(
            markers[0::2],
            markers[1::2],
            ["first_trade", "second_trade"],
            ["+1.00%", "-3.00%"],
        ):
            self.assertEqual(arrow.get("trade_id"), trade_id)
            self.assertEqual(label.get("trade_id"), trade_id)
            self.assertEqual(label.get("text"), expected_text)
            self.assertEqual(int(label["time"]), int(arrow["time"]))
            self.assertEqual(label["position"], arrow["position"])

    def test_load_trade_markers_skips_text_for_open_entry(self) -> None:
        entry1 = pd.Timestamp("2026-04-01T10:00:00Z")
        exit1 = pd.Timestamp("2026-04-01T10:15:00Z")
        entry_open = pd.Timestamp("2026-04-02T09:00:00Z")
        exit_open_placeholder = pd.Timestamp("2026-04-02T09:30:00Z")
        df = pd.DataFrame(
            {
                "trade_id": ["t1", "t_open"],
                "venue": ["kucoin"] * 2,
                "symbol": ["SOL-USDT"] * 2,
                "entry_ts": [entry1, entry_open],
                "exit_ts": [exit1, exit_open_placeholder],
                "side": ["long", "long"],
                "qty": [1.0, 1.0],
                "entry_price": [82.5, 83.0],
                "exit_price": [83.0, 84.0],
                "pnl_pct": [0.6, 1.2],
                "exit_event": ["tp_exit", "tp_exit"],
            }
        )
        markers = ds.load_trade_markers(
            max_points=100,
            _trades_df=df,
            open_entry_ts=int(entry_open.timestamp()),
        )
        at_open = [m for m in markers if int(m["time"]) == int(entry_open.timestamp())]
        # The "open" trade keeps its arrow but drops the pnl-text companion.
        self.assertEqual(len(at_open), 1)
        self.assertEqual(at_open[0]["shape"], "arrowUp")
        self.assertEqual(at_open[0]["text"], "")
        # The other trade still ships both arrow + text.
        at_closed = [m for m in markers if int(m["time"]) == int(entry1.timestamp())]
        self.assertEqual(len(at_closed), 2)

    def test_load_trade_markers_emits_open_arrow_from_live_levels(self) -> None:
        # When ``execution_state.json`` carries an open position the loader
        # appends a direction-colored arrow at that timestamp even though
        # it isn't in ``closed_trades``. ``time + shape`` match dedupes
        # the legacy live-entry marker that the chart endpoint emits
        # separately.
        open_entry = int(pd.Timestamp("2026-02-20T05:00:00Z").timestamp())
        (self.tmp_path / "execution_state.json").write_text(
            json.dumps(
                {
                    "side": "short",
                    "entry_bar_ts": open_entry,
                    "sl": 99.1,
                }
            ),
            encoding="utf-8",
        )
        # Empty trades.parquet so only the open marker comes out.
        pd.DataFrame(
            columns=["entry_ts", "exit_ts", "side", "pnl_pct"]
        ).to_parquet(self.tmp_path / "trades.parquet", index=False)
        markers = load_trade_markers(max_points=100)
        self.assertEqual(len(markers), 1)
        open_marker = markers[0]
        self.assertEqual(int(open_marker["time"]), open_entry)
        self.assertEqual(open_marker["shape"], "arrowDown")
        self.assertEqual(open_marker["position"], "aboveBar")
        self.assertEqual(open_marker["color"], "#ef4444")
        self.assertEqual(open_marker["text"], "")

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

    def test_build_equity_curve_emits_real_entry_time_for_postgres_rows(self) -> None:
        # Simulate the postgres `closed_trades` direct-row path used by
        # ``build_trading_diary(live_only=True)``: real entry_ts must round-trip
        # to the equity curve's ``entry_time`` as the *real* epoch seconds
        # value, never silently collapsed to 0 / 1.
        entry1 = pd.Timestamp("2026-05-15T14:00:00Z")
        exit1 = pd.Timestamp("2026-05-15T15:32:38Z")
        entry2 = pd.Timestamp("2026-05-16T09:00:00Z")
        exit2 = pd.Timestamp("2026-05-16T10:00:00Z")
        df = pd.DataFrame(
            {
                "trade_id": ["t_real_a", "t_real_b"],
                "venue": ["kucoin", "kucoin"],
                "symbol": ["SOL-USDT", "SOL-USDT"],
                "entry_ts": [entry1, entry2],
                "exit_ts": [exit1, exit2],
                "side": ["long", "short"],
                "qty": [1.0, 1.0],
                "entry_price": [91.38, 92.12],
                "exit_price": [89.82, 91.55],
                "pnl_pct": [-1.71, 0.62],
                "exit_event": ["sl_exit", "tp_exit"],
                "strategy": ["live_executor", "live_executor"],
            }
        )
        equity = ds.build_equity_curve(
            max_points=100,
            symbol="SOL-USDT",
            venue="kucoin",
            live_only=True,
            include_reconstructed=False,
            allow_file_fallback=False,
            allow_fill_reconstruction=False,
            _trades_df=df,
        )
        trades = equity.get("trades", [])
        self.assertEqual(len(trades), 2)
        seen_entry_times = {int(t["entry_time"]) for t in trades}
        self.assertEqual(
            seen_entry_times,
            {int(entry1.timestamp()), int(entry2.timestamp())},
        )
        # Every entry_time must look like a recent timestamp — anything close
        # to the unix epoch is the 1970-bug the user reported.
        for t in trades:
            self.assertGreater(
                int(t["entry_time"]),
                int(pd.Timestamp("2020-01-01", tz="UTC").timestamp()),
            )

    def test_build_equity_curve_emits_null_for_garbage_entry_time(self) -> None:
        # `closed_trades.entry_ts` is NOT NULL at the schema level, so older
        # buggy writers (e.g. NaT->0 fallbacks) end up with rows pinned to
        # 1970-01-01T00:00:01Z. Surface those as `null` so the frontend can
        # render "—" instead of showing 1/1/1970 in the tooltip.
        bogus_entry = pd.Timestamp("1970-01-01T00:00:01Z")  # epoch 1s
        real_exit = pd.Timestamp("2026-05-15T15:32:38Z")
        df = pd.DataFrame(
            {
                "trade_id": ["t_bad", "t_nat"],
                "venue": ["kucoin", "kucoin"],
                "symbol": ["SOL-USDT", "SOL-USDT"],
                "entry_ts": [bogus_entry, pd.NaT],
                "exit_ts": [real_exit, real_exit + pd.Timedelta(minutes=5)],
                "side": ["long", "short"],
                "qty": [1.0, 1.0],
                "entry_price": [91.38, 92.0],
                "exit_price": [89.82, 91.5],
                "pnl_pct": [-1.71, 0.5],
                "exit_event": ["sl_exit", "tp_exit"],
                "strategy": ["live_executor", "live_executor"],
            }
        )
        equity = ds.build_equity_curve(
            max_points=100,
            symbol="SOL-USDT",
            venue="kucoin",
            live_only=True,
            include_reconstructed=False,
            allow_file_fallback=False,
            allow_fill_reconstruction=False,
            _trades_df=df,
        )
        trades = equity.get("trades", [])
        # NaT entry rows are still surfaced (exit_ts + pnl are valid), and the
        # epoch-1 row must be surfaced too — both with ``entry_time=None`` so
        # the tooltip can fall back to "—" instead of printing 1970.
        self.assertEqual(len(trades), 2)
        for t in trades:
            self.assertIsNone(t.get("entry_time"))

    def test_build_equity_curve_falls_back_to_payload_entry_time(self) -> None:
        opened_at = pd.Timestamp("2026-05-14T08:15:00Z")
        entry_bar_ts = pd.Timestamp("2026-05-14T09:20:00Z")
        real_exit = pd.Timestamp("2026-05-15T15:32:38Z")
        df = pd.DataFrame(
            {
                "trade_id": ["t_opened_at", "t_entry_bar"],
                "venue": ["kucoin", "kucoin"],
                "symbol": ["SOL-USDT", "SOL-USDT"],
                "entry_ts": [
                    pd.Timestamp("1970-01-01T00:00:01Z"),
                    pd.NaT,
                ],
                "exit_ts": [real_exit, real_exit + pd.Timedelta(minutes=5)],
                "side": ["long", "short"],
                "qty": [1.0, 1.0],
                "entry_price": [91.38, 92.0],
                "exit_price": [89.82, 91.5],
                "pnl_pct": [-1.71, 0.5],
                "exit_event": ["sl_exit", "tp_exit"],
                "strategy": ["live_executor", "live_executor"],
                "payload_json": [
                    {"opened_at": opened_at.isoformat()},
                    {"entry_bar_ts": int(entry_bar_ts.timestamp() * 1000)},
                ],
            }
        )
        equity = ds.build_equity_curve(
            max_points=100,
            symbol="SOL-USDT",
            venue="kucoin",
            live_only=True,
            include_reconstructed=False,
            allow_file_fallback=False,
            allow_fill_reconstruction=False,
            _trades_df=df,
        )
        trades = equity.get("trades", [])
        self.assertEqual(len(trades), 2)
        self.assertEqual(
            [int(t["entry_time"]) for t in trades],
            [int(opened_at.timestamp()), int(entry_bar_ts.timestamp())],
        )

    def test_reconstruct_trades_from_execution_fills_emits_real_entry_time(self) -> None:
        # The fills-reconstruction code path is a second producer of
        # ``entry_time``; make sure it likewise carries through real
        # timestamps, not epoch.
        ts0 = pd.Timestamp("2026-04-10T08:00:00Z")
        fills = pd.DataFrame(
            {
                "ts": [ts0, ts0 + pd.Timedelta(minutes=5)],
                "seq": [1, 2],
                "side": ["buy", "sell"],
                "qty": [1.0, 1.0],
                "price": [100.0, 102.0],
            }
        )
        out = ds._reconstruct_trades_from_execution_fills_df(
            fills_df=fills, max_points=100, source="test"
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(int(out[0]["entry_time"]), int(ts0.timestamp()))
        self.assertGreater(
            int(out[0]["entry_time"]),
            int(pd.Timestamp("2020-01-01", tz="UTC").timestamp()),
        )

    @patch("quant.execution.dashboard_state.load_closed_trades_from_postgres")
    def test_dashboard_performance_uses_closed_trade_rows_and_breakeven_bucket(self, mock_load_trades) -> None:
        mock_load_trades.return_value = pd.DataFrame(
            {
                "trade_id": ["a1", "a2", "b1", "c1", "reco1", "sim1"],
                "venue": ["kucoin", "kucoin", "kucoin", "kucoin", "kucoin", "kucoin"],
                "symbol": ["SOL-USDT", "SOL-USDT", "SOL-USDT", "SOL-USDT", "SOL-USDT", "SOL-USDT"],
                "entry_ts": [
                    pd.Timestamp("2026-02-20T10:00:00Z"),
                    pd.Timestamp("2026-02-20T10:00:00Z"),
                    pd.Timestamp("2026-02-20T11:00:00Z"),
                    pd.Timestamp("2026-02-20T12:00:00Z"),
                    pd.Timestamp("2026-02-20T13:00:00Z"),
                    pd.Timestamp("2026-02-20T14:00:00Z"),
                ],
                "exit_ts": [
                    pd.Timestamp("2026-02-20T10:05:00Z"),
                    pd.Timestamp("2026-02-20T10:08:00Z"),
                    pd.Timestamp("2026-02-20T11:07:00Z"),
                    pd.Timestamp("2026-02-20T12:09:00Z"),
                    pd.Timestamp("2026-02-20T13:09:00Z"),
                    pd.Timestamp("2026-02-20T14:09:00Z"),
                ],
                "side": ["long", "long", "short", "long", "long", "long"],
                "qty": [1.0, 3.0, 2.0, 1.0, 1.0, 1.0],
                "entry_price": [100.0, 100.0, 105.0, 110.0, 111.0, 112.0],
                "exit_price": [102.0, 104.0, 104.0, 110.0, 112.0, 113.0],
                "pnl_pct": [2.0, 4.0, -1.0, 0.0, 0.9, 0.8],
                "exit_event": ["tp1", "tp2", "sl_exit", "be_exit", "fills_reconstructed", "tp1"],
                "strategy": [
                    "live_executor",
                    "live_executor",
                    "live_executor",
                    "live_executor",
                    "dashboard_fills_reconstruction",
                    "sim_backtest",
                ],
            }
        )

        perf = ds.build_dashboard_performance(symbol="SOL-USDT", venue="kucoin", max_points=100)
        self.assertEqual(int(perf["trade_count"]), 4)
        self.assertEqual(int(perf["winning_trade_count"]), 2)
        self.assertEqual(int(perf["losing_trade_count"]), 1)
        self.assertEqual(int(perf["breakeven_trade_count"]), 1)
        self.assertEqual(
            int(perf["winning_trade_count"])
            + int(perf["losing_trade_count"])
            + int(perf["breakeven_trade_count"]),
            int(perf["trade_count"]),
        )
        self.assertAlmostEqual(float(perf["average_gain"]), (2.0 + 4.0 - 1.0 + 0.0) / 4.0, places=4)
        self.assertEqual(perf["source"], "postgres:closed_trades")


if __name__ == "__main__":
    unittest.main()
