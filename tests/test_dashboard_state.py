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
    build_combined_equity,
    build_regime_overlay,
    load_kraken_equity_history,
    load_kraken_metrics,
    load_active_levels,
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

    def test_load_levels(self) -> None:
        levels = load_active_levels()
        self.assertAlmostEqual(float(levels["sl"]), 99.1, places=6)
        self.assertAlmostEqual(float(levels["tp2"]), 105.5, places=6)

    def test_regime_overlay(self) -> None:
        overlay = build_regime_overlay(symbol="SOL-USDT", hours=24 * 30)
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

    def test_kraken_metrics_and_equity_loaders(self) -> None:
        metrics_path = self.tmp_path / "kraken_metrics.json"
        equity_path = self.tmp_path / "kraken_equity.csv"
        os.environ["KRAKEN_METRICS_JSON"] = str(metrics_path)
        os.environ["KRAKEN_EQUITY_CSV"] = str(equity_path)

        metrics_path.write_text(
            json.dumps({"ts": "2026-02-20T00:00:00Z", "equity_usd": 1000.0, "wallet_usd": 990.0, "upnl_usd": 10.0}),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {"ts": int(pd.Timestamp("2026-02-20T00:00:00Z").timestamp()), "equity_usd": 1000.0},
                {"ts": int(pd.Timestamp("2026-02-20T01:00:00Z").timestamp()), "equity_usd": 1010.0},
            ]
        ).to_csv(equity_path, index=False)

        m = load_kraken_metrics()
        h = load_kraken_equity_history(max_points=100)
        self.assertEqual(float(m.get("equity_usd", 0.0)), 1000.0)
        self.assertEqual(len(h.get("points", [])), 2)

    def test_kraken_equity_loader_accepts_legacy_timestamp_and_equity_columns(self) -> None:
        equity_path = self.tmp_path / "kraken_equity_legacy.csv"
        os.environ["KRAKEN_EQUITY_CSV"] = str(equity_path)
        pd.DataFrame(
            [
                {"time": "2026-02-20T00:00:00Z", "equity": 1000.0},
                {"time": "2026-02-20T01:00:00Z", "equity": 1015.0},
            ]
        ).to_csv(equity_path, index=False)

        h = load_kraken_equity_history(max_points=100)
        pts = h.get("points", [])
        self.assertEqual(len(pts), 2)
        self.assertEqual(int(pts[0]["time"]), int(pd.Timestamp("2026-02-20T00:00:00Z").timestamp()))
        self.assertAlmostEqual(float(pts[1]["equity"]), 1015.0, places=6)

    def test_build_combined_equity(self) -> None:
        kucoin = [{"time": 100, "equity": 200.0}, {"time": 200, "equity": 210.0}]
        kraken = [{"time": 100, "equity": 300.0}, {"time": 300, "equity": 320.0}]
        out = build_combined_equity(kucoin_points=kucoin, kraken_points_usd=kraken)
        pts = out.get("points", [])
        self.assertTrue(len(pts) >= 2)
        self.assertEqual(int(pts[0]["time"]), 100)
        self.assertAlmostEqual(float(pts[0]["equity"]), 500.0, places=6)


if __name__ == "__main__":
    unittest.main()
