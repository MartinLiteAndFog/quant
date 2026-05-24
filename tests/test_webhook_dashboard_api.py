from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

import quant.execution.webhook_server as ws
from quant.execution.webhook_server import api_dashboard_chart, api_regime_latest, api_dashboard_statespace, api_status, dashboard, api_gate_solusd
from quant.regime import RegimeDecision, RegimeService, RegimeStore


class WebhookDashboardApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        os.environ["REGIME_DB_PATH"] = str(root / "regime.db")
        os.environ["DASHBOARD_RENKO_PARQUET"] = str(root / "renko.parquet")
        os.environ["DASHBOARD_TRADES_PARQUET"] = str(root / "trades.parquet")
        os.environ["DASHBOARD_LEVELS_JSON"] = str(root / "execution_state.json")
        os.environ["GATE_CONF_ARTIFACT_DIR"] = str(root / "artifacts")
        os.environ["GATE_DAILY_PATH"] = str(root / "gate_daily.csv")
        os.environ["GATE_DAILY_TS_COL"] = "ts"
        os.environ["GATE_DAILY_COL"] = "gate_on_2of3"
        os.environ["GATE_DAILY_OFF_PATH"] = str(root / "gate_daily_off.csv")
        os.environ["GATE_DAILY_OFF_COL"] = "gate_off_2of3"
        os.environ["GATE_ON_MEANS"] = "trend"
        os.environ["GATE_CONF_HORIZONS_MINUTES"] = "5,30,120,240"
        os.environ["GATE_CONF_CACHE_SEC"] = "0"
        os.environ["GATE_CONF_NOW_MODE"] = "last_ts"
        os.environ["DASHBOARD_TRADE_ALLOW_FILE_FALLBACK"] = "1"

        renko = pd.DataFrame(
            {
                "ts": pd.date_range("2026-02-20", periods=2, freq="h", tz="UTC"),
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
            }
        )
        renko.to_parquet(root / "renko.parquet", index=False)
        pd.DataFrame(
            [
                {
                    "entry_ts": "2026-02-20T00:00:00Z",
                    "exit_ts": "2026-02-20T01:00:00Z",
                    "side": 1,
                    "entry_price": 100.0,
                    "exit_price": 101.0,
                    "pnl_pct": 1.0,
                    "exit_event": "tp_exit",
                }
            ]
        ).to_parquet(root / "trades.parquet", index=False)
        # Include open position snapshot to exercise dashboard fallback marker.
        entry_bar_ts = int(pd.Timestamp("2026-02-20T00:00:00Z").timestamp())
        (root / "execution_state.json").write_text(
            f'{{"sl":99.0,"ttp":102.0,"tp1":103.0,"tp2":104.0,"side":1,"entry_px":100.5,"entry_bar_ts":{entry_bar_ts}}}',
            encoding="utf-8",
        )
        art = root / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "ts": pd.to_datetime(
                    [
                        "2026-02-20T00:00:00Z",
                        "2026-02-20T00:01:00Z",
                        "2026-02-20T00:02:00Z",
                    ],
                    utc=True,
                ),
                "voxel_id": [1, 2, 1],
            }
        ).to_parquet(art / "voxel_map.parquet", index=False)
        pd.DataFrame(
            {
                "voxel_id": [1, 2],
                "pi": [0.6, 0.4],
            }
        ).to_parquet(art / "voxel_stats.parquet", index=False)
        pd.DataFrame(
            {
                "from_voxel_id": [1, 1, 2, 2],
                "to_voxel_id": [1, 2, 2, 1],
                "p": [0.7, 0.2, 0.6, 0.3],
            }
        ).to_parquet(art / "transitions_topk.parquet", index=False)
        pd.DataFrame(
            {
                "voxel_id": [1, 2],
                "basin_id": [10, 20],
            }
        ).to_parquet(art / "basins_v02_components.parquet", index=False)
        pd.DataFrame(
            {
                "ts": ["2026-02-19T00:00:00Z", "2026-02-20T00:00:00Z"],
                "gate_on_2of3": [0, 1],
            }
        ).to_csv(root / "gate_daily.csv", index=False)
        pd.DataFrame(
            {
                "ts": ["2026-02-19T00:00:00Z", "2026-02-20T00:00:00Z"],
                "gate_off_2of3": [1, 0],
            }
        ).to_csv(root / "gate_daily_off.csv", index=False)
        pd.DataFrame(
            [
                {"time": int(pd.Timestamp("2026-02-20T00:00:00Z").timestamp()), "equity": 1000.0},
                {"time": int(pd.Timestamp("2026-02-20T01:00:00Z").timestamp()), "equity": 1010.0},
            ]
        ).to_parquet(root / "equity_history.parquet", index=False)
        os.environ["DASHBOARD_EQUITY_PARQUET"] = str(root / "equity_history.parquet")
        pd.DataFrame(
            [
                {"ts": int(pd.Timestamp("2026-02-20T00:00:00Z").timestamp()), "equity_usd": 500.0},
                {"ts": int(pd.Timestamp("2026-02-20T01:00:00Z").timestamp()), "equity_usd": 505.0},
            ]
        ).to_csv(root / "kraken_equity.csv", index=False)
        os.environ["KRAKEN_EQUITY_CSV"] = str(root / "kraken_equity.csv")
        (root / "kraken_metrics.json").write_text('{"equity_usd":505.0,"position_side":"long"}', encoding="utf-8")
        os.environ["KRAKEN_METRICS_JSON"] = str(root / "kraken_metrics.json")

        svc = RegimeService(RegimeStore())
        svc.upsert_decision(
            RegimeDecision(
                ts="2026-02-20T00:00:00Z",
                symbol="SOL-USDT",
                gate_on=1,
                regime_state="trend",
                regime_score=0.8,
                confidence=0.8,
                reason_code="seed",
            )
        )
        ws._STATUS_CACHE.clear()
        ws._POSITION_CACHE.clear()
        ws._CHART_CACHE.clear()

    def tearDown(self) -> None:
        ws._STATUS_CACHE.clear()
        ws._POSITION_CACHE.clear()
        ws._CHART_CACHE.clear()
        os.environ.pop("DASHBOARD_API_CACHE_SEC", None)
        os.environ.pop("KUCOIN_FUTURES_API_KEY", None)
        os.environ.pop("WEBHOOK_TOKEN", None)
        self.tmp.cleanup()

    def test_chart_filters_historic_trade_markers_before_first_bar(self) -> None:
        root = Path(self.tmp.name)
        # Bars cover Feb 20 only (see setUp). Seed trades that include both
        # an "ancient" trade well before the bars window and a fresh trade
        # within it. Only the fresh trade should be represented on the chart;
        # older trades must not be anchored to the first visible bar.
        pd.DataFrame(
            [
                # Ancient trade: would otherwise be stacked at the chart
                # start because its timestamps predate the first bar.
                {
                    "trade_id": "ancient_trade",
                    "entry_ts": "2025-12-01T00:00:00Z",
                    "exit_ts": "2025-12-01T00:30:00Z",
                    "side": 1,
                    "entry_price": 70.0,
                    "exit_price": 71.0,
                    "pnl_pct": 1.4,
                    "exit_event": "tp_exit",
                },
                # Fresh trade: lives squarely inside the Renko bar window.
                {
                    "trade_id": "fresh_trade",
                    "entry_ts": "2026-02-20T00:30:00Z",
                    "exit_ts": "2026-02-20T00:45:00Z",
                    "side": 1,
                    "entry_price": 100.0,
                    "exit_price": 101.0,
                    "pnl_pct": 1.0,
                    "exit_event": "tp_exit",
                },
            ]
        ).to_parquet(root / "trades.parquet", index=False)
        # Make sure the test reads from this seeded file rather than from
        # any background Postgres data picked up by the live executor.
        os.environ["DASHBOARD_TRADE_ALLOW_FILE_FALLBACK"] = "1"
        ws._CHART_CACHE.clear()

        body = api_dashboard_chart(symbol="SOL-USDT", hours=48, max_points=1000)
        self.assertTrue(body.get("ok"))
        bars = body.get("bars", [])
        self.assertTrue(bars, "test setUp must seed at least one Renko bar")
        first_bar_ts = int(bars[0]["time"])
        last_bar_ts = int(bars[-1]["time"])
        trade_markers = [
            m
            for m in body.get("markers", [])
            if m.get("trade_id") in {"ancient_trade", "fresh_trade"}
        ]
        self.assertEqual(len(trade_markers), 2)
        for m in trade_markers:
            mt = int(m.get("time", 0))
            self.assertGreaterEqual(
                mt,
                first_bar_ts,
                f"Marker {m!r} predates the first bar at {first_bar_ts}",
            )
            self.assertLessEqual(mt, last_bar_ts)
        ancient_ts = int(pd.Timestamp("2025-12-01T00:00:00Z").timestamp())
        ancient_markers = [
            m
            for m in trade_markers
            if int(m.get("original_time", 0)) == ancient_ts
        ]
        self.assertEqual(ancient_markers, [])
        self.assertFalse(
            any(int(m.get("original_time", 0)) < first_bar_ts for m in trade_markers)
        )

        # The fresh trade is inside the visible window, but between the two
        # Renko bars. Lightweight-charts only renders markers on existing
        # series times, so keep the original trade time for diagnostics while
        # rendering the marker on the nearest-left visible bar.
        fresh_ts = int(pd.Timestamp("2026-02-20T00:30:00Z").timestamp())
        fresh_markers = [m for m in trade_markers if m.get("trade_id") == "fresh_trade"]
        self.assertEqual([int(m.get("time", 0)) for m in fresh_markers], [first_bar_ts, first_bar_ts])
        self.assertEqual([m.get("text") for m in fresh_markers], ["", "+1.00%"])
        self.assertEqual({int(m.get("original_time", 0)) for m in fresh_markers}, {fresh_ts})
        self.assertEqual({m.get("time_anchor") for m in fresh_markers}, {"asof_bar"})

    def test_chart_includes_closed_trade_overlapping_window(self) -> None:
        root = Path(self.tmp.name)
        pd.DataFrame(
            {
                "ts": pd.date_range("2026-02-20T02:00:00Z", periods=2, freq="h"),
                "open": [102.0, 103.0],
                "high": [103.0, 104.0],
                "low": [101.0, 102.0],
                "close": [102.5, 103.5],
            }
        ).to_parquet(root / "renko.parquet", index=False)
        pd.DataFrame(
            [
                {
                    "trade_id": "fully_old_trade",
                    "entry_ts": "2026-02-20T00:00:00Z",
                    "exit_ts": "2026-02-20T01:00:00Z",
                    "side": "long",
                    "entry_price": 100.0,
                    "exit_price": 101.0,
                    "pnl_pct": 1.0,
                    "exit_event": "tp_exit",
                },
                {
                    "trade_id": "overlap_trade",
                    "entry_ts": "2026-02-20T01:30:00Z",
                    "exit_ts": "2026-02-20T02:30:00Z",
                    "side": "short",
                    "entry_price": 104.0,
                    "exit_price": 102.0,
                    "pnl_pct": 1.92,
                    "exit_event": "tp_exit",
                },
            ]
        ).to_parquet(root / "trades.parquet", index=False)
        os.environ["DASHBOARD_TRADE_ALLOW_FILE_FALLBACK"] = "1"
        ws._CHART_CACHE.clear()

        body = api_dashboard_chart(symbol="SOL-USDT", hours=48, max_points=1000)

        self.assertTrue(body.get("ok"))
        first_bar_ts = int(body["bars"][0]["time"])
        old_markers = [
            m for m in body.get("markers", [])
            if m.get("trade_id") == "fully_old_trade"
        ]
        overlap_markers = [
            m for m in body.get("markers", [])
            if m.get("trade_id") == "overlap_trade"
        ]
        self.assertEqual(old_markers, [])
        self.assertEqual(
            overlap_markers,
            [],
            "pre-window entries must not be snapped onto the first visible bar",
        )

    def test_chart_maps_stop_loss_marker_to_visible_renko_bar(self) -> None:
        root = Path(self.tmp.name)
        pd.DataFrame(
            {
                "ts": pd.date_range("2026-02-20T02:00:00Z", periods=2, freq="h", tz="UTC"),
                "open": [102.0, 103.0],
                "high": [103.0, 104.0],
                "low": [101.0, 102.0],
                "close": [102.5, 103.5],
            }
        ).to_parquet(root / "renko.parquet", index=False)
        pd.DataFrame(
            [
                {
                    "trade_id": "sl_visible_exit",
                    "entry_ts": "2026-02-20T01:30:00Z",
                    "exit_ts": "2026-02-20T02:30:00Z",
                    "side": "long",
                    "entry_price": 103.0,
                    "exit_price": 101.5,
                    "pnl_pct": -1.46,
                    "exit_event": "stop_loss",
                },
                {
                    "trade_id": "tp_visible_exit",
                    "entry_ts": "2026-02-20T02:15:00Z",
                    "exit_ts": "2026-02-20T02:45:00Z",
                    "side": "long",
                    "entry_price": 102.0,
                    "exit_price": 103.0,
                    "pnl_pct": 0.98,
                    "exit_event": "tp_exit",
                },
            ]
        ).to_parquet(root / "trades.parquet", index=False)
        os.environ["DASHBOARD_TRADE_ALLOW_FILE_FALLBACK"] = "1"
        ws._CHART_CACHE.clear()

        body = api_dashboard_chart(symbol="SOL-USDT", hours=48, max_points=1000)

        self.assertTrue(body.get("ok"))
        first_bar_ts = int(body["bars"][0]["time"])
        sl_markers = [
            m
            for m in body.get("markers", [])
            if m.get("trade_id") == "sl_visible_exit"
            and m.get("marker_kind") == "sl_exit"
        ]
        self.assertEqual(len(sl_markers), 1)
        sl_marker = sl_markers[0]
        self.assertEqual(int(sl_marker["time"]), first_bar_ts)
        self.assertEqual(int(sl_marker["original_time"]), int(pd.Timestamp("2026-02-20T02:30:00Z").timestamp()))
        self.assertEqual(sl_marker.get("time_anchor"), "asof_bar")
        self.assertEqual(sl_marker.get("text"), "×")
        self.assertEqual(str(sl_marker.get("color")).lower(), "#ef4444")
        self.assertFalse(
            any(
                m.get("trade_id") == "tp_visible_exit"
                and m.get("marker_kind") == "sl_exit"
                for m in body.get("markers", [])
            )
        )

    def test_chart_maps_decision_entry_marker_to_visible_renko_bar(self) -> None:
        ws._CHART_CACHE.clear()
        decision_ts = pd.Timestamp("2026-02-20T00:30:00Z")
        exit_ts = pd.Timestamp("2026-02-20T00:45:00Z")
        closed_trades = pd.DataFrame(
            [
                {
                    "trade_id": "bad-entry-decision-trade",
                    "venue": "kucoin",
                    "symbol": "SOL-USDT",
                    "entry_ts": pd.Timestamp("1970-01-01T00:00:01Z"),
                    "exit_ts": exit_ts,
                    "side": "long",
                    "qty": 1.0,
                    "entry_price": 100.0,
                    "exit_price": 101.0,
                    "pnl_pct": 1.0,
                    "exit_event": "tp_exit",
                    "strategy": "live_executor",
                    "source_action_event_id": "decision-action-1",
                    "payload_json": {},
                }
            ]
        )
        decisions = [
            {
                "decision_id": "decision-chart-1",
                "ts": decision_ts.isoformat(),
                "venue": "kucoin",
                "symbol": "SOL-USDT",
                "decision_kind": "entry",
                "direction": "long",
                "source_action_event_id": "decision-action-1",
                "seq": 1,
                "payload_json": {},
            }
        ]

        with patch.object(ws, "load_closed_trades_from_postgres", return_value=closed_trades), \
             patch("quant.execution.dashboard_state._load_decision_rows_for_trade_markers", return_value=decisions):
            body = api_dashboard_chart(symbol="SOL-USDT", hours=48, max_points=1000)

        self.assertTrue(body.get("ok"))
        first_bar_ts = int(body["bars"][0]["time"])
        trade_markers = [
            m
            for m in body.get("markers", [])
            if m.get("trade_id") == "bad-entry-decision-trade"
        ]
        self.assertEqual(len(trade_markers), 2)
        self.assertEqual([int(m["time"]) for m in trade_markers], [first_bar_ts, first_bar_ts])
        self.assertEqual({int(m["original_time"]) for m in trade_markers}, {int(decision_ts.timestamp())})
        self.assertEqual({m.get("time_anchor") for m in trade_markers}, {"asof_bar"})
        self.assertEqual([m.get("text") for m in trade_markers], ["", "+1.00%"])
        self.assertNotIn(
            int(pd.Timestamp("1970-01-01T00:00:01Z").timestamp()),
            {int(m.get("original_time", 0)) for m in trade_markers},
        )

    def test_chart_payload_shape(self) -> None:
        body = api_dashboard_chart(symbol="SOL-USDT", hours=48, max_points=1000)
        self.assertTrue(body.get("ok"))
        self.assertIn("bars", body)
        self.assertIn("markers", body)
        self.assertIn("levels", body)
        self.assertIn("regime", body)
        self.assertTrue(len(body["bars"]) >= 1)
        self.assertIn("gate_confidence", body)
        self.assertIn("gate_confidence_error", body)
        self.assertIn("open_position", body)
        self.assertEqual(body.get("segments"), [])
        self.assertEqual(body.get("kraken_metrics"), {})
        self.assertEqual(body.get("equity_kraken"), [])
        self.assertEqual(body.get("equity_kraken_source"), "none")
        self.assertEqual(body.get("equity_combined"), [])
        self.assertEqual(body.get("equity_combined_source"), "none")
        self.assertEqual(body.get("equity_total"), body.get("equity_real"))
        self.assertEqual(body.get("equity_total_source"), body.get("equity_real_source"))
        self.assertEqual(
            [c.get("key") for c in body.get("equity_components", [])],
            ["kucoin"],
        )

    @patch("quant.execution.webhook_server.get_live_gate_state")
    def test_chart_payload_exposes_day_regime_separately_from_trade_regime(self, mock_gate_state) -> None:
        mock_gate_state.return_value = {
            "gate_countertrend_on": 1,
            "gate_trend_on": 0,
            "gate_on": 0,
            "source": "daily_csv",
        }
        body = api_dashboard_chart(symbol="SOL-USDT", hours=48, max_points=1000)
        self.assertTrue(body.get("ok"))
        self.assertIsNone(body.get("regime_state"))
        self.assertEqual(body.get("day_regime_state"), "countertrend")

    @patch("quant.execution.webhook_server.get_live_gate_state")
    def test_gate_solusd_endpoint(self, mock_gate_state) -> None:
        mock_gate_state.return_value = {
            "ts": "2026-02-20T00:00:00Z",
            "gate_on": 1,
            "gate_off": 0,
            "source": "postgres_daily_gate",
            "primary": "countertrend",
            "gate_countertrend_on": 1,
            "gate_trend_on": 0,
            "gate_on_ts": "2026-02-20T00:00:00Z",
            "gate_off_ts": "2026-02-20T00:00:00Z",
            "gate_on_age_sec": 5.0,
            "gate_off_age_sec": 5.0,
        }
        body = api_gate_solusd()
        self.assertIn("gate_on", body)
        self.assertIn("gate_off", body)
        self.assertIn("ts", body)
        self.assertEqual(body.get("source"), "postgres_daily_gate")
        self.assertIn("gate_on_ts", body)
        self.assertIn("gate_off_ts", body)

    def test_chart_includes_live_entry_marker_when_trades_missing(self) -> None:
        # New marker contract: the live (open) trade is represented by a
        # plain direction-colored arrow (long -> arrowUp belowBar in green)
        # with empty text. The legacy blue "live entry" label has been
        # superseded by load_trade_markers' open-trade arrow.
        root = Path(self.tmp.name)
        os.environ["DASHBOARD_TRADES_PARQUET"] = str(root / "missing_trades.parquet")
        body = api_dashboard_chart(symbol="SOL-USDT", hours=48, max_points=1000)
        self.assertTrue(body.get("ok"))
        self.assertIsInstance(body.get("open_position"), dict)
        entry_ts = int(pd.Timestamp("2026-02-20T00:00:00Z").timestamp())
        live_arrows = [
            m
            for m in body.get("markers", [])
            if int(m.get("time", 0)) == entry_ts
            and str(m.get("shape", "")) == "arrowUp"
        ]
        self.assertTrue(live_arrows)
        live_arrow = live_arrows[0]
        self.assertEqual(str(live_arrow.get("text", "")), "")
        self.assertEqual(str(live_arrow.get("position", "")), "belowBar")
        self.assertEqual(str(live_arrow.get("color", "")).lower(), "#22c55e")

    def test_chart_includes_live_entry_marker_when_open_entry_predates_bars(self) -> None:
        root = Path(self.tmp.name)
        # Simulate the live dashboard after the Renko window has rolled past
        # the open trade's original entry. Closed-trade markers should still be
        # constrained to in-window entries, but the current open trade needs a
        # visible direction arrow.
        pd.DataFrame(
            {
                "ts": pd.date_range("2026-02-20T02:00:00Z", periods=2, freq="h"),
                "open": [102.0, 103.0],
                "high": [103.0, 104.0],
                "low": [101.0, 102.0],
                "close": [102.5, 103.5],
            }
        ).to_parquet(root / "renko.parquet", index=False)
        os.environ["DASHBOARD_TRADES_PARQUET"] = str(root / "missing_trades.parquet")
        (root / "execution_state.json").write_text(
            '{"sl":105.0,"side":"short","entry_px":101.0,"entry_bar_ts":"2026-02-20T00:00:00Z"}',
            encoding="utf-8",
        )
        ws._CHART_CACHE.clear()

        body = api_dashboard_chart(symbol="SOL-USDT", hours=48, max_points=1000)

        self.assertTrue(body.get("ok"))
        first_bar_ts = int(body["bars"][0]["time"])
        live_arrows = [
            m
            for m in body.get("markers", [])
            if int(m.get("time", 0)) == first_bar_ts
            and str(m.get("shape", "")) == "arrowDown"
        ]
        self.assertTrue(live_arrows)
        live_arrow = live_arrows[0]
        self.assertEqual(str(live_arrow.get("text", "")), "")
        self.assertEqual(str(live_arrow.get("position", "")), "aboveBar")
        self.assertEqual(str(live_arrow.get("color", "")).lower(), "#ef4444")

    def test_chart_keeps_live_entry_marker_when_closed_markers_exist(self) -> None:
        root = Path(self.tmp.name)
        pd.DataFrame(
            {
                "ts": pd.date_range("2026-02-20T02:00:00Z", periods=2, freq="h"),
                "open": [102.0, 103.0],
                "high": [103.0, 104.0],
                "low": [101.0, 102.0],
                "close": [102.5, 103.5],
            }
        ).to_parquet(root / "renko.parquet", index=False)
        pd.DataFrame(
            [
                {
                    "trade_id": "visible_closed_trade",
                    "entry_ts": "2026-02-20T02:30:00Z",
                    "exit_ts": "2026-02-20T02:45:00Z",
                    "side": "long",
                    "entry_price": 102.0,
                    "exit_price": 103.0,
                    "pnl_pct": 0.98,
                    "exit_event": "tp_exit",
                }
            ]
        ).to_parquet(root / "trades.parquet", index=False)
        (root / "execution_state.json").write_text(
            '{"sl":105.0,"side":"short","entry_px":101.0,"entry_bar_ts":"2026-02-20T00:00:00Z"}',
            encoding="utf-8",
        )
        os.environ["DASHBOARD_TRADE_ALLOW_FILE_FALLBACK"] = "1"
        ws._CHART_CACHE.clear()

        body = api_dashboard_chart(symbol="SOL-USDT", hours=48, max_points=1000)

        self.assertTrue(body.get("ok"))
        first_bar_ts = int(body["bars"][0]["time"])
        live_arrows = [
            m
            for m in body.get("markers", [])
            if int(m.get("time", 0)) == first_bar_ts
            and str(m.get("shape", "")) == "arrowDown"
            and str(m.get("text", "")) == ""
        ]
        self.assertTrue(live_arrows)
        self.assertTrue(
            any(m.get("trade_id") == "visible_closed_trade" for m in body.get("markers", []))
        )

    def test_chart_handles_string_entry_bar_ts(self) -> None:
        root = Path(self.tmp.name)
        os.environ["DASHBOARD_TRADES_PARQUET"] = str(root / "missing_trades.parquet")
        (root / "execution_state.json").write_text(
            '{"sl":99.0,"side":"long","entry_px":83.0,"entry_bar_ts":"2026-02-20T00:00:00Z"}',
            encoding="utf-8",
        )
        body = api_dashboard_chart(symbol="SOL-USDT", hours=48, max_points=1000)
        self.assertTrue(body.get("ok"))
        # Direction arrow renders at the parsed entry_bar_ts with empty
        # text (new contract — see test above).
        entry_ts = int(pd.Timestamp("2026-02-20T00:00:00Z").timestamp())
        live_arrows = [
            m
            for m in body.get("markers", [])
            if int(m.get("time", 0)) == entry_ts
            and str(m.get("shape", "")) == "arrowUp"
        ]
        self.assertTrue(live_arrows)
        self.assertEqual(str(live_arrows[0].get("text", "")), "")
        self.assertIsInstance(body.get("levels", {}).get("entry_bar_ts"), int)

    def test_chart_falls_back_to_expected_trades_when_levels_missing_entry(self) -> None:
        root = Path(self.tmp.name)
        os.environ["DASHBOARD_TRADES_PARQUET"] = str(root / "missing_trades.parquet")
        (root / "execution_state.json").write_text('{"sl":81.7}', encoding="utf-8")
        (root / "expected_trades.jsonl").write_text(
            "\n".join(
                [
                    '{"ts":"2026-02-20T00:00:00Z","symbol":"SOL-USDT","side":"long","action":"entry","qty":20,"expected_px":83.0}',
                    '{"ts":"2026-02-20T00:10:00Z","symbol":"SOL-USDT","side":"short","action":"exit_flip","qty":20,"expected_px":82.5}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        body = api_dashboard_chart(symbol="SOL-USDT", hours=48, max_points=1000)
        self.assertTrue(body.get("ok"))
        self.assertIsInstance(body.get("markers"), list)
        op = body.get("open_position")
        if op is not None:
            self.assertIsInstance(op, dict)
            self.assertIn(op.get("side"), ("long", "short"))

    def test_regime_latest_endpoint(self) -> None:
        body = api_regime_latest(symbol="SOL-USDT")
        self.assertTrue(body.get("ok"))
        self.assertEqual(body["symbol"], "SOL-USDT")
        self.assertIsNotNone(body["regime"])

    def test_gate_confidence_handles_mixed_timestamp_precision(self) -> None:
        root = Path(self.tmp.name)
        gate_parquet = root / "gate_daily.parquet"
        gate_ts_us = pd.Series(
            pd.DatetimeIndex(
                pd.to_datetime(
                    ["2026-02-19T00:00:00Z", "2026-02-20T00:00:00Z"],
                    utc=True,
                )
            ).as_unit("us")
        )
        pd.DataFrame({"ts": gate_ts_us, "gate_on_2of3": [0, 1]}).to_parquet(gate_parquet, index=False)

        os.environ["GATE_DAILY_PATH"] = str(gate_parquet)
        os.environ["GATE_CONF_CACHE_SEC"] = "0"

        body = api_dashboard_chart(symbol="SOL-USDT", hours=48, max_points=1000)
        self.assertTrue(body.get("ok"))
        self.assertEqual(body.get("gate_confidence_error"), "temporarily_disabled")
        self.assertIsNone(body.get("gate_confidence"))


    def test_chart_payload_includes_regime_scores(self) -> None:
        body = api_dashboard_chart(symbol="SOL-USDT", hours=48, max_points=1000)
        self.assertTrue(body.get("ok"))
        self.assertIn("regime_scores", body)
        self.assertIn("regime_forecast", body)
        self.assertIsInstance(body["regime_scores"], list)
        self.assertIsInstance(body["regime_forecast"], list)
        if body["regime_scores"]:
            self.assertIn("time", body["regime_scores"][0])
            self.assertIn("score", body["regime_scores"][0])

    def test_statespace_endpoint_shape(self) -> None:
        body = api_dashboard_statespace(window_hours=48)
        self.assertIn("trajectory", body)
        self.assertIn("current", body)
        self.assertIn("recent_density", body)
        self.assertIn("density_bg", body)

    def test_dashboard_html_uses_refresh_env_defaults(self) -> None:
        os.environ["DASHBOARD_UI_REFRESH_MS"] = "2500"
        os.environ["DASHBOARD_STATESPACE_REFRESH_MS"] = "12000"
        html = dashboard()
        self.assertIn("const uiRefreshMsDefault = 2500;", html)
        self.assertIn("const ssRefreshMsDefault = 12000;", html)
        self.assertIn("id=\"manual-token\"", html)
        self.assertIn("id=\"manual-action\"", html)
        self.assertIn("/api/manual/order", html)
        self.assertIn("visibilitychange", html)
        self.assertIn("refreshNow(", html)

    def test_manual_order_accepts_token_from_json_body(self) -> None:
        os.environ["WEBHOOK_TOKEN"] = "LeoVeKetem"
        client = TestClient(ws.app)

        with patch("quant.execution.webhook_server._kucoin_broker", return_value=object()), patch(
            "quant.execution.manual_orders.execute_manual_action",
            return_value={"ok": True, "action": "cancel_short"},
        ) as exec_mock:
            resp = client.post(
                "/api/manual/order",
                json={"action": "cancel_short", "token": "LeoVeKetem"},
            )

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body.get("ok"))
        exec_mock.assert_called_once()

    def test_chart_response_is_cached(self) -> None:
        """Second call within TTL should return cached response."""
        import time
        os.environ["DASHBOARD_API_CACHE_SEC"] = "10"

        r1 = api_dashboard_chart(symbol="SOL-USDT", hours=168, max_points=100)
        ts1 = r1.get("ts")
        self.assertTrue(r1["ok"])

        time.sleep(0.05)
        r2 = api_dashboard_chart(symbol="SOL-USDT", hours=168, max_points=100)
        ts2 = r2.get("ts")

        self.assertEqual(ts1, ts2, "Second call should return cached response")

    def test_chart_equity_curve_uses_closed_trades_not_decisions(self) -> None:
        """Trade-mode equity must keep the full closed_trades history."""
        from unittest.mock import patch

        import quant.execution.webhook_server as ws

        ws._CHART_CACHE.clear()
        closed_trades = pd.DataFrame(
            [
                {
                    "trade_id": "ct-1",
                    "venue": "kucoin",
                    "symbol": "SOL-USDT",
                    "entry_ts": "2026-02-20T00:00:00Z",
                    "exit_ts": "2026-02-20T00:10:00Z",
                    "side": "long",
                    "qty": 1.0,
                    "entry_price": 100.0,
                    "exit_price": 101.0,
                    "pnl_pct": 1.0,
                    "strategy": "live_executor",
                    "exit_event": "tp_exit",
                },
                {
                    "trade_id": "ct-2",
                    "venue": "kucoin",
                    "symbol": "SOL-USDT",
                    "entry_ts": "2026-02-20T00:20:00Z",
                    "exit_ts": "2026-02-20T00:30:00Z",
                    "side": "short",
                    "qty": 1.0,
                    "entry_price": 101.0,
                    "exit_price": 103.0,
                    "pnl_pct": -2.0,
                    "strategy": "live_executor",
                    "exit_event": "sl_exit",
                },
                {
                    "trade_id": "ct-3",
                    "venue": "kucoin",
                    "symbol": "SOL-USDT",
                    "entry_ts": "2026-02-20T00:40:00Z",
                    "exit_ts": "2026-02-20T00:50:00Z",
                    "side": "long",
                    "qty": 1.0,
                    "entry_price": 103.0,
                    "exit_price": 106.0,
                    "pnl_pct": 3.0,
                    "strategy": "live_executor",
                    "exit_event": "tp_exit",
                },
            ]
        )
        sparse_decision_payload = {
            "curve": {
                "points": [
                    {"time": 1, "pnl_pct": 10.0, "cum_pct": 10.0},
                    {"time": 2, "pnl_pct": -5.0, "cum_pct": 5.0},
                ],
                "source": "postgres:trade_decisions+closed_trades",
            },
            "performance": {},
            "needs_backfill": True,
        }

        with patch.object(ws, "load_closed_trades_from_postgres", return_value=closed_trades), \
             patch.object(ws, "build_decision_dashboard_payload", return_value=sparse_decision_payload, create=True) as decision_mock, \
             patch.object(ws, "_schedule_auto_backfill_trade_decisions", return_value={"scheduled": True}) as schedule_mock, \
             patch.object(ws, "_maybe_auto_backfill_trade_decisions", return_value=None) as backfill_mock:
            body = api_dashboard_chart(symbol="SOL-USDT", hours=48, max_points=100)

        self.assertTrue(body.get("ok"))
        self.assertEqual(body.get("equity_source"), "preloaded")
        self.assertEqual(len(body.get("equity_curve", [])), 3)
        self.assertEqual(
            [float(p["pnl_pct"]) for p in body["equity_curve"]],
            [1.0, -2.0, 3.0],
        )
        decision_mock.assert_not_called()
        schedule_mock.assert_not_called()
        backfill_mock.assert_not_called()

    def test_api_status_uses_cache_within_ttl(self) -> None:
        os.environ["KUCOIN_FUTURES_API_KEY"] = "x"
        os.environ["DASHBOARD_API_CACHE_SEC"] = "120"

        class _DummyBroker:
            def __init__(self):
                self.ticker_calls = 0
                self.balance_calls = 0

            def get_best_bid_ask(self, symbol):
                self.ticker_calls += 1
                return (100.0, 101.0)

            def get_account_balance(self, currency="USDT"):
                self.balance_calls += 1
                return {"equity": 123.0}

        b = _DummyBroker()
        with patch("quant.execution.webhook_server._kucoin_broker", return_value=b):
            a = api_status(symbol="SOL-USDT")
            c = api_status(symbol="SOL-USDT")
        self.assertTrue(a.get("ok"))
        self.assertTrue(c.get("ok"))
        self.assertEqual(b.ticker_calls, 1)
        self.assertEqual(b.balance_calls, 1)


if __name__ == "__main__":
    unittest.main()
