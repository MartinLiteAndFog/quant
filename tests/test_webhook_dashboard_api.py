from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant.execution.webhook_server import api_dashboard_chart, api_regime_latest
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
        os.environ["GATE_ON_MEANS"] = "trend"
        os.environ["GATE_CONF_HORIZONS_MINUTES"] = "5,30,120,240"
        os.environ["GATE_CONF_CACHE_SEC"] = "0"
        os.environ["GATE_CONF_NOW_MODE"] = "last_ts"

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
                {"entry_ts": "2026-02-20T00:00:00Z", "exit_ts": "2026-02-20T01:00:00Z", "side": 1, "exit_event": "tp_exit"}
            ]
        ).to_parquet(root / "trades.parquet", index=False)
        (root / "execution_state.json").write_text('{"sl":99.0,"ttp":102.0,"tp1":103.0,"tp2":104.0}', encoding="utf-8")
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

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_chart_payload_shape(self) -> None:
        body = api_dashboard_chart(symbol="SOL-USDT", hours=48, max_points=1000)
        self.assertTrue(body.get("ok"))
        self.assertIn("bars", body)
        self.assertIn("markers", body)
        self.assertIn("levels", body)
        self.assertIn("regime", body)
        self.assertTrue(len(body["bars"]) >= 1)
        self.assertIn("gate_confidence", body)
        gc = body["gate_confidence"]
        self.assertIsInstance(gc, dict)
        self.assertIn("horizons", gc)
        self.assertTrue(len(gc["horizons"]) >= 1)
        self.assertIn("p_trend_voxel", gc["horizons"][0])

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
        self.assertIsNone(body.get("gate_confidence_error"))
        self.assertIsInstance(body.get("gate_confidence"), dict)


if __name__ == "__main__":
    unittest.main()
