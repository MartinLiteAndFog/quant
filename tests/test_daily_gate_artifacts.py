import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from quant.execution.daily_gate_artifacts import build_daily_gate_artifacts, write_daily_gate_artifacts


class DailyGateArtifactsTests(unittest.TestCase):
    def test_build_daily_gate_artifacts_uses_last_row_per_day_and_derives_off(self) -> None:
        df = pd.DataFrame(
            {
                "ts": [
                    "2026-03-20T00:05:00Z",
                    "2026-03-20T23:55:00Z",
                    "2026-03-21T00:10:00Z",
                    "2026-03-21T23:50:00Z",
                ],
                "gate_base_2of3": [0, 1, 1, 0],
            }
        )

        on_df, off_df = build_daily_gate_artifacts(df, on_source_col="gate_base_2of3")

        self.assertEqual(on_df.to_dict("records"), [
            {"ts": "2026-03-20T00:00:00Z", "gate_on_2of3": 1},
            {"ts": "2026-03-21T00:00:00Z", "gate_on_2of3": 0},
        ])
        self.assertEqual(off_df.to_dict("records"), [
            {"ts": "2026-03-20T00:00:00Z", "gate_off_2of3": 0},
            {"ts": "2026-03-21T00:00:00Z", "gate_off_2of3": 1},
        ])

    def test_write_daily_gate_artifacts_builds_from_renko_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            renko_path = root / "renko.parquet"
            on_path = root / "live" / "gate_daily.csv"
            off_path = root / "live" / "gate_daily_off.csv"

            ts = pd.date_range("2026-03-10T00:00:00Z", periods=5000, freq="5min", tz="UTC")
            base = 100.0 + np.linspace(0.0, 18.0, len(ts))
            wave = 1.5 * np.sin(np.linspace(0, 80, len(ts)))
            close = base + wave
            open_ = np.roll(close, 1)
            open_[0] = close[0] - 0.25
            spread = 0.35 + 0.10 * np.sin(np.linspace(0, 50, len(ts)))
            pd.DataFrame(
                {
                    "ts": ts,
                    "open": open_,
                    "high": np.maximum(open_, close) + spread,
                    "low": np.minimum(open_, close) - spread,
                    "close": close,
                }
            ).to_parquet(renko_path, index=False)

            write_daily_gate_artifacts(
                input_path=renko_path,
                out_on_path=on_path,
                out_off_path=off_path,
                persist_to_postgres=False,
                publish_latest=False,
            )

            on_df = pd.read_csv(on_path)
            off_df = pd.read_csv(off_path)

            self.assertIn("ts", on_df.columns)
            self.assertIn("gate_on_2of3", on_df.columns)
            self.assertIn("ts", off_df.columns)
            self.assertIn("gate_off_2of3", off_df.columns)
            self.assertGreaterEqual(len(on_df), 2)
            self.assertEqual(len(on_df), len(off_df))
            self.assertTrue(((on_df["gate_on_2of3"] + off_df["gate_off_2of3"]) == 1).all())

    @patch("quant.execution.daily_gate_artifacts.publish_latest_daily_gate_snapshot")
    @patch("quant.execution.daily_gate_artifacts.upsert_daily_gate_history")
    def test_write_daily_gate_artifacts_persists_history_and_latest_snapshot(self, mock_upsert, mock_publish) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            renko_path = root / "renko.parquet"

            ts = pd.date_range("2026-03-10T00:00:00Z", periods=5000, freq="5min", tz="UTC")
            base = 100.0 + np.linspace(0.0, 18.0, len(ts))
            wave = 1.5 * np.sin(np.linspace(0, 80, len(ts)))
            close = base + wave
            open_ = np.roll(close, 1)
            open_[0] = close[0] - 0.25
            spread = 0.35 + 0.10 * np.sin(np.linspace(0, 50, len(ts)))
            pd.DataFrame(
                {
                    "ts": ts,
                    "open": open_,
                    "high": np.maximum(open_, close) + spread,
                    "low": np.minimum(open_, close) - spread,
                    "close": close,
                }
            ).to_parquet(renko_path, index=False)

            write_daily_gate_artifacts(
                input_path=renko_path,
                symbol="SOL-USDT",
            )

            self.assertGreaterEqual(mock_upsert.call_count, 2)
            mock_publish.assert_called_once()

    @patch(
        "quant.execution.daily_gate_artifacts.load_live_renko_bricks_from_postgres",
        return_value=pd.DataFrame(
            {
                "ts": pd.date_range("2026-03-10T00:00:00Z", periods=5000, freq="5min", tz="UTC"),
                "open": 100.0 + np.linspace(0.0, 10.0, 5000),
                "high": 100.4 + np.linspace(0.0, 10.0, 5000),
                "low": 99.6 + np.linspace(0.0, 10.0, 5000),
                "close": 100.2 + np.linspace(0.0, 10.0, 5000),
            }
        ),
    )
    @patch("quant.execution.daily_gate_artifacts.publish_latest_daily_gate_snapshot")
    @patch("quant.execution.daily_gate_artifacts.upsert_daily_gate_history")
    def test_write_daily_gate_artifacts_reads_renko_from_postgres_first(
        self,
        mock_upsert,
        mock_publish,
        mock_load_pg,
    ) -> None:
        write_daily_gate_artifacts(
            input_path="/definitely/missing/renko.parquet",
            symbol="SOL-USDT",
        )

        mock_load_pg.assert_called_once()
        self.assertGreaterEqual(mock_upsert.call_count, 2)
        mock_publish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
