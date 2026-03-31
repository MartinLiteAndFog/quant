import tempfile
import unittest
from pathlib import Path

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

    def test_write_daily_gate_artifacts_builds_from_predictions_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            predictions_path = root / "predictions.parquet"
            on_path = root / "live" / "gate_daily.csv"
            off_path = root / "live" / "gate_daily_off.csv"

            ts = pd.date_range("2026-03-18T00:00:00Z", periods=3000, freq="min", tz="UTC")
            close = 100.0 + np.linspace(0.0, 8.0, len(ts)) + np.sin(np.linspace(0, 30, len(ts)))
            v_temporal = 0.15 + 0.05 * np.sin(np.linspace(0, 40, len(ts)))
            v_obs_mean = 0.25 + 0.04 * np.cos(np.linspace(0, 25, len(ts)))
            pd.DataFrame(
                {
                    "ts": ts,
                    "close": close,
                    "v_temporal": v_temporal,
                    "v_obs_mean": v_obs_mean,
                }
            ).to_parquet(predictions_path, index=False)

            write_daily_gate_artifacts(
                input_path=predictions_path,
                out_on_path=on_path,
                out_off_path=off_path,
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


if __name__ == "__main__":
    unittest.main()
