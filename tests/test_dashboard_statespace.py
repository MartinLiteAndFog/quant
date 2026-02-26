from __future__ import annotations

import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import pandas as pd


class TestStateSpaceWriter(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        n = 500
        ts = pd.date_range("2026-02-01", periods=n, freq="5min", tz="UTC")
        close = 100.0 + np.cumsum(np.random.default_rng(42).normal(0, 0.05, n))
        renko = pd.DataFrame({
            "ts": ts,
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.3,
            "close": close,
        })
        renko.to_parquet(self.root / "renko.parquet", index=False)
        self._env_patch = unittest.mock.patch.dict(os.environ, {
            "DASHBOARD_RENKO_PARQUET": str(self.root / "renko.parquet"),
            "DASHBOARD_STATESPACE_PARQUET": str(self.root / "state_space.parquet"),
        })
        self._env_patch.start()

    def tearDown(self) -> None:
        self._env_patch.stop()
        self.tmp.cleanup()

    def test_refresh_writes_parquet(self) -> None:
        from quant.execution.dashboard_statespace import refresh_state_space_cache
        info = refresh_state_space_cache()
        self.assertTrue(info["ok"])
        out_path = Path(os.environ["DASHBOARD_STATESPACE_PARQUET"])
        self.assertTrue(out_path.exists())
        df = pd.read_parquet(out_path)
        for col in ("ts", "X_raw", "Y_res", "Z_res", "conf_x", "conf_y", "conf_z"):
            self.assertIn(col, df.columns, f"Missing column: {col}")
        self.assertGreater(len(df), 0)
        for col in ("X_raw", "Y_res", "Z_res"):
            self.assertTrue(df[col].dropna().between(-2.0, 2.0).all(), f"{col} out of range")

    def test_load_trajectory_filters_by_window(self) -> None:
        from quant.execution.dashboard_statespace import refresh_state_space_cache, load_state_space_trajectory
        refresh_state_space_cache()
        full = load_state_space_trajectory(window_hours=9999)
        short = load_state_space_trajectory(window_hours=1)
        self.assertLessEqual(len(short["trajectory"]), len(full["trajectory"]))
        self.assertIn("current", full)
        for key in ("x", "y", "z", "conf_x", "conf_y", "conf_z"):
            self.assertIn(key, full["current"])

    def test_load_trajectory_returns_empty_when_no_data(self) -> None:
        os.environ["DASHBOARD_STATESPACE_PARQUET"] = str(self.root / "nonexistent.parquet")
        from quant.execution.dashboard_statespace import load_state_space_trajectory
        result = load_state_space_trajectory(window_hours=8)
        self.assertEqual(len(result["trajectory"]), 0)
        self.assertIsNone(result["current"])

    def test_refresh_rejects_short_renko(self) -> None:
        n = 10
        ts = pd.date_range("2026-02-01", periods=n, freq="5min", tz="UTC")
        close = 100.0 + np.cumsum(np.random.default_rng(7).normal(0, 0.05, n))
        short_renko = pd.DataFrame({
            "ts": ts,
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.3,
            "close": close,
        })
        short_path = self.root / "renko_short.parquet"
        short_renko.to_parquet(short_path, index=False)
        os.environ["DASHBOARD_RENKO_PARQUET"] = str(short_path)
        from quant.execution.dashboard_statespace import refresh_state_space_cache
        info = refresh_state_space_cache()
        self.assertFalse(info["ok"])
        self.assertIn("renko_too_short", info["reason"])


if __name__ == "__main__":
    unittest.main()
