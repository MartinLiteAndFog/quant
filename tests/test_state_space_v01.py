from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant.state_space.config import StateSpaceConfig
from quant.state_space.pipeline import compute_state_space
from quant.state_space.residualize import ols_residual
from quant.state_space.utils_robust import robust_z


class StateSpaceV01Tests(unittest.TestCase):
    def test_robust_z_smoke(self) -> None:
        x = pd.Series(np.linspace(1.0, 200.0, 200))
        z = robust_z(x, window=20)
        self.assertEqual(len(z), len(x))
        self.assertTrue(z.iloc[:19].isna().all())
        self.assertTrue(np.isfinite(z.iloc[40:]).all())

    def test_ols_residual_smoke(self) -> None:
        x = pd.Series(np.arange(1.0, 201.0))
        y = 2.0 + 3.0 * x
        resid = ols_residual(y, x)
        finite = resid.dropna()
        self.assertGreater(len(finite), 0)
        self.assertLess(float(finite.abs().max()), 1e-9)

    def test_pipeline_output_columns(self) -> None:
        n = 600
        rng = np.random.default_rng(7)
        ts = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
        close = 100.0 + np.cumsum(rng.normal(0.0, 0.5, size=n))
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        high = np.maximum(open_, close) + np.abs(rng.normal(0.1, 0.05, size=n))
        low = np.minimum(open_, close) - np.abs(rng.normal(0.1, 0.05, size=n))
        volume = rng.integers(100, 1000, size=n).astype(float)
        df = pd.DataFrame(
            {
                "ts": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
        out = compute_state_space(df, StateSpaceConfig())
        required = {
            "ts",
            "X_raw",
            "Y_raw",
            "Z_raw",
            "Y_res",
            "Z_res",
            "signal_x",
            "reliability_x",
            "disagreement_x",
            "conf_x",
            "signal_y",
            "reliability_y",
            "disagreement_y",
            "conf_y",
            "signal_z",
            "reliability_z",
            "disagreement_z",
            "conf_z",
        }
        self.assertTrue(required.issubset(set(out.columns)))
        self.assertEqual(len(out), len(df))


if __name__ == "__main__":
    unittest.main()
