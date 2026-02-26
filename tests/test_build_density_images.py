from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class TestDensityBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        n = 500
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "ts": pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"),
            "X_raw": rng.normal(0, 0.3, n).clip(-1, 1),
            "Y_res": rng.normal(0, 0.3, n).clip(-1, 1),
            "Z_res": rng.normal(0, 0.3, n).clip(-1, 1),
        })
        df.to_parquet(self.root / "state_space.parquet", index=False)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_build_density_pngs(self) -> None:
        from scripts.build_density_images import build_density_images
        out_dir = self.root / "density"
        build_density_images(
            state_space_path=self.root / "state_space.parquet",
            out_dir=out_dir,
        )
        for name in ("density_bg_xy.png", "density_bg_xz.png", "density_bg_yz.png"):
            p = out_dir / name
            self.assertTrue(p.exists(), f"Missing: {name}")
            self.assertGreater(p.stat().st_size, 100, f"Empty: {name}")


if __name__ == "__main__":
    unittest.main()
