from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quant.state_space_transitions.config import TransitionConfig
from quant.state_space_transitions.pipeline import run_pipeline
from quant.state_space_transitions.voxelizer import voxelize_dataframe


class StateSpaceTransitionsTests(unittest.TestCase):
    def _make_df(self, n: int = 200) -> pd.DataFrame:
        ts = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
        x = np.linspace(-1.0, 1.0, n)
        y = np.sin(np.linspace(0.0, 6.0, n))
        z = np.cos(np.linspace(0.0, 4.0, n))
        return pd.DataFrame({"ts": ts, "X_raw": x, "Y_res": y, "Z_res": z})

    def test_voxelize_dataframe_outputs_expected_columns(self) -> None:
        df = self._make_df(300)
        cfg = TransitionConfig(n_bins=16)
        out, edges, mids = voxelize_dataframe(df, cfg)
        for col in ["ix", "iy", "iz", "voxel_id", "cx", "cy", "cz"]:
            self.assertIn(col, out.columns)
        self.assertEqual(len(edges), 3)
        self.assertEqual(len(mids), 3)
        self.assertTrue((out["voxel_id"] >= 0).all())

    def test_run_pipeline_writes_required_outputs(self) -> None:
        df = self._make_df(1000)
        cfg = TransitionConfig(n_bins=12, topk=5, run_id="test-run", n_min_voxel=1)
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "transitions"
            result = run_pipeline(df=df, cfg=cfg, out_dir=out_dir, basin_k=8)
            self.assertGreater(result["n_rows"], 0)

            required = [
                "edges.json",
                "df_with_voxels.parquet",
                "edges_counts.parquet",
                "transitions_topk.parquet",
                "voxel_stats.parquet",
                "basin_voxels.parquet",
                "basin_transitions.parquet",
                "basin_stats.parquet",
            ]
            for name in required:
                self.assertTrue((out_dir / name).exists(), name)

            with open(out_dir / "edges.json", "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertIn("n_bins", payload)
            self.assertIn("edges", payload)
            self.assertIn("mids", payload)


if __name__ == "__main__":
    unittest.main()
