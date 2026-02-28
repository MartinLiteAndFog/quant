from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "trim_statespace_parquet.py"
    spec = importlib.util.spec_from_file_location("trim_statespace_parquet", str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load script module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrimStateSpaceParquetTests(unittest.TestCase):
    def test_numeric_ts_unit_detection(self) -> None:
        mod = _load_script_module()
        s = pd.Series([1704067200, 1704067260], dtype=np.int64)  # sec
        out = mod.parse_ts_to_utc(s)
        self.assertTrue(pd.api.types.is_datetime64tz_dtype(out.dtype))
        self.assertEqual(str(out.iloc[0]), "2024-01-01 00:00:00+00:00")

    def test_stable_sort_and_dedup_keep_first(self) -> None:
        mod = _load_script_module()
        df = pd.DataFrame(
            {
                "ts": ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"],
                "v": [10, 20, 30],
            }
        )
        cleaned = mod.prepare_dataframe(df, ts_col="ts", months=None)
        self.assertEqual(len(cleaned), 2)
        # keep first duplicate occurrence after stable sort
        self.assertEqual(int(cleaned.iloc[0]["v"]), 10)


if __name__ == "__main__":
    unittest.main()
