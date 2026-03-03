import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant.execution.gate_provider import get_live_gate_from_statespace, get_live_gate_state


def _recent_ts(periods: int, freq: str = "5min") -> pd.DatetimeIndex:
    """Generate timestamps ending near 'now' so the max-age check passes."""
    end = pd.Timestamp.now("UTC").floor("min")
    return pd.date_range(end=end, periods=periods, freq=freq, tz="UTC")


class TestLiveGateFromStatespace(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ss_path = Path(self.tmp.name) / "state_space_latest.parquet"
        os.environ["DASHBOARD_STATESPACE_PARQUET"] = str(self.ss_path)

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("DASHBOARD_STATESPACE_PARQUET", None)
        for k in list(os.environ):
            if k.startswith("LIVE_GATE_"):
                del os.environ[k]

    def test_returns_gate_on_when_conditions_met(self):
        df = pd.DataFrame({
            "ts": _recent_ts(5),
            "X_raw": [0.05, 0.03, 0.02, 0.01, 0.0],
            "Y_res": [0.6, 0.7, 0.8, 0.7, 0.75],
            "Z_res": [-0.5, -0.4, -0.3, -0.35, -0.3],
        })
        df.to_parquet(self.ss_path, index=False)
        result = get_live_gate_from_statespace()
        self.assertEqual(result["gate_on"], 1)
        self.assertEqual(result["source"], "statespace_live")

    def test_returns_gate_off_when_trending(self):
        df = pd.DataFrame({
            "ts": _recent_ts(5),
            "X_raw": [0.8, 0.85, 0.9, 0.88, 0.92],
            "Y_res": [0.1, 0.05, 0.0, -0.1, -0.05],
            "Z_res": [0.6, 0.7, 0.8, 0.75, 0.85],
        })
        df.to_parquet(self.ss_path, index=False)
        result = get_live_gate_from_statespace()
        self.assertEqual(result["gate_on"], 0)

    def test_returns_none_when_file_missing(self):
        os.environ["DASHBOARD_STATESPACE_PARQUET"] = "/nonexistent/path.parquet"
        result = get_live_gate_from_statespace()
        self.assertIsNone(result)

    def test_custom_thresholds_via_env(self):
        os.environ["LIVE_GATE_DRIFT_THRESH"] = "0.01"
        df = pd.DataFrame({
            "ts": _recent_ts(3),
            "X_raw": [0.05, 0.04, 0.03],
            "Y_res": [-0.1, -0.1, -0.1],
            "Z_res": [-0.5, -0.5, -0.5],
        })
        df.to_parquet(self.ss_path, index=False)
        result = get_live_gate_from_statespace()
        self.assertEqual(result["g1_drift"], 0)
        self.assertEqual(result["g2_elasticity"], 0)
        self.assertEqual(result["gate_on"], 0)

    def test_returns_none_when_data_too_old(self):
        old_ts = pd.date_range("2020-01-01", periods=3, freq="5min", tz="UTC")
        df = pd.DataFrame({
            "ts": old_ts,
            "X_raw": [0.0, 0.0, 0.0],
            "Y_res": [0.5, 0.5, 0.5],
            "Z_res": [-0.5, -0.5, -0.5],
        })
        df.to_parquet(self.ss_path, index=False)
        result = get_live_gate_from_statespace()
        self.assertIsNone(result)


class TestGateStatePriorityChain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ss_path = Path(self.tmp.name) / "state_space_latest.parquet"
        self.gate_csv = Path(self.tmp.name) / "gate.csv"
        os.environ["DASHBOARD_STATESPACE_PARQUET"] = str(self.ss_path)
        os.environ["PC_PREDICTIONS_PARQUET"] = "/nonexistent/predictions.parquet"
        os.environ["PC_GATE_CSV"] = str(self.gate_csv)

    def tearDown(self):
        self.tmp.cleanup()
        for k in ["DASHBOARD_STATESPACE_PARQUET", "PC_PREDICTIONS_PARQUET", "PC_GATE_CSV", "GATE_PREFER_CSV"]:
            os.environ.pop(k, None)
        for k in list(os.environ):
            if k.startswith("LIVE_GATE_"):
                del os.environ[k]

    def test_prefers_csv_when_gate_prefer_csv_set(self):
        """GATE_PREFER_CSV=1 (explicit): exact backtest parity from gate_base_2of3"""
        os.environ["GATE_PREFER_CSV"] = "1"
        pd.DataFrame({
            "ts": ["2026-03-01T12:00:00Z"],
            "gate_base_2of3": [1],
        }).to_csv(self.gate_csv, index=False)
        df = pd.DataFrame({
            "ts": _recent_ts(3),
            "X_raw": [0.0, 0.0, 0.0],
            "Y_res": [0.5, 0.5, 0.5],
            "Z_res": [-0.5, -0.5, -0.5],
        })
        df.to_parquet(self.ss_path, index=False)
        result = get_live_gate_state()
        self.assertEqual(result["source"], "gate_csv")
        self.assertEqual(result["gate_on"], 1)

    def test_prefers_statespace_when_gate_prefer_csv_off(self):
        os.environ["GATE_PREFER_CSV"] = "0"
        pd.DataFrame({"ts": ["2026-03-01T12:00:00Z"], "gate_base_2of3": [0]}).to_csv(self.gate_csv, index=False)
        df = pd.DataFrame({
            "ts": _recent_ts(3),
            "X_raw": [0.0, 0.0, 0.0],
            "Y_res": [0.5, 0.5, 0.5],
            "Z_res": [-0.5, -0.5, -0.5],
        })
        df.to_parquet(self.ss_path, index=False)
        result = get_live_gate_state()
        self.assertEqual(result["source"], "statespace_live")

    def test_falls_back_to_default_when_no_data(self):
        os.environ["PC_GATE_CSV"] = "/nonexistent/gate.csv"
        result = get_live_gate_state()
        self.assertEqual(result["source"], "default_off")


if __name__ == "__main__":
    unittest.main()
