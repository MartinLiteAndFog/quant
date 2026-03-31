import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant.execution.CHOPgate import get_live_gate_state


def _daily_ts(days_ago: int) -> str:
    return (pd.Timestamp.now("UTC").normalize() - pd.Timedelta(days=days_ago)).strftime("%Y-%m-%dT00:00:00Z")


class TestDailyCsvLiveGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.on_csv = root / "gate_daily.csv"
        self.off_csv = root / "gate_daily_off.csv"
        os.environ["GATE_DAILY_PATH"] = str(self.on_csv)
        os.environ["GATE_DAILY_COL"] = "gate_on_2of3"
        os.environ["GATE_DAILY_OFF_PATH"] = str(self.off_csv)
        os.environ["GATE_DAILY_OFF_COL"] = "gate_off_2of3"
        os.environ.pop("LIVE_GATE_PRIMARY", None)

    def tearDown(self):
        self.tmp.cleanup()
        for k in [
            "GATE_DAILY_PATH",
            "GATE_DAILY_COL",
            "GATE_DAILY_OFF_PATH",
            "GATE_DAILY_OFF_COL",
            "LIVE_GATE_PRIMARY",
            "LIVE_RENKO_PATH",
        ]:
            os.environ.pop(k, None)
        for k in list(os.environ):
            if k.startswith("LIVE_GATE_"):
                del os.environ[k]

    def test_reads_latest_applicable_daily_csv_rows(self):
        pd.DataFrame(
            {
                "ts": [_daily_ts(2), _daily_ts(0)],
                "gate_on_2of3": [0, 1],
            }
        ).to_csv(self.on_csv, index=False)
        pd.DataFrame(
            {
                "ts": [_daily_ts(2), _daily_ts(0)],
                "gate_off_2of3": [1, 0],
            }
        ).to_csv(self.off_csv, index=False)
        os.environ["LIVE_GATE_PRIMARY"] = "on"

        result = get_live_gate_state()

        self.assertEqual(result["source"], "daily_csv")
        self.assertEqual(result["gate_countertrend_on"], 1)
        self.assertEqual(result["gate_trend_on"], 0)
        self.assertEqual(result["gate_on"], 1)
        self.assertEqual(result["gate_off"], 0)

    def test_ignores_future_daily_rows(self):
        future_ts = (pd.Timestamp.now("UTC").normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
        pd.DataFrame(
            {
                "ts": [_daily_ts(1), future_ts],
                "gate_on_2of3": [1, 0],
            }
        ).to_csv(self.on_csv, index=False)
        pd.DataFrame(
            {
                "ts": [_daily_ts(1), future_ts],
                "gate_off_2of3": [0, 1],
            }
        ).to_csv(self.off_csv, index=False)
        os.environ["LIVE_GATE_PRIMARY"] = "on"

        result = get_live_gate_state()

        self.assertEqual(result["gate_countertrend_on"], 1)
        self.assertEqual(result["gate_trend_on"], 0)
        self.assertEqual(result["ts"], _daily_ts(1))

    def test_primary_off_uses_off_gate_column(self):
        pd.DataFrame({"ts": [_daily_ts(0)], "gate_on_2of3": [0]}).to_csv(self.on_csv, index=False)
        pd.DataFrame({"ts": [_daily_ts(0)], "gate_off_2of3": [1]}).to_csv(self.off_csv, index=False)

        result = get_live_gate_state()

        self.assertEqual(result["gate_countertrend_on"], 0)
        self.assertEqual(result["gate_trend_on"], 1)
        self.assertEqual(result["gate_on"], 1)
        self.assertEqual(result["gate_off"], 0)

    def test_missing_daily_csv_defaults_safe_off(self):
        result = get_live_gate_state()

        self.assertEqual(result["gate_on"], 0)
        self.assertEqual(result["gate_off"], 1)
        self.assertEqual(result["source"], "default_off")
        self.assertIn("error", result)

    def test_missing_daily_csv_does_not_fall_back_to_renko_gate(self):
        renko_path = Path(self.tmp.name) / "renko_latest.parquet"
        ts = pd.date_range(end=pd.Timestamp.now("UTC").floor("min"), periods=220, freq="5min", tz="UTC")
        close = pd.Series(range(len(ts)), dtype=float) + 100.0
        renko = pd.DataFrame(
            {
                "ts": ts,
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close + 0.25,
            }
        )
        renko.to_parquet(renko_path, index=False)
        os.environ["LIVE_RENKO_PATH"] = str(renko_path)

        result = get_live_gate_state()

        self.assertEqual(result["source"], "default_off")
        self.assertEqual(result["gate_on"], 0)
        self.assertEqual(result["gate_off"], 1)
        self.assertIn("daily_csv_error", result)


if __name__ == "__main__":
    unittest.main()
