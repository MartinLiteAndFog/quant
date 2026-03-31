import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    @patch("quant.execution.CHOPgate._read_live_gate_from_redis")
    @patch("quant.execution.CHOPgate.load_latest_daily_gate_from_postgres")
    def test_prefers_postgres_daily_gate_history_over_redis(self, mock_pg, mock_redis):
        mock_pg.return_value = {
            "ts": _daily_ts(0),
            "gate_on": 1,
            "gate_off": 0,
            "source": "postgres_daily_gate",
            "primary": "on",
            "gate_countertrend_on": 1,
            "gate_trend_on": 0,
            "gate_on_ts": _daily_ts(0),
            "gate_off_ts": _daily_ts(0),
            "gate_on_age_sec": 10.0,
            "gate_off_age_sec": 10.0,
        }
        mock_redis.return_value = {
            "ts": _daily_ts(0),
            "gate_on": 0,
            "gate_off": 1,
            "source": "redis",
            "gate_countertrend_on": 0,
            "gate_trend_on": 1,
        }

        result = get_live_gate_state()

        self.assertEqual(result["source"], "postgres_daily_gate")
        self.assertEqual(result["gate_countertrend_on"], 1)
        self.assertEqual(result["gate_trend_on"], 0)
        mock_pg.assert_called_once()
        mock_redis.assert_not_called()

    @patch("quant.execution.CHOPgate._read_live_gate_from_redis", return_value=None)
    @patch("quant.execution.CHOPgate.load_latest_daily_gate_from_postgres")
    def test_primary_off_uses_postgres_trend_gate(self, mock_pg, mock_redis):
        mock_pg.return_value = {
            "ts": _daily_ts(0),
            "gate_on": 1,
            "gate_off": 0,
            "source": "postgres_daily_gate",
            "primary": "off",
            "gate_countertrend_on": 0,
            "gate_trend_on": 1,
            "gate_on_ts": _daily_ts(0),
            "gate_off_ts": _daily_ts(0),
            "gate_on_age_sec": 10.0,
            "gate_off_age_sec": 10.0,
        }

        result = get_live_gate_state()

        self.assertEqual(result["source"], "postgres_daily_gate")
        self.assertEqual(result["gate_countertrend_on"], 0)
        self.assertEqual(result["gate_trend_on"], 1)
        self.assertEqual(result["gate_on"], 1)
        self.assertEqual(result["gate_off"], 0)

    @patch("quant.execution.CHOPgate._read_live_gate_from_redis")
    @patch("quant.execution.CHOPgate.load_latest_daily_gate_from_postgres", return_value=None)
    def test_uses_redis_when_postgres_missing(self, mock_pg, mock_redis):
        mock_redis.return_value = {
            "ts": _daily_ts(0),
            "gate_on": 0,
            "gate_off": 1,
            "source": "redis",
            "gate_countertrend_on": 0,
            "gate_trend_on": 1,
        }

        result = get_live_gate_state()

        self.assertEqual(result["source"], "redis")
        self.assertEqual(result["gate_countertrend_on"], 0)
        self.assertEqual(result["gate_trend_on"], 1)
        mock_pg.assert_called_once()
        mock_redis.assert_called_once()

    @patch("quant.execution.CHOPgate._read_live_gate_from_redis", return_value=None)
    @patch("quant.execution.CHOPgate.load_latest_daily_gate_from_postgres", return_value=None)
    def test_missing_postgres_and_redis_forces_countertrend_fallback(self, mock_pg, mock_redis):
        result = get_live_gate_state()

        self.assertEqual(result["source"], "forced_countertrend")
        self.assertEqual(result["gate_on"], 1)
        self.assertEqual(result["gate_off"], 0)
        self.assertEqual(result["gate_countertrend_on"], 1)
        self.assertEqual(result["gate_trend_on"], 0)
        mock_pg.assert_called_once()
        mock_redis.assert_called_once()


if __name__ == "__main__":
    unittest.main()
