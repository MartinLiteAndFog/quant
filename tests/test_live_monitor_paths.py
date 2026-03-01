from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from quant.execution.live_monitor import ExpectedTrade, load_expected_trades, record_expected


class LiveMonitorPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.expected_path = self.tmp_path / "expected_trades_custom.jsonl"
        os.environ["DASHBOARD_EXPECTED_TRADES_JSONL"] = str(self.expected_path)
        os.environ["QUANT_LIVE_DIR"] = str(self.tmp_path / "other_live_dir")

    def tearDown(self) -> None:
        os.environ.pop("DASHBOARD_EXPECTED_TRADES_JSONL", None)
        os.environ.pop("QUANT_LIVE_DIR", None)
        self.tmp.cleanup()

    def test_record_expected_uses_dashboard_expected_path_when_set(self) -> None:
        record_expected(
            ExpectedTrade(
                ts="2026-02-27T17:30:00Z",
                symbol="SOL-USDT",
                side="short",
                action="exit_sl",
                qty=5.0,
                expected_px=83.0,
                note="event=manual_flatten_short source=test",
            )
        )
        self.assertTrue(self.expected_path.exists())
        df = load_expected_trades()
        self.assertEqual(len(df), 1)
        self.assertEqual(str(df.iloc[0]["action"]), "exit_sl")


if __name__ == "__main__":
    unittest.main()
