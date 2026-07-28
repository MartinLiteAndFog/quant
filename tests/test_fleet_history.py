from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from quant.execution.cashflow_sync import _coverage_start
from quant.execution.fleet_api import _effective_since, _history_start_ts
from quant.execution.fleet_history import fleet_history_start


class FleetHistoryStartTests(unittest.TestCase):
    def test_default_is_shared_audited_start(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            expected = pd.Timestamp("2026-07-22T00:00:00Z")
            self.assertEqual(fleet_history_start(), expected)
            self.assertEqual(_history_start_ts(), expected)
            self.assertEqual(
                _coverage_start(pd.Timestamp("2026-07-28T12:00:00Z"), initial=True),
                expected,
            )

    def test_selected_range_cannot_cross_global_floor(self) -> None:
        with patch.dict(
            "os.environ",
            {"FLEET_HISTORY_START": "2026-07-22"},
            clear=False,
        ):
            with patch(
                "quant.execution.fleet_api._hours_cutoff_ts",
                return_value=pd.Timestamp("2026-07-20T00:00:00Z"),
            ):
                self.assertEqual(
                    _effective_since(168),
                    pd.Timestamp("2026-07-22T00:00:00Z"),
                )

    def test_invalid_value_fails_closed_to_audited_start(self) -> None:
        with patch.dict(
            "os.environ",
            {"FLEET_HISTORY_START": "not-a-date"},
            clear=False,
        ):
            self.assertEqual(
                fleet_history_start(),
                pd.Timestamp("2026-07-22T00:00:00Z"),
            )

    def test_explicit_off_remains_supported(self) -> None:
        with patch.dict(
            "os.environ",
            {"FLEET_HISTORY_START": "off"},
            clear=False,
        ):
            self.assertIsNone(fleet_history_start())


if __name__ == "__main__":
    unittest.main()
