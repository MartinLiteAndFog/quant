"""Unit tests for the autonomous equity snapshot writer (P0 fleet fix)."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import quant.execution.equity_snapshot_writer as writer


class WriteOnceTests(unittest.TestCase):
    def test_writes_snapshot_row_on_ok_fetch(self) -> None:
        with patch.object(
            writer,
            "_fetch_equity",
            return_value={"ok": True, "equity": 54.2, "currency": "USDT"},
        ), patch("quant.execution.event_store.insert_equity_snapshot") as ins:
            ok = writer.write_equity_snapshot_once(
                venue="kucoin", account="sol-pilot-canonical"
            )
        self.assertTrue(ok)
        row = ins.call_args[0][0]
        self.assertEqual(row["venue"], "kucoin")
        self.assertEqual(row["account"], "sol-pilot-canonical")
        self.assertEqual(row["equity"], 54.2)
        self.assertEqual(row["source"], "equity_snapshot_writer")

    def test_skips_on_failed_fetch(self) -> None:
        with patch.object(
            writer,
            "_fetch_equity",
            return_value={"ok": False, "error": "kucoin_credentials_missing"},
        ), patch("quant.execution.event_store.insert_equity_snapshot") as ins:
            ok = writer.write_equity_snapshot_once(venue="kucoin", account="x")
        self.assertFalse(ok)
        self.assertFalse(ins.called)

    def test_skips_on_zero_equity(self) -> None:
        with patch.object(
            writer,
            "_fetch_equity",
            return_value={"ok": True, "equity": 0.0, "currency": "USDT"},
        ), patch("quant.execution.event_store.insert_equity_snapshot") as ins:
            ok = writer.write_equity_snapshot_once(venue="kucoin", account="x")
        self.assertFalse(ok)
        self.assertFalse(ins.called)


class StartGateTests(unittest.TestCase):
    def test_env_zero_disables_even_when_default_on(self) -> None:
        with patch.dict(os.environ, {"FLEET_EQUITY_WRITER_ENABLED": "0"}, clear=False):
            t = writer.start_equity_snapshot_writer(
                venue="kucoin", account="x", default_enabled=True
            )
        self.assertIsNone(t)

    def test_missing_account_refuses_to_start(self) -> None:
        with patch.dict(os.environ, {"FLEET_EQUITY_WRITER_ENABLED": "1"}, clear=False):
            t = writer.start_equity_snapshot_writer(
                venue="kucoin", account="", default_enabled=True
            )
        self.assertIsNone(t)

    def test_interval_floor(self) -> None:
        with patch.dict(os.environ, {"FLEET_EQUITY_WRITER_SEC": "5"}, clear=False):
            self.assertEqual(writer._interval_sec(), 120.0)


if __name__ == "__main__":
    unittest.main()
