"""Regression tests for inbound TradingView signal logging.

Context: the first version of `_log_inbound_signal` wrote the raw action string
("entry", "tp1", ...) into `signal_events.signal` — a smallint constrained to
(-1, 0, 1) — and the raw side ("buy"/"sell"/None) into `signal_side`, which is
NOT NULL and constrained to ('short', 'flat', 'long'). Every insert therefore
raised, the exception was swallowed, and `signal_events` sat empty fleet-wide
while TradingView signals were actually arriving. `/diag/timeline` reported
"signals: 0", which read as "no signals arrived" rather than "the writer is
broken".

These tests pin the row to the DDL in src/quant/sql/001_events.sql so that
schema drift fails here instead of silently in production.
"""

from __future__ import annotations

import json
import unittest
from contextlib import contextmanager
from typing import Any, Dict, List
from unittest.mock import patch

from quant.execution.bot_webhook import _log_inbound_signal, _signal_code

# Mirrors the check constraints on signal_events in src/quant/sql/001_events.sql.
VALID_SIGNAL_CODES = {-1, 0, 1}
VALID_SIGNAL_SIDES = {"short", "flat", "long"}
# Columns declared NOT NULL without a default on signal_events.
REQUIRED_NOT_NULL = (
    "event_id",
    "ts",
    "strategy",
    "config_hash",
    "symbol",
    "signal",
    "signal_side",
    "signal_family",
    "signal_kind",
)


class _FakeCursor:
    def __init__(self, sink: List[Dict[str, Any]]) -> None:
        self._sink = sink

    def execute(self, sql: str, params: Any = None) -> None:
        if isinstance(params, dict):
            self._sink.append(params)

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class _FakeConn:
    def __init__(self, sink: List[Dict[str, Any]]) -> None:
        self._sink = sink

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._sink)

    def __enter__(self) -> "_FakeConn":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _capture(payload: Dict[str, Any], disposition: str = "accepted") -> List[Dict[str, Any]]:
    """Run the logger against a fake connection, returning the parameter dicts
    that would actually reach Postgres.

    Stubbing at the connection rather than at `insert_signal_event` is
    deliberate: `insert_signal_event` fills in defaults (config_hash,
    source_type, ...), so those are part of the row the database sees. Asserting
    on the caller's dict would test the wrong boundary and miss exactly the kind
    of column/constraint drift these tests exist to catch.
    """
    sink: List[Dict[str, Any]] = []

    @contextmanager
    def _fake_get_conn() -> Any:
        yield _FakeConn(sink)

    with patch("quant.execution.event_store.get_conn", _fake_get_conn):
        _log_inbound_signal(payload, disposition=disposition, detail="unit-test")
    return sink


class TestSignalCodeMapping(unittest.TestCase):
    def test_directional_actions_carry_side(self) -> None:
        self.assertEqual(_signal_code("entry", "buy"), 1)
        self.assertEqual(_signal_code("entry", "sell"), -1)
        self.assertEqual(_signal_code("flip", "buy"), 1)
        self.assertEqual(_signal_code("flip", "sell"), -1)

    def test_reducing_actions_are_flat(self) -> None:
        for action in ("exit", "tp1", "tp2", "sl"):
            with self.subTest(action=action):
                self.assertEqual(_signal_code(action, ""), 0)

    def test_unknown_or_incomplete_input_is_flat_not_an_error(self) -> None:
        # A malformed signal asserts no direction; it must still be recordable,
        # because an unparseable inbound signal is exactly what we want to see
        # in the timeline.
        for action, side in (("", ""), ("bogus", "buy"), ("entry", ""), ("flip", "hold")):
            with self.subTest(action=action, side=side):
                self.assertEqual(_signal_code(action, side), 0)

    def test_every_mapping_output_satisfies_the_check_constraint(self) -> None:
        for action in ("entry", "exit", "flip", "tp1", "tp2", "sl", "bogus", ""):
            for side in ("buy", "sell", "", "garbage"):
                with self.subTest(action=action, side=side):
                    self.assertIn(_signal_code(action, side), VALID_SIGNAL_CODES)


def _payload_of(row: Dict[str, Any]) -> Dict[str, Any]:
    """payload_json is serialised to a JSON string before it reaches the DB."""
    return json.loads(row["payload_json"])


class TestInboundSignalRow(unittest.TestCase):
    def _assert_row_valid(self, row: Dict[str, Any]) -> None:
        for col in REQUIRED_NOT_NULL:
            self.assertIsNotNone(row.get(col), f"NOT NULL column {col} was None")
        self.assertIn(row["signal"], VALID_SIGNAL_CODES)
        self.assertIn(row["signal_side"], VALID_SIGNAL_SIDES)
        # The regression itself: `signal` must be an integer code, never the
        # raw action string.
        self.assertIsInstance(row["signal"], int)

    def test_entry_buy_records_a_long(self) -> None:
        rows = _capture({"action": "entry", "side": "buy", "symbol": "SOL-USDT"})
        self.assertEqual(len(rows), 1)
        self._assert_row_valid(rows[0])
        self.assertEqual(rows[0]["signal"], 1)
        self.assertEqual(rows[0]["signal_side"], "long")

    def test_flip_sell_records_a_short(self) -> None:
        rows = _capture({"action": "flip", "side": "sell", "symbol": "SOL-USDT"})
        self._assert_row_valid(rows[0])
        self.assertEqual(rows[0]["signal"], -1)
        self.assertEqual(rows[0]["signal_side"], "short")

    def test_reducing_action_records_flat_and_keeps_the_action(self) -> None:
        rows = _capture({"action": "tp2", "symbol": "SOL-USDT"})
        self._assert_row_valid(rows[0])
        self.assertEqual(rows[0]["signal"], 0)
        self.assertEqual(rows[0]["signal_side"], "flat")
        # The typed columns only model direction, so "which of tp1/tp2/sl fired"
        # must survive somewhere — otherwise the timeline cannot tell them apart.
        self.assertEqual(rows[0]["signal_kind"], "tp2")
        self.assertEqual(_payload_of(rows[0])["action"], "tp2")

    def test_every_disposition_produces_a_valid_row(self) -> None:
        for disposition in (
            "accepted",
            "skipped_bot_mismatch",
            "rejected_not_ready",
            "rejected_parse_error",
        ):
            with self.subTest(disposition=disposition):
                rows = _capture({"action": "entry", "side": "buy"}, disposition=disposition)
                self.assertEqual(len(rows), 1)
                self._assert_row_valid(rows[0])
                self.assertEqual(_payload_of(rows[0])["disposition"], disposition)

    def test_garbage_payload_still_records_a_valid_row(self) -> None:
        # The rejected_parse_error path receives exactly the payloads that failed
        # validation, so it must never depend on them being well formed.
        for payload in ({}, {"action": None}, {"action": "???", "side": 12345}):
            with self.subTest(payload=payload):
                rows = _capture(payload, disposition="rejected_parse_error")
                self.assertEqual(len(rows), 1)
                self._assert_row_valid(rows[0])

    def test_symbol_falls_back_when_absent(self) -> None:
        rows = _capture({"action": "exit"})
        self.assertTrue(rows[0]["symbol"], "symbol must never be empty (NOT NULL)")

    def test_logging_failure_never_breaks_signal_execution(self) -> None:
        # Diagnostics must stay non-fatal: a DB outage cannot cost us a trade.
        with patch(
            "quant.execution.event_store.insert_signal_event",
            side_effect=RuntimeError("db down"),
        ):
            _log_inbound_signal({"action": "entry", "side": "buy"}, disposition="accepted")


if __name__ == "__main__":
    unittest.main()
