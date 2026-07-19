from __future__ import annotations

import csv
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from quant.execution.execution_calibration import (
    ExecutionDecision,
    aggregate_calibration_records,
    export_calibration_csv,
    observe_oms_call,
)
from quant.execution.oms import OmsResult
from quant.execution import event_store


def _decision(**overrides):
    values = {
        "decision_ts": "2026-07-15T10:00:00.000Z",
        "venue": "kraken",
        "strategy": "live_executor_2",
        "symbol": "SOL-USDT",
        "action": "exit_tp2",
        "exit_reason": "tp2_exit",
        "side": "sell",
        "requested_qty": 2.0,
        "reference_bid": 99.9,
        "reference_ask": 100.1,
        "reference_mid": 100.0,
        "reduce_only": True,
    }
    values.update(overrides)
    return ExecutionDecision(**values)


def test_observe_oms_call_captures_venue_fill_fields() -> None:
    clock_values = iter(("2026-07-15T10:00:00.100Z", "2026-07-15T10:00:00.500Z"))
    result = OmsResult(
        ok=True,
        mode="PO",
        details={
            "order_id": "order-1",
            "client_id": "quant:SOL:tp2",
            "side": "sell",
            "qty": 2.0,
            "filled_qty": 1.75,
            "avg_fill_price": 100.2,
            "acknowledged_ts": "2026-07-15T10:00:00.200Z",
            "filled_ts": "2026-07-15T10:00:00.450Z",
            "liquidity": "maker",
            "fee": 0.035,
            "fee_currency": "USD",
            "fee_bps": 2.0,
            "requotes": 1,
            "reduce_only": True,
        },
    )

    captured = observe_oms_call(lambda: result, decision=_decision(), clock=lambda: next(clock_values))
    row = captured.record

    assert captured.result is result
    assert row["status"] == "filled"
    assert row["filled_qty"] == pytest.approx(1.75)
    assert row["filled_qty_inferred"] is False
    assert row["avg_fill_price"] == pytest.approx(100.2)
    assert row["liquidity"] == "maker"
    assert row["fee_bps"] == pytest.approx(2.0)
    assert row["submit_to_ack_ms"] == pytest.approx(100.0)
    assert row["submit_to_fill_ms"] == pytest.approx(350.0)
    assert row["slippage_bps"] == pytest.approx(-20.0)
    assert row["timing_precision"] == "exchange"


def test_observe_oms_call_marks_return_boundary_and_fallback() -> None:
    clock_values = iter(("2026-07-15T10:00:00.100Z", "2026-07-15T10:00:00.900Z"))
    result = OmsResult(
        ok=True,
        mode="FB_MKT_REQUOTE_MAX",
        details={
            "order_id": "order-2",
            "client_id": "FB_MKT_REQUOTE_MAX:1",
            "side": "buy",
            "qty": 3.0,
            "kind": "market",
        },
    )

    row = observe_oms_call(
        lambda: result,
        decision=_decision(action="enter_long", side="buy", requested_qty=3.0, reduce_only=False),
        clock=lambda: next(clock_values),
    ).record

    assert row["acknowledged_ts"] == "2026-07-15T10:00:00.900000Z"
    assert row["filled_ts"] == "2026-07-15T10:00:00.900000Z"
    assert row["timing_precision"] == "oms_return_boundary"
    assert row["filled_qty"] == pytest.approx(3.0)
    assert row["filled_qty_inferred"] is True
    assert row["fallback_used"] is True
    assert row["liquidity"] == "taker"
    assert row["fee"] is None
    assert row["avg_fill_price"] is None


def test_observe_oms_call_records_sanitized_rejection_and_reraises() -> None:
    records = []
    clock_values = iter(("2026-07-15T10:00:00.100Z", "2026-07-15T10:00:00.200Z"))

    def fail():
        raise RuntimeError("secret-key-should-not-be-recorded")

    with pytest.raises(RuntimeError, match="secret-key"):
        observe_oms_call(
            fail,
            decision=_decision(),
            clock=lambda: next(clock_values),
            sink=records.append,
        )

    assert len(records) == 1
    assert records[0]["status"] == "rejected"
    assert records[0]["reject_reason"] == "order_call_failed:RuntimeError"
    assert "secret" not in str(records[0])


def test_export_and_aggregation_are_deterministic(tmp_path: Path) -> None:
    base = observe_oms_call(
        lambda: OmsResult(True, "ENTRY_MKT", {"order_id": "b", "side": "buy", "qty": 1.0}),
        decision=_decision(
            decision_ts="2026-07-15T10:00:01Z",
            action="enter_long",
            side="buy",
            requested_qty=1.0,
            reduce_only=False,
        ),
        clock=iter(("2026-07-15T10:00:01.100Z", "2026-07-15T10:00:01.200Z")).__next__,
    ).record
    earlier = dict(base)
    earlier.update(
        telemetry_id="cal_earlier",
        decision_ts="2026-07-15T09:59:00.000000Z",
        submitted_ts="2026-07-15T09:59:00.100000Z",
        rejected=True,
        status="rejected",
        fallback_used=True,
    )

    output = tmp_path / "calibration.csv"
    assert export_calibration_csv([base, earlier], output) == 2
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["telemetry_id"] for row in rows] == ["cal_earlier", base["telemetry_id"]]

    summary = aggregate_calibration_records([base, earlier])
    assert len(summary) == 1
    assert summary[0]["attempts"] == 2
    assert summary[0]["fill_rate"] == pytest.approx(0.5)
    assert summary[0]["reject_rate"] == pytest.approx(0.5)
    assert summary[0]["fallback_rate"] == pytest.approx(1.0)


class _Cursor:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.executions = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, sql, params=None):
        self.executions.append((sql, params))

    def fetchall(self):
        return list(self.rows)


class _Connection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def test_event_store_insert_and_load_calibration() -> None:
    record = observe_oms_call(
        lambda: OmsResult(True, "ENTRY_MKT", {"order_id": "db-1", "side": "buy", "qty": 1.0}),
        decision=_decision(action="enter_long", side="buy", requested_qty=1.0, reduce_only=False),
        clock=iter(("2026-07-15T10:00:00.100Z", "2026-07-15T10:00:00.200Z")).__next__,
    ).record
    insert_cursor = _Cursor()

    @contextmanager
    def insert_conn():
        yield _Connection(insert_cursor)

    with (
        patch.object(event_store, "ensure_execution_calibration_schema"),
        patch.object(event_store, "get_conn", insert_conn),
    ):
        event_store.insert_execution_calibration(record)

    assert "on conflict (telemetry_id) do nothing" in insert_cursor.executions[0][0].lower()
    assert insert_cursor.executions[0][1]["telemetry_id"] == record["telemetry_id"]

    columns = (
        "telemetry_id", "decision_ts", "submitted_ts", "acknowledged_ts", "filled_ts", "result_ts",
        "venue", "strategy", "symbol", "action", "exit_reason", "side",
        "reference_bid", "reference_ask", "reference_mid", "requested_qty", "filled_qty", "avg_fill_price",
        "order_id", "client_oid", "order_type", "liquidity", "fee", "fee_currency", "fee_bps", "requotes",
        "fallback_used", "rejected", "reject_reason", "reduce_only", "status",
        "submit_to_ack_ms", "submit_to_fill_ms", "decision_to_result_ms", "slippage_bps",
        "timing_precision", "filled_qty_inferred", "fill_price_source", "fee_source",
    )
    load_cursor = _Cursor([tuple(record[key] for key in columns)])

    @contextmanager
    def load_conn():
        yield _Connection(load_cursor)

    with (
        patch.object(event_store, "ensure_execution_calibration_schema"),
        patch.object(event_store, "get_conn", load_conn),
    ):
        loaded = event_store.load_execution_calibration(symbol="SOL-USDT")

    assert loaded == [record]
    assert load_cursor.executions[0][1]["symbol"] == "SOL-USDT"
