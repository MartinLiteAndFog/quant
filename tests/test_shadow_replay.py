from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from quant.execution.shadow_replay import GatePolicy, compare_evidence, load_jsonl, normalize_evidence


def _rows() -> list[dict]:
    return [
        {
            "replay_kind": "signal",
            "event_id": "sig-1",
            "match_id": "bar-100:signal",
            "ts": "2026-07-15T10:00:00Z",
            "seq": 1,
            "venue": "kraken",
            "symbol": "SOL-USDT",
            "signal": "imba_cross_long",
            "signal_side": "long",
            "regime_on": True,
        },
        {
            "replay_kind": "state",
            "event_id": "act-1",
            "match_id": "bar-100:decision",
            "ts": "2026-07-15T10:00:00.100Z",
            "seq": 2,
            "venue": "kraken",
            "symbol": "SOL-USDT",
            "engine_action": "enter_long",
            "position_before": 0,
            "position_after": 1,
            "qty_before": 0,
            "qty_after": 2,
        },
        {
            "replay_kind": "order",
            "event_id": "ord-event-1",
            "match_id": "entry:bar-100",
            "ts": "2026-07-15T10:00:00.200Z",
            "seq": 3,
            "venue": "kraken",
            "symbol": "SOL-USDT",
            "execution_stage": "submitted",
            "side": "buy",
            "qty": 2,
            "price": 150.0,
            "reduce_only": False,
            "status": "accepted",
        },
        {
            "replay_kind": "fill",
            "telemetry_id": "fill-event-1",
            "match_id": "entry:bar-100",
            "decision_ts": "2026-07-15T10:00:00Z",
            "filled_ts": "2026-07-15T10:00:00.500Z",
            "seq": 4,
            "venue": "kraken",
            "symbol": "SOL-USDT",
            "side": "buy",
            "filled_qty": 2,
            "avg_fill_price": 150.01,
            "fee_bps": 2.0,
            "liquidity": "taker",
            "status": "filled",
        },
    ]


def _compare(expected_rows: list[dict], actual_rows: list[dict], policy: GatePolicy | None = None) -> dict:
    return compare_evidence(
        normalize_evidence(expected_rows, stream="expected"),
        normalize_evidence(actual_rows, stream="actual"),
        policy,
    )


def test_complete_parity_passes_and_report_is_deterministic() -> None:
    rows = _rows()
    first = _compare(rows, rows)
    second = _compare(list(reversed(rows)), [*rows, rows[2]])

    assert first == second
    assert first["gate"]["passed"] is True
    assert first["gate"]["fail_closed"] is False
    assert first["matched_count"] == 4
    assert first["divergence_counts"] == {"signal": 0, "state": 0, "order": 0, "fill": 0}


@pytest.mark.parametrize(
    ("index", "field", "value", "kind"),
    [
        (0, "signal", "imba_cross_short", "signal"),
        (1, "position_after", -1, "state"),
        (2, "qty", 3.0, "order"),
        (3, "avg_fill_price", 151.0, "fill"),
    ],
)
def test_field_mismatches_are_classified(index: int, field: str, value: object, kind: str) -> None:
    expected = _rows()
    actual = [dict(row) for row in expected]
    actual[index][field] = value

    report = _compare(expected, actual)

    assert report["gate"]["passed"] is False
    assert report["divergence_counts"][kind] == 1
    divergence = next(item for item in report["divergences"] if item["class"] == kind)
    assert divergence["issue"] == "field_mismatch"


def test_numeric_tolerances_and_divergence_thresholds_are_configurable() -> None:
    expected = _rows()
    actual = [dict(row) for row in expected]
    actual[2]["qty"] = 2.0001
    actual[3]["avg_fill_price"] = 150.0101
    policy = GatePolicy(
        qty_abs_tolerance=0.001,
        price_abs_tolerance=0.001,
        max_divergences={"signal": 0, "state": 0, "order": 0, "fill": 0},
    )
    assert _compare(expected, actual, policy)["gate"]["passed"] is True

    actual = [dict(row) for row in expected]
    actual[0]["signal"] = "different"
    allowed = GatePolicy(max_divergences={"signal": 1, "state": 0, "order": 0, "fill": 0})
    assert _compare(expected, actual, allowed)["gate"]["passed"] is True


def test_missing_and_unexpected_records_have_stable_divergence_ids() -> None:
    expected = _rows()
    actual = [dict(row) for row in expected[1:]]
    actual.append(
        {
            **expected[0],
            "event_id": "sig-extra",
            "match_id": "bar-101:signal",
            "seq": 5,
        }
    )
    first = _compare(expected, actual)
    second = _compare(expected, list(reversed(actual)))

    assert first == second
    assert {item["issue"] for item in first["divergences"]} >= {"missing_actual", "unexpected_actual"}
    assert all(item["divergence_id"].startswith("div_") for item in first["divergences"])


def test_semantic_sequence_mismatch_is_detected_but_line_order_is_ignored() -> None:
    expected = _rows()
    assert _compare(expected, list(reversed(expected)))["gate"]["passed"] is True

    actual = [dict(row) for row in expected]
    actual[1]["seq"], actual[2]["seq"] = actual[2]["seq"], actual[1]["seq"]
    report = _compare(expected, actual)

    assert report["gate"]["passed"] is False
    assert any(item["issue"] == "sequence_mismatch" for item in report["divergences"])


def test_malformed_and_missing_required_evidence_fail_closed(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"replay_kind":"signal"}\nnot-json\n', encoding="utf-8")
    missing = tmp_path / "missing.jsonl"

    report = compare_evidence(
        load_jsonl(malformed, stream="expected"),
        load_jsonl(missing, stream="actual"),
    )

    assert report["gate"]["passed"] is False
    assert report["gate"]["fail_closed"] is True
    assert report["evidence_errors"]
    assert any("missing required kind" in reason for reason in report["gate"]["reasons"])


def test_conflicting_duplicate_source_id_fails_closed() -> None:
    rows = _rows()
    conflict = dict(rows[2])
    conflict["qty"] = 99
    expected = normalize_evidence([*rows, conflict], stream="expected")
    report = compare_evidence(expected, normalize_evidence(rows, stream="actual"))

    assert report["gate"]["fail_closed"] is True
    assert any("conflicting duplicate" in error for error in report["evidence_errors"])


def test_cli_writes_machine_readable_report_and_uses_gate_exit_code(tmp_path: Path) -> None:
    expected_path = tmp_path / "expected.jsonl"
    actual_path = tmp_path / "actual.jsonl"
    report_path = tmp_path / "report.json"
    rendered = "".join(json.dumps(row, sort_keys=True) + "\n" for row in _rows())
    expected_path.write_text(rendered, encoding="utf-8")
    actual_path.write_text(rendered, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = "src" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/shadow_replay.py",
            "--expected",
            str(expected_path),
            "--actual",
            str(actual_path),
            "--report",
            str(report_path),
        ],
        cwd=Path(__file__).parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout_report = json.loads(result.stdout)
    file_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert stdout_report == file_report
    assert file_report["gate"]["passed"] is True
