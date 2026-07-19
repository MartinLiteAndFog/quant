"""Deterministic shadow-execution replay and production parity gate.

The module is deliberately read-only: it consumes JSON-like evidence from the
signal/action/execution/calibration streams and never imports a broker or OMS.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "shadow-replay/v1"
REPLAY_KINDS = ("signal", "state", "order", "fill")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_id(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:32]
    return f"{prefix}_{digest}"


def _first(row: Mapping[str, Any], *keys: str) -> Any:
    payload = row.get("payload_json")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            payload = None
    for source in (row, payload if isinstance(payload, Mapping) else {}):
        for key in keys:
            value = source.get(key)
            if value not in (None, ""):
                return value
    return None


def _float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return None


def _timestamp(value: Any) -> Tuple[str, float]:
    if value in (None, ""):
        raise ValueError("missing timestamp")
    raw = str(value).strip()
    parsed = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        dt = datetime.fromisoformat(parsed)
    except ValueError as exc:
        raise ValueError(f"invalid timestamp {raw!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z"), dt.timestamp() * 1000.0


def _token(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value).strip().lower()


def _infer_kind(row: Mapping[str, Any]) -> str:
    explicit = _token(_first(row, "replay_kind", "kind", "event_kind", "record_kind"))
    aliases = {"decision": "state", "action": "state", "execution": "order"}
    if explicit in aliases:
        explicit = aliases[explicit]
    if explicit in REPLAY_KINDS:
        return str(explicit)
    stage = _token(_first(row, "execution_stage", "stage")) or ""
    status = _token(_first(row, "status")) or ""
    if "fill" in stage or "fill" in status or _first(row, "filled_qty", "avg_fill_price", "filled_ts") is not None:
        return "fill"
    if stage or _first(row, "order_id", "client_oid", "client_id", "requested_qty") is not None:
        return "order"
    if _first(row, "engine_action", "position_before", "position_after", "engine_mode_after") is not None:
        return "state"
    if _first(row, "signal", "signal_kind", "signal_side") is not None:
        return "signal"
    raise ValueError("cannot infer replay kind")


def _match_id(row: Mapping[str, Any], kind: str) -> str:
    explicit = _first(row, "match_id", "replay_match_id", "correlation_id")
    if explicit is not None:
        return str(explicit)
    if kind == "signal":
        value = _first(row, "source_event_id", "event_id", "decision_id")
    elif kind == "state":
        value = _first(row, "source_signal_event_id", "source_event_id", "event_id", "decision_id")
    else:
        value = _first(row, "client_oid", "client_id", "order_id", "source_action_event_id", "telemetry_id")
    if value is None:
        raise ValueError(f"{kind} evidence has no stable match identity")
    return str(value)


def _values(row: Mapping[str, Any], kind: str) -> Dict[str, Any]:
    if kind == "signal":
        values = {
            "signal": _token(_first(row, "signal", "signal_kind")),
            "side": _token(_first(row, "signal_side", "side", "direction")),
            "regime_on": _bool(_first(row, "regime_on")),
            "gate_name": _token(_first(row, "gate_name")),
        }
    elif kind == "state":
        values = {
            "action": _token(_first(row, "engine_action", "action")),
            "position_before": _int(_first(row, "position_before")),
            "position_after": _int(_first(row, "position_after")),
            "qty_before": _float(_first(row, "qty_before")),
            "qty_after": _float(_first(row, "qty_after")),
            "mode_before": _token(_first(row, "engine_mode_before", "state_before")),
            "mode_after": _token(_first(row, "engine_mode_after", "state_after")),
            "reason": _token(_first(row, "reason_code", "exit_reason")),
            "blocked": _bool(_first(row, "blocked")),
        }
    elif kind == "order":
        values = {
            "stage": _token(_first(row, "execution_stage", "stage", "order_type")),
            "side": _token(_first(row, "side", "action_side")),
            "qty": _float(_first(row, "qty", "requested_qty")),
            "price": _float(_first(row, "price", "limit_price", "stop_price")),
            "reduce_only": _bool(_first(row, "reduce_only")),
            "status": _token(_first(row, "status")),
            "reject_reason": _token(_first(row, "reject_reason")),
        }
    else:
        values = {
            "side": _token(_first(row, "side", "action_side")),
            "qty": _float(_first(row, "filled_qty", "qty")),
            "price": _float(_first(row, "avg_fill_price", "fill_price", "price")),
            "fee": _float(_first(row, "fee", "fee_amount")),
            "fee_bps": _float(_first(row, "fee_bps")),
            "liquidity": _token(_first(row, "liquidity")),
            "status": _token(_first(row, "status")),
        }
    return {key: value for key, value in values.items() if value is not None}


def _validate_semantics(kind: str, values: Mapping[str, Any]) -> None:
    required = {
        "signal": ("signal",),
        "state": ("action",),
        "order": ("side", "qty"),
        "fill": ("side", "qty", "price"),
    }[kind]
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"{kind} evidence missing required fields: {', '.join(missing)}")
    for name in ("qty", "qty_before", "qty_after", "price", "fee"):
        if name in values and float(values[name]) < 0:
            raise ValueError(f"{kind} evidence has negative {name}")


@dataclass(frozen=True)
class ReplayRecord:
    kind: str
    match_id: str
    record_id: str
    source_id: str
    ts: str
    ts_ms: float
    seq: Optional[int]
    venue: str
    symbol: str
    values: Dict[str, Any]

    def public_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row.pop("ts_ms", None)
        return row


@dataclass(frozen=True)
class EvidenceBatch:
    stream: str
    records: Tuple[ReplayRecord, ...] = ()
    errors: Tuple[str, ...] = ()


def normalize_record(row: Mapping[str, Any], *, stream: str, line_no: int) -> ReplayRecord:
    """Normalize one existing signal/action/execution/calibration row."""
    if not isinstance(row, Mapping):
        raise ValueError("row is not an object")
    kind = _infer_kind(row)
    match_id = _match_id(row, kind)
    if kind == "fill":
        timestamp_value = _first(row, "ts", "filled_ts", "result_ts", "submitted_ts", "decision_ts")
    elif kind == "order":
        timestamp_value = _first(row, "ts", "submitted_ts", "acknowledged_ts", "result_ts", "decision_ts")
    else:
        timestamp_value = _first(row, "ts", "decision_ts", "result_ts")
    ts, ts_ms = _timestamp(timestamp_value)
    venue = str(_first(row, "venue") or "").strip().upper()
    symbol = str(_first(row, "symbol") or "").strip().upper()
    if not venue or not symbol:
        raise ValueError("missing venue or symbol")
    values = _values(row, kind)
    _validate_semantics(kind, values)
    source = _first(row, "event_id", "telemetry_id", "decision_id")
    source_id = str(source) if source is not None else _stable_id("src", row)
    identity = {"kind": kind, "source_id": source_id}
    return ReplayRecord(
        kind=kind,
        match_id=match_id,
        record_id=_stable_id("replay", identity),
        source_id=source_id,
        ts=ts,
        ts_ms=ts_ms,
        seq=_int(_first(row, "seq")),
        venue=venue,
        symbol=symbol,
        values=values,
    )


def _normalize_indexed(
    rows: Iterable[Tuple[int, Mapping[str, Any]]],
    *,
    stream: str,
    initial_errors: Iterable[str] = (),
) -> EvidenceBatch:
    records: Dict[str, ReplayRecord] = {}
    errors = list(initial_errors)
    for line_no, row in rows:
        try:
            record = normalize_record(row, stream=stream, line_no=line_no)
        except (TypeError, ValueError) as exc:
            errors.append(f"{stream}:{line_no}: {exc}")
            continue
        previous = records.get(record.record_id)
        if previous is not None and previous != record:
            errors.append(f"{stream}:{line_no}: conflicting duplicate source id {record.source_id!r}")
            continue
        records[record.record_id] = record
    ordered = tuple(sorted(records.values(), key=_record_sort_key))
    return EvidenceBatch(stream=stream, records=ordered, errors=tuple(sorted(errors)))


def normalize_evidence(rows: Iterable[Mapping[str, Any]], *, stream: str) -> EvidenceBatch:
    """Normalize and idempotently deduplicate an in-memory evidence stream."""

    return _normalize_indexed(enumerate(rows, start=1), stream=stream)


def load_jsonl(path: Path, *, stream: str) -> EvidenceBatch:
    """Load JSONL without raising; evidence problems are returned fail-closed."""

    if not path.is_file():
        return EvidenceBatch(stream=stream, errors=(f"{stream}: evidence file not found: {path}",))
    rows = []
    errors = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"{stream}:{line_no}: invalid JSON: {exc.msg}")
                continue
            if not isinstance(value, Mapping):
                errors.append(f"{stream}:{line_no}: row is not an object")
                continue
            rows.append((line_no, value))
    return _normalize_indexed(rows, stream=stream, initial_errors=errors)


def _record_sort_key(record: ReplayRecord) -> Tuple[Any, ...]:
    return (
        record.seq is None,
        record.seq if record.seq is not None else 0,
        record.ts_ms,
        record.kind,
        record.match_id,
        record.record_id,
    )


@dataclass(frozen=True)
class GatePolicy:
    max_divergences: Mapping[str, int] = field(
        default_factory=lambda: {kind: 0 for kind in REPLAY_KINDS}
    )
    required_kinds: Tuple[str, ...] = REPLAY_KINDS
    qty_abs_tolerance: float = 1e-9
    qty_rel_tolerance: float = 1e-6
    price_abs_tolerance: float = 1e-9
    price_bps_tolerance: float = 0.01
    timestamp_tolerance_ms: float = 1_000.0

    def __post_init__(self) -> None:
        unknown = set(self.required_kinds) - set(REPLAY_KINDS)
        if unknown:
            raise ValueError(f"unknown required replay kinds: {sorted(unknown)}")
        if any(int(self.max_divergences.get(kind, 0)) < 0 for kind in REPLAY_KINDS):
            raise ValueError("divergence thresholds cannot be negative")
        tolerances = (
            self.qty_abs_tolerance,
            self.qty_rel_tolerance,
            self.price_abs_tolerance,
            self.price_bps_tolerance,
            self.timestamp_tolerance_ms,
        )
        if any(value < 0 or not math.isfinite(value) for value in tolerances):
            raise ValueError("tolerances must be finite and non-negative")


def _pair_records(records: Sequence[ReplayRecord]) -> Dict[Tuple[str, str, int], ReplayRecord]:
    counters: Dict[Tuple[str, str], int] = {}
    paired: Dict[Tuple[str, str, int], ReplayRecord] = {}
    for record in sorted(records, key=_record_sort_key):
        base = (record.kind, record.match_id)
        occurrence = counters.get(base, 0)
        counters[base] = occurrence + 1
        paired[(record.kind, record.match_id, occurrence)] = record
    return paired


def _numeric_equal(field_name: str, expected: float, actual: float, policy: GatePolicy) -> bool:
    delta = abs(actual - expected)
    if field_name in {"qty", "qty_before", "qty_after"}:
        return delta <= max(policy.qty_abs_tolerance, abs(expected) * policy.qty_rel_tolerance)
    if field_name == "price":
        relative = abs(expected) * policy.price_bps_tolerance / 10_000.0
        return delta <= max(policy.price_abs_tolerance, relative)
    return delta <= 1e-12


def _mismatch_fields(expected: ReplayRecord, actual: ReplayRecord, policy: GatePolicy) -> Tuple[str, ...]:
    mismatches = []
    if expected.venue != actual.venue:
        mismatches.append("venue")
    if expected.symbol != actual.symbol:
        mismatches.append("symbol")
    if abs(expected.ts_ms - actual.ts_ms) > policy.timestamp_tolerance_ms:
        mismatches.append("timestamp")
    for field_name, expected_value in sorted(expected.values.items()):
        if field_name not in actual.values:
            mismatches.append(field_name)
            continue
        actual_value = actual.values[field_name]
        if isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
            if not isinstance(actual_value, (int, float)) or isinstance(actual_value, bool):
                mismatches.append(field_name)
            elif not _numeric_equal(field_name, float(expected_value), float(actual_value), policy):
                mismatches.append(field_name)
        elif expected_value != actual_value:
            mismatches.append(field_name)
    return tuple(mismatches)


def _divergence(
    kind: str,
    issue: str,
    key: Tuple[str, str, int],
    *,
    fields: Sequence[str] = (),
    expected: Optional[ReplayRecord] = None,
    actual: Optional[ReplayRecord] = None,
) -> Dict[str, Any]:
    identity = {
        "kind": kind,
        "issue": issue,
        "match_id": key[1],
        "occurrence": key[2],
        "fields": sorted(fields),
    }
    return {
        "divergence_id": _stable_id("div", identity),
        "class": kind,
        "issue": issue,
        "match_id": key[1],
        "occurrence": key[2],
        "fields": sorted(fields),
        "expected_record_id": expected.record_id if expected else None,
        "actual_record_id": actual.record_id if actual else None,
    }


def compare_evidence(expected: EvidenceBatch, actual: EvidenceBatch, policy: Optional[GatePolicy] = None) -> Dict[str, Any]:
    """Compare expected/actual streams and return a deterministic gate report."""

    policy = policy or GatePolicy()
    expected_map = _pair_records(expected.records)
    actual_map = _pair_records(actual.records)
    divergences = []
    matched = 0
    for key in sorted(set(expected_map) | set(actual_map)):
        expected_record = expected_map.get(key)
        actual_record = actual_map.get(key)
        kind = key[0]
        if expected_record is None:
            divergences.append(_divergence(kind, "unexpected_actual", key, actual=actual_record))
        elif actual_record is None:
            divergences.append(_divergence(kind, "missing_actual", key, expected=expected_record))
        else:
            matched += 1
            fields = _mismatch_fields(expected_record, actual_record, policy)
            if fields:
                divergences.append(
                    _divergence(kind, "field_mismatch", key, fields=fields, expected=expected_record, actual=actual_record)
                )

    # Sequence is semantic evidence. File-line ordering is ignored; seq/timestamp
    # determines the order, making shuffled and duplicate JSONL input idempotent.
    common = set(expected_map) & set(actual_map)
    expected_order = [key for key, _ in sorted(expected_map.items(), key=lambda item: _record_sort_key(item[1])) if key in common]
    actual_order = [key for key, _ in sorted(actual_map.items(), key=lambda item: _record_sort_key(item[1])) if key in common]
    actual_positions = {key: index for index, key in enumerate(actual_order)}
    for expected_index, key in enumerate(expected_order):
        if actual_positions.get(key) != expected_index:
            divergences.append(
                _divergence(key[0], "sequence_mismatch", key, fields=("sequence",), expected=expected_map[key], actual=actual_map[key])
            )

    by_identity = {item["divergence_id"]: item for item in divergences}
    divergences = [by_identity[key] for key in sorted(by_identity)]
    counts = {kind: sum(1 for item in divergences if item["class"] == kind) for kind in REPLAY_KINDS}
    errors = tuple(sorted((*expected.errors, *actual.errors)))
    expected_kinds = {record.kind for record in expected.records}
    actual_kinds = {record.kind for record in actual.records}
    fail_closed_reasons = list(errors)
    if not expected.records:
        fail_closed_reasons.append("expected evidence is empty")
    if not actual.records:
        fail_closed_reasons.append("actual evidence is empty")
    for kind in policy.required_kinds:
        if kind not in expected_kinds:
            fail_closed_reasons.append(f"expected evidence missing required kind: {kind}")
        if kind not in actual_kinds:
            fail_closed_reasons.append(f"actual evidence missing required kind: {kind}")
    fail_closed_reasons = sorted(set(fail_closed_reasons))
    threshold_reasons = [
        f"{kind} divergences {counts[kind]} exceed threshold {int(policy.max_divergences.get(kind, 0))}"
        for kind in REPLAY_KINDS
        if counts[kind] > int(policy.max_divergences.get(kind, 0))
    ]
    gate_passed = not fail_closed_reasons and not threshold_reasons
    policy_dict = {
        "max_divergences": {kind: int(policy.max_divergences.get(kind, 0)) for kind in REPLAY_KINDS},
        "required_kinds": list(policy.required_kinds),
        "qty_abs_tolerance": policy.qty_abs_tolerance,
        "qty_rel_tolerance": policy.qty_rel_tolerance,
        "price_abs_tolerance": policy.price_abs_tolerance,
        "price_bps_tolerance": policy.price_bps_tolerance,
        "timestamp_tolerance_ms": policy.timestamp_tolerance_ms,
    }
    evidence = {
        "expected_digest": _stable_id("evidence", [record.public_dict() for record in expected.records]),
        "actual_digest": _stable_id("evidence", [record.public_dict() for record in actual.records]),
        "expected_kinds": sorted(expected_kinds),
        "actual_kinds": sorted(actual_kinds),
    }
    report_body = {
        "schema_version": SCHEMA_VERSION,
        "evidence": evidence,
        "expected_count": len(expected.records),
        "actual_count": len(actual.records),
        "matched_count": matched,
        "divergence_counts": counts,
        "divergences": divergences,
        "evidence_errors": list(errors),
        "gate": {
            "passed": gate_passed,
            "fail_closed": bool(fail_closed_reasons),
            "reasons": [*fail_closed_reasons, *threshold_reasons],
            "policy": policy_dict,
        },
    }
    return {"report_id": _stable_id("shadow", report_body), **report_body}
