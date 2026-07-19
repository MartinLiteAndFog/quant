from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, TypeVar


T = TypeVar("T")


CALIBRATION_COLUMNS: Tuple[str, ...] = (
    "telemetry_id",
    "decision_ts",
    "submitted_ts",
    "acknowledged_ts",
    "filled_ts",
    "result_ts",
    "venue",
    "strategy",
    "symbol",
    "action",
    "exit_reason",
    "side",
    "reference_bid",
    "reference_ask",
    "reference_mid",
    "requested_qty",
    "filled_qty",
    "avg_fill_price",
    "order_id",
    "client_oid",
    "order_type",
    "liquidity",
    "fee",
    "fee_currency",
    "fee_bps",
    "requotes",
    "fallback_used",
    "rejected",
    "reject_reason",
    "reduce_only",
    "status",
    "submit_to_ack_ms",
    "submit_to_fill_ms",
    "decision_to_result_ms",
    "slippage_bps",
    "timing_precision",
    "filled_qty_inferred",
    "fill_price_source",
    "fee_source",
)


@dataclass(frozen=True)
class ExecutionDecision:
    decision_ts: str
    venue: str
    strategy: str
    symbol: str
    action: str
    exit_reason: Optional[str]
    side: Optional[str]
    requested_qty: float
    reference_bid: Optional[float] = None
    reference_ask: Optional[float] = None
    reference_mid: Optional[float] = None
    reduce_only: Optional[bool] = None


@dataclass(frozen=True)
class CapturedExecution:
    result: Any
    record: Dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _result_parts(result: Any) -> tuple[bool, str, Mapping[str, Any]]:
    if isinstance(result, Mapping):
        ok = bool(result.get("ok", False))
        mode = str(result.get("mode") or "")
        details = result.get("details")
    else:
        ok = bool(getattr(result, "ok", False))
        mode = str(getattr(result, "mode", "") or "")
        details = getattr(result, "details", None)
    return ok, mode, details if isinstance(details, Mapping) else {}


def _parse_ts(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        raw = str(value).strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(value: Any) -> Optional[str]:
    dt = _parse_ts(value)
    if dt is None:
        return None
    return dt.isoformat().replace("+00:00", "Z")


def _elapsed_ms(start: Any, end: Any) -> Optional[float]:
    start_dt = _parse_ts(start)
    end_dt = _parse_ts(end)
    if start_dt is None or end_dt is None:
        return None
    return max(0.0, (end_dt - start_dt).total_seconds() * 1000.0)


def _liquidity(mode: str, details: Mapping[str, Any]) -> Optional[str]:
    explicit = str(_first(details, ("liquidity", "liquidity_type", "maker_taker")) or "").lower()
    if explicit in ("maker", "taker"):
        return explicit
    mode_u = mode.upper()
    if mode_u in ("L1", "L2", "PO") or "MAKER" in mode_u:
        return "maker"
    if any(token in mode_u for token in ("MKT", "MARKET", "FALLBACK", "FB_", "SL_")):
        return "taker"
    return None


def _order_type(mode: str, details: Mapping[str, Any]) -> Optional[str]:
    explicit = _first(details, ("order_type", "orderType", "kind"))
    if explicit not in (None, ""):
        return str(explicit).lower()
    mode_u = mode.upper()
    if mode_u in ("L1", "L2", "PO"):
        return "limit"
    if "MKT" in mode_u or "MARKET" in mode_u:
        return "market"
    return None


def _slippage_bps(side: Optional[str], reference_mid: Optional[float], avg_fill_price: Optional[float]) -> Optional[float]:
    if reference_mid is None or reference_mid <= 0 or avg_fill_price is None or avg_fill_price <= 0:
        return None
    raw = (avg_fill_price / reference_mid - 1.0) * 10_000.0
    if str(side or "").lower() == "sell":
        raw = -raw
    return raw


def _telemetry_id(record: Mapping[str, Any]) -> str:
    stable = "|".join(
        str(record.get(key) or "")
        for key in (
            "venue",
            "strategy",
            "symbol",
            "decision_ts",
            "submitted_ts",
            "action",
            "order_id",
            "client_oid",
        )
    )
    return "cal_" + hashlib.sha256(stable.encode("utf-8")).hexdigest()[:32]


def build_calibration_record(
    *,
    decision: ExecutionDecision,
    result: Any,
    submitted_ts: str,
    result_ts: str,
    exception_type: Optional[str] = None,
) -> Dict[str, Any]:
    ok, mode, details = _result_parts(result)
    rejected = bool(exception_type) or not ok

    acknowledged_ts = _iso(
        _first(details, ("acknowledged_ts", "ack_ts", "acknowledged_at", "submit_ack_ts"))
    )
    filled_ts = _iso(_first(details, ("filled_ts", "fill_ts", "filled_at", "last_fill_ts")))
    timing_precision = "exchange"
    if acknowledged_ts is None and _first(details, ("order_id", "orderId")) not in (None, ""):
        acknowledged_ts = _iso(result_ts)
        timing_precision = "oms_return_boundary"
    if filled_ts is None and ok:
        filled_ts = _iso(result_ts)
        timing_precision = "oms_return_boundary"
    if acknowledged_ts is None and filled_ts is None:
        timing_precision = "executor_boundary"

    requested_qty = max(0.0, float(decision.requested_qty))
    filled_qty = _float(_first(details, ("filled_qty", "executed_qty", "cum_qty", "cumQty", "filled")))
    filled_qty_inferred = False
    if filled_qty is None and ok:
        filled_qty = _float(_first(details, ("qty", "size")))
        if filled_qty is None:
            filled_qty = requested_qty
        filled_qty_inferred = True

    avg_fill_price = _float(
        _first(details, ("avg_fill_price", "fill_price", "average_price", "avgPrice", "price"))
    )
    fill_price_source = "venue" if avg_fill_price is not None else None

    fee = _float(_first(details, ("fee", "fee_amount", "commission", "fill_fee")))
    fee_bps = _float(_first(details, ("fee_bps", "commission_bps")))
    fee_currency_raw = _first(details, ("fee_currency", "commission_asset", "feeCurrency"))
    fee_source = "venue" if fee is not None or fee_bps is not None else None

    requotes = _int(_first(details, ("requotes", "reprice_count", "retry_count")))
    mode_u = mode.upper()
    fallback_used = bool(
        _first(details, ("fallback_used",))
        or any(token in mode_u for token in ("FB_", "FALLBACK", "_MKT", "MARKET"))
    )
    reject_reason_raw = _first(details, ("reject_reason", "error_code", "reason"))
    if exception_type:
        reject_reason = f"order_call_failed:{exception_type}"
    elif rejected:
        reject_reason = str(reject_reason_raw or mode or "oms_rejected")[:160]
    else:
        reject_reason = None

    side = str(_first(details, ("side",)) or decision.side or "").lower() or None
    record: Dict[str, Any] = {
        "telemetry_id": None,
        "decision_ts": _iso(decision.decision_ts),
        "submitted_ts": _iso(submitted_ts),
        "acknowledged_ts": acknowledged_ts,
        "filled_ts": filled_ts,
        "result_ts": _iso(result_ts),
        "venue": str(decision.venue),
        "strategy": str(decision.strategy),
        "symbol": str(decision.symbol),
        "action": str(decision.action),
        "exit_reason": str(decision.exit_reason) if decision.exit_reason else None,
        "side": side,
        "reference_bid": _float(decision.reference_bid),
        "reference_ask": _float(decision.reference_ask),
        "reference_mid": _float(decision.reference_mid),
        "requested_qty": requested_qty,
        "filled_qty": filled_qty,
        "avg_fill_price": avg_fill_price,
        "order_id": str(_first(details, ("order_id", "orderId")) or "") or None,
        "client_oid": str(_first(details, ("client_id", "client_oid", "cliOrdId")) or "") or None,
        "order_type": _order_type(mode, details),
        "liquidity": _liquidity(mode, details),
        "fee": fee,
        "fee_currency": str(fee_currency_raw) if fee_currency_raw not in (None, "") else None,
        "fee_bps": fee_bps,
        "requotes": requotes,
        "fallback_used": fallback_used,
        "rejected": rejected,
        "reject_reason": reject_reason,
        "reduce_only": bool(_first(details, ("reduce_only",))) if _first(details, ("reduce_only",)) is not None else decision.reduce_only,
        "status": "rejected" if rejected else "filled" if filled_ts else "acknowledged",
        "submit_to_ack_ms": _elapsed_ms(submitted_ts, acknowledged_ts),
        "submit_to_fill_ms": _elapsed_ms(submitted_ts, filled_ts),
        "decision_to_result_ms": _elapsed_ms(decision.decision_ts, result_ts),
        "slippage_bps": _slippage_bps(side, _float(decision.reference_mid), avg_fill_price),
        "timing_precision": timing_precision,
        "filled_qty_inferred": filled_qty_inferred,
        "fill_price_source": fill_price_source,
        "fee_source": fee_source,
    }
    record["telemetry_id"] = _telemetry_id(record)
    return {key: record.get(key) for key in CALIBRATION_COLUMNS}


def observe_oms_call(
    call: Callable[[], T],
    *,
    decision: ExecutionDecision,
    clock: Callable[[], str] = utc_now_iso,
    sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    result_selector: Optional[Callable[[T], Any]] = None,
) -> CapturedExecution:
    submitted_ts = clock()
    try:
        result = call()
    except Exception as exc:
        result_ts = clock()
        record = build_calibration_record(
            decision=decision,
            result=None,
            submitted_ts=submitted_ts,
            result_ts=result_ts,
            exception_type=type(exc).__name__,
        )
        if sink is not None:
            sink(record)
        raise

    result_ts = clock()
    observed_result = result_selector(result) if result_selector is not None else result
    record = build_calibration_record(
        decision=decision,
        result=observed_result,
        submitted_ts=submitted_ts,
        result_ts=result_ts,
    )
    if sink is not None:
        sink(record)
    return CapturedExecution(result=result, record=record)


def normalize_calibration_records(records: Iterable[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    rows = [{key: row.get(key) for key in CALIBRATION_COLUMNS} for row in records]
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("decision_ts") or ""),
            str(row.get("submitted_ts") or ""),
            str(row.get("telemetry_id") or ""),
        ),
    )


def _percentile(values: Iterable[Any], q: float) -> Optional[float]:
    clean = sorted(value for value in (_float(v) for v in values) if value is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    return clean[lo] + (clean[hi] - clean[lo]) * (pos - lo)


def _mean(values: Iterable[Any]) -> Optional[float]:
    clean = [value for value in (_float(v) for v in values) if value is not None]
    return sum(clean) / len(clean) if clean else None


def aggregate_calibration_records(records: Iterable[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    groups: Dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in normalize_calibration_records(records):
        key = tuple(str(row.get(name) or "unknown") for name in ("venue", "symbol", "action", "order_type", "liquidity"))
        groups.setdefault(key, []).append(row)

    out: list[Dict[str, Any]] = []
    for key in sorted(groups):
        rows = groups[key]
        n = len(rows)
        filled = sum(1 for row in rows if str(row.get("status")) == "filled")
        rejected = sum(1 for row in rows if bool(row.get("rejected")))
        fallback = sum(1 for row in rows if bool(row.get("fallback_used")))
        out.append(
            {
                "venue": key[0],
                "symbol": key[1],
                "action": key[2],
                "order_type": key[3],
                "liquidity": key[4],
                "attempts": n,
                "fill_rate": filled / n,
                "reject_rate": rejected / n,
                "fallback_rate": fallback / n,
                "requotes_mean": _mean(row.get("requotes") for row in rows),
                "submit_to_fill_ms_p50": _percentile((row.get("submit_to_fill_ms") for row in rows), 0.50),
                "submit_to_fill_ms_p95": _percentile((row.get("submit_to_fill_ms") for row in rows), 0.95),
                "slippage_bps_p50": _percentile((row.get("slippage_bps") for row in rows), 0.50),
                "slippage_bps_p95": _percentile((row.get("slippage_bps") for row in rows), 0.95),
                "fee_bps_mean": _mean(row.get("fee_bps") for row in rows),
            }
        )
    return out


def load_calibration_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"calibration row {line_no} is not an object")
            rows.append(value)
    return normalize_calibration_records(rows)


def export_calibration_csv(
    records: Iterable[Mapping[str, Any]],
    path: Path,
    *,
    aggregate: bool = False,
) -> int:
    rows = aggregate_calibration_records(records) if aggregate else normalize_calibration_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if aggregate and rows else list(CALIBRATION_COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
