"""Immutable forward protocol and fail-closed checkpoint scoring.

This module deliberately has no database, network, broker, or order dependency.  It
turns a frozen protocol plus an exported observation ledger into a deterministic
report.  A pass is only eligibility for shadow-champion review, never permission to
trade live.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL = Path(__file__).with_name("forward_protocol.json")
UTC = timezone.utc
T_CRITICAL_95_TWO_SIDED = (
    math.inf,
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


def _utc(value: Any, *, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True)
class ForwardProtocol:
    schema_version: int
    protocol_id: str
    candidate_id: str
    symbol: str
    source: str
    artifact_sha256: str
    candidate_spec_sha256: str
    observer_code_sha256: Mapping[str, str]
    warmup_start: datetime
    evidence_start: datetime
    evidence_end: datetime
    checkpoint_at: datetime
    outcome_maturity_minutes: int
    base_cost_bps: float
    stress_cost_bps: float
    minimum_formal_trades: int
    minimum_bar_coverage: float
    maximum_drawdown_bps: float
    minimum_mean_net_bps: float
    minimum_lcb95_net_bps: float
    minimum_stress_mean_net_bps: float
    promotion_scope: str
    live_orders_permitted: bool
    raw: Mapping[str, Any]
    protocol_sha256: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ForwardProtocol":
        raw = dict(data)
        schema_version = int(raw.get("schema_version", 0))
        if schema_version not in {1, 2}:
            raise ValueError("unsupported forward protocol schema")
        warmup_start = _utc(raw["warmup_start"], field="warmup_start")
        evidence_start = _utc(raw["evidence_start"], field="evidence_start")
        evidence_end = _utc(raw["evidence_end"], field="evidence_end")
        checkpoint_at = _utc(raw["checkpoint_at"], field="checkpoint_at")
        outcome_maturity_minutes = int(raw["outcome_maturity_minutes"])
        if outcome_maturity_minutes < 1:
            raise ValueError("outcome_maturity_minutes must be positive")
        if not warmup_start < evidence_start < evidence_end:
            raise ValueError("protocol timestamps must satisfy warmup < evidence start < evidence end")
        if checkpoint_at < evidence_end + timedelta(minutes=outcome_maturity_minutes):
            raise ValueError("checkpoint must allow the last formal outcome to mature")
        if bool(raw.get("live_orders_permitted", True)):
            raise ValueError("brain-forward protocol must forbid live orders")
        artifact_sha256 = str(raw["artifact_sha256"]).lower()
        candidate_spec_sha256 = str(raw["candidate_spec_sha256"]).lower()
        for name, digest in (("artifact_sha256", artifact_sha256), ("candidate_spec_sha256", candidate_spec_sha256)):
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"{name} must be a SHA-256 digest")
        observer_code_sha256 = {
            str(name): str(digest).lower()
            for name, digest in dict(raw["observer_code_sha256"]).items()
        }
        required_code = {"runtime.py", "service.py", "store.py"}
        if schema_version == 2:
            required_code |= {
                "service_v2.py",
                "evidence.py",
                "variants.py",
                "variant_store.py",
            }
        if set(observer_code_sha256) != required_code:
            raise ValueError(f"observer_code_sha256 must contain exactly {sorted(required_code)}")
        for name, digest in observer_code_sha256.items():
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"observer_code_sha256[{name!r}] must be a SHA-256 digest")
        minimum_bar_coverage = float(raw["minimum_bar_coverage"])
        if not 0.0 < minimum_bar_coverage <= 1.0:
            raise ValueError("minimum_bar_coverage must be in (0, 1]")
        minimum_formal_trades = int(raw["minimum_formal_trades"])
        if minimum_formal_trades < 2:
            raise ValueError("minimum_formal_trades must be at least two")
        base_cost = float(raw["base_cost_bps"])
        stress_cost = float(raw["stress_cost_bps"])
        if base_cost < 0.0 or stress_cost < base_cost:
            raise ValueError("stress cost must be at least the non-negative base cost")
        return cls(
            schema_version=schema_version,
            protocol_id=str(raw["protocol_id"]),
            candidate_id=str(raw["candidate_id"]),
            symbol=str(raw["symbol"]),
            source=str(raw["source"]),
            artifact_sha256=artifact_sha256,
            candidate_spec_sha256=candidate_spec_sha256,
            observer_code_sha256=observer_code_sha256,
            warmup_start=warmup_start,
            evidence_start=evidence_start,
            evidence_end=evidence_end,
            checkpoint_at=checkpoint_at,
            outcome_maturity_minutes=outcome_maturity_minutes,
            base_cost_bps=base_cost,
            stress_cost_bps=stress_cost,
            minimum_formal_trades=minimum_formal_trades,
            minimum_bar_coverage=minimum_bar_coverage,
            maximum_drawdown_bps=float(raw["maximum_drawdown_bps"]),
            minimum_mean_net_bps=float(raw["minimum_mean_net_bps"]),
            minimum_lcb95_net_bps=float(raw["minimum_lcb95_net_bps"]),
            minimum_stress_mean_net_bps=float(raw["minimum_stress_mean_net_bps"]),
            promotion_scope=str(raw["promotion_scope"]),
            live_orders_permitted=False,
            raw=raw,
            protocol_sha256=hashlib.sha256(_canonical_json(raw)).hexdigest(),
        )

    @classmethod
    def load(cls, path: Path = PROTOCOL) -> "ForwardProtocol":
        protocol_path = path
        source_path = Path("/app/src/quant/brain_forward/forward_protocol.json")
        if not protocol_path.exists() and source_path.exists():
            protocol_path = source_path
        return cls.from_dict(json.loads(protocol_path.read_text(encoding="utf-8")))

    def assert_runtime(self, *, symbol: str, source: str, artifact_sha256: str) -> None:
        mismatches = []
        if symbol != self.symbol:
            mismatches.append(f"symbol={symbol!r}")
        if source != self.source:
            mismatches.append(f"source={source!r}")
        if artifact_sha256.lower() != self.artifact_sha256:
            mismatches.append("artifact_sha256")
        package_dir = Path(__file__).parent
        source_dir = Path("/app/src/quant/brain_forward")
        for name, expected in self.observer_code_sha256.items():
            code_path = package_dir / name
            if not code_path.exists() and (source_dir / name).exists():
                code_path = source_dir / name
            actual = hashlib.sha256(code_path.read_bytes()).hexdigest() if code_path.exists() else "missing"
            if actual != expected:
                mismatches.append(f"observer_code_sha256:{name}")
        if mismatches:
            raise RuntimeError("forward protocol/runtime mismatch: " + ", ".join(mismatches))

    def phase_at(self, value: Any) -> str:
        timestamp = _utc(value, field="event_ts")
        if timestamp < self.warmup_start:
            return "pre_protocol"
        if timestamp < self.evidence_start:
            return "warmup"
        if timestamp < self.evidence_end:
            return "formal"
        return "post_epoch"

    def accepts_observation(self, value: Any) -> bool:
        return self.phase_at(value) in {"warmup", "formal"}


def _t_lcb95(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    degrees = len(values) - 1
    critical = T_CRITICAL_95_TWO_SIDED[degrees] if degrees < len(T_CRITICAL_95_TWO_SIDED) else 1.96
    return mean - critical * std / math.sqrt(len(values))


def _max_drawdown_bps(values: Sequence[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    maximum = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        maximum = max(maximum, peak - cumulative)
    return maximum


def _coverage(protocol: ForwardProtocol, timestamps: Iterable[Any]) -> tuple[int, int, float]:
    coverage_end = protocol.evidence_end + timedelta(minutes=protocol.outcome_maturity_minutes)
    expected = int((coverage_end - protocol.evidence_start).total_seconds() // 60)
    observed = {
        _utc(value, field="bar timestamp").replace(second=0, microsecond=0)
        for value in timestamps
        if protocol.evidence_start <= _utc(value, field="bar timestamp") < coverage_end
    }
    count = len(observed)
    return count, expected, count / expected if expected else 0.0


def build_checkpoint_report(
    protocol: ForwardProtocol,
    decisions: Iterable[Mapping[str, Any]],
    trades: Iterable[Mapping[str, Any]],
    bar_timestamps: Iterable[Any],
    *,
    as_of: Any,
) -> dict[str, Any]:
    """Score one frozen candidate on its locked formal epoch.

    Every malformed or contradictory record becomes a failed gate.  The
    function never grants live-trading permission.
    """

    evaluated_at = _utc(as_of, field="as_of")
    decision_rows = [dict(row) for row in decisions]
    rows = [dict(row) for row in trades]
    failures: list[str] = []
    formal_decision_ids: set[str] = set()
    for position, row in enumerate(decision_rows):
        try:
            decision_id = str(row["decision_id"])
            event_ts = _utc(row["event_ts"], field="event_ts")
            if decision_id in formal_decision_ids:
                failures.append(f"duplicate formal decision_id at row {position}")
            formal_decision_ids.add(decision_id)
            if row.get("protocol_id") != protocol.protocol_id or row.get("protocol_sha256") != protocol.protocol_sha256:
                failures.append(f"decision protocol identity mismatch at row {position}")
            if str(row.get("artifact_sha256", "")).lower() != protocol.artifact_sha256:
                failures.append(f"decision artifact identity mismatch at row {position}")
            if row.get("symbol") != protocol.symbol or row.get("source") != protocol.source:
                failures.append(f"decision market source mismatch at row {position}")
            if row.get("evidence_phase") != "formal" or protocol.phase_at(event_ts) != "formal":
                failures.append(f"non-formal decision at row {position}")
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"malformed decision at row {position}: {exc}")
    normalized: list[tuple[datetime, datetime, datetime, float, float]] = []
    seen_ids: set[str] = set()
    for position, row in enumerate(rows):
        try:
            decision_id = str(row["decision_id"])
            event_ts = _utc(row["event_ts"], field="event_ts")
            entry_ts = _utc(row["entry_ts"], field="entry_ts")
            exit_ts = _utc(row["exit_ts"], field="exit_ts")
            gross_bps = float(row["gross_bps"])
            net_bps = float(row["net_bps"])
            if decision_id in seen_ids:
                failures.append(f"duplicate decision_id at row {position}")
            seen_ids.add(decision_id)
            if decision_id not in formal_decision_ids:
                failures.append(f"trade without formal decision at row {position}")
            if row.get("protocol_id") != protocol.protocol_id or row.get("protocol_sha256") != protocol.protocol_sha256:
                failures.append(f"protocol identity mismatch at row {position}")
            if str(row.get("artifact_sha256", "")).lower() != protocol.artifact_sha256:
                failures.append(f"artifact identity mismatch at row {position}")
            if row.get("symbol") != protocol.symbol or row.get("source") != protocol.source:
                failures.append(f"market source mismatch at row {position}")
            if row.get("evidence_phase") != "formal" or protocol.phase_at(event_ts) != "formal":
                failures.append(f"non-formal trade at row {position}")
            if entry_ts != event_ts + timedelta(minutes=1):
                failures.append(f"entry is not the next minute at row {position}")
            if not entry_ts <= exit_ts <= event_ts + timedelta(minutes=5):
                failures.append(f"exit horizon mismatch at row {position}")
            if not math.isclose(net_bps, gross_bps - protocol.base_cost_bps, abs_tol=1e-9):
                failures.append(f"base cost mismatch at row {position}")
            normalized.append((event_ts, entry_ts, exit_ts, gross_bps, net_bps))
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"malformed trade at row {position}: {exc}")
    normalized.sort(key=lambda item: (item[1], item[2]))
    for previous, current in zip(normalized, normalized[1:]):
        if current[1] <= previous[2]:
            failures.append("overlapping formal trades")
            break

    net_values = [item[4] for item in normalized]
    gross_values = [item[3] for item in normalized]
    mean_net = statistics.fmean(net_values) if net_values else None
    lcb95 = _t_lcb95(net_values)
    stress_values = [gross - protocol.stress_cost_bps for gross in gross_values]
    stress_mean = statistics.fmean(stress_values) if stress_values else None
    drawdown = _max_drawdown_bps(net_values)
    observed_minutes, expected_minutes, coverage = _coverage(protocol, bar_timestamps)

    gate_values = {
        "checkpoint_open": evaluated_at >= protocol.checkpoint_at,
        "formal_trade_count": len(normalized) >= protocol.minimum_formal_trades,
        "bar_coverage": coverage >= protocol.minimum_bar_coverage,
        "mean_net_positive": mean_net is not None and mean_net > protocol.minimum_mean_net_bps,
        "lcb95_net_positive": lcb95 is not None and lcb95 > protocol.minimum_lcb95_net_bps,
        "stress_mean_positive": stress_mean is not None and stress_mean > protocol.minimum_stress_mean_net_bps,
        "drawdown_within_limit": drawdown <= protocol.maximum_drawdown_bps,
        "ledger_integrity": not failures,
        "live_orders_forbidden": not protocol.live_orders_permitted,
    }
    eligible = all(gate_values.values())
    if evaluated_at < protocol.checkpoint_at:
        verdict = "locked_before_checkpoint"
    elif eligible:
        verdict = "eligible_for_shadow_champion_review"
    else:
        verdict = "retain_research_candidate"
    body: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.protocol_sha256,
        "candidate_id": protocol.candidate_id,
        "artifact_sha256": protocol.artifact_sha256,
        "evaluated_at": evaluated_at.isoformat(),
        "evidence_window": {
            "start": protocol.evidence_start.isoformat(),
            "end": protocol.evidence_end.isoformat(),
            "checkpoint_at": protocol.checkpoint_at.isoformat(),
        },
        "metrics": {
            "formal_trades": len(normalized),
            "formal_decisions": len(formal_decision_ids),
            "mean_net_bps": mean_net,
            "student_t_lcb95_net_bps": lcb95,
            "stress_cost_bps": protocol.stress_cost_bps,
            "stress_mean_net_bps": stress_mean,
            "max_cumulative_drawdown_bps": drawdown,
            "observed_minutes": observed_minutes,
            "expected_minutes": expected_minutes,
            "bar_coverage": coverage,
        },
        "gates": gate_values,
        "integrity_failures": failures,
        "verdict": verdict,
        "promotion_scope": protocol.promotion_scope,
        "live_orders_permitted": False,
    }
    body["report_sha256"] = hashlib.sha256(_canonical_json(body)).hexdigest()
    return body


def _load_json_records(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"{path} must contain a JSON list")
        return value
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a frozen brain-forward evidence checkpoint")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--decisions", type=Path, required=True, help="JSON list or JSONL decision ledger")
    parser.add_argument("--trades", type=Path, required=True, help="JSON list or JSONL trade ledger")
    parser.add_argument("--bar-timestamps", type=Path, required=True, help="JSON list or JSONL timestamps")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    protocol = ForwardProtocol.load(args.protocol)
    decisions = _load_json_records(args.decisions)
    trades = _load_json_records(args.trades)
    bars = _load_json_records(args.bar_timestamps)
    timestamps = [row.get("ts") if isinstance(row, Mapping) else row for row in bars]
    report = build_checkpoint_report(protocol, decisions, trades, timestamps, as_of=args.as_of)
    rendered = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if report["verdict"] == "eligible_for_shadow_champion_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
