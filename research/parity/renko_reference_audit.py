"""Offline, fail-closed TradingView vs local Renko brick/alert parity DIAGNOSTICS.

Research only: no exchange access, order generation or claimed TV parity.
Input CSVs are explicitly UTC-normalized; brick ts means the SOURCE 5m
opening time plus TradingView's synthetic millisecond sequencing, not
the time at which that brick could have been known.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path

UTC = timezone.utc
FIVE_MIN = 300
MSEC = timedelta(milliseconds=1)


class ParityInputError(ValueError):
    pass


def utc(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ParityInputError(f"invalid timestamp: {value}") from exc
    if dt.tzinfo is None:
        raise ParityInputError(f"timezone missing, UTC required: {value}")
    return dt.astimezone(UTC)


def dec(value: str, label: str) -> Decimal:
    try:
        v = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ParityInputError(f"invalid {label}: {value}") from exc
    if not v.is_finite():
        raise ParityInputError(f"nonfinite {label}")
    return v


def aligned(ts: datetime) -> bool:
    return ts.microsecond == 0 and int(ts.timestamp()) % FIVE_MIN == 0


def source_bucket(ts: datetime) -> datetime:
    return datetime.fromtimestamp((int(ts.timestamp()) // FIVE_MIN) * FIVE_MIN, UTC)


@dataclass(frozen=True)
class Brick:
    display_at: datetime
    source_open_at: datetime
    known_at: datetime
    sequence: int
    direction: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal

    @property
    def has_wick(self) -> bool:
        return self.high > max(self.open, self.close) or self.low < min(self.open, self.close)


@dataclass(frozen=True)
class SourceBar:
    open_at: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal


@dataclass(frozen=True)
class Alert:
    reference_id: str
    direction: int
    confirmed_5m_close_at: datetime


def csv_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ParityInputError(f"{path}: missing columns {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ParityInputError(f"{path}: empty CSV")
    return rows


def _prices(row: dict[str, str]) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    o, h, l, c = (dec(row[k], k) for k in ("open", "high", "low", "close"))
    if l > min(o, c) or h < max(o, c) or l > h or l <= 0:
        raise ParityInputError("OHLC geometry invalid")
    return o, h, l, c


def read_bricks(path: Path, *, box: Decimal) -> tuple[Brick, ...]:
    rows = csv_rows(path, {"ts", "open", "high", "low", "close"})
    output: list[Brick] = []
    last_ts: datetime | None = None
    last_bucket: datetime | None = None
    seq = 0
    for row in rows:
        stamp = utc(row["ts"])
        bucket = source_bucket(stamp)
        if stamp < bucket or stamp >= bucket + timedelta(minutes=5):
            raise ParityInputError("Renko timestamp escaped source bucket")
        if last_ts is not None and stamp <= last_ts:
            raise ParityInputError("Renko timestamps must be strictly increasing")
        seq = seq + 1 if bucket == last_bucket else 0
        # The timestamp is a display index only; never use it as order availability.
        if stamp != bucket + seq * MSEC:
            raise ParityInputError("expected 0,1,2,... ms offsets per 5m source bucket")
        o, h, l, c = _prices(row)
        delta = c - o
        if abs(delta - box) <= box / 1000000:
            direction = 1
        elif abs(delta + box) <= box / 1000000:
            direction = -1
        else:
            raise ParityInputError("brick body must have exactly one box")
        if row.get("direction") not in (None, "") and int(row["direction"]) != direction:
            raise ParityInputError("direction disagrees with brick body")
        output.append(Brick(stamp, bucket, bucket + timedelta(minutes=5),
                            seq, direction, o, h, l, c))
        last_ts, last_bucket = stamp, bucket
    return tuple(output)


def read_source(path: Path) -> dict[datetime, SourceBar]:
    output: dict[datetime, SourceBar] = {}
    previous: datetime | None = None
    for row in csv_rows(path, {"open_at", "open", "high", "low", "close"}):
        stamp = utc(row["open_at"])
        if not aligned(stamp):
            raise ParityInputError("source 5m timestamp not UTC aligned")
        if previous is not None and stamp != previous + timedelta(minutes=5):
            raise ParityInputError("source 5m bars have gap, duplicate or disorder")
        o, h, l, c = _prices(row)
        output[stamp] = SourceBar(stamp, o, h, l, c)
        previous = stamp
    return output


def read_alerts(path: Path) -> tuple[Alert, ...]:
    ids: set[str] = set()
    times: set[datetime] = set()
    output: list[Alert] = []
    for row in csv_rows(path, {"reference_id", "direction", "confirmed_5m_close_at"}):
        ident = row["reference_id"].strip()
        try:
            direction = int(row["direction"])
        except ValueError as exc:
            raise ParityInputError("alert direction must be +1 or -1") from exc
        stamp = utc(row["confirmed_5m_close_at"])
        if not ident or direction not in (-1, 1) or not aligned(stamp):
            raise ParityInputError("invalid confirmed 5m alert")
        if ident in ids or stamp in times:
            raise ParityInputError("duplicate alert identity or source-close time")
        ids.add(ident)
        times.add(stamp)
        output.append(Alert(ident, direction, stamp))
    return tuple(sorted(output, key=lambda x: x.confirmed_5m_close_at))


def _group(bricks: tuple[Brick, ...]) -> dict[datetime, tuple[Brick, ...]]:
    groups: dict[datetime, list[Brick]] = {}
    for b in bricks:
        groups.setdefault(b.source_open_at, []).append(b)
    return {key: tuple(value) for key, value in groups.items()}


def compare(reference: tuple[Brick, ...], local: tuple[Brick, ...],
            source: dict[datetime, SourceBar] | None = None,
            *, box: Decimal, max_examples: int = 30) -> dict:
    if box <= 0 or not box.is_finite():
        raise ParityInputError("box must be finite and > 0")
    rr, ll = _group(reference), _group(local)
    counts = {
        "reference_bricks": len(reference), "local_bricks": len(local),
        "brick_delta_local_minus_reference": len(local) - len(reference),
        "reference_wick_bricks": sum(b.has_wick for b in reference),
        "local_wick_bricks": sum(b.has_wick for b in local),
        "bucket_count_reference": len(rr), "bucket_count_local": len(ll),
        "different_brick_count_buckets": 0,
        "body_mismatch_bricks": 0, "direction_mismatch_bricks": 0,
        "wick_mismatch_bricks": 0, "display_timestamp_mismatch_bricks": 0,
        "source_range_violations": 0, "reference_buckets_without_source": 0,
        "reference_geometry_anomalies": 0,
    }
    mismatches: list[dict[str, str | int]] = []
    def problem(at: datetime, reason: str, **kwargs) -> None:
        if len(mismatches) < max_examples:
            mismatches.append({"source_open_at": at.isoformat(), "known_at":
                               (at + timedelta(minutes=5)).isoformat(),
                               "reason": reason, **kwargs})

    # A source bar high/low only bounds the possible wick: its precise
    # TradingView assignment requires the real feed and historical engine.
    if source is not None:
        for at, bricks in rr.items():
            bar = source.get(at)
            if bar is None:
                counts["reference_buckets_without_source"] += 1
                problem(at, "reference_bucket_missing_in_source")
                continue
            for b in bricks:
                if b.high > bar.high or b.low < bar.low:
                    counts["source_range_violations"] += 1
                    problem(at, "reference_brick_outside_source_5m_range",
                            sequence=b.sequence)
    previous: Brick | None = None
    for b in reference:
        if previous is not None:
            same = b.direction == previous.direction
            expected_open = previous.close if same else previous.close + b.direction * box
            expected_close = expected_open + b.direction * box
            if abs(b.open - expected_open) > box / 1000000 or abs(b.close - expected_close) > box / 1000000:
                counts["reference_geometry_anomalies"] += 1
                problem(b.source_open_at, "reference_geometry_anomaly", sequence=b.sequence)
        previous = b

    for at in sorted(set(rr) | set(ll)):
        left, right = rr.get(at, ()), ll.get(at, ())
        if len(left) != len(right):
            counts["different_brick_count_buckets"] += 1
            problem(at, "brick_count", reference=len(left), local=len(right))
        for a, b in zip(left, right):
            if a.direction != b.direction:
                counts["direction_mismatch_bricks"] += 1
                problem(at, "direction", sequence=a.sequence,
                        reference=a.direction, local=b.direction)
            if a.open != b.open or a.close != b.close:
                counts["body_mismatch_bricks"] += 1
                problem(at, "body", sequence=a.sequence,
                        reference=f"{a.open}->{a.close}", local=f"{b.open}->{b.close}")
            if a.high != b.high or a.low != b.low:
                counts["wick_mismatch_bricks"] += 1
                problem(at, "wick_or_high_low", sequence=a.sequence,
                        reference=f"{a.low}/{a.high}", local=f"{b.low}/{b.high}")
            if a.display_at != b.display_at:
                counts["display_timestamp_mismatch_bricks"] += 1
                problem(at, "display_timestamp", sequence=a.sequence)
    anchor = None
    if reference and local:
        first_delta = reference[0].open - local[0].open
        anchor = {
            "first_open_reference_minus_local": str(first_delta),
            "whole_box_offset": first_delta % box == 0,
            "possible_anchor_difference_not_proven": first_delta != 0,
        }
    return {
        "schema": "research-renko-reference-audit-v1",
        "parity_verified": False,
        "candidate_only": True,
        "live_orders_sent": 0,
        "trade_target_signals_exported": 0,
        "reference_source_identity_verified": False,
        "source_5m_available": source is not None,
        "wick_reconstruction_verified": False,
        "reason_not_verified": "Requires original same-instrument source, TradingView reference and independent confirmed alerts; exact historical wick/anchor logic not established",
        "counts": counts, "anchor_diagnostic": anchor, "first_mismatches": mismatches,
    }


def compare_alerts(reference: tuple[Alert, ...], local: tuple[Alert, ...]) -> dict:
    r = {a.confirmed_5m_close_at: a.direction for a in reference}
    l = {a.confirmed_5m_close_at: a.direction for a in local}
    same = set(r) & set(l)
    bad = sorted(at.isoformat() for at in same if r[at] != l[at])
    missing = sorted(at.isoformat() for at in set(r) - set(l))
    extra = sorted(at.isoformat() for at in set(l) - set(r))
    return {
        "reference_count": len(r), "local_count": len(l),
        "exact_direction_and_time": len(same) - len(bad),
        "direction_mismatch_times": bad,
        "missing_local_times": missing, "extra_local_times": extra,
        "matches_supplied_alerts": bool(r) and not (bad or missing or extra),
        "parity_verified": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tv-bricks", required=True, type=Path)
    p.add_argument("--local-bricks", required=True, type=Path)
    p.add_argument("--source-5m", type=Path)
    p.add_argument("--tv-alerts", type=Path)
    p.add_argument("--local-alerts", type=Path)
    p.add_argument("--box", default="0.1")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if (args.tv_alerts is None) != (args.local_alerts is None):
        p.error("both alert streams must be provided together")
    if args.output.exists():
        p.error("output exists; refusing overwrite")
    box = dec(args.box, "box")
    report = compare(read_bricks(args.tv_bricks, box=box),
                     read_bricks(args.local_bricks, box=box),
                     read_source(args.source_5m) if args.source_5m else None,
                     box=box)
    files = [args.tv_bricks, args.local_bricks, args.source_5m,
             args.tv_alerts, args.local_alerts]
    report["input_sha256"] = {str(path): sha256(path.read_bytes()).hexdigest()
                              for path in files if path is not None}
    report["alert_comparison"] = (compare_alerts(read_alerts(args.tv_alerts),
                                                 read_alerts(args.local_alerts))
                                  if args.tv_alerts else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": report["counts"],
                      "parity_verified": False}, sort_keys=True))


if __name__ == "__main__":
    main()
