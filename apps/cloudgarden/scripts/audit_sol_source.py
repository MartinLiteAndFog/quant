#!/usr/bin/env python3
"""Read-only audit of the private, frozen SOL Parquet file. No uploads or data substitution."""
from __future__ import annotations
import argparse
from datetime import timezone, timedelta
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path

EXPECTED_SHA = "fa3e689ddbc42b35b89f66db4240521a330896e2d89fc2d2df8624bc749fda64"
EXPECTED = (3_109_186, 10, 1_424, "2020-08-11T06:00:00Z", "2026-07-11T09:29:00Z")
SOURCE = Path(__file__).resolve().parents[3] / "research/solusdt_robust_growth_v1/immutable/SOL-USDT_1m_20260711T092900Z.parquet"
FIELDS = ("open", "high", "low", "close", "volume")
PRECISION = Decimal("0.00000001")

def audit(source: Path) -> dict[str, object]:
    import pyarrow.parquet as parquet
    if not source.is_file():
        raise FileNotFoundError("Private original is not mounted; source audit BLOCKED.")
    digest = sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    sha = digest.hexdigest()
    if sha != EXPECTED_SHA:
        raise ValueError("Frozen original SHA256 mismatch; refusing to substitute sources.")
    reader = parquet.ParquetFile(source)
    count = gaps = missing = 0
    first = last = previous_minute = None
    for batch in reader.iter_batches(batch_size=32_768, columns=["ts", *FIELDS]):
        columns = batch.to_pydict()
        for idx, ts in enumerate(columns["ts"]):
            if ts is None or ts.tzinfo is None or ts.utcoffset() != timedelta(0):
                raise ValueError("Timestamp missing explicit UTC timezone.")
            ts = ts.astimezone(timezone.utc)
            if ts.second or ts.microsecond or getattr(ts, "nanosecond", 0):
                raise ValueError("Timestamp is off the UTC minute grid.")
            minute = int(ts.timestamp()) // 60
            if previous_minute is not None:
                delta = minute - previous_minute
                if delta <= 0:
                    raise ValueError("Duplicate or unordered timestamp.")
                if delta > 1:
                    gaps += 1
                    missing += delta - 1
            previous_minute = minute
            first = first or ts
            last = ts
            values = []
            for field in FIELDS:
                raw = columns[field][idx]
                if raw is None:
                    raise ValueError("Null OHLCV value.")
                try:
                    value = Decimal(str(raw))
                    if not value.is_finite() or value != value.quantize(PRECISION):
                        raise ValueError("OHLCV exceeds eight-decimal contract.")
                except InvalidOperation as exc:
                    raise ValueError("Invalid OHLCV number.") from exc
                values.append(value)
            op, high, low, close, volume = values
            if min(op, high, low, close) <= 0 or volume < 0:
                raise ValueError("Nonpositive price or negative volume.")
            if high < max(op, low, close) or low > min(op, high, close):
                raise ValueError("OHLC range impossible.")
            count += 1
    if count != reader.metadata.num_rows:
        raise ValueError("Scanned Parquet rows do not reconcile with metadata.")
    start = first.isoformat().replace("+00:00", "Z") if first else None
    end = last.isoformat().replace("+00:00", "Z") if last else None
    if (count, gaps, missing, start, end) != EXPECTED:
        raise ValueError("Count, UTC bounds or gap totals differ from frozen source reference.")
    if (int(last.timestamp()) - int(first.timestamp())) // 60 + 1 != count + missing:
        raise ValueError("Time coverage not reconciled to count plus missing minutes.")
    return {
        "schemaVersion": 1, "status": "original_bytes_and_ohlcv_structure_verified",
        "sourceSha256": sha, "originalBytes": source.stat().st_size,
        "totalBars": count, "gapCount": gaps, "missingMinutes": missing,
        "duplicateCount": 0, "firstUtc": start, "lastUtc": end,
        "provenance": {
            "venue": "unverified", "spotOrFutures": "unverified",
            "exactInstrument": "unverified", "contractMultiplier": "unverified",
            "timestampSemantics": "assumed_open_unverified",
            "licenseAndRedistributionRights": "unverified",
            "fundingHistory": "unavailable_unverified",
        },
        "strategyAndVenueParity": "not_verified",
        "marketPerformanceVerdict": "not_comparable", "rawMarketDataIncluded": False,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--report", type=Path, help="Write metadata-only JSON")
    args = parser.parse_args()
    result = json.dumps(audit(args.source), sort_keys=True, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(result, encoding="utf-8")
    print(result, end="")

if __name__ == "__main__":
    main()
