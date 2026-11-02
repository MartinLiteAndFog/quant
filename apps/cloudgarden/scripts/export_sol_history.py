#!/usr/bin/env python3
"""Export the immutable local SOL archive into validated daily browser assets.

Run from the repository: .venv/bin/python apps/cloudgarden/scripts/export_sol_history.py
The local data assets are ignored by Git. No candle is repaired or forward-filled.
"""
from __future__ import annotations

import argparse
from datetime import timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as parquet


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / "research/solusdt_robust_growth_v1/immutable/SOL-USDT_1m_20260711T092900Z.parquet"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "public/market"
EIGHT_PLACES = Decimal("0.00000001")
FIELDS = ("open", "high", "low", "close", "volume")


def decimal_string(value: object) -> tuple[Decimal, str]:
    number = Decimal(str(value))
    try:
        valid = number.is_finite() and number == number.quantize(EIGHT_PLACES)
    except InvalidOperation:
        valid = False
    if not valid:
        raise ValueError(f"Source value cannot be preserved at eight decimal places: {value!r}")
    text = format(number, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return number, "0" if number == 0 else text


def export_history(source: Path, output: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    source_sha = digest.hexdigest()
    reader = parquet.ParquetFile(source)
    output.mkdir(parents=True, exist_ok=True)
    days: list[dict[str, object]] = []
    rows: list[list[str]] = []
    current_date = ""
    day_gaps = day_missing = total_missing = total_rows = 0
    previous_seconds: int | None = None

    def flush_day() -> None:
        if not rows:
            return
        filename = f"{current_date}.json"
        target = output / filename
        temporary = output / f".{filename}.tmp"
        payload = (json.dumps(rows, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        temporary.write_bytes(payload)
        temporary.replace(target)
        days.append({"date": current_date, "barCount": len(rows), "gapCount": day_gaps,
                     "missingMinutes": day_missing, "file": filename,
                     "sha256": hashlib.sha256(payload).hexdigest()})

    for batch in reader.iter_batches(batch_size=16_384, columns=["ts", *FIELDS]):
        columns = batch.to_pydict()
        for index, timestamp in enumerate(columns["ts"]):
            if timestamp is None or timestamp.utcoffset() is None:
                raise ValueError("Source timestamp must have a UTC timezone")
            timestamp = timestamp.astimezone(timezone.utc)
            if timestamp.second or timestamp.microsecond or getattr(timestamp, "nanosecond", 0):
                raise ValueError(f"Source timestamp is not aligned to a minute: {timestamp}")
            seconds = int(timestamp.timestamp())
            missing = 0
            if previous_seconds is not None:
                interval = seconds - previous_seconds
                if interval <= 0:
                    raise ValueError(f"Duplicate or out-of-order timestamp: {timestamp}")
                if interval % 60:
                    raise ValueError(f"Source interval is not whole minutes: {interval}")
                missing = interval // 60 - 1
            date = timestamp.strftime("%Y-%m-%d")
            if date != current_date:
                flush_day()
                rows = []
                day_gaps = day_missing = 0
                current_date = date
            values = [decimal_string(columns[field][index]) for field in FIELDS]
            op, high, low, close, volume = (value[0] for value in values)
            if min(op, high, low, close) <= 0 or volume < 0:
                raise ValueError(f"Invalid price or volume at {timestamp}")
            if high < max(op, low, close) or low > min(op, high, close):
                raise ValueError(f"Invalid OHLC range at {timestamp}")
            rows.append([timestamp.isoformat().replace("+00:00", "Z"), *(value[1] for value in values)])
            total_rows += 1
            previous_seconds = seconds
            if missing:
                day_gaps += 1
                day_missing += missing
                total_missing += missing
    flush_day()
    if not days:
        raise ValueError("Source contains no candles")
    if total_rows != reader.metadata.num_rows:
        raise ValueError("Export count differs from parquet metadata")
    manifest: dict[str, object] = {
        "schemaVersion": 1,
        "source": "Local downloaded SOL-USDT minute archive (venue unverified)",
        "sourceSha256": source_sha,
        "symbol": "SOL-USDT",
        "venue": "unverified",
        "marketType": "unverified",
        "license": "unverified; local private use",
        "timestampSemantics": "open",
        "intervalSeconds": 60,
        "firstDate": days[0]["date"],
        "lastDate": days[-1]["date"],
        "totalBars": total_rows,
        "missingMinutes": total_missing,
        "duplicateCount": 0,
        "days": days,
    }
    temporary = output / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = export_history(args.source, args.output)
    print(json.dumps({key: value for key, value in manifest.items() if key != "days"}, indent=2))
    print(f"Daily assets: {len(manifest['days'])}; output: {args.output}")


if __name__ == "__main__":
    main()
