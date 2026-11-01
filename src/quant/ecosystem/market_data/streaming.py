"""ECO-19a: fail-closed, bounded-memory UTC OHLCV CSV/.gz -> indexed SQLite.

The importer NEVER generates missing candles or labels source data as real
without user-supplied provenance. The 5-year ingest path intentionally does
not call ECO-02 normalize(), which materializes all rows in memory.

Source layouts:
- canonical: CSV header timestamp,open,high,low,close,volume (timezone ISO)
- binance_klines: headerless 12-column klines with opening epoch milliseconds.
  This is a *layout parser*, not a claim the venue/feed matches TradingView.

SQLite is an intermediate immutable research index. No exchange operations.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone, time
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
import csv
import gzip
import json
import os
import sqlite3
from typing import Iterator, Mapping, TextIO
from uuid import uuid4

from quant.ecosystem.market_data import Provenance, normalize, timestamp

UTC = timezone.utc
VERSION = 1
BATCH = 4096
COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
INPUT_LAYOUTS = ("canonical", "binance_klines")
MAX_GAP_DETAILS = 64
HARD_BOUND = Decimal("1e20")


class StreamImportError(ValueError):
    """No partially validated OHLCV input can be admitted as a dataset."""


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _decimal(raw: str, name: str, row: int, *, positive: bool = True) -> str:
    try:
        value = Decimal(raw)
    except (TypeError, InvalidOperation, ValueError) as exc:
        raise StreamImportError(f"line {row}: invalid {name}") from exc
    if (not value.is_finite() or abs(value) > HARD_BOUND
            or (value <= 0 if positive else value < 0)):
        raise StreamImportError(f"line {row}: invalid/negative/nonfinite {name}")
    return format(value.normalize(), "f")


def _open_utc(raw: str, layout: str, row: int) -> datetime:
    try:
        if layout == "binance_klines":
            if not raw.isascii() or not raw.isdecimal():
                raise ValueError("open time must be epoch milliseconds")
            millis = int(raw)
            if millis < 0 or millis % 1000:
                raise ValueError("millisecond stamp must be an exact second")
            dt = datetime.fromtimestamp(millis // 1000, UTC)
        else:
            dt = timestamp(raw)
        if dt.microsecond:
            raise ValueError("only second-aligned UTC opens accepted")
        return dt
    except (OverflowError, OSError, ValueError, TypeError) as exc:
        raise StreamImportError(f"line {row}: invalid timezone-aware open timestamp") from exc


def _raw_csv_rows(handle: TextIO, layout: str) -> Iterator[tuple[int, dict[str, str]]]:
    if layout == "canonical":
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise StreamImportError("CSV requires a unique header")
        if not set(COLUMNS).issubset(reader.fieldnames):
            raise StreamImportError("CSV missing required columns: " + ",".join(COLUMNS))
        for line, row in enumerate(reader, 2):
            if any(row.get(k) is None for k in COLUMNS):
                raise StreamImportError(f"line {line}: malformed/incomplete candle")
            yield line, row
    else:
        reader = csv.reader(handle)
        for line, fields in enumerate(reader, 1):
            if len(fields) != 12:
                raise StreamImportError(f"line {line}: expected exactly 12 kline fields")
            yield line, {
                "timestamp": fields[0], "open": fields[1],
                "high": fields[2], "low": fields[3],
                "close": fields[4], "volume": fields[5],
                "close_timestamp": fields[6],
            }


def _decode_rows(path: Path, layout: str) -> Iterator[tuple[int, dict[str, str]]]:
    if str(path).endswith(".gz"):
        handler = gzip.open(path, mode="rt", encoding="utf-8", newline="")
    else:
        handler = path.open(mode="r", encoding="utf-8", newline="")
    try:
        with handler as stream:
            yield from _raw_csv_rows(stream, layout)
    except (OSError, UnicodeError, csv.Error, EOFError) as exc:
        raise StreamImportError("source CSV/gzip is corrupt or unreadable") from exc


def _sha256_raw(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while piece := stream.read(1024 * 1024):
            digest.update(piece)
    return digest.hexdigest()


def _boundary(value: str | datetime | None, label: str, interval: int) -> datetime | None:
    if value is None:
        return None
    try:
        dt = timestamp(value)
    except Exception as exc:
        raise StreamImportError(f"{label} must be a timezone-aware UTC timestamp") from exc
    if dt.microsecond or int(dt.timestamp()) % interval:
        raise StreamImportError(f"{label} must align to the source interval")
    return dt


def _gap(begin: datetime, end: datetime, interval: int,
         gaps: list[dict], totals: dict[str, int]) -> None:
    if end <= begin:
        return
    seconds = int((end - begin).total_seconds())
    if seconds % interval:
        raise StreamImportError("unaligned gap bounds")
    count = seconds // interval
    totals["missing_bars"] += count
    totals["gap_runs"] += 1
    if len(gaps) < MAX_GAP_DETAILS:
        gaps.append({
            "start": begin.isoformat(), "end": end.isoformat(),
            "missing_bars": count,
        })


def ingest_ohlcv(
    source: str | Path,
    destination: str | Path,
    provenance: Provenance,
    *, layout: str = "canonical",
    expected_start: str | datetime | None = None,
    expected_end: str | datetime | None = None,
    reject_gaps: bool = False,
) -> dict:
    """Stream and atomically validate a source into an indexed local SQLite file.

    An import cannot be resumed at a different supplied source hash, and
    existing destinations are never overwritten. Reports use open-time UTC.
    The first/last observed day is flagged as partial unless all bars exist.
    """
    src, dest = Path(source), Path(destination)
    if not src.is_file() or not src.stat().st_size:
        raise StreamImportError("nonempty source file required")
    if dest.exists():
        raise StreamImportError("destination exists: refusing overwrite")
    if dest.resolve() == src.resolve():
        raise StreamImportError("input and output cannot be the same file")
    if layout not in INPUT_LAYOUTS:
        raise StreamImportError("unsupported CSV layout")
    if not isinstance(provenance, Provenance):
        raise StreamImportError("explicit ECO-02 Provenance is required")
    if type(reject_gaps) is not bool:
        raise StreamImportError("reject_gaps must be boolean")
    interval = provenance.interval_seconds
    if interval <= 0 or 86400 % interval:
        raise StreamImportError("interval must divide exactly into a UTC day")
    start = _boundary(expected_start, "expected_start", interval)
    end = _boundary(expected_end, "expected_end", interval)
    if start and end and start >= end:
        raise StreamImportError("expected_end must follow expected_start")

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.parent / f".{dest.name}.{uuid4().hex}.partial"
    db: sqlite3.Connection | None = None
    try:
        db = sqlite3.connect(str(temp))
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("CREATE TABLE candles (ts INTEGER PRIMARY KEY, open TEXT NOT NULL,"
                   " high TEXT NOT NULL, low TEXT NOT NULL, close TEXT NOT NULL,"
                   " volume TEXT NOT NULL)")
        db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        hasher = sha256()
        totals = {"missing_bars": 0, "gap_runs": 0}
        gaps: list[dict] = []
        count = 0
        first: datetime | None = None
        previous: datetime | None = None
        daily_counts: dict[str, int] = {}
        chunk: list[tuple] = []

        with db:
            for line, record in _decode_rows(src, layout):
                at = _open_utc(record["timestamp"], layout, line)
                unix = int(at.timestamp())
                if unix % interval:
                    raise StreamImportError(f"line {line}: candle not aligned to {interval}s")
                if previous is not None and at <= previous:
                    raise StreamImportError(f"line {line}: duplicate or out-of-order candle")
                if start is not None and at < start or end is not None and at >= end:
                    raise StreamImportError(f"line {line}: candle outside declared coverage")
                if previous is not None:
                    _gap(previous+timedelta(seconds=interval), at, interval, gaps, totals)
                elif start is not None:
                    _gap(start, at, interval, gaps, totals)
                numbers = tuple(_decimal(record[key], key, line, positive=(key != "volume"))
                                for key in COLUMNS[1:])
                op, high, low, close, volume = (Decimal(x) for x in numbers)
                if high < max(op, close, low) or low > min(op, close, high):
                    raise StreamImportError(f"line {line}: impossible OHLC")
                if layout == "binance_klines":
                    try:
                        close_ms = int(record["close_timestamp"])
                    except (TypeError, ValueError) as exc:
                        raise StreamImportError(f"line {line}: invalid close timestamp") from exc
                    if close_ms != unix*1000 + interval*1000 - 1:
                        raise StreamImportError(f"line {line}: incorrect candle close boundary")
                if first is None:
                    first = at
                previous = at
                count += 1
                daily_counts[at.date().isoformat()] = daily_counts.get(
                    at.date().isoformat(), 0
                ) + 1
                record_tuple = (unix, *numbers)
                hasher.update((_canonical(record_tuple)+"\n").encode("utf-8"))
                chunk.append(record_tuple)
                if len(chunk) >= BATCH:
                    db.executemany("INSERT INTO candles VALUES (?,?,?,?,?,?)", chunk)
                    chunk.clear()
            if not count or first is None or previous is None:
                raise StreamImportError("input has no data")
            if chunk:
                db.executemany("INSERT INTO candles VALUES (?,?,?,?,?,?)", chunk)
            if end is not None:
                _gap(previous+timedelta(seconds=interval), end, interval, gaps, totals)
            if reject_gaps and totals["missing_bars"]:
                raise StreamImportError("missing OHLCV intervals: gaps are not interpolated")
            expected_day_bars = 86400 // interval
            partial_days = [
                {"day": day, "observed_bars": n}
                for day, n in daily_counts.items() if n != expected_day_bars
            ]
            provenance_dict = asdict(provenance)
            canonical_hash = hasher.hexdigest()
            fingerprint = sha256(_canonical({
                "schema": VERSION, "provenance": provenance_dict,
                "candles_sha256": canonical_hash, "rows": count,
            }).encode()).hexdigest()
            report = {
                "schema_version": VERSION,
                "status": "validated_index_only; not a trading backtest",
                "source_name": src.name,
                "source_layout": layout,
                "source_file_sha256": _sha256_raw(src),
                "canonical_candles_sha256": canonical_hash,
                "dataset_fingerprint": fingerprint,
                "source_bytes": src.stat().st_size,
                "rows": count,
                "observed_first_open": first.isoformat(),
                "observed_end_exclusive": (
                    previous+timedelta(seconds=interval)
                ).isoformat(),
                "expected_start": start.isoformat() if start else None,
                "expected_end": end.isoformat() if end else None,
                "coverage_claim": ("explicit_expected_window" if start and end
                                   else "observed_window_only"),
                "missing_bars": totals["missing_bars"],
                "gap_runs": totals["gap_runs"],
                "gap_details_truncated": totals["gap_runs"] > MAX_GAP_DETAILS,
                "gap_details": gaps,
                "partial_utc_days": partial_days[:64],
                "partial_days_total": len(partial_days),
                "max_row_buffer": BATCH,
                "provenance": provenance_dict,
                "parity_with_TradingView_verified": False,
                "signal_engine_connected": False,
                "oos_verified": False,
                "live_orders_sent": 0,
            }
            db.execute("INSERT INTO metadata VALUES (?,?)", ("manifest", _canonical(report)))
            db.execute("INSERT INTO metadata VALUES (?,?)", ("schema_version", str(VERSION)))
        db.close()
        db = None
        if dest.exists():
            raise StreamImportError("destination appeared during import")
        os.replace(temp, dest)
        return report
    finally:
        if db is not None:
            db.close()
        if temp.exists():
            temp.unlink()


def get_manifest(database: str | Path) -> dict:
    """Read the persisted provenance without scanning millions of OHLCV rows."""
    with sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True) as db:
        found = db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
        version = db.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
        if not found or not version or int(version[0]) != VERSION:
            raise StreamImportError("unrecognized indexed market format")
        obj = json.loads(found[0])
        if obj.get("schema_version") != VERSION:
            raise StreamImportError("corrupt market manifest")
        return obj


def iter_candles(
    database: str | Path,
    start: str | datetime,
    end: str | datetime,
) -> Iterator[dict[str, str]]:
    """Query only [start,end) using SQLite primary index, never full-memory load."""
    lo, hi = timestamp(start), timestamp(end)
    if lo >= hi:
        raise StreamImportError("window end must follow start")
    with sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True) as db:
        cursor = db.execute(
            "SELECT ts,open,high,low,close,volume FROM candles WHERE ts>=? AND ts<? ORDER BY ts",
            (int(lo.timestamp()), int(hi.timestamp())),
        )
        for ts, op, high, low, close, vol in cursor:
            yield {
                "timestamp": datetime.fromtimestamp(ts, UTC).isoformat(),
                "open": op, "high": high, "low": low,
                "close": close, "volume": vol,
            }


def read_window_as_dataset(
    database: str | Path,
    start: str | datetime,
    end: str | datetime,
    *, max_bars: int = 10080,
):
    """Small bounded ECO-02 in-memory Dataset for a specific research window.

    Fingerprints on returned Dataset cover just this selected window; use the
    persisted manifest for original full-source content fingerprint.
    """
    if type(max_bars) is not int or not 1 <= max_bars <= 100_000:
        raise StreamImportError("window must have a finite 1..100000 bar budget")
    report = get_manifest(database)
    prov = Provenance(**report["provenance"])
    lo, hi = timestamp(start), timestamp(end)
    if hi <= lo or (hi-lo).total_seconds() / prov.interval_seconds > max_bars:
        raise StreamImportError("requested window is too large")
    return normalize(iter_candles(database, lo, hi), prov, start=lo, end=hi)
