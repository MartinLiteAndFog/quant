#!/usr/bin/env python3
"""Export local BTC-USDT minute candles with exact values, gaps and byte hashes.

Run: .venv/bin/python apps/cloudgarden/scripts/export_btc_history.py
Raw assets remain local. No repaired candles, synthetic volume or forward filling.
The directory name identifies the downloaded archive; venue provenance has not
been independently reconciled to an exchange receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from export_sol_history import export_history

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / "data/raw/exchange=binance-vision/symbol=BTC-USDT/timeframe=1m/BTC-USDT_1m_20200812T000000Z_20260624T235900Z.parquet"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "public/market-btc"


def export_btc_history(source: Path, output: Path) -> dict[str, object]:
    """Share the strict minute/OHLCV validator; replace instrument metadata only."""
    manifest = export_history(source, output)
    manifest.update({
        "source": "Local downloaded BTC-USDT minute archive (binance-vision directory; venue unverified)",
        "symbol": "BTC-USDT",
        "venue": "unverified",
        "marketType": "unverified",
        "license": "unverified; local private use",
    })
    temporary = output / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "manifest.json")
    # Verify what the browser will read, not only the in-memory source count.
    written_rows = 0
    for day in manifest["days"]:
        payload = (output / day["file"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != day["sha256"]:
            raise ValueError(f"Written BTC day checksum mismatch: {day['date']}")
        candles = json.loads(payload)
        if len(candles) != day["barCount"] or any(row[0][:10] != day["date"] for row in candles):
            raise ValueError(f"Written BTC day count/date mismatch: {day['date']}")
        written_rows += len(candles)
    if written_rows != manifest["totalBars"]:
        raise ValueError("Written BTC archive total differs from source")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = export_btc_history(args.source, args.output)
    print(json.dumps({key: value for key, value in manifest.items() if key != "days"}, indent=2))
    print(f"Daily assets: {len(manifest['days'])}; output: {args.output}")


if __name__ == "__main__":
    main()
