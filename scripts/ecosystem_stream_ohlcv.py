#!/usr/bin/env python3
"""Offline, bounded-memory SOL OHLCV CSV[.gz] -> indexed read-only research store.

Examples:
 PYTHONPATH=src python scripts/ecosystem_stream_ohlcv.py \
   --source /path/to/candles.csv.gz --layout canonical \
   --provenance /path/to/provenance.json --output /path/to/sol.sqlite

 The source and provenance must be genuinely supplied. This script does
 not fetch historical prices or invent TradingView signal parity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant.ecosystem.market_data import Provenance
from quant.ecosystem.market_data.streaming import (
    StreamImportError, ingest_ohlcv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--layout", choices=("canonical","binance_klines"),
                        default="canonical")
    parser.add_argument("--provenance", type=Path, required=True,
                        help="explicit ECO-02 Provenance JSON, including source and license")
    parser.add_argument("--output", type=Path, required=True, help="new .sqlite, never overwritten")
    parser.add_argument("--expected-start", help="inclusive UTC open, example 2021-01-01T00:00:00Z")
    parser.add_argument("--expected-end", help="exclusive UTC open, example 2026-01-01T00:00:00Z")
    parser.add_argument("--reject-gaps", action="store_true",
                        help="fail completely on missing timestamps")
    args = parser.parse_args()
    try:
        details=json.loads(args.provenance.read_text(encoding="utf-8"))
        if type(details) is not dict:
            raise ValueError("provenance must be a JSON object")
        provenance=Provenance(**details)
        manifest=ingest_ohlcv(
            args.source,args.output,provenance,
            layout=args.layout,
            expected_start=args.expected_start,
            expected_end=args.expected_end,
            reject_gaps=args.reject_gaps,
        )
    except (StreamImportError, ValueError, TypeError, OSError, KeyError) as exc:
        parser.exit(status=2, message=f"INPUT REJECTED; no validated OHLCV store: {exc}\n")
    print(json.dumps(manifest,sort_keys=True,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
