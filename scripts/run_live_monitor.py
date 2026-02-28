#!/usr/bin/env python3
# scripts/run_live_monitor.py
"""
Fetch actual fills from KuCoin, match to expected trades (data/live/expected_trades.jsonl),
report predicted vs actual (slippage, optional PnL diff). Use for live overwatch.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from quant.execution.live_monitor import (
    fetch_actual_fills_from_kucoin,
    load_expected_trades,
    match_expected_to_actual,
    report_predicted_vs_actual,
)

def main() -> None:
    ap = argparse.ArgumentParser(description="Predicted vs actual: match expected trades to KuCoin fills")
    ap.add_argument("--symbol", default="SOL-USDT", help="Symbol (e.g. SOL-USDT)")
    ap.add_argument("--expected-dir", default=None, help="Dir with expected_trades.jsonl (default: data/live)")
    ap.add_argument("--out", default=None, help="Output CSV path for matched report")
    ap.add_argument("--since", default=None, help="Only expected trades since this ISO ts")
    ap.add_argument("--window-sec", type=float, default=120.0, help="Match window seconds")
    ap.add_argument("--limit", type=int, default=200, help="Max fills to fetch from KuCoin")
    args = ap.parse_args()

    expected_dir = Path(args.expected_dir) if args.expected_dir else None
    expected = load_expected_trades(live_dir=expected_dir, since_ts=args.since)
    if expected.empty:
        print("No expected trades found. Record some via record_expected() when executing signals.")
        return

    print(f"Loaded {len(expected)} expected trades")
    actual = fetch_actual_fills_from_kucoin(symbol=args.symbol, limit=args.limit)
    print(f"Fetched {len(actual)} actual fills from KuCoin")

    matched = match_expected_to_actual(
        expected,
        actual,
        time_window_sec=args.window_sec,
    )
    print(f"Matched {len(matched)} trades")

    out_path = Path(args.out) if args.out else None
    summary = report_predicted_vs_actual(matched, out_path=out_path)
    print("Summary:", summary)
    if out_path:
        print("Wrote", out_path)


if __name__ == "__main__":
    main()
