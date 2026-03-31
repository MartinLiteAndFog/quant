#!/usr/bin/env python3
from __future__ import annotations

import argparse

from quant.execution.daily_gate_artifacts import write_daily_gate_artifacts


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build live-ready CHOP/ADX/ER daily gate artifacts from Renko OHLC."
    )
    ap.add_argument(
        "--input-path",
        default=None,
        help="Renko OHLC CSV/parquet. If omitted, uses LIVE_RENKO_PATH, DASHBOARD_RENKO_PARQUET, or /data/live/renko_latest.parquet.",
    )
    ap.add_argument(
        "--out-on-csv",
        default=None,
        help="Optional debug CSV for countertrend/ON gate.",
    )
    ap.add_argument(
        "--out-off-csv",
        default=None,
        help="Optional debug CSV for trend/OFF gate.",
    )
    ap.add_argument("--symbol", default=None, help="Gate symbol for Postgres/Redis persistence.")
    args = ap.parse_args()

    import os

    input_path = (
        args.input_path
        or os.getenv("LIVE_RENKO_PATH")
        or os.getenv("DASHBOARD_RENKO_PARQUET")
        or "/data/live/renko_latest.parquet"
    )

    on_df, off_df = write_daily_gate_artifacts(
        input_path=input_path,
        out_on_path=args.out_on_csv,
        out_off_path=args.out_off_csv,
        symbol=args.symbol,
    )

    print("INFO persisted daily gate history rows=", len(on_df), "source=postgres_daily_gate")
    print("INFO attempted latest daily gate snapshot publish source=redis")
    if args.out_on_csv and args.out_off_csv:
        print("INFO wrote debug CSV artifacts")


if __name__ == "__main__":
    main()
