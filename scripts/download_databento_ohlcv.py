#!/usr/bin/env python3
"""
Download historical OHLCV bars from Databento and write them to the repo's
canonical parquet schema:

  ts (UTC), open, high, low, close, volume

Example:
  python scripts/download_databento_ohlcv.py \
    --dataset GLBX.MDP3 \
    --symbol NQ.v.0 \
    --stype-in continuous \
    --schema ohlcv-1m \
    --start 2025-04-01 \
    --end 2026-04-01
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _load_databento():
    try:
        import databento as db  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: databento\n"
            "Install it with: python -m pip install databento"
        ) from exc
    return db


def _normalize_store_to_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Databento returned no rows for the requested range")

    out = df.copy()
    if "ts_event" in out.columns:
        out = out.rename(columns={"ts_event": "ts"})
    elif out.index.name == "ts_event":
        out = out.reset_index().rename(columns={"ts_event": "ts"})
    elif "ts" not in out.columns:
        out = out.reset_index()
        if "ts_event" in out.columns:
            out = out.rename(columns={"ts_event": "ts"})

    need = {"ts", "open", "high", "low", "close"}
    missing = need - set(out.columns)
    if missing:
        raise ValueError(f"Databento OHLCV response missing columns: {sorted(missing)}")

    if "volume" not in out.columns:
        out["volume"] = 0.0

    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["ts", "open", "high", "low", "close"])
    out = out.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    return out[["ts", "open", "high", "low", "close", "volume"]]


def _default_output(exchange: str, symbol: str, timeframe: str) -> Path:
    safe_symbol = symbol.replace("/", "-")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("data/raw") / f"exchange={exchange}" / f"symbol={safe_symbol}" / f"timeframe={timeframe}"
    return out_dir / f"{safe_symbol}_{timeframe}_{stamp}.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="GLBX.MDP3", help="Databento dataset, e.g. GLBX.MDP3")
    ap.add_argument("--schema", default="ohlcv-1m", help="Databento schema, e.g. ohlcv-1m")
    ap.add_argument("--symbol", default="NQ.v.0", help="Databento symbol, e.g. NQ.v.0 or NQM6")
    ap.add_argument(
        "--stype-in",
        default="continuous",
        choices=["continuous", "raw_symbol", "parent", "instrument_id"],
        help="Input symbology type for --symbol",
    )
    ap.add_argument("--start", required=True, help="Inclusive UTC start date/time")
    ap.add_argument("--end", required=True, help="Exclusive/inclusive UTC end date/time per Databento API")
    ap.add_argument("--exchange", default="databento", help="Exchange/source label used in output path")
    ap.add_argument("--timeframe", default="1m", help="Repo timeframe label used in output path")
    ap.add_argument("--out", default=None, help="Output parquet path; default writes under data/raw/...")
    args = ap.parse_args()

    key = os.getenv("DATABENTO_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set DATABENTO_API_KEY before running this script")

    db = _load_databento()
    client = db.Historical(key)
    store = client.timeseries.get_range(
        dataset=args.dataset,
        symbols=args.symbol,
        schema=args.schema,
        stype_in=args.stype_in,
        start=args.start,
        end=args.end,
    )
    df = _normalize_store_to_ohlcv(store.to_df())

    out_path = Path(args.out) if args.out else _default_output(args.exchange, args.symbol, args.timeframe)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"rows={len(df)}")
    print(f"start={df['ts'].iloc[0]}  end={df['ts'].iloc[-1]}")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
