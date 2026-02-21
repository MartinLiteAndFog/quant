# scripts/ingest_pipe_ohlcv.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLS = ["ts", "open", "high", "low", "close", "volume", "quote_vol", "taker_base", "taker_quote", "count"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to pipe-delimited .csv.gz")
    ap.add_argument("--symbol", required=True, help="e.g. SOL-USDT")
    ap.add_argument("--exchange", default="local")
    ap.add_argument("--timeframe", default="1m")
    ap.add_argument("--out-ts", required=True, help="e.g. 20260215T133011Z (used in filename)")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        raise FileNotFoundError(str(inp))

    df = pd.read_csv(inp, sep="|", header=None, names=COLS, compression="gzip")

    # types
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    # invariants
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

    out_dir = Path("data/raw") / f"exchange={args.exchange}" / f"symbol={args.symbol}" / f"timeframe={args.timeframe}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.symbol}_{args.timeframe}_{args.out_ts}.parquet"

    df = df[["ts", "open", "high", "low", "close", "volume"]]
    df.to_parquet(out_path, index=False)

    # report
    dt = df["ts"].diff().dt.total_seconds().dropna()
    print("saved:", out_path)
    print("rows:", len(df))
    print("range:", df["ts"].iloc[0], "->", df["ts"].iloc[-1])
    print("dup ts:", int(df["ts"].duplicated().sum()))
    print("dt_sec median:", float(dt.median()))
    print("dt_sec p99:", float(dt.quantile(0.99)))
    print("dt_sec max:", float(dt.max()))


if __name__ == "__main__":
    main()
