# scripts/check_parquet_anomalies.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_ohlcv_parquet(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    df = pd.read_parquet(p)

    # Handle ts either as column or datetime index
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts"})
        else:
            raise ValueError("parquet missing 'ts' column and not datetime-indexed")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    # Ensure numeric columns exist (some parquet may have extra cols; we only need these)
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"parquet missing column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Spot OHLCV anomalies around a timestamp + global jump/outlier scan.")
    ap.add_argument("--parquet", required=True, help="Path to OHLCV parquet with ts/open/high/low/close")
    ap.add_argument("--ts", required=True, help="Center timestamp, e.g. 2023-12-28T08:16:00Z")
    ap.add_argument("--minutes", type=int, default=30, help="Window size in minutes on each side (default: 30)")

    ap.add_argument("--roll", type=int, default=2000, help="Rolling median window (rows) for outlier scan")
    ap.add_argument("--k", type=float, default=0.25, help="Outlier threshold vs rolling median (fraction), default 0.25")

    args = ap.parse_args()

    df = _read_ohlcv_parquet(args.parquet)

    t0 = pd.to_datetime(args.ts, utc=True)
    t1 = t0 - pd.Timedelta(minutes=args.minutes)
    t2 = t0 + pd.Timedelta(minutes=args.minutes)

    w = df[(df["ts"] >= t1) & (df["ts"] <= t2)][["ts", "open", "high", "low", "close"]].copy()

    print("\n=== WINDOW ===")
    print(f"range: {t1} -> {t2}")
    if len(w) == 0:
        print("(no rows in window)")
    else:
        print(w.to_string(index=False))

    s = df["close"].astype(float)

    # Rolling median deviation
    med = s.rolling(args.roll, min_periods=max(50, args.roll // 10)).median()
    dev = (s - med).abs() / (med.abs() + 1e-12)

    bad = df[dev > args.k][["ts", "close"]].copy()
    bad["roll_median"] = med[dev > args.k].values
    bad["dev_frac"] = dev[dev > args.k].values

    print("\n=== BASIC STATS ===")
    print("rows:", len(df))
    print("ts range:", df["ts"].iloc[0], "->", df["ts"].iloc[-1])
    print("close min/median/max:", float(s.min()), float(s.median()), float(s.max()))
    print("outliers (dev > %.4f): %d" % (args.k, len(bad)))

    if len(bad):
        bad2 = bad.sort_values("dev_frac", ascending=False).head(30)
        print("\n=== TOP OUTLIERS (by rolling-median deviation) ===")
        print(bad2.to_string(index=False))

    # 1-bar jumps
    prev = s.shift(1)
    jump_abs = (s - prev).abs()
    jump_pct = jump_abs / (prev.abs() + 1e-12)

    j = df[["ts"]].copy()
    j["close"] = s
    j["jump_abs"] = jump_abs
    j["jump_pct"] = jump_pct

    j2 = j.sort_values("jump_pct", ascending=False).head(30)
    print("\n=== TOP 1-BAR JUMPS (pct) ===")
    print(j2.to_string(index=False))


if __name__ == "__main__":
    main()
