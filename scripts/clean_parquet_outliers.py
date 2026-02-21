# scripts/clean_parquet_outliers.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


REQ_COLS = ["open", "high", "low", "close"]


def _read_ohlcv_parquet(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    df = pd.read_parquet(p)

    # Handle ts either as column or datetime index
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex) and getattr(df.index, "name", None) in (None, "ts"):
            df = df.reset_index().rename(columns={"index": "ts"})
        elif isinstance(df.index, pd.DatetimeIndex):
            # keep the index name if it exists
            name = df.index.name or "ts"
            df = df.reset_index().rename(columns={name: "ts"})
        else:
            raise ValueError("parquet missing 'ts' column and not datetime-indexed")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    for c in REQ_COLS:
        if c not in df.columns:
            raise ValueError(f"parquet missing column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # volume is optional
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # Drop rows with NaN OHLC early (they’re unusable)
    df = df.dropna(subset=REQ_COLS).reset_index(drop=True)

    # Ensure invariants (some sources violate this)
    df["high"] = df[REQ_COLS].max(axis=1)
    df["low"] = df[REQ_COLS].min(axis=1)

    return df


def _rolling_median_dev(close: pd.Series, roll: int) -> pd.Series:
    roll = int(roll)
    med = close.rolling(roll, min_periods=max(50, roll // 10)).median()
    dev = (close - med).abs() / (med.abs() + 1e-12)
    return dev


def _find_stuck_segments(df: pd.DataFrame, min_len: int, require_zero_volume: bool) -> pd.Series:
    """
    Flags bars that belong to "freeze/stuck" segments:
      - close unchanged
      - AND bar has zero range (high==low==open==close)  (typical feed-freeze)
      - AND (optionally) volume==0 if volume exists
    """
    close = df["close"].astype(float)
    same_close = close.diff().fillna(0.0) == 0.0

    zero_range = (df["high"] == df["low"]) & (df["open"] == df["close"]) & (df["high"] == df["close"])

    if require_zero_volume and "volume" in df.columns:
        vol_ok = df["volume"].fillna(0.0) == 0.0
    else:
        vol_ok = pd.Series(True, index=df.index)

    stuck_bar = same_close & zero_range & vol_ok

    # group runs where stuck_bar is True
    run_id = (stuck_bar != stuck_bar.shift(1)).cumsum()
    run_len = stuck_bar.groupby(run_id).transform("sum")  # length of run for each row
    stuck_long = stuck_bar & (run_len >= int(min_len))
    return stuck_long


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Clean OHLCV parquet by dropping (default) or repairing obvious feed glitches/outliers."
    )
    ap.add_argument("--parquet", required=True, help="Input OHLCV parquet (ts + open/high/low/close [+volume])")
    ap.add_argument("--out", required=True, help="Output parquet path")

    # Mode
    ap.add_argument(
        "--mode",
        choices=["drop", "repair_prev"],
        default="drop",
        help="drop = remove flagged bars (most trustworthy). repair_prev = replace flagged bars with prev close.",
    )

    # Outlier thresholds
    ap.add_argument("--roll", type=int, default=2000, help="Rolling median window (rows) for dev scan")
    ap.add_argument("--dev_k", type=float, default=0.25, help="|close-roll_median| / |median| threshold")
    ap.add_argument("--jump_k", type=float, default=0.20, help="1-bar jump threshold vs prev close (fraction)")
    ap.add_argument("--range_k", type=float, default=0.20, help="(high-low)/prev_close threshold (fraction)")
    ap.add_argument("--oc_k", type=float, default=0.20, help="|close-open|/prev_close threshold (fraction)")

    # Stuck/freeze detection
    ap.add_argument("--stuck_len", type=int, default=10, help="Min length (bars) of stuck segments to flag")
    ap.add_argument(
        "--stuck_require_zero_volume",
        action="store_true",
        help="If volume exists, require volume==0 for stuck detection (stricter).",
    )

    # Iteration
    ap.add_argument("--max_passes", type=int, default=5, help="Re-run adjacency-based checks after cleaning")

    args = ap.parse_args()

    df0 = _read_ohlcv_parquet(args.parquet)

    # We keep a log of why things were flagged (for transparency)
    reasons_cols = ["bad_dev", "bad_jump", "bad_range", "bad_oc", "bad_stuck"]
    total_flagged_all_passes = 0

    df = df0.copy()

    for pass_i in range(1, int(args.max_passes) + 1):
        close = df["close"].astype(float)
        prev_close = close.shift(1)

        # avoid exploding on first row
        denom = prev_close.abs() + 1e-12

        dev = _rolling_median_dev(close, roll=int(args.roll))
        jump_pct = (close - prev_close).abs() / denom
        range_pct = (df["high"].astype(float) - df["low"].astype(float)).abs() / denom
        oc_pct = (df["close"].astype(float) - df["open"].astype(float)).abs() / denom

        bad_dev = dev > float(args.dev_k)
        bad_jump = jump_pct > float(args.jump_k)
        bad_range = range_pct > float(args.range_k)
        bad_oc = oc_pct > float(args.oc_k)
        bad_stuck = _find_stuck_segments(df, min_len=int(args.stuck_len), require_zero_volume=bool(args.stuck_require_zero_volume))

        bad = (bad_dev | bad_jump | bad_range | bad_oc | bad_stuck)

        # never auto-flag the very first bar purely because prev_close is NaN
        if len(df) > 0:
            bad.iloc[0] = bad.iloc[0] & (bad_dev.iloc[0] | bad_stuck.iloc[0])

        flagged = int(bad.sum())
        total_flagged_all_passes += flagged

        print(f"\n=== CLEAN PASS {pass_i} ===")
        print("rows_in:", len(df))
        print("flagged:", flagged)

        if flagged == 0:
            break

        if args.mode == "drop":
            # Drop flagged bars (most trustworthy: don’t invent prices)
            df = df.loc[~bad].copy().reset_index(drop=True)
        else:
            # repair_prev: set flagged bar OHLC to previous close (conservative, but synthetic)
            df = df.copy()
            for idx in np.flatnonzero(bad.values):
                if idx == 0:
                    # fallback: keep as-is for the first bar
                    continue
                base = float(df.at[idx - 1, "close"])
                df.at[idx, "open"] = base
                df.at[idx, "high"] = base
                df.at[idx, "low"] = base
                df.at[idx, "close"] = base
            df["high"] = df[REQ_COLS].max(axis=1)
            df["low"] = df[REQ_COLS].min(axis=1)

        # maintain ordering / uniqueness
        df = df.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    # Final sanity report
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    denom = prev_close.abs() + 1e-12
    jump_abs = (close - prev_close).abs()

    print("\n=== FINAL STATS ===")
    print("rows_out:", len(df))
    print("ts range:", df["ts"].iloc[0], "->", df["ts"].iloc[-1])
    print("median |Δclose|:", float(jump_abs.median(skipna=True)))
    print("p99    |Δclose|:", float(jump_abs.quantile(0.99)))
    print("max    |Δclose|:", float(jump_abs.max(skipna=True)))
    print("max jump pct:", float(((jump_abs / denom).max(skipna=True))))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Write with ts as a column (consistent for your other scripts)
    df.to_parquet(out, index=False)
    print("wrote:", str(out))


if __name__ == "__main__":
    main()
