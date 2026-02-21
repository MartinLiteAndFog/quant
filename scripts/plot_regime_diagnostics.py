#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _read_ohlcv_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts"})
        else:
            raise ValueError("parquet missing 'ts' column and not datetime-indexed")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    need = {"open", "high", "low", "close"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"parquet missing columns: {sorted(missing)}")

    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


def _true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr


def _choppiness(df: pd.DataFrame, n: int) -> pd.Series:
    """
    CHOP = 100 * log10( sum(TR,n) / (HH(n)-LL(n)) ) / log10(n)
    """
    n = int(n)
    tr = _true_range(df)
    sum_tr = tr.rolling(n, min_periods=n).sum()
    hh = df["high"].astype(float).rolling(n, min_periods=n).max()
    ll = df["low"].astype(float).rolling(n, min_periods=n).min()
    denom = (hh - ll).replace(0.0, np.nan)
    chop = 100.0 * np.log10(sum_tr / denom) / np.log10(float(n))
    return chop


def _wilder_smooth(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / float(n), adjust=False).mean()


def _adx(df: pd.DataFrame, n: int) -> pd.Series:
    n = int(n)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    up = high.diff()
    down = -low.diff()

    dm_plus = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    dm_minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    tr = _true_range(df)
    atr = _wilder_smooth(tr, n)
    sm_plus = _wilder_smooth(dm_plus, n)
    sm_minus = _wilder_smooth(dm_minus, n)

    di_plus = 100.0 * (sm_plus / atr.replace(0.0, np.nan))
    di_minus = 100.0 * (sm_minus / atr.replace(0.0, np.nan))

    dx = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0.0, np.nan)
    adx = _wilder_smooth(dx, n)
    return adx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--run-dir", required=True, help="data/runs/<run_id> that contains regime.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--last-days", type=int, default=365)

    ap.add_argument("--chop-len", type=int, default=14)
    ap.add_argument("--chop-on", type=float, default=58.0)
    ap.add_argument("--chop-off", type=float, default=52.0)

    ap.add_argument("--adx-len", type=int, default=14)
    ap.add_argument("--adx-on", type=float, default=18.0)
    ap.add_argument("--adx-off", type=float, default=25.0)

    args = ap.parse_args()

    bars = _read_ohlcv_parquet(args.parquet)
    bars = bars[["ts", "open", "high", "low", "close"]].copy()

    # last N days window
    end_ts = bars["ts"].max()
    start_ts = end_ts - pd.Timedelta(days=int(args.last_days))
    w = bars[(bars["ts"] >= start_ts) & (bars["ts"] <= end_ts)].reset_index(drop=True)

    if len(w) == 0:
        raise SystemExit("No rows in requested window.")

    # indicators
    chop = _choppiness(w, args.chop_len)
    adx = _adx(w, args.adx_len)

    # regime from run
    run_dir = Path(args.run_dir)
    reg_path = run_dir / "regime.parquet"
    if not reg_path.exists():
        raise SystemExit(f"Missing {reg_path}. Did you run renko_runner with regime output enabled?")

    reg = pd.read_parquet(reg_path)
    if "ts" not in reg.columns or "regime_on" not in reg.columns:
        raise SystemExit("regime.parquet must contain columns: ts, regime_on")

    reg["ts"] = pd.to_datetime(reg["ts"], utc=True, errors="coerce")
    reg = reg.dropna(subset=["ts"]).drop_duplicates("ts", keep="last").sort_values("ts")

    # align to window timestamps
    reg = reg.set_index("ts")["regime_on"].astype(int)
    reg = reg.reindex(pd.DatetimeIndex(w["ts"]), method="ffill")
    reg = reg.fillna(0).astype(int)

    on_rate = 100.0 * float((reg == 1).mean())

    # --- Plot ---
    fig = plt.figure(figsize=(16, 9))

    # Price panel
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(w["ts"].values, w["close"].astype(float).values)
    ax1.set_title(f"SOL-USDC | last {args.last_days}d | regime_on rate={on_rate:.2f}%")
    ax1.set_ylabel("Price")

    # CHOP/ADX panel
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    ax2.plot(w["ts"].values, chop.values, label="CHOP")
    ax2.axhline(args.chop_on, linestyle="--", linewidth=1)
    ax2.axhline(args.chop_off, linestyle="--", linewidth=1)
    ax2.set_ylabel("CHOP")

    ax2b = ax2.twinx()
    ax2b.plot(w["ts"].values, adx.values, label="ADX")
    ax2b.axhline(args.adx_on, linestyle="--", linewidth=1)
    ax2b.axhline(args.adx_off, linestyle="--", linewidth=1)
    ax2b.set_ylabel("ADX")

    # Regime panel
    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
    ax3.plot(w["ts"].values, reg.values)
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_ylabel("regime_on")
    ax3.set_xlabel("Time")

    # Keep legend simple (matplotlib doesn't merge twin legends nicely)
    ax2.legend(loc="upper left")
    ax2b.legend(loc="upper right")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print(f"WROTE: {out}")


if __name__ == "__main__":
    main()
