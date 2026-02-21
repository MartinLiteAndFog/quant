#!/usr/bin/env python3
# scripts/plot_equity_price_vol.py

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_price(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts"})
        else:
            raise ValueError("Parquet must contain 'ts' column or have a DatetimeIndex.")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    if "close" not in df.columns:
        raise ValueError("Parquet must contain 'close' column.")

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df[["ts", "close"]]


def _load_equity(equity_path: str) -> pd.DataFrame:
    eq = pd.read_parquet(equity_path)
    if "ts" not in eq.columns:
        if isinstance(eq.index, pd.DatetimeIndex):
            eq = eq.reset_index().rename(columns={"index": "ts"})
        else:
            raise ValueError("Equity parquet must contain 'ts' column or have a DatetimeIndex.")

    eq["ts"] = pd.to_datetime(eq["ts"], utc=True, errors="coerce")
    eq = eq.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    if "equity" not in eq.columns:
        raise ValueError("Equity parquet must contain 'equity' column.")

    eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
    eq = eq.dropna(subset=["equity"]).reset_index(drop=True)
    return eq[["ts", "equity"]]


def _apply_window(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if start:
        s = pd.Timestamp(start)
        if s.tzinfo is None:
            s = s.tz_localize("UTC")
        out = out[out["ts"] >= s]
    if end:
        e = pd.Timestamp(end)
        if e.tzinfo is None:
            e = e.tz_localize("UTC")
        out = out[out["ts"] <= e]
    return out.reset_index(drop=True)


def _rolling_vol_pct(price: pd.Series, window: int) -> pd.Series:
    # 1m-ish pct returns volatility in percent (not annualized)
    r = price.pct_change()
    vol = r.rolling(window=window, min_periods=max(3, window // 5)).std()
    return vol * 100.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot price + equity + rolling volatility (no trade markers).")
    ap.add_argument("--parquet", required=True, help="OHLCV parquet with ts + close")
    ap.add_argument("--equity", required=True, help="equity.parquet with ts + equity")
    ap.add_argument("--start", default=None, help="ISO timestamp, e.g. 2025-02-01T00:00:00Z")
    ap.add_argument("--end", default=None, help="ISO timestamp, e.g. 2026-02-01T00:00:00Z")
    ap.add_argument("--vol-window", type=int, default=240, help="Rolling window in bars (default 240 ~ 4h on 1m data)")
    ap.add_argument("--out", default="data/plots/price_equity_vol.png")
    ap.add_argument("--title", default="Price + Equity + Volatility")
    args = ap.parse_args()

    price = _load_price(args.parquet)
    eq = _load_equity(args.equity)

    price = _apply_window(price, args.start, args.end)
    eq = _apply_window(eq, args.start, args.end)

    if len(price) == 0:
        raise ValueError("No price rows in the selected window.")
    if len(eq) == 0:
        raise ValueError("No equity rows in the selected window.")

    # Normalize price and equity to 1.0 at start (so they’re comparable)
    price_norm = price["close"] / float(price["close"].iloc[0])
    eq_norm = eq["equity"] / float(eq["equity"].iloc[0])

    # Align equity to price time axis (forward-fill to show smooth equity line)
    eq_aligned = pd.DataFrame({"ts": price["ts"]}).merge(eq[["ts", "equity"]], on="ts", how="left").sort_values("ts")
    eq_aligned["equity"] = eq_aligned["equity"].ffill()
    eq_aligned = eq_aligned.dropna(subset=["equity"]).reset_index(drop=True)
    eq_aligned_norm = eq_aligned["equity"] / float(eq_aligned["equity"].iloc[0])

    vol_pct = _rolling_vol_pct(price["close"], window=int(args.vol_window))

    # --- Plot: 2 panels (top: normalized price+equity, bottom: volatility %) ---
    fig = plt.figure(figsize=(22, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(price["ts"], price_norm, linewidth=1.2, label="Price (normalized)")
    ax1.plot(eq_aligned["ts"], eq_aligned_norm, linewidth=1.2, label="Equity (normalized)")
    ax1.set_title(args.title)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left")

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(price["ts"], vol_pct, linewidth=1.0, label=f"Rolling vol ({args.vol_window} bars, %)")
    ax2.grid(True, alpha=0.25)
    ax2.set_ylabel("Vol %")
    ax2.legend(loc="upper left")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    plt.close()

    print("WROTE:", args.out)
    print("PRICE rows:", len(price), "EQUITY rows:", len(eq), "VOL window:", args.vol_window)


if __name__ == "__main__":
    main()
