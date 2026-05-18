#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_price(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    if "ts" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "ts"})
    if "ts" not in df.columns:
        raise ValueError("price parquet missing 'ts'")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last")
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"price parquet missing '{c}'")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df[["ts", "open", "high", "low", "close"]]


def _read_trades(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    for c in ["entry_ts", "exit_ts", "side"]:
        if c not in df.columns:
            raise ValueError(f"trades parquet missing '{c}'")
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True, errors="coerce")
    df["side"] = pd.to_numeric(df["side"], errors="coerce")
    df = df.dropna(subset=["entry_ts", "exit_ts", "side"]).sort_values("entry_ts").reset_index(drop=True)
    return df


def _read_equity(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    if "ts" not in df.columns:
        raise ValueError("equity parquet missing 'ts'")
    eq_col = "equity_5x" if "equity_5x" in df.columns else "equity"
    if eq_col not in df.columns:
        raise ValueError("equity parquet missing equity column")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df[eq_col] = pd.to_numeric(df[eq_col], errors="coerce")
    df = df.dropna(subset=["ts", eq_col]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    return df[["ts", eq_col]].rename(columns={eq_col: "equity"})


def _asof_price(ts_index: pd.DatetimeIndex, prices: np.ndarray, t: pd.Timestamp) -> float | None:
    loc = ts_index.searchsorted(pd.Timestamp(t), side="right") - 1
    if loc < 0 or loc >= len(ts_index):
        return None
    return float(prices[loc])


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot EURUSD price in blue with red equity and entry markers.")
    ap.add_argument("--price", required=True, help="OHLCV parquet with ts/open/high/low/close")
    ap.add_argument("--trades", required=True, help="Trades parquet with entry_ts/side")
    ap.add_argument("--equity", required=True, help="Equity parquet with ts/equity_5x or equity")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--title", default="EURUSD Price + Leveraged Equity")
    args = ap.parse_args()

    price = _read_price(Path(args.price))
    trades = _read_trades(Path(args.trades))
    equity = _read_equity(Path(args.equity))

    px_ts = pd.DatetimeIndex(price["ts"])
    px_close = price["close"].astype(float).to_numpy()

    # Scale equity onto the price axis for a single-panel overlay.
    eq = equity.copy()
    pmin, pmax = float(np.min(px_close)), float(np.max(px_close))
    emin, emax = float(eq["equity"].min()), float(eq["equity"].max())
    if emax - emin <= 1e-12:
        eq["equity_scaled"] = pmin
    else:
        eq["equity_scaled"] = (eq["equity"] - emin) / (emax - emin) * (pmax - pmin) + pmin

    long_x, long_y, short_x, short_y = [], [], [], []
    for _, row in trades.iterrows():
        y = _asof_price(px_ts, px_close, row["entry_ts"])
        if y is None:
            continue
        if float(row["side"]) > 0:
            long_x.append(row["entry_ts"])
            long_y.append(y)
        elif float(row["side"]) < 0:
            short_x.append(row["entry_ts"])
            short_y.append(y)

    fig, ax = plt.subplots(figsize=(18, 8))

    # Equity first so it sits visually in the back.
    ax.plot(eq["ts"], eq["equity_scaled"], color="red", linewidth=2.0, alpha=0.28, label="Equity 5x (scaled)")
    ax.plot(px_ts, px_close, color="blue", linewidth=1.2, alpha=0.95, label="EURUSD")

    if long_x:
        ax.scatter(long_x, long_y, marker="^", s=55, color="green", edgecolors="none", label="Long entry", zorder=5)
    if short_x:
        ax.scatter(short_x, short_y, marker="v", s=55, color="red", edgecolors="none", label="Short entry", zorder=5)

    ax.set_title(args.title)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("EURUSD")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"WROTE: {out}")


if __name__ == "__main__":
    main()
