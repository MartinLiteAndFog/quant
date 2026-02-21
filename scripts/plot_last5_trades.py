#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def load_fills(fills_path: str, fill_col: str) -> pd.DataFrame:
    df = pd.read_parquet(fills_path)
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts"})
        else:
            raise ValueError("fills parquet missing 'ts' column and not datetime-indexed")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    if fill_col not in df.columns:
        raise ValueError(f"fills parquet missing fill_col='{fill_col}' (cols={list(df.columns)})")

    df[fill_col] = pd.to_numeric(df[fill_col], errors="coerce")
    df = df.dropna(subset=[fill_col]).reset_index(drop=True)
    return df[["ts", fill_col]].copy()


def load_trades(trades_path: str) -> pd.DataFrame:
    t = pd.read_parquet(trades_path)
    need = {"entry_ts", "exit_ts", "side", "entry_px", "exit_px"}
    missing = need - set(t.columns)
    if missing:
        raise ValueError(f"trades parquet missing columns: {sorted(missing)}")

    t["entry_ts"] = pd.to_datetime(t["entry_ts"], utc=True, errors="coerce")
    t["exit_ts"] = pd.to_datetime(t["exit_ts"], utc=True, errors="coerce")
    t = t.dropna(subset=["entry_ts", "exit_ts"]).sort_values("exit_ts").reset_index(drop=True)

    for c in ["side", "entry_px", "exit_px"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t = t.dropna(subset=["side", "entry_px", "exit_px"]).reset_index(drop=True)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="e.g. data/runs/IMBA_OPPEXIT_fee20_gateAnalysis")
    ap.add_argument("--fills-parquet", required=True)
    ap.add_argument("--fill-col", default="fill_ohlc4")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default=None, help="optional output png path")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    trades_path = run_dir / "trades_real.parquet"
    if not trades_path.exists():
        raise FileNotFoundError(f"trades file not found: {trades_path}")

    t = load_trades(str(trades_path))
    if len(t) == 0:
        raise ValueError("no closed trades found in trades_real.parquet")

    n = int(max(1, args.n))
    last = t.tail(n).copy()

    t0 = last["entry_ts"].min()
    t1 = last["exit_ts"].max()

    fills = load_fills(args.fills_parquet, args.fill_col)
    seg = fills[(fills["ts"] >= t0) & (fills["ts"] <= t1)].copy()
    if len(seg) == 0:
        raise ValueError(f"no fills rows in range {t0}..{t1}")

    import matplotlib.pyplot as plt

    fig_h = 2.6 * n
    fig, axes = plt.subplots(n, 1, figsize=(14, fig_h), sharex=False)
    if n == 1:
        axes = [axes]

    for i, (_, tr) in enumerate(last.iterrows()):
        ax = axes[i]
        a = tr["entry_ts"]
        b = tr["exit_ts"]
        side = int(tr["side"])
        entry_px = float(tr["entry_px"])
        exit_px = float(tr["exit_px"])

        s = seg[(seg["ts"] >= a) & (seg["ts"] <= b)].copy()
        if len(s) == 0:
            pad = pd.Timedelta(hours=6)
            s = seg[(seg["ts"] >= (a - pad)) & (seg["ts"] <= (b + pad))].copy()

        ax.plot(s["ts"].values, s[args.fill_col].values, linewidth=1.0)

        ax.scatter([a], [entry_px], marker="^" if side == 1 else "v")
        ax.scatter([b], [exit_px], marker="x")

        gross = (side * (exit_px - entry_px)) / abs(entry_px) if entry_px != 0 else np.nan
        ax.set_title(
            f"Trade {-n+i+1 if n>1 else -1}: side={side:+d}  entry={entry_px:.4f}  exit={exit_px:.4f}  gross={gross*100:.3f}%   [{a} -> {b}]",
            fontsize=10,
            loc="left",
        )
        ax.grid(True, linewidth=0.4)

    fig.tight_layout()

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"INFO wrote {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
