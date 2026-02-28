#!/usr/bin/env python3
"""
Visualise Monte Carlo: histograms of return / max DD / Sharpe + sample equity paths.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

from quant.backtest.fill_model import FillModelParams, apply_fill_model

# Reuse bootstrap + path_stats from monte_carlo_trades
def bootstrap_paths(r: np.ndarray, n_paths: int, seed: int | None) -> np.ndarray:
    n = len(r)
    rng = np.random.default_rng(seed)
    return rng.choice(r, size=(n_paths, n), replace=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True, help="trades.parquet")
    ap.add_argument("--paths", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--maker-fill", action="store_true")
    ap.add_argument("--fee-bps", type=float, default=15.0)
    ap.add_argument("--out", default=None, help="Output PNG path")
    ap.add_argument("--sample-paths", type=int, default=80, help="Number of equity paths to draw")
    args = ap.parse_args()

    t = pd.read_parquet(args.trades)
    r_col = "pnl_pct"
    if args.maker_fill and "pnl_pct_adj" not in t.columns:
        params = FillModelParams(fee_bps_roundtrip=args.fee_bps / 10_000.0)
        t = apply_fill_model(t, params=params, seed=args.seed)
        r_col = "pnl_pct_adj"
    elif "pnl_pct_adj" in t.columns:
        r_col = "pnl_pct_adj"
    r = pd.to_numeric(t[r_col], errors="coerce").dropna().astype(float).values
    n = len(r)
    paths = bootstrap_paths(r, args.paths, args.seed)

    # Equity curves for sample paths (index = trade number)
    eq_paths = np.cumprod(1.0 + paths, axis=1)  # (n_paths, n_trades)
    total_return_pct = (eq_paths[:, -1] - 1.0) * 100.0
    peak = np.maximum.accumulate(eq_paths, axis=1)
    dd_pct = (eq_paths / peak - 1.0) * 100.0
    max_dd_pct = np.min(dd_pct, axis=1)
    mean_r = np.mean(paths, axis=1)
    std_r = np.std(paths, axis=1, ddof=1)
    std_r = np.where(std_r <= 0, np.nan, std_r)
    sharpe_ann = (mean_r / std_r * np.sqrt(n / (n / 400.0))) if np.any(np.isfinite(std_r)) else np.full(args.paths, np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1) Histogram total return %
    ax = axes[0, 0]
    ax.hist(total_return_pct, bins=60, color="steelblue", alpha=0.7, edgecolor="white")
    for p in [5, 25, 50, 75, 95]:
        v = np.percentile(total_return_pct, p)
        ax.axvline(v, color="red" if p == 50 else "gray", linestyle="--", alpha=0.8, label=f"p{p}" if p in (5, 50, 95) else "")
    ax.set_xlabel("Total return %")
    ax.set_ylabel("Count")
    ax.set_title("Monte Carlo: Total return (bootstrap)")
    ax.legend(loc="upper right", fontsize=8)

    # 2) Histogram max drawdown %
    ax = axes[0, 1]
    ax.hist(max_dd_pct, bins=60, color="coral", alpha=0.7, edgecolor="white")
    for p in [5, 50, 95]:
        v = np.percentile(max_dd_pct, p)
        ax.axvline(v, color="darkred" if p == 50 else "gray", linestyle="--", alpha=0.8)
    ax.set_xlabel("Max drawdown %")
    ax.set_ylabel("Count")
    ax.set_title("Monte Carlo: Max drawdown")

    # 3) Sample equity paths
    ax = axes[1, 0]
    n_show = min(args.sample_paths, args.paths)
    idx_show = np.linspace(0, args.paths - 1, n_show, dtype=int)
    for i in idx_show:
        ax.plot(eq_paths[i], color="gray", alpha=0.25, linewidth=0.8)
    ax.plot(np.median(eq_paths, axis=0), color="darkblue", linewidth=2, label="Median path")
    ax.plot(np.percentile(eq_paths, 5, axis=0), color="red", linestyle="--", alpha=0.8, label="p5")
    ax.plot(np.percentile(eq_paths, 95, axis=0), color="green", linestyle="--", alpha=0.8, label="p95")
    ax.set_xlabel("Trade number")
    ax.set_ylabel("Equity (start=1)")
    ax.set_title("Sample equity paths")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4) Histogram Sharpe
    ax = axes[1, 1]
    sh = sharpe_ann[np.isfinite(sharpe_ann)]
    if len(sh):
        ax.hist(sh, bins=50, color="seagreen", alpha=0.7, edgecolor="white")
        for p in [5, 50, 95]:
            v = np.percentile(sh, p)
            ax.axvline(v, color="darkgreen" if p == 50 else "gray", linestyle="--", alpha=0.8)
    ax.set_xlabel("Sharpe (ann.)")
    ax.set_ylabel("Count")
    ax.set_title("Monte Carlo: Sharpe ratio")

    title = f"Monte Carlo ({args.paths} paths, bootstrap)" + (" + maker fill" if args.maker_fill else "")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    out = Path(args.out) if args.out else Path(args.trades).parent / "monte_carlo_plot.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
