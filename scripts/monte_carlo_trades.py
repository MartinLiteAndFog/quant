#!/usr/bin/env python3
"""
Monte Carlo simulation on trade returns: bootstrap resampling, equity paths, percentiles.

Use after backtest (and optionally after fill model) to get distribution of outcomes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quant.backtest.fill_model import FillModelParams, apply_fill_model


def bootstrap_paths(
    r: np.ndarray,
    n_paths: int = 1000,
    seed: int | None = None,
) -> np.ndarray:
    """r = trade returns (1d); returns (n_paths, len(r)) with resampled order."""
    n = len(r)
    rng = np.random.default_rng(seed)
    paths = rng.choice(r, size=(n_paths, n), replace=True)
    return paths


def path_stats(path_returns: np.ndarray) -> dict:
    """path_returns: (n_trades,) per path. Return total_return_pct, max_dd_pct, sharpe_ann."""
    eq = np.cumprod(1.0 + path_returns)
    total_return_pct = (float(eq[-1]) - 1.0) * 100.0
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak - 1.0) * 100.0
    max_dd_pct = float(dd.min())
    n = len(path_returns)
    mean_r = float(np.mean(path_returns))
    std_r = float(np.std(path_returns, ddof=1)) if n > 1 else 0.0
    trades_per_year = n / max((n / 400.0), 1e-6)
    sharpe_ann = (mean_r / std_r * np.sqrt(trades_per_year)) if std_r > 0 else np.nan
    return {"total_return_pct": total_return_pct, "max_drawdown_pct": max_dd_pct, "sharpe_ann": sharpe_ann}


def run_monte_carlo(
    trades_path: Path,
    r_col: str = "pnl_pct",
    n_paths: int = 1000,
    seed: int | None = None,
    apply_maker_fill: bool = False,
    fee_bps: float = 15.0,
) -> pd.DataFrame:
    """Load trades, optionally apply fill model, bootstrap and return path stats DataFrame."""
    t = pd.read_parquet(trades_path)
    if r_col not in t.columns and "pnl_pct_adj" in t.columns:
        r_col = "pnl_pct_adj"
    if r_col not in t.columns:
        r_col = "pnl_pct" if "pnl_pct" in t.columns else None
    if r_col is None:
        raise ValueError("trades need pnl_pct or pnl_pct_adj")
    if apply_maker_fill and "pnl_pct_adj" not in t.columns:
        params = FillModelParams(fee_bps_roundtrip=fee_bps / 10_000.0)
        t = apply_fill_model(t, params=params, seed=seed)
        r_col = "pnl_pct_adj"
    r = pd.to_numeric(t[r_col], errors="coerce").dropna().astype(float).values
    if len(r) < 20:
        raise ValueError("Too few trade returns for bootstrap")
    paths = bootstrap_paths(r, n_paths=n_paths, seed=seed)
    rows = [path_stats(paths[i]) for i in range(n_paths)]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Monte Carlo bootstrap on trade returns")
    ap.add_argument("--trades", required=True, help="trades.parquet (or with pnl_pct_adj)")
    ap.add_argument("--paths", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--maker-fill", action="store_true", help="Apply maker fill model before bootstrap")
    ap.add_argument("--fee-bps", type=float, default=15.0)
    ap.add_argument("--out-csv", type=str, default=None)
    ap.add_argument("--percentiles", type=str, default="5,25,50,75,95")
    args = ap.parse_args()

    df = run_monte_carlo(
        Path(args.trades),
        n_paths=args.paths,
        seed=args.seed,
        apply_maker_fill=args.maker_fill,
        fee_bps=args.fee_bps,
    )
    pct = [float(x) for x in args.percentiles.split(",")]
    print("=== Monte Carlo (bootstrap trade returns) ===")
    print(f"Paths: {len(df)}  Maker-fill: {args.maker_fill}")
    print("\nPercentiles:")
    for col in ["total_return_pct", "max_drawdown_pct", "sharpe_ann"]:
        q = np.nanpercentile(df[col].values, pct)
        print(f"  {col}: " + "  ".join(f"p{p}={q[i]:.2f}" for i, p in enumerate(pct)))
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
