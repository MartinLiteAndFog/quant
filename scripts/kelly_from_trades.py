#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def growth(L: float, r: np.ndarray) -> float:
    x = 1.0 + L * r
    if np.any(x <= 0):
        return -np.inf
    return float(np.mean(np.log(x)))


def kelly_search(r: np.ndarray, L_max: float = 20.0, n_grid: int = 20001) -> dict:
    # Hard upper bound from worst loss to avoid log(<=0)
    r_min = float(np.min(r))
    if r_min < 0:
        L_bankrupt = 0.999999 / abs(r_min)  # slightly inside
        L_hi = min(L_max, L_bankrupt)
    else:
        L_hi = L_max

    grid = np.linspace(0.0, max(1e-9, L_hi), n_grid)
    vals = np.array([growth(L, r) for L in grid], dtype=float)

    j = int(np.nanargmax(vals))
    L_star = float(grid[j])
    g_star = float(vals[j])

    return {
        "L_star": L_star,
        "g_star": g_star,
        "L_hi_used": float(L_hi),
        "r_min": r_min,
        "r_mean": float(np.mean(r)),
        "r_std": float(np.std(r, ddof=1)) if len(r) > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades-real", required=True, help="path to trades_real.parquet")
    ap.add_argument("--col", default="pnl_pct_real", help="return column (default pnl_pct_real)")
    ap.add_argument("--L-max", type=float, default=20.0)
    ap.add_argument("--grid", type=int, default=20001)
    args = ap.parse_args()

    df = pd.read_parquet(args.trades_real)
    if args.col not in df.columns:
        raise SystemExit(f"missing col '{args.col}' in {list(df.columns)}")

    r = pd.to_numeric(df[args.col], errors="coerce").dropna().astype(float).values
    if len(r) < 50:
        raise SystemExit(f"too few returns: n={len(r)}")

    out = kelly_search(r, L_max=float(args.L_max), n_grid=int(args.grid))
    L = out["L_star"]

    print("=== Kelly (max E[log(1+L*r)]) ===")
    print(f"n_trades: {len(r)}")
    print(f"r_mean:   {out['r_mean']:.6f}  ({out['r_mean']*100:.4f}%)")
    print(f"r_std:    {out['r_std']:.6f}  ({out['r_std']*100:.4f}%)")
    print(f"r_min:    {out['r_min']:.6f}  ({out['r_min']*100:.4f}%)")
    print(f"L_hi_used (bankruptcy bound): {out['L_hi_used']:.4f}")
    print(f"Kelly L*: {L:.4f}")
    print(f"0.5 Kelly: {0.5*L:.4f}")
    print(f"0.25 Kelly: {0.25*L:.4f}")

    # sanity: growth at those
    for name, Lx in [("Kelly", L), ("Half", 0.5*L), ("Quarter", 0.25*L)]:
        print(f"g({name}) = {growth(Lx, r):.8f}")


if __name__ == "__main__":
    main()

