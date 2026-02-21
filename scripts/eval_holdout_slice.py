# scripts/eval_holdout_slice.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def prod_return(pnl: pd.Series) -> float:
    pnl = pd.to_numeric(pnl, errors="coerce").dropna()
    if len(pnl) == 0:
        return 0.0
    return float((1.0 + pnl).prod() - 1.0)


def max_drawdown_from_equity(equity: pd.Series) -> float:
    e = pd.to_numeric(equity, errors="coerce").dropna()
    if len(e) == 0:
        return 0.0
    peak = e.cummax()
    dd = (e / peak) - 1.0
    return float(dd.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="data/runs/<RUN_ID>")
    ap.add_argument("--start", required=True, help="UTC start ts, e.g. 2025-05-01")
    args = ap.parse_args()

    run = Path(args.run)
    start = pd.to_datetime(args.start, utc=True)

    trades_p = run / "trades.parquet"
    eq_p = run / "equity.parquet"
    stats_p = run / "stats.json"

    tr = pd.read_parquet(trades_p).copy()
    tr["entry_ts"] = pd.to_datetime(tr["entry_ts"], utc=True)
    tr["exit_ts"] = pd.to_datetime(tr["exit_ts"], utc=True)

    tr_h = tr[tr["entry_ts"] >= start].copy()

    ret_all = prod_return(tr["pnl_pct"])
    ret_h = prod_return(tr_h["pnl_pct"])

    print("RUN:", run.name)
    print("TRADES all:", len(tr), "range:", tr["entry_ts"].min(), "->", tr["entry_ts"].max(), "return_pct:", 100 * ret_all)
    print("TRADES holdout:", len(tr_h), "start>=", start, "range:", (tr_h["entry_ts"].min() if len(tr_h) else None),
          "->", (tr_h["entry_ts"].max() if len(tr_h) else None), "return_pct:", 100 * ret_h)

    if eq_p.exists():
        eq = pd.read_parquet(eq_p).copy()
        # try to find time column
        if "ts" in eq.columns:
            eq["ts"] = pd.to_datetime(eq["ts"], utc=True)
            eq = eq.sort_values("ts").set_index("ts")
        else:
            eq.index = pd.to_datetime(eq.index, utc=True)
            eq = eq.sort_index()

        # find equity-like column
        eq_col = None
        for c in ["equity", "equity_pct", "equity_curve", "balance"]:
            if c in eq.columns:
                eq_col = c
                break
        if eq_col is None:
            # fallback: pick first numeric col
            num_cols = [c for c in eq.columns if pd.api.types.is_numeric_dtype(eq[c])]
            eq_col = num_cols[0] if num_cols else None

        if eq_col is not None:
            eq_all = eq[eq_col]
            eq_h = eq_all.loc[start:]
            dd_all = max_drawdown_from_equity(eq_all)
            dd_h = max_drawdown_from_equity(eq_h)

            print("EQUITY col:", eq_col)
            print("DD all:", 100 * dd_all)
            print("DD holdout:", 100 * dd_h)
        else:
            print("No numeric equity column found in equity.parquet")
    else:
        print("No equity.parquet found")

    if stats_p.exists():
        try:
            stats = pd.read_json(stats_p, typ="series")
            print("fee_bps:", float(stats.get("fee_bps", np.nan)), "trades_total:", int(stats.get("trades", -1)))
        except Exception:
            pass


if __name__ == "__main__":
    main()
