# scripts/leverage_sweep.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load_returns(trades_path: str, r_col: str) -> tuple[pd.Series, pd.Series | None]:
    p = Path(trades_path)
    if not p.exists():
        raise FileNotFoundError(f"trades file not found: {trades_path}")

    df = pd.read_parquet(p)

    if r_col not in df.columns:
        raise ValueError(f"Missing return column '{r_col}'. Available: {list(df.columns)}")

    r = pd.to_numeric(df[r_col], errors="coerce").dropna().astype(float)

    # optional timestamps to estimate CAGR
    ts = None
    for cand in ["exit_ts", "ts", "exit_time"]:
        if cand in df.columns:
            t = pd.to_datetime(df[cand], utc=True, errors="coerce")
            t = t.loc[r.index] if len(t) == len(df) else t
            ts = t
            break

    return r.reset_index(drop=True), (None if ts is None else ts.reset_index(drop=True))


def metrics_for_L(r: np.ndarray, L: float, equity0: float = 10_000.0) -> dict:
    m = 1.0 + L * r
    min_m = float(np.min(m)) if len(m) else 1.0

    if min_m <= 0.0:
        # bankruptcy / invalid under log utility
        return {
            "L": float(L),
            "final": np.nan,
            "ret_pct": np.nan,
            "maxDD_pct": np.nan,
            "min_1pLr": float(min_m),
            "g_meanlog": np.nan,
        }

    eq = equity0 * np.cumprod(m)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    maxdd_pct = float(np.min(dd) * 100.0)

    g = float(np.mean(np.log(m)))  # per-trade mean log-growth

    return {
        "L": float(L),
        "final": float(eq[-1]),
        "ret_pct": float(eq[-1] / equity0 - 1.0) * 100.0,
        "maxDD_pct": maxdd_pct,
        "min_1pLr": float(min_m),
        "g_meanlog": g,
    }


def add_cagr_calmar(rows: list[dict], ts: pd.Series | None, equity0: float = 10_000.0) -> list[dict]:
    if ts is None or ts.isna().all() or len(ts) < 2:
        # no time info -> keep blank
        for d in rows:
            d["years"] = np.nan
            d["cagr_pct"] = np.nan
            d["calmar"] = np.nan
        return rows

    t0 = pd.to_datetime(ts.iloc[0], utc=True)
    t1 = pd.to_datetime(ts.iloc[-1], utc=True)
    years = float((t1 - t0).total_seconds() / (365.25 * 24 * 3600))
    years = max(years, 1e-9)

    for d in rows:
        final = d.get("final", np.nan)
        maxdd = d.get("maxDD_pct", np.nan)
        if not np.isfinite(final):
            d["years"] = years
            d["cagr_pct"] = np.nan
            d["calmar"] = np.nan
            continue

        cagr = (final / equity0) ** (1.0 / years) - 1.0
        d["years"] = years
        d["cagr_pct"] = float(cagr * 100.0)
        d["calmar"] = float(cagr / max(1e-12, abs(maxdd) / 100.0)) if np.isfinite(maxdd) else np.nan

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades-real", required=True, help="Path to trades_real.parquet")
    ap.add_argument("--r-col", default="pnl_pct_real", help="Return column to use (default: pnl_pct_real)")
    ap.add_argument("--equity0", type=float, default=10_000.0)
    ap.add_argument("--Ls", type=str, default="0.5,1.0,1.25,1.5,1.92,2.0", help="Comma-separated leverage values")
    ap.add_argument("--max-L", type=float, default=None, help="If set, sweep L from 0 to max-L in steps")
    ap.add_argument("--step", type=float, default=0.1, help="Step for sweep when --max-L is used")
    args = ap.parse_args()

    r, ts = load_returns(args.trades_real, args.r_col)
    r_np = r.to_numpy(dtype=float)

    if args.max_L is not None:
        Ls = np.arange(0.0, float(args.max_L) + 1e-12, float(args.step))
    else:
        Ls = np.array([float(x.strip()) for x in str(args.Ls).split(",") if x.strip() != ""])

    rows = [metrics_for_L(r_np, float(L), equity0=float(args.equity0)) for L in Ls]
    rows = add_cagr_calmar(rows, ts, equity0=float(args.equity0))

    out = pd.DataFrame(rows)

    # nice formatting order
    cols = [
        "L",
        "final",
        "ret_pct",
        "maxDD_pct",
        "min_1pLr",
        "g_meanlog",
        "years",
        "cagr_pct",
        "calmar",
    ]
    out = out[[c for c in cols if c in out.columns]]

    # show best by mean log-growth (Kelly objective)
    if "g_meanlog" in out.columns:
        best_idx = out["g_meanlog"].astype(float).idxmax()
        best = out.loc[best_idx].to_dict()
        print("\n=== BEST (by mean log-growth g) ===")
        print(
            f"L={best['L']:.4f}  final={best['final']:.2f}  ret={best['ret_pct']:.2f}%  "
            f"maxDD={best['maxDD_pct']:.2f}%  min(1+Lr)={best['min_1pLr']:.6f}  g={best['g_meanlog']:.8f}"
        )
        if np.isfinite(best.get("cagr_pct", np.nan)):
            print(f"years={best['years']:.2f}  CAGR={best['cagr_pct']:.2f}%  Calmar={best['calmar']:.2f}")

    # print table
    with pd.option_context("display.max_rows", 500, "display.width", 200, "display.float_format", lambda x: f"{x:.6f}"):
        print("\n=== LEVERAGE TABLE ===")
        print(out.to_string(index=False))

    # also: best by Calmar if available
    if "calmar" in out.columns and out["calmar"].notna().any():
        bestc = out.loc[out["calmar"].astype(float).idxmax()].to_dict()
        print("\n=== BEST (by Calmar) ===")
        print(
            f"L={bestc['L']:.4f}  final={bestc['final']:.2f}  ret={bestc['ret_pct']:.2f}%  "
            f"maxDD={bestc['maxDD_pct']:.2f}%  CAGR={bestc['cagr_pct']:.2f}%  Calmar={bestc['calmar']:.2f}"
        )


if __name__ == "__main__":
    main()
