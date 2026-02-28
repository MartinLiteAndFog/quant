# scripts/pc_calibration_report.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def find_pcol(df: pd.DataFrame, h: int) -> str:
    for c in (f"p_up_{h}", f"pup_{h}"):
        if c in df.columns:
            return c
    raise ValueError(f"Missing p column for horizon={h} (expected p_up_{h} or pup_{h})")


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def compute_realized(df: pd.DataFrame, h: int, close_col: str) -> tuple[np.ndarray, np.ndarray]:
    close = df[close_col].astype(float).to_numpy()
    y = np.full(len(close), np.nan, dtype=float)
    if len(close) > h:
        y[:-h] = np.log(close[h:] / close[:-h])
    y_up = (y > 0).astype(float)
    return y_up, y


def calibration_bins(p: np.ndarray, y_up: np.ndarray, bins: np.ndarray, min_n: int):
    idx = np.digitize(p, bins) - 1
    rows = []
    for i in range(len(bins) - 1):
        m = idx == i
        n = int(m.sum())
        if n >= min_n:
            rows.append(
                (float(bins[i]), float(bins[i + 1]), n, float(p[m].mean()), float(y_up[m].mean()))
            )
    return rows


def tail_stats(p: np.ndarray, y_up: np.ndarray, thresholds=(0.90, 0.95, 0.99)):
    pred_up = (p > 0.5).astype(float)
    out = {}
    for th in thresholds:
        m = p >= th
        out[f"frac_p>={th:.2f}"] = float(m.mean())
        out[f"acc_p>={th:.2f}"] = float((pred_up[m] == y_up[m]).mean()) if m.any() else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description="Calibration report from predictions.parquet")
    ap.add_argument("--predictions", required=True, help="Path to predictions.parquet")
    ap.add_argument("--close-col", default="close")
    ap.add_argument("--horizons", default="1,5,15,60")
    ap.add_argument("--min-n", type=int, default=1000)
    ap.add_argument("--bins", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99,1.01")
    ap.add_argument("--out", default="", help="Optional CSV output path for bin table")
    args = ap.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    df = pd.read_parquet(pred_path)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if args.close_col not in df.columns:
        raise ValueError(f"Missing close column '{args.close_col}' in predictions.parquet")

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    bins = np.array([float(x.strip()) for x in args.bins.split(",") if x.strip()], dtype=float)

    all_rows = []
    print(f"USING: {pred_path} rows={len(df)}")

    for h in horizons:
        pcol = find_pcol(df, h)
        p = df[pcol].astype(float).to_numpy()

        y_up, y = compute_realized(df, h, close_col=args.close_col)

        m = np.isfinite(p) & np.isfinite(y)
        p = p[m]
        y_up = y_up[m]
        y = y[m]

        if len(p) == 0:
            print(f"\n=== h={h} === no valid rows")
            continue

        acc = float(((p > 0.5).astype(float) == y_up).mean())
        bs = brier_score(p, y_up)
        ll = log_loss(p, y_up)
        tails = tail_stats(p, y_up)

        print(f"\n=== h={h} === n={len(p)} acc@0.5={acc:.4f} brier={bs:.6f} logloss={ll:.6f}")
        for k, v in tails.items():
            print(f"  {k}: {v:.6f}" if "frac" in k else f"  {k}: {v:.4f}")

        rows = calibration_bins(p, y_up, bins=bins, min_n=args.min_n)
        print("  bins: lo-hi   n      p_mean  up_rate")
        for lo, hi, n, pmean, upr in rows:
            print(f"  {lo:.2f}-{hi:.2f}  {n:7d}  {pmean:.3f}  {upr:.3f}")
            all_rows.append({"h": h, "bin_lo": lo, "bin_hi": hi, "n": n, "p_mean": pmean, "up_rate": upr})

        mean_y_hi = float(np.mean(y[p >= 0.90])) if np.any(p >= 0.90) else float("nan")
        mean_y_lo = float(np.mean(y[p <= 0.10])) if np.any(p <= 0.10) else float("nan")
        print(f"  mean_y_when_p>=0.90: {mean_y_hi:.6f}")
        print(f"  mean_y_when_p<=0.10: {mean_y_lo:.6f}")

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(outp, index=False)
        print(f"\nWROTE: {outp}")


if __name__ == "__main__":
    main()