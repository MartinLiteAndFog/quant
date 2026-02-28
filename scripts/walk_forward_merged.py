#!/usr/bin/env python3
"""
Quantitative analysis of merged strategies (ON + OFF) and walk-forward simulation.

Modes:
  --analysis-only:      Load trades from --run-dir, compute stats (full + by strategy).
  --walk-forward:       Run rolling train/test folds, backtest each test window (fixed params).
  --walk-forward-fit:   Same but fit params on train (grid search), evaluate on test with best params.

Reuses fold logic from build_gate_walkforward (make_folds).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(idx: pd.DatetimeIndex, train_days: int, test_days: int, step_days: int) -> List[Fold]:
    """Rolling train/test folds (same logic as build_gate_walkforward)."""
    idx = pd.DatetimeIndex(sorted(idx.unique()))
    start = idx.min().normalize()
    end = idx.max().normalize()
    folds: List[Fold] = []
    t0 = start + pd.Timedelta(days=train_days)
    while True:
        test_start = t0
        test_end = test_start + pd.Timedelta(days=test_days - 1)
        train_end = test_start - pd.Timedelta(days=1)
        train_start = train_end - pd.Timedelta(days=train_days - 1)
        if test_end > end:
            break
        folds.append(Fold(train_start, train_end, test_start, test_end))
        t0 = t0 + pd.Timedelta(days=step_days)
    return folds


def _ensure_utc(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, utc=True, errors="coerce")
    return s


def trade_metrics(trades: pd.DataFrame, fee_bps: float = 15.0, r_col: str = "pnl_pct") -> dict[str, Any]:
    """Compute quantitative stats from a trades DataFrame (optional strategy column)."""
    if trades is None or len(trades) == 0:
        return {
            "n_trades": 0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ann": np.nan,
            "calmar": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "avg_trade_pct": np.nan,
            "span_years": np.nan,
        }
    t = trades.copy()
    if r_col not in t.columns:
        r_col = "pnl_pct" if "pnl_pct" in t.columns else None
    if r_col is None:
        return {"n_trades": len(t), "total_return_pct": np.nan, "max_drawdown_pct": np.nan}
    r = pd.to_numeric(t[r_col], errors="coerce").dropna().astype(float).values
    if len(r) == 0:
        return {"n_trades": 0, "total_return_pct": 0.0, "max_drawdown_pct": 0.0}
    # Net: if strategy "off" we subtract fee (caller can pass pre-adjusted pnl or we assume merged format)
    eq = np.cumprod(1.0 + r)
    total_return_pct = (float(eq[-1]) - 1.0) * 100.0
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak - 1.0) * 100.0
    max_drawdown_pct = float(dd.min())
    n = len(r)
    win_rate = float(np.mean(r > 0)) if n else np.nan
    gross_profit = np.sum(r[r > 0])
    gross_loss = np.abs(np.sum(r[r < 0]))
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else np.nan)
    avg_trade_pct = float(np.mean(r)) * 100.0
    # Sharpe (annualized): mean(r)/std(r) * sqrt(trades_per_year). Approximate trades_per_year from span.
    span_years = np.nan
    if "exit_ts" in t.columns:
        ts = _ensure_utc(t["exit_ts"]).dropna()
        if len(ts) >= 2:
            span_days = (ts.max() - ts.min()).total_seconds() / (24 * 3600)
            span_years = span_days / 365.25
    if span_years is np.nan or span_years <= 0:
        span_years = n / 400.0  # assume ~400 trades/year if no ts
    trades_per_year = n / max(span_years, 1e-6)
    std_r = np.std(r, ddof=1) if n > 1 else 0.0
    sharpe_ann = (float(np.mean(r)) / std_r * np.sqrt(trades_per_year)) if std_r > 0 else np.nan
    calmar = (total_return_pct / 100.0) / (max(abs(max_drawdown_pct) / 100.0, 1e-12)) if max_drawdown_pct != 0 else np.nan
    return {
        "n_trades": n,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ann": float(sharpe_ann) if np.isfinite(sharpe_ann) else np.nan,
        "calmar": float(calmar) if np.isfinite(calmar) else np.nan,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade_pct": avg_trade_pct,
        "span_years": span_years,
    }


def analysis_report(run_dir: Path, fee_bps: float = 15.0) -> pd.DataFrame:
    """Load trades from run_dir and produce metrics (full + by strategy)."""
    trades_path = run_dir / "trades.parquet"
    if not trades_path.exists():
        raise FileNotFoundError(f"No trades.parquet in {run_dir}")
    t = pd.read_parquet(trades_path)
    fee_rt = fee_bps / 10_000.0
    # For OFF trades subtract fee (merged runner stores gross for TP2)
    def net_pnl(row):
        r = float(row["pnl_pct"])
        if str(row.get("strategy", "")).strip().lower() == "off":
            r -= fee_rt
        return r
    if "strategy" in t.columns:
        t = t.copy()
        t["pnl_net"] = t.apply(net_pnl, axis=1)
    else:
        t["pnl_net"] = t["pnl_pct"]
    rows = []
    # Full
    m = trade_metrics(t, fee_bps=fee_bps, r_col="pnl_net")
    rows.append({"segment": "full", **m})
    # By strategy
    if "strategy" in t.columns:
        for strat in ["on", "off"]:
            sub = t[t["strategy"].astype(str).str.strip().str.lower() == strat]
            m = trade_metrics(sub, fee_bps=fee_bps, r_col="pnl_net")
            rows.append({"segment": strat, **m})
    return pd.DataFrame(rows)


def _run_merged_backtest(
    parquet_path: str,
    signals_jsonl: str,
    regime_csv: str,
    regime_col: str,
    regime_csv_off: str,
    regime_col_off: str,
    fee_bps: float,
    run_id: str,
    extra_args: List[str],
    timeout: int = 300,
) -> Path:
    """Run merged backtest; return path to run dir (data/runs/<run_id>). extra_args e.g. ['--tp1-pct', '0.02']."""
    cmd = [
        sys.executable, "-m", "quant.backtest.renko_runner",
        "--parquet", str(parquet_path),
        "--signals-jsonl", signals_jsonl,
        "--regime-csv", regime_csv,
        "--regime-col", regime_col,
        "--regime-csv-off", regime_csv_off,
        "--regime-col-off", regime_col_off,
        "--fee-bps", str(fee_bps),
        "--run-id", run_id,
    ] + list(extra_args)
    subprocess.run(cmd, check=True, timeout=timeout)
    return Path("data/runs") / run_id


def _trades_net_and_metrics(run_dir: Path, fee_bps: float) -> Tuple[pd.DataFrame, dict]:
    """Load trades from run_dir, add pnl_net (fee off for OFF), return (trades with pnl_net, metrics dict)."""
    t = pd.read_parquet(run_dir / "trades.parquet")
    fee_rt = fee_bps / 10_000.0
    if "strategy" in t.columns:
        t = t.copy()
        t["pnl_net"] = t.apply(
            lambda row: float(row["pnl_pct"]) - (fee_rt if str(row.get("strategy", "")).strip().lower() == "off" else 0),
            axis=1,
        )
    else:
        t["pnl_net"] = t["pnl_pct"]
    m = trade_metrics(t, fee_bps=fee_bps, r_col="pnl_net")
    return t, m


def _parse_grid(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def run_walk_forward_fit(
    parquet_path: str,
    signals_jsonl: str,
    regime_csv: str,
    regime_col: str,
    regime_csv_off: str,
    regime_col_off: str,
    train_days: int,
    test_days: int,
    step_days: int,
    fee_bps: float,
    out_dir: Path,
    grid_tp1: List[float],
    grid_tp2: List[float],
    grid_ttp: List[float],
    fit_objective: str = "sharpe_ann",
) -> pd.DataFrame:
    """Walk-forward with parameter fit on train: grid search, best by fit_objective, then OOS on test."""
    bars = pd.read_parquet(parquet_path)
    bars["ts"] = pd.to_datetime(bars["ts"], utc=True, errors="coerce")
    bars = bars.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    idx = pd.DatetimeIndex(bars["ts"]).normalize().unique()
    folds = make_folds(idx, train_days, test_days, step_days)
    if not folds:
        raise ValueError("No folds produced.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    for k, f in enumerate(folds):
        mask_train = (bars["ts"] >= f.train_start) & (bars["ts"] <= f.train_end + pd.Timedelta(days=1))
        mask_test = (bars["ts"] >= f.test_start) & (bars["ts"] <= f.test_end + pd.Timedelta(days=1))
        bar_train = bars.loc[mask_train].copy()
        bar_test = bars.loc[mask_test].copy()
        if len(bar_train) < 100 or len(bar_test) < 50:
            continue
        with tempfile.TemporaryDirectory(prefix="wf_fit_") as tmp:
            tmp = Path(tmp)
            train_parquet = tmp / "bars_train.parquet"
            test_parquet = tmp / "bars_test.parquet"
            bar_train.to_parquet(train_parquet, index=False)
            bar_test.to_parquet(test_parquet, index=False)
            best_score = -np.inf
            best_extra: List[str] = []
            for tp1 in grid_tp1:
                for tp2 in grid_tp2:
                    for ttp in grid_ttp:
                        if tp2 <= tp1:
                            continue
                        run_id_train = f"wf_fit_f{k}_tp1{tp1}_tp2{tp2}_ttp{ttp}"
                        extra = [
                            "--tp1-pct", str(tp1),
                            "--tp2-pct", str(tp2),
                            "--ttp-trail-pct", str(ttp),
                        ]
                        try:
                            _run_merged_backtest(
                                str(train_parquet),
                                signals_jsonl, regime_csv, regime_col,
                                regime_csv_off, regime_col_off,
                                fee_bps, run_id_train, extra, timeout=180,
                            )
                            run_dir = Path("data/runs") / run_id_train
                            _, m = _trades_net_and_metrics(run_dir, fee_bps)
                            score = m.get(fit_objective, np.nan)
                            if np.isfinite(score) and score > best_score:
                                best_score = score
                                best_extra = extra
                        except Exception:
                            continue
            if not best_extra:
                continue
            run_id_test = f"wf_fit_f{k}_test"
            _run_merged_backtest(
                str(test_parquet),
                signals_jsonl, regime_csv, regime_col,
                regime_csv_off, regime_col_off,
                fee_bps, run_id_test, best_extra, timeout=300,
            )
            run_dir_test = Path("data/runs") / run_id_test
            if (run_dir_test / "trades.parquet").exists():
                _, m = _trades_net_and_metrics(run_dir_test, fee_bps)
                tp1_val = best_extra[best_extra.index("--tp1-pct") + 1] if "--tp1-pct" in best_extra else ""
                tp2_val = best_extra[best_extra.index("--tp2-pct") + 1] if "--tp2-pct" in best_extra else ""
                ttp_val = best_extra[best_extra.index("--ttp-trail-pct") + 1] if "--ttp-trail-pct" in best_extra else ""
                fold_rows.append({
                    "fold": k,
                    "test_start": str(f.test_start),
                    "test_end": str(f.test_end),
                    "train_start": str(f.train_start),
                    "train_end": str(f.train_end),
                    "best_tp1_pct": tp1_val,
                    "best_tp2_pct": tp2_val,
                    "best_ttp_pct": ttp_val,
                    "train_best_score": best_score,
                    **m,
                })
    return pd.DataFrame(fold_rows)


def run_walk_forward(
    parquet_path: str,
    signals_jsonl: str,
    regime_csv: str,
    regime_col: str,
    regime_csv_off: str,
    regime_col_off: str,
    train_days: int,
    test_days: int,
    step_days: int,
    fee_bps: float,
    out_dir: Path,
) -> pd.DataFrame:
    """Run backtest on each test fold; return DataFrame of OOS metrics per fold."""
    bars = pd.read_parquet(parquet_path)
    bars["ts"] = pd.to_datetime(bars["ts"], utc=True, errors="coerce")
    bars = bars.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    idx = pd.DatetimeIndex(bars["ts"]).normalize().unique()
    folds = make_folds(idx, train_days, test_days, step_days)
    if not folds:
        raise ValueError("No folds produced. Check date range vs train_days, test_days, step_days.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    for k, f in enumerate(folds):
        # Slice bars to test window (inclusive)
        mask = (bars["ts"] >= f.test_start) & (bars["ts"] <= f.test_end + pd.Timedelta(days=1))
        bar_test = bars.loc[mask].copy()
        if len(bar_test) < 50:
            continue
        with tempfile.TemporaryDirectory(prefix="wf_merged_") as tmp:
            tmp = Path(tmp)
            test_parquet = tmp / "bars_test.parquet"
            bar_test.to_parquet(test_parquet, index=False)
            run_id = f"wf_fold_{k}"
            cmd = [
                sys.executable, "-m", "quant.backtest.renko_runner",
                "--parquet", str(test_parquet),
                "--signals-jsonl", signals_jsonl,
                "--regime-csv", regime_csv,
                "--regime-col", regime_col,
                "--regime-csv-off", regime_csv_off,
                "--regime-col-off", regime_col_off,
                "--fee-bps", str(fee_bps),
                "--run-id", run_id,
            ]
            subprocess.run(cmd, check=True, timeout=300)
            run_fold_dir = Path("data/runs") / run_id
            if (run_fold_dir / "trades.parquet").exists():
                t = pd.read_parquet(run_fold_dir / "trades.parquet")
                fee_rt = fee_bps / 10_000.0
                if "strategy" in t.columns:
                    t = t.copy()
                    t["pnl_net"] = t.apply(
                        lambda row: float(row["pnl_pct"]) - (fee_rt if str(row.get("strategy", "")).strip().lower() == "off" else 0),
                        axis=1,
                    )
                else:
                    t["pnl_net"] = t["pnl_pct"]
                m = trade_metrics(t, fee_bps=fee_bps, r_col="pnl_net")
                fold_rows.append({
                    "fold": k,
                    "test_start": str(f.test_start),
                    "test_end": str(f.test_end),
                    "train_start": str(f.train_start),
                    "train_end": str(f.train_end),
                    **m,
                })
    return pd.DataFrame(fold_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantitative analysis and walk-forward for merged strategies")
    ap.add_argument("--run-dir", type=str, default=None, help="Run dir with trades.parquet (for --analysis-only or baseline)")
    ap.add_argument("--analysis-only", action="store_true", help="Only run quantitative analysis on --run-dir")
    ap.add_argument("--walk-forward", action="store_true", help="Run walk-forward simulation")
    ap.add_argument("--parquet", type=str, default=None)
    ap.add_argument("--signals-jsonl", type=str, default=None)
    ap.add_argument("--regime-csv", type=str, default=None)
    ap.add_argument("--regime-col", type=str, default="gate_on_2of3")
    ap.add_argument("--regime-csv-off", type=str, default=None)
    ap.add_argument("--regime-col-off", type=str, default="gate_off_2of3")
    ap.add_argument("--fee-bps", type=float, default=15.0)
    ap.add_argument("--train-days", type=int, default=252, help="Train window (days) per fold")
    ap.add_argument("--test-days", type=int, default=63, help="Test window (days) per fold")
    ap.add_argument("--step-days", type=int, default=21, help="Step forward (days) between folds")
    ap.add_argument("--out-dir", type=str, default="data/runs/walk_forward_merged")
    ap.add_argument("--out-csv", type=str, default=None, help="Write fold metrics to this CSV")
    # Walk-forward with fit (grid search on train)
    ap.add_argument("--walk-forward-fit", action="store_true", help="WF with param fit on train (grid search), OOS on test")
    ap.add_argument("--grid-tp1", type=str, default="0.01,0.015,0.02", help="Comma-separated tp1_pct values for grid")
    ap.add_argument("--grid-tp2", type=str, default="0.025,0.03,0.04", help="Comma-separated tp2_pct values for grid")
    ap.add_argument("--grid-ttp", type=str, default="0.01,0.012", help="Comma-separated ttp_trail_pct values for grid")
    ap.add_argument("--fit-objective", type=str, default="sharpe_ann", choices=["sharpe_ann", "total_return_pct", "calmar"], help="Metric to maximize on train")
    args = ap.parse_args()

    if args.analysis_only:
        if not args.run_dir:
            raise SystemExit("--analysis-only requires --run-dir")
        run_dir = Path(args.run_dir)
        df = analysis_report(run_dir, fee_bps=args.fee_bps)
        print("=== Quantitative analysis (merged strategies) ===")
        print(df.to_string(index=False))
        return

    if args.walk_forward_fit:
        if not all([args.parquet, args.signals_jsonl, args.regime_csv, args.regime_csv_off]):
            raise SystemExit("--walk-forward-fit requires --parquet, --signals-jsonl, --regime-csv, --regime-csv-off")
        grid_tp1 = _parse_grid(args.grid_tp1)
        grid_tp2 = _parse_grid(args.grid_tp2)
        grid_ttp = _parse_grid(args.grid_ttp)
        if not grid_tp1 or not grid_tp2 or not grid_ttp:
            raise SystemExit("--grid-tp1, --grid-tp2, --grid-ttp must each have at least one value")
        out_dir = Path(args.out_dir)
        df_folds = run_walk_forward_fit(
            parquet_path=args.parquet,
            signals_jsonl=args.signals_jsonl,
            regime_csv=args.regime_csv,
            regime_col=args.regime_col,
            regime_csv_off=args.regime_csv_off,
            regime_col_off=args.regime_col_off,
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            fee_bps=args.fee_bps,
            out_dir=out_dir,
            grid_tp1=grid_tp1,
            grid_tp2=grid_tp2,
            grid_ttp=grid_ttp,
            fit_objective=args.fit_objective,
        )
        if args.out_csv:
            Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
            df_folds.to_csv(args.out_csv, index=False)
            print(f"Wrote {args.out_csv}")
        print("=== Walk-forward FIT OOS metrics (per fold, params from train grid) ===")
        print(df_folds.to_string(index=False))
        if len(df_folds):
            print("\n=== OOS summary (fitted) ===")
            print(f"  Folds: {len(df_folds)}")
            print(f"  Mean total_return_pct: {df_folds['total_return_pct'].mean():.2f}")
            print(f"  Mean max_drawdown_pct:  {df_folds['max_drawdown_pct'].mean():.2f}")
            print(f"  Mean sharpe_ann:         {df_folds['sharpe_ann'].mean():.4f}")
            print(f"  Mean win_rate:           {df_folds['win_rate'].mean()*100:.2f}%")
        return

    if args.walk_forward:
        if not all([args.parquet, args.signals_jsonl, args.regime_csv, args.regime_csv_off]):
            raise SystemExit("--walk-forward requires --parquet, --signals-jsonl, --regime-csv, --regime-csv-off")
        out_dir = Path(args.out_dir)
        df_folds = run_walk_forward(
            parquet_path=args.parquet,
            signals_jsonl=args.signals_jsonl,
            regime_csv=args.regime_csv,
            regime_col=args.regime_col,
            regime_csv_off=args.regime_csv_off,
            regime_col_off=args.regime_col_off,
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            fee_bps=args.fee_bps,
            out_dir=out_dir,
        )
        if args.out_csv:
            Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
            df_folds.to_csv(args.out_csv, index=False)
            print(f"Wrote {args.out_csv}")
        print("=== Walk-forward OOS metrics (per fold) ===")
        print(df_folds.to_string(index=False))
        if len(df_folds):
            print("\n=== OOS summary ===")
            print(f"  Folds: {len(df_folds)}")
            print(f"  Mean total_return_pct: {df_folds['total_return_pct'].mean():.2f}")
            print(f"  Mean max_drawdown_pct:  {df_folds['max_drawdown_pct'].mean():.2f}")
            print(f"  Mean sharpe_ann:         {df_folds['sharpe_ann'].mean():.4f}")
            print(f"  Mean win_rate:           {df_folds['win_rate'].mean()*100:.2f}%")
        return

    # Default: analysis on run_dir if provided
    if args.run_dir:
        run_dir = Path(args.run_dir)
        df = analysis_report(run_dir, fee_bps=args.fee_bps)
        print("=== Quantitative analysis (merged strategies) ===")
        print(df.to_string(index=False))
    else:
        print("Provide --run-dir for analysis, or --walk-forward with parquet/signals/regime args.")


if __name__ == "__main__":
    main()
