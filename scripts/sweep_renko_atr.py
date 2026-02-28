# scripts/sweep_renko_atr.py
"""
Sweep Renko: (1) ATR-adaptive grid (atr_period × k), (2) baseline fixed box 0.1, (3) constant ATR box.
Run backtest for each, collect equity, plot all curves on one chart.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# build_renko_fixed lives in scripts/
_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent


def _parse_floats(s: str) -> List[float]:
    if not s or not str(s).strip():
        return []
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def _parse_ints(s: str) -> List[int]:
    if not s or not str(s).strip():
        return []
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def build_renko(
    ohlc_path: str,
    out_path: str,
    *,
    box: float | None = None,
    box_atr: float | None = None,
    box_atr_constant: float | None = None,
    atr_period: int = 14,
    atr_median: bool = False,
    gap_sec: float = 180.0,
) -> bool:
    cmd = [
        sys.executable,
        str(_SCRIPTS / "build_renko_fixed.py"),
        "--in", ohlc_path,
        "--out", out_path,
        "--gap-sec", str(gap_sec),
    ]
    if box is not None:
        cmd += ["--box", str(box)]
    elif box_atr is not None:
        cmd += ["--box-atr", str(box_atr), "--atr-period", str(atr_period)]
    elif box_atr_constant is not None:
        cmd += ["--box-atr-constant", str(box_atr_constant), "--atr-period", str(atr_period)]
        if atr_median:
            cmd += ["--atr-median"]
    else:
        return False
    rc = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return rc == 0


def run_backtest(
    parquet: str,
    signals_jsonl: str,
    run_id: str,
    logs_dir: Path,
    *,
    fee_bps: float = 4.0,
    regime_csv: str | None = None,
    regime_csv_off: str | None = None,
    regime_col: str = "gate_on_2of3",
    regime_col_off: str = "gate_off_2of3",
    fills_parquet: str | None = None,
    fill_col: str = "close",
    extra_args: List[str] | None = None,
) -> int:
    cmd = [
        sys.executable, "-m", "quant.backtest.renko_runner",
        "--parquet", parquet,
        "--signals-jsonl", signals_jsonl,
        "--fee-bps", str(fee_bps),
        "--run-id", run_id,
    ]
    if regime_csv:
        cmd += ["--regime-csv", regime_csv, "--regime-col", regime_col]
    if regime_csv_off:
        cmd += ["--regime-csv-off", regime_csv_off, "--regime-col-off", regime_col_off]
    if fills_parquet:
        cmd += ["--fills-parquet", fills_parquet, "--fill-col", fill_col]
    if extra_args:
        cmd += extra_args
    log_path = logs_dir / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n\n")
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(_REPO))
    return rc


def load_equity(run_id: str, use_real: bool = False) -> pd.DataFrame | None:
    run_dir = _REPO / "data" / "runs" / run_id
    name = "equity_real.parquet" if use_real else "equity.parquet"
    p = run_dir / name
    if not p.exists():
        return None
    eq = pd.read_parquet(p)
    if "ts" not in eq.columns or "equity" not in eq.columns:
        return None
    eq["ts"] = pd.to_datetime(eq["ts"], utc=True)
    eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
    eq = eq.dropna(subset=["equity"]).sort_values("ts").reset_index(drop=True)
    return eq


def _run_id_to_label(run_id: str, prefix: str) -> str:
    """e.g. RENKO_ATR_baseline_box0p1 -> box=0.1 (baseline), RENKO_ATR_atr_adaptive_p14_k1p0 -> ATR adaptive k=1.0 p=14"""
    s = run_id[len(prefix) :].lstrip("_")
    if s.startswith("baseline_box"):
        box = s.replace("baseline_box", "").replace("p", ".")
        return f"box={box} (baseline)"
    if s.startswith("atr_adaptive_"):
        rest = s.replace("atr_adaptive_", "")
        # p7_k1p0 -> p=7 k=1.0
        parts = rest.split("_")
        p = parts[0].replace("p", "")
        k = parts[1].replace("k", "").replace("p", ".")
        return f"ATR adaptive k={k} p={p}"
    if s.startswith("atr_const_"):
        rest = s.replace("atr_const_", "")
        parts = rest.split("_")
        k = parts[0].replace("k", "").replace("p", ".") if parts else "?"
        p = parts[1].replace("p", "") if len(parts) > 1 else "?"
        return f"ATR constant k={k} p={p}"
    return run_id


def _plot_existing_runs(run_prefix: str, out_path: str, use_real: bool = False) -> None:
    """Load equity from data/runs/<prefix>_*/ and plot."""
    runs_dir = _REPO / "data" / "runs"
    name = "equity_real.parquet" if use_real else "equity.parquet"
    equity_series: dict[str, pd.Series] = {}
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir() or not (d.name == run_prefix or d.name.startswith(run_prefix + "_")):
            continue
        p = d / name
        if not p.exists():
            continue
        eq = load_equity(d.name, use_real=use_real)
        if eq is not None and len(eq):
            label = _run_id_to_label(d.name, run_prefix)
            equity_series[label] = eq.set_index("ts")["equity"]
    if not equity_series:
        print(f"No runs found with prefix {run_prefix} and {name}")
        return
    _, aligned = align_equity_curves(equity_series)
    if aligned.empty:
        print("Aligned equity empty")
        return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 7))
    for label in aligned.columns:
        ax.plot(aligned.index, aligned[label].values, label=label, alpha=0.85)
    ax.set_ylabel("Equity (norm. 1.0)")
    ax.set_xlabel("Time (UTC)")
    ax.set_title("Renko sweep: baseline + ATR-adaptive + ATR-constant")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    out = Path(out_path)
    if not out.is_absolute():
        out = _REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Wrote", out)


def align_equity_curves(series_dict: dict[str, pd.Series]) -> Tuple[pd.DatetimeIndex, pd.DataFrame]:
    """Union of timestamps, forward-fill each series; normalize to start 1.0."""
    if not series_dict:
        return pd.DatetimeIndex([]), pd.DataFrame()
    all_ts = pd.concat([s.dropna().index.to_series() for s in series_dict.values()]).drop_duplicates().sort_values()
    all_ts = pd.DatetimeIndex(pd.to_datetime(all_ts, utc=True))
    out = pd.DataFrame(index=all_ts)
    for label, ser in series_dict.items():
        s = ser.reindex(all_ts).ffill()
        # normalize to 1.0 at first valid
        first_val = s.dropna()
        if len(first_val):
            s = s / float(first_val.iloc[0])
        out[label] = s
    return all_ts, out


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep Renko (ATR adaptive + baseline + constant), plot equity curves")
    ap.add_argument("--ohlc", default=None, help="OHLC parquet to build Renko from (not needed for --plot-only)")
    ap.add_argument("--signals-jsonl", default=None, help="Signals JSONL for backtest (not needed for --plot-only)")
    ap.add_argument("--out", default="data/plots/sweep_renko_atr_equity.png", help="Output plot path")
    ap.add_argument("--sweep-dir", default=None, help="Dir for temp Renko parquets (default: data/sweeps/renko_atr_<tag>)")
    ap.add_argument("--atr-periods", default="7,14,21", help="Comma-separated ATR periods for adaptive sweep")
    ap.add_argument("--k-grid", default="0.5,1.0,1.5", help="Comma-separated k for adaptive (and one constant)")
    ap.add_argument("--baseline-box", type=float, default=0.1, help="Fixed box baseline (your current choice)")
    ap.add_argument("--constant-atr-k", type=float, default=1.0, help="k for constant ATR run (single run)")
    ap.add_argument("--constant-atr-period", type=int, default=14, help="ATR period for constant run")
    ap.add_argument("--fee-bps", type=float, default=4.0)
    ap.add_argument(
        "--regime-csv",
        default="data/regimes/SOLUSDT_tv5mIMBA_gate2of3_qch0.4_qadx0.6_qer0.3_daily.csv",
        help="Gate ON daily CSV (default: SOLUSDT gate2of3).",
    )
    ap.add_argument(
        "--regime-csv-off",
        default="data/regimes/SOLUSDT_tv5mIMBA_gate2of3_qch0.4_qadx0.6_qer0.3_daily_OFF.csv",
        help="Gate OFF daily CSV for TP2 (default: SOLUSDT _OFF).",
    )
    ap.add_argument("--regime-col", default="gate_on_2of3", help="Column name in regime-csv (default: gate_on_2of3).")
    ap.add_argument("--regime-col-off", default="gate_off_2of3", help="Column name in regime-csv-off (default: gate_off_2of3).")
    ap.add_argument("--fills-parquet", default=None, help="If set, use equity_real for all runs")
    ap.add_argument("--fill-col", default="close")
    ap.add_argument("--run-prefix", default="RENKO_ATR", help="Prefix for run_id (avoid overwriting)")
    ap.add_argument("--plot-only", action="store_true", help="Skip sweep; plot equity from existing runs with this prefix.")
    args = ap.parse_args()

    if args.plot_only:
        _plot_existing_runs(
            run_prefix=args.run_prefix,
            out_path=args.out,
            use_real=bool(args.fills_parquet),
        )
        return

    if not args.ohlc or not args.signals_jsonl:
        ap.error("--ohlc and --signals-jsonl required unless --plot-only")
    atr_periods = _parse_ints(args.atr_periods) or [14]
    k_grid = _parse_floats(args.k_grid) or [1.0]
    use_real = bool(args.fills_parquet)

    from datetime import datetime, timezone
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else _REPO / "data" / "sweeps" / f"renko_atr_{tag}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    renko_dir = sweep_dir / "renko"
    renko_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = sweep_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ohlc_path = str(Path(args.ohlc).resolve() if not Path(args.ohlc).is_absolute() else Path(args.ohlc))
    if not Path(ohlc_path).exists():
        ohlc_path = str(_REPO / args.ohlc)
    if not Path(ohlc_path).exists():
        raise FileNotFoundError(f"OHLC not found: {args.ohlc}")
    sp = Path(args.signals_jsonl)
    signals_path = str(sp.resolve()) if sp.is_absolute() or sp.exists() else str(_REPO / args.signals_jsonl)
    if not Path(signals_path).exists():
        raise FileNotFoundError(f"Signals not found: {args.signals_jsonl}")

    equity_series: dict[str, pd.Series] = {}
    run_ids: List[str] = []

    # 1) Baseline: fixed box 0.1
    baseline_parquet = str(renko_dir / "baseline_box0.1.parquet")
    if build_renko(ohlc_path, baseline_parquet, box=args.baseline_box):
        run_id = f"{args.run_prefix}_baseline_box{args.baseline_box}".replace(".", "p")
        run_ids.append(run_id)
        rc = run_backtest(
            baseline_parquet, signals_path, run_id, logs_dir,
            fee_bps=args.fee_bps, regime_csv=args.regime_csv, regime_csv_off=args.regime_csv_off,
            regime_col=args.regime_col, regime_col_off=args.regime_col_off,
            fills_parquet=args.fills_parquet, fill_col=args.fill_col,
        )
        eq = load_equity(run_id, use_real=use_real)
        if eq is not None and len(eq):
            equity_series[f"box={args.baseline_box} (baseline)"] = eq.set_index("ts")["equity"]
        print(f"Baseline box={args.baseline_box} -> {run_id} rc={rc}")

    # 2) ATR-adaptive sweep (period × k)
    for period in atr_periods:
        for k in k_grid:
            out_name = f"atr_adaptive_p{period}_k{k}".replace(".", "p")
            parquet = str(renko_dir / f"{out_name}.parquet")
            if build_renko(ohlc_path, parquet, box_atr=k, atr_period=period):
                run_id = f"{args.run_prefix}_{out_name}"
                run_ids.append(run_id)
                rc = run_backtest(
                    parquet, signals_path, run_id, logs_dir,
                    fee_bps=args.fee_bps, regime_csv=args.regime_csv, regime_csv_off=args.regime_csv_off,
                    regime_col=args.regime_col, regime_col_off=args.regime_col_off,
                    fills_parquet=args.fills_parquet, fill_col=args.fill_col,
                )
                eq = load_equity(run_id, use_real=use_real)
                if eq is not None and len(eq):
                    equity_series[f"ATR adaptive k={k} p={period}"] = eq.set_index("ts")["equity"]
                print(f"ATR adaptive k={k} p={period} -> {run_id} rc={rc}")

    # 3) Constant ATR (one run)
    const_parquet = str(renko_dir / f"atr_const_k{args.constant_atr_k}_p{args.constant_atr_period}.parquet").replace(".", "p")
    if build_renko(
        ohlc_path, const_parquet,
        box_atr_constant=args.constant_atr_k,
        atr_period=args.constant_atr_period,
    ):
        run_id = f"{args.run_prefix}_atr_const_k{args.constant_atr_k}_p{args.constant_atr_period}".replace(".", "p")
        run_ids.append(run_id)
        rc = run_backtest(
            const_parquet, signals_path, run_id, logs_dir,
            fee_bps=args.fee_bps, regime_csv=args.regime_csv, regime_csv_off=args.regime_csv_off,
            regime_col=args.regime_col, regime_col_off=args.regime_col_off,
            fills_parquet=args.fills_parquet, fill_col=args.fill_col,
        )
        eq = load_equity(run_id, use_real=use_real)
        if eq is not None and len(eq):
            equity_series[f"ATR constant k={args.constant_atr_k} p={args.constant_atr_period}"] = eq.set_index("ts")["equity"]
        print(f"ATR constant k={args.constant_atr_k} p={args.constant_atr_period} -> {run_id} rc={rc}")

    if not equity_series:
        print("No equity data collected. Check logs in", logs_dir)
        return

    # Align and normalize
    _, aligned = align_equity_curves(equity_series)
    if aligned.empty:
        print("Aligned equity empty")
        return

    # Plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 7))
    for label in aligned.columns:
        ax.plot(aligned.index, aligned[label].values, label=label, alpha=0.85)
    ax.set_ylabel("Equity (norm. 1.0)")
    ax.set_xlabel("Time (UTC)")
    ax.set_title("Renko sweep: baseline + ATR-adaptive + ATR-constant")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Wrote", out_path)
    print("Sweep dir", sweep_dir)


if __name__ == "__main__":
    main()
