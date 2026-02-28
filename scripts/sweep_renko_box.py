# scripts/sweep_renko_box.py
"""
Sweep fixed Renko box sizes (e.g. from 0.07 upwards). Same equity calculation as renko_runner.
Build Renko per box → run backtest → plot all equity curves on one chart.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent


def _parse_floats(s: str) -> List[float]:
    if not s or not str(s).strip():
        return []
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def build_renko(ohlc_path: str, out_path: str, box: float, gap_sec: float = 180.0) -> bool:
    cmd = [
        sys.executable,
        str(_SCRIPTS / "build_renko_fixed.py"),
        "--in", ohlc_path,
        "--out", out_path,
        "--box", str(box),
        "--gap-sec", str(gap_sec),
    ]
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


def align_equity_curves(series_dict: dict[str, pd.Series]) -> Tuple[pd.DatetimeIndex, pd.DataFrame]:
    if not series_dict:
        return pd.DatetimeIndex([]), pd.DataFrame()
    all_ts = pd.concat([s.dropna().index.to_series() for s in series_dict.values()]).drop_duplicates().sort_values()
    all_ts = pd.DatetimeIndex(pd.to_datetime(all_ts, utc=True))
    out = pd.DataFrame(index=all_ts)
    for label, ser in series_dict.items():
        s = ser.reindex(all_ts).ffill()
        first_val = s.dropna()
        if len(first_val):
            s = s / float(first_val.iloc[0])
        out[label] = s
    return all_ts, out


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep fixed Renko box sizes, plot equity curves")
    ap.add_argument("--ohlc", required=True, help="OHLC parquet to build Renko from")
    ap.add_argument("--signals-jsonl", required=True, help="Signals JSONL for backtest")
    ap.add_argument("--out", default="data/plots/sweep_renko_box_equity.png", help="Output plot path")
    ap.add_argument(
        "--box-grid",
        default="0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15",
        help="Comma-separated box sizes (default: 0.07 .. 0.15)",
    )
    ap.add_argument("--fee-bps", type=float, default=4.0)
    ap.add_argument(
        "--regime-csv",
        default="data/regimes/SOLUSDT_tv5mIMBA_gate2of3_qch0.4_qadx0.6_qer0.3_daily.csv",
    )
    ap.add_argument(
        "--regime-csv-off",
        default="data/regimes/SOLUSDT_tv5mIMBA_gate2of3_qch0.4_qadx0.6_qer0.3_daily_OFF.csv",
    )
    ap.add_argument("--regime-col", default="gate_on_2of3")
    ap.add_argument("--regime-col-off", default="gate_off_2of3")
    ap.add_argument("--fills-parquet", default=None)
    ap.add_argument("--fill-col", default="close")
    ap.add_argument("--run-prefix", default="RENKO_BOX", help="Prefix for run_id")
    ap.add_argument("--sweep-dir", default=None)
    args = ap.parse_args()

    box_sizes = _parse_floats(args.box_grid)
    if not box_sizes:
        raise SystemExit("--box-grid must list at least one box size")
    use_real = bool(args.fills_parquet)

    from datetime import datetime, timezone
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else _REPO / "data" / "sweeps" / f"renko_box_{tag}"
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
    for box in box_sizes:
        safe = str(box).replace(".", "p")
        parquet = str(renko_dir / f"box_{safe}.parquet")
        if not build_renko(ohlc_path, parquet, box=box):
            print(f"WARN build failed box={box}")
            continue
        run_id = f"{args.run_prefix}_box{safe}"
        rc = run_backtest(
            parquet, signals_path, run_id, logs_dir,
            fee_bps=args.fee_bps,
            regime_csv=args.regime_csv, regime_csv_off=args.regime_csv_off,
            regime_col=args.regime_col, regime_col_off=args.regime_col_off,
            fills_parquet=args.fills_parquet, fill_col=args.fill_col,
        )
        eq = load_equity(run_id, use_real=use_real)
        if eq is not None and len(eq):
            equity_series[f"box={box}"] = eq.set_index("ts")["equity"]
        print(f"box={box} -> {run_id} rc={rc}")

    if not equity_series:
        print("No equity data collected. Check logs in", logs_dir)
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
    ax.set_title("Renko fixed box sweep (same equity formula)")
    ax.legend(loc="best", fontsize=9)
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
