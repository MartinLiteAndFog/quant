#!/usr/bin/env python3
"""
Plot merged strategy equity (0.7 Kelly leverage) in red over SOL-USDT price in blue.

Default leverages: ON (countertrades) 3.11x, OFF (TP2) 7.34x — from Kelly on merged
trades at 15 bps (run scripts/kelly_from_trades on strategy-split trades to recompute).
Use --parquet with the longer renko file (FROM_20210924) to include full history.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Headless backend for saving to file
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt


# 0.7 Kelly from merged run at 15 bps: ON L*=4.44 -> 3.11, OFF L*=10.49 -> 7.34
LEV_ON = 3.11
LEV_OFF = 7.34
FEE_BPS_DEFAULT = 15.0


def _read_parquet_ts(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    if ts_col not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": ts_col})
    if ts_col not in df.columns:
        raise ValueError(f"missing '{ts_col}'")
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).drop_duplicates(ts_col, keep="last").reset_index(drop=True)
    return df


def load_price_from_parquet(parquet_path: str, price_col: str = "close") -> pd.Series:
    df = pd.read_parquet(parquet_path, columns=["ts", price_col])
    df = _read_parquet_ts(df, "ts")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col]).reset_index(drop=True)
    s = pd.Series(df[price_col].values, index=pd.DatetimeIndex(df["ts"]), name=price_col)
    return s[~s.index.duplicated(keep="last")]


def build_leveraged_equity_from_merged_trades(
    trades_path: Path,
    fee_bps: float = 15.0,
    lev_on: float = LEV_ON,
    lev_off: float = LEV_OFF,
    initial_capital: float = 1.0,
    cap_leverage: float | None = None,
) -> pd.DataFrame:
    """
    trades.parquet must have: exit_ts, pnl_pct, strategy ('on' | 'off').
    Flip (on) pnl is already net of fee from backtest; off we subtract fee_bps roundtrip.

    Without cap_leverage: equity = initial * cumprod(1 + L*r) — assumes every trade
    is sized at L×current equity, so compounding explodes (unrealistic at scale).
    With cap_leverage: use min(L, cap) so position size doesn't grow without bound.
    """
    t = pd.read_parquet(trades_path)
    for c in ["exit_ts", "pnl_pct", "strategy"]:
        if c not in t.columns:
            raise ValueError(f"trades parquet missing '{c}' (cols={list(t.columns)})")

    t["exit_ts"] = pd.to_datetime(t["exit_ts"], utc=True, errors="coerce")
    t["pnl_pct"] = pd.to_numeric(t["pnl_pct"], errors="coerce")
    t = t.dropna(subset=["exit_ts", "pnl_pct"]).sort_values("exit_ts").reset_index(drop=True)

    fee_rt = float(fee_bps) / 10_000.0

    def net_return(row):
        r = float(row["pnl_pct"])
        if str(row.get("strategy", "")).strip().lower() == "off":
            r = r - fee_rt
        return r

    t["r_net"] = t.apply(net_return, axis=1)
    L_arr = t["strategy"].apply(lambda s: lev_on if str(s).strip().lower() == "on" else lev_off).astype(float).values
    if cap_leverage is not None and float(cap_leverage) > 0:
        L_arr = np.minimum(L_arr, float(cap_leverage))
    m = 1.0 + L_arr * t["r_net"].astype(float).values
    if np.any(m <= 0):
        raise ValueError("Leveraged return produced non-positive multiplier (bankruptcy); check fee/leverage.")
    eq = float(initial_capital) * np.cumprod(m)
    return pd.DataFrame({"ts": t["exit_ts"].values, "equity": eq})


def run_merged_backtest(
    parquet: str,
    signals_jsonl: str,
    regime_csv: str,
    regime_col: str,
    regime_csv_off: str,
    regime_col_off: str,
    run_id: str,
    fee_bps: float = 15.0,
) -> Path:
    run_dir = Path("data/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "quant.backtest.renko_runner",
        "--parquet", parquet,
        "--signals-jsonl", signals_jsonl,
        "--regime-csv", regime_csv,
        "--regime-col", regime_col,
        "--regime-csv-off", regime_csv_off,
        "--regime-col-off", regime_col_off,
        "--fee-bps", str(fee_bps),
        "--run-id", run_id,
    ]
    subprocess.run(cmd, check=True)
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot merged equity (0.7 Kelly) in red over SOL price in blue")
    ap.add_argument("--parquet", required=True, help="Renko (or OHLC) parquet with ts, close for price series")
    ap.add_argument("--run-dir", default=None, help="Use existing run dir; if not set, run backtest first")
    ap.add_argument("--signals-jsonl", default=None)
    ap.add_argument("--regime-csv", default=None)
    ap.add_argument("--regime-col", default="gate_on_2of3")
    ap.add_argument("--regime-csv-off", default=None)
    ap.add_argument("--regime-col-off", default="gate_off_2of3")
    ap.add_argument("--fee-bps", type=float, default=FEE_BPS_DEFAULT)
    ap.add_argument("--run-id", default="merged_15bps_kelly")
    ap.add_argument("--lev-on", type=float, default=LEV_ON, help=f"Leverage for ON/countertrades (0.7 Kelly default {LEV_ON})")
    ap.add_argument("--lev-off", type=float, default=LEV_OFF, help=f"Leverage for OFF/TP2 (0.7 Kelly default {LEV_OFF})")
    ap.add_argument("--initial-capital", type=float, default=1.0)
    ap.add_argument("--cap-leverage", type=float, default=None, help="Cap leverage (e.g. 2) so equity does not compound without bound; default no cap")
    ap.add_argument("--resample", default="4h", help="Resample price for plot (e.g. 1h, 4h, 1D)")
    ap.add_argument("--out", default=None, help="Output png path")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir is None:
        if not all([args.signals_jsonl, args.regime_csv, args.regime_csv_off]):
            raise SystemExit("Without --run-dir you must pass --signals-jsonl, --regime-csv, --regime-csv-off")
        run_dir = run_merged_backtest(
            parquet=args.parquet,
            signals_jsonl=args.signals_jsonl,
            regime_csv=args.regime_csv,
            regime_col=args.regime_col,
            regime_csv_off=args.regime_csv_off,
            regime_col_off=args.regime_col_off,
            run_id=args.run_id,
            fee_bps=args.fee_bps,
        )
    else:
        run_dir = Path(run_dir)

    trades_path = run_dir / "trades.parquet"
    if not trades_path.exists():
        raise FileNotFoundError(f"No trades.parquet in {run_dir}")

    # Leveraged equity from merged trades (fee applied to off; lev_on / lev_off)
    eq_df = build_leveraged_equity_from_merged_trades(
        trades_path,
        fee_bps=args.fee_bps,
        lev_on=float(args.lev_on),
        lev_off=float(args.lev_off),
        initial_capital=args.initial_capital,
        cap_leverage=getattr(args, "cap_leverage", None),
    )
    eq_df = _read_parquet_ts(eq_df, "ts")
    eq_df["equity"] = pd.to_numeric(eq_df["equity"], errors="coerce")

    # Price series (full window)
    price = load_price_from_parquet(args.parquet, "close")
    price_rs = price.resample(args.resample).last().dropna()

    # Align equity to price index for plotting (step: last equity at or before each price ts)
    eq_ts = pd.to_datetime(eq_df["ts"], utc=True)
    eq_val = eq_df["equity"].astype(float).values
    target_ts = price_rs.index
    idx = np.searchsorted(eq_ts.values, target_ts.values, side="right") - 1
    idx = np.clip(idx, 0, len(eq_val) - 1)
    equity_on_grid = pd.Series(eq_val[idx], index=target_ts, name="equity")
    # Before first trade, equity = initial
    first_ts = eq_ts.iloc[0] if len(eq_ts) else None
    if first_ts is not None:
        equity_on_grid.loc[equity_on_grid.index < first_ts] = float(args.initial_capital)

    # Plot: blue = SOL price (background), red = equity (foreground)
    fig, ax1 = plt.subplots(figsize=(16, 7))

    ax1.plot(price_rs.index, price_rs.values, color="blue", linewidth=1.2, alpha=0.85, label="SOLUSDT")
    ax1.set_ylabel("SOLUSDT price", color="blue", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_xlabel("Time (UTC)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(equity_on_grid.index, equity_on_grid.values, color="red", linewidth=1.5, alpha=0.95, label=f"Equity ({args.lev_on:.1f}x / {args.lev_off:.1f}x)")
    ax2.set_ylabel("Equity", color="red", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.legend(loc="upper right")

    cap_str = f", cap L={args.cap_leverage}" if getattr(args, "cap_leverage", None) is not None else ""
    title = args.title or f"Merged strategies ({args.lev_on:.2f}x ON / {args.lev_off:.2f}x OFF, {args.fee_bps} bps{cap_str}) | {run_dir.name}"
    ax1.set_title(title)
    ax1.grid(True, alpha=0.25)
    fig.tight_layout()

    out_path = Path(args.out) if args.out else (run_dir / "merged_equity_solusdt.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"INFO wrote {out_path}")


if __name__ == "__main__":
    main()
