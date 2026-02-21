# scripts/plot_equity_gate_price.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _read_parquet_ts(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    if ts_col not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": ts_col})
        else:
            raise ValueError(f"missing '{ts_col}' and not datetime-indexed")
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).drop_duplicates(ts_col, keep="last").reset_index(drop=True)
    return df


def load_equity_from_run(run_dir: str, initial_capital: float = 10_000.0) -> pd.DataFrame:
    run = Path(run_dir)

    eqp = run / "equity_real.parquet"
    if eqp.exists():
        eq = pd.read_parquet(eqp)
        eq = _read_parquet_ts(eq, "ts")
        if "equity" not in eq.columns:
            raise ValueError(f"{eqp} missing 'equity'")
        eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
        eq = eq.dropna(subset=["equity"]).reset_index(drop=True)
        return eq[["ts", "equity"]].copy()

    # fallback: rebuild from trades_real
    tp = run / "trades_real.parquet"
    if not tp.exists():
        raise FileNotFoundError(f"missing equity_real.parquet and trades_real.parquet in {run}")

    t = pd.read_parquet(tp)
    # robust column pick
    for c in ["pnl_pct_real", "pnl_pct"]:
        if c in t.columns:
            r_col = c
            break
    else:
        raise ValueError("trades_real missing pnl column (expected pnl_pct_real or pnl_pct)")

    # find exit timestamp col
    exit_col = "exit_ts" if "exit_ts" in t.columns else ("ts" if "ts" in t.columns else None)
    if exit_col is None:
        raise ValueError("trades_real missing exit_ts/ts")

    t[exit_col] = pd.to_datetime(t[exit_col], utc=True, errors="coerce")
    t[r_col] = pd.to_numeric(t[r_col], errors="coerce")
    t = t.dropna(subset=[exit_col, r_col]).sort_values(exit_col).reset_index(drop=True)

    eq = float(initial_capital) * np.cumprod(1.0 + t[r_col].astype(float).values)
    out = pd.DataFrame({"ts": t[exit_col].values, "equity": eq})
    out = _read_parquet_ts(out, "ts")
    return out


def load_price_series(fills_parquet: str, price_col: str = "close") -> pd.Series:
    df = pd.read_parquet(fills_parquet, columns=["ts", price_col])
    df = _read_parquet_ts(df, "ts")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col]).reset_index(drop=True)
    s = pd.Series(df[price_col].values, index=pd.DatetimeIndex(df["ts"]), name=price_col)
    s = s[~s.index.duplicated(keep="last")]
    return s


def load_gate_series(gate_csv: str, ts_col: str = "ts", gate_col: str = "gate_on") -> pd.Series:
    g = pd.read_csv(gate_csv)
    if ts_col not in g.columns:
        raise ValueError(f"gate csv missing ts_col='{ts_col}' (cols={list(g.columns)})")
    if gate_col not in g.columns:
        raise ValueError(f"gate csv missing gate_col='{gate_col}' (cols={list(g.columns)})")

    g = g[[ts_col, gate_col]].copy()
    g[ts_col] = pd.to_datetime(g[ts_col], utc=True, errors="coerce")
    g = g.dropna(subset=[ts_col]).sort_values(ts_col).drop_duplicates(ts_col, keep="last").reset_index(drop=True)

    v = pd.to_numeric(g[gate_col], errors="coerce")
    if v.isna().all():
        v = g[gate_col].astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
    v = v.fillna(0).astype(int).clip(0, 1)

    s = pd.Series(v.values.astype(bool), index=pd.DatetimeIndex(g[ts_col]), name="gate_on")
    s = s[~s.index.duplicated(keep="last")]
    return s


def asof_align_bool(target_index: pd.DatetimeIndex, src: pd.Series, default_off: bool = True) -> pd.Series:
    # align gate to target timestamps via backward asof (ffill from last known gate row)
    b = pd.DataFrame({"ts": target_index})
    r = pd.DataFrame({"ts": src.index, "gate_on": src.astype(int).values}).sort_values("ts")
    m = pd.merge_asof(b.sort_values("ts"), r, on="ts", direction="backward", allow_exact_matches=True)
    if default_off:
        m["gate_on"] = m["gate_on"].fillna(0).astype(int)
    else:
        m["gate_on"] = m["gate_on"].fillna(1).astype(int)
    out = pd.Series(m["gate_on"].astype(bool).values, index=pd.DatetimeIndex(m["ts"]), name="gate_on")
    return out


def spans_from_bool(s: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    # contiguous True segments -> (start,end)
    if s is None or len(s) == 0:
        return []
    x = s.astype(bool).values
    idx = s.index
    spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    in_on = False
    start = None
    for i, on in enumerate(x):
        if on and not in_on:
            in_on = True
            start = idx[i]
        elif (not on) and in_on:
            end = idx[i]
            spans.append((start, end))
            in_on = False
            start = None
    if in_on and start is not None:
        spans.append((start, idx[-1]))
    return spans


def build_step_equity_on_index(eq_events: pd.DataFrame, target_index: pd.DatetimeIndex, initial_capital: float) -> pd.Series:
    # step function: equity updates at eq_events.ts, forward-filled on target_index
    if eq_events is None or len(eq_events) == 0:
        return pd.Series(np.full(len(target_index), float(initial_capital)), index=target_index, name="equity")

    e = eq_events.copy()
    e = _read_parquet_ts(e, "ts")
    e["equity"] = pd.to_numeric(e["equity"], errors="coerce")
    e = e.dropna(subset=["equity"]).reset_index(drop=True)

    # asof backward: for each target ts pick last equity <= ts; before first event -> initial_capital
    b = pd.DataFrame({"ts": target_index})
    r = e[["ts", "equity"]].sort_values("ts")
    m = pd.merge_asof(b.sort_values("ts"), r, on="ts", direction="backward", allow_exact_matches=True)
    m["equity"] = m["equity"].fillna(float(initial_capital))
    return pd.Series(m["equity"].values.astype(float), index=pd.DatetimeIndex(m["ts"]), name="equity")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Run folder containing equity_real.parquet or trades_real.parquet")
    ap.add_argument("--fills-parquet", required=True, help="Minute fills parquet with ts + price col (e.g. close)")
    ap.add_argument("--price-col", default="close", help="Price column in fills parquet (default: close)")
    ap.add_argument("--gate-csv", default=None, help="Optional gate CSV with ts + gate_on")
    ap.add_argument("--gate-ts-col", default="ts")
    ap.add_argument("--gate-col", default="gate_on")
    ap.add_argument("--gate-default-off", action="store_true", help="Default OFF before first gate row")
    ap.add_argument("--initial-capital", type=float, default=10_000.0)
    ap.add_argument("--resample", default="4H", help="Resample rule for plotting axis (default: 4H). e.g. 1H, 6H, 1D")
    ap.add_argument("--out", default=None, help="Output png path (default: <run-dir>/equity_gate_price.png)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.out) if args.out else (run_dir / "equity_gate_price.png")

    # 1) price series (downsample for plotting)
    price = load_price_series(str(args.fills_parquet), str(args.price_col))
    price_rs = price.resample(args.resample).last().dropna()

    # 2) equity events -> step equity on the same index as price_rs
    eq_events = load_equity_from_run(str(run_dir), initial_capital=float(args.initial_capital))
    equity_step = build_step_equity_on_index(eq_events, price_rs.index, float(args.initial_capital))

    # 3) gate align to price_rs index (optional)
    gate_rs = None
    spans = []
    if args.gate_csv:
        gate = load_gate_series(str(args.gate_csv), ts_col=str(args.gate_ts_col), gate_col=str(args.gate_col))
        gate_rs = asof_align_bool(price_rs.index, gate, default_off=bool(args.gate_default_off))
        spans = spans_from_bool(gate_rs)

    # 4) plot
    fig, ax = plt.subplots(figsize=(16, 6))
    ax2 = ax.twinx()

    # gate shading FIRST (background)
    if spans:
        for (a, b) in spans:
            # light-blue background stripes
            ax.axvspan(a, b, alpha=0.10)

    # equity (left axis)
    ax.plot(equity_step.index, equity_step.values, linewidth=1.5)
    ax.set_ylabel("Equity")

    # price (right axis) — BLUE line as requested
    ax2.plot(price_rs.index, price_rs.values, linewidth=1.0, color="blue")
    ax2.set_ylabel("SOLUSDT price")

    # cosmetics
    title = args.title or f"Equity (left) + SOLUSDT (right) with Gate Windows (shaded) | {run_dir.name}"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    # annotate ON-rate if gate present
    if gate_rs is not None and len(gate_rs):
        on_rate = 100.0 * float(gate_rs.mean())
        ax.text(
            0.01, 0.02,
            f"Gate ON-rate (on plotted grid): {on_rate:.2f}%",
            transform=ax.transAxes,
            fontsize=10,
            alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.25", alpha=0.15),
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"INFO wrote plot {out_path}")


if __name__ == "__main__":
    main()
