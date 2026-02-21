# src/quant/backtest/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# Events that realize PnL (i.e. they close a position)
EXIT_EVENTS = {"signal_flip_exit", "sl_exit", "tp_exit"}


@dataclass
class Metrics:
    rows: int
    start: str
    end: str
    total_return_pct: float
    max_drawdown_pct: float
    turnover_sum: float


def _sort_events(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or len(events) == 0:
        return pd.DataFrame(columns=["ts", "event", "pnl_pct", "price", "side"])

    ev = events.copy()
    if "ts" not in ev.columns:
        raise ValueError("events missing column: ts")

    ev["ts"] = pd.to_datetime(ev["ts"], utc=True)

    # IMPORTANT: keep deterministic order within same timestamp
    if "seq" in ev.columns:
        ev = ev.sort_values(["ts", "seq"]).reset_index(drop=True)
    else:
        # stable sort preserves original row order for identical ts
        ev = ev.sort_values(["ts"], kind="mergesort").reset_index(drop=True)

    return ev


def compute_equity_curve(
    bricks: pd.DataFrame,
    events: pd.DataFrame,
    initial_equity: float = 10_000.0,
) -> pd.DataFrame:
    """
    Returns equity over time on brick timestamps.

    Equity update rule:
      - only EXIT_EVENTS realize pnl_pct
      - pnl_pct is already signed (+ for win, - for loss) and should INCLUDE fees if you want fees in equity
      - equity := equity * (1 + pnl_pct)
    """
    if "close" not in bricks.columns:
        raise ValueError("bricks must contain column 'close'")
    if bricks.index.name != "ts":
        # accept ts-indexed df, but enforce datetime index
        pass

    idx = pd.to_datetime(bricks.index, utc=True)
    eq = pd.DataFrame({"ts": idx})
    eq["equity"] = float(initial_equity)

    ev = _sort_events(events)

    if len(ev) == 0:
        eq = eq.set_index("ts")
        return eq

    # Keep only realized exits with numeric pnl_pct
    need_cols = {"event", "pnl_pct", "ts"}
    missing = [c for c in need_cols if c not in ev.columns]
    if missing:
        raise ValueError(f"events missing columns: {missing}")

    realized = ev[ev["event"].isin(EXIT_EVENTS)].copy()
    if len(realized) == 0:
        eq = eq.set_index("ts")
        return eq

    realized["pnl_pct"] = pd.to_numeric(realized["pnl_pct"], errors="coerce")
    realized = realized.dropna(subset=["pnl_pct"])
    realized = realized[["ts", "pnl_pct"]].copy()

    # Aggregate multiple exits on same ts (rare but possible): compound in-sequence.
    # If there are multiple realized exits at same ts, the correct compounding is:
    # equity *= Π(1 + pnl_i)
    realized["one_plus"] = 1.0 + realized["pnl_pct"].astype(float)
    realized = realized.groupby("ts", as_index=False)["one_plus"].prod()

    # Map the factor to brick timeline (forward-fill last known equity)
    # We'll apply factors at their exact ts, then forward-fill.
    fac = realized.set_index("ts")["one_plus"].sort_index()

    eq = eq.set_index("ts")
    eq["factor"] = 1.0
    common = eq.index.intersection(fac.index)
    if len(common) > 0:
        eq.loc[common, "factor"] = fac.loc[common].values

    eq["equity"] = float(initial_equity) * eq["factor"].cumprod()
    eq = eq.drop(columns=["factor"])
    return eq


def compute_stats(
    bricks: pd.DataFrame,
    events: pd.DataFrame,
    initial_equity: float = 10_000.0,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Returns (stats_dict, equity_df indexed by ts).
    """
    if len(bricks) == 0:
        raise ValueError("bricks is empty")

    eq = compute_equity_curve(bricks=bricks, events=events, initial_equity=initial_equity)

    start = pd.to_datetime(bricks.index[0], utc=True)
    end = pd.to_datetime(bricks.index[-1], utc=True)

    equity0 = float(initial_equity)
    equity1 = float(eq["equity"].iloc[-1])
    total_return_pct = (equity1 / equity0 - 1.0) * 100.0

    peak = eq["equity"].cummax()
    dd = (eq["equity"] / peak - 1.0) * 100.0
    max_drawdown_pct = float(dd.min())

    # turnover: count entries + exits if present, otherwise keep 0
    turnover_sum = 0.0
    if events is not None and len(events) > 0 and "event" in events.columns:
        turnover_sum = float(
            events["event"].isin({"entry", *EXIT_EVENTS}).sum()
        )

    stats = Metrics(
        rows=int(len(bricks)),
        start=str(start),
        end=str(end),
        total_return_pct=float(total_return_pct),
        max_drawdown_pct=float(max_drawdown_pct),
        turnover_sum=float(turnover_sum),
    ).__dict__

    return stats, eq
