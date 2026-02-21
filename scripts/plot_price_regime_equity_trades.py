#!/usr/bin/env python3
# scripts/plot_price_regime_equity_trades.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Loaders
# ----------------------------

def _load_ohlcv_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # accept either ts column or datetime index
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts"})
        else:
            raise ValueError("parquet missing 'ts' column and not datetime-indexed")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "close" not in df.columns:
        raise ValueError("parquet missing 'close'")

    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


def _load_events_parquet(path: str) -> pd.DataFrame:
    ev = pd.read_parquet(path)
    if "ts" not in ev.columns:
        raise ValueError("events.parquet missing 'ts'")
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True, errors="coerce")
    ev = ev.dropna(subset=["ts"]).copy()

    if "seq" in ev.columns:
        ev = ev.sort_values(["ts", "seq"], kind="mergesort")
    else:
        ev = ev.sort_values(["ts"], kind="mergesort")
        ev["seq"] = np.arange(len(ev), dtype=int)

    return ev.reset_index(drop=True)


def _load_gate_csv(path: str, ts_col: str = "ts", gate_col: str = "gate_on") -> pd.DataFrame:
    g = pd.read_csv(path)
    if ts_col not in g.columns:
        raise ValueError(f"gate csv missing '{ts_col}'")
    if gate_col not in g.columns:
        raise ValueError(f"gate csv missing '{gate_col}'")

    g[ts_col] = pd.to_datetime(g[ts_col], utc=True, errors="coerce")
    g = g.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    g[gate_col] = pd.to_numeric(g[gate_col], errors="coerce").fillna(0).astype(int)
    g[gate_col] = (g[gate_col] != 0).astype(int)
    return g[[ts_col, gate_col]].rename(columns={ts_col: "ts", gate_col: "gate_on"})


# ----------------------------
# Trade pairing + equity
# ----------------------------

def _pair_trades(events: pd.DataFrame) -> pd.DataFrame:
    """
    Pair entry -> next exit (tp_exit / sl_exit / signal_flip_exit).
    One trade ends at exit; flips will create the next trade via next 'entry'.
    """
    exits = {"tp_exit", "sl_exit", "signal_flip_exit"}
    out: List[Dict] = []
    open_entry = None

    for _, r in events.iterrows():
        e = str(r.get("event", ""))
        if e == "entry":
            open_entry = r
            continue
        if e in exits and open_entry is not None:
            out.append(
                dict(
                    entry_ts=open_entry["ts"],
                    entry_side=int(open_entry.get("side", 0)),
                    entry_px=float(open_entry.get("price", np.nan)),
                    exit_ts=r["ts"],
                    exit_event=e,
                    exit_px=float(r.get("price", np.nan)),
                    pnl_pct=float(r.get("pnl_pct", np.nan)),
                )
            )
            open_entry = None

    t = pd.DataFrame(out)
    if t.empty:
        return t

    t = t.sort_values(["entry_ts", "exit_ts"]).reset_index(drop=True)
    return t


def _equity_from_trades(trades: pd.DataFrame, initial: float = 10_000.0) -> pd.DataFrame:
    """
    Equity sampled at exits. Assumes pnl_pct already includes fees (as in your runner).
    """
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["ts", "equity"])

    t = trades.dropna(subset=["exit_ts", "pnl_pct"]).copy()
    t = t.sort_values("exit_ts")
    one_plus = 1.0 + t["pnl_pct"].astype(float).values
    equity = initial * np.cumprod(one_plus)
    return pd.DataFrame({"ts": pd.to_datetime(t["exit_ts"], utc=True), "equity": equity})


# ----------------------------
# Regime shading (ON segments)
# ----------------------------

def _gate_segments(gate_daily: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Return list of (start, end) intervals where gate_on == 1.
    Assumes daily rows; we shade from ts to next ts (or same-day end).
    """
    if gate_daily is None or gate_daily.empty:
        return []

    g = gate_daily.copy().sort_values("ts").reset_index(drop=True)
    on = g["gate_on"].astype(int).values
    ts = g["ts"].values

    segs: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    i = 0
    n = len(g)
    while i < n:
        if on[i] == 1:
            start = pd.Timestamp(ts[i]).to_pydatetime()
            j = i + 1
            while j < n and on[j] == 1:
                j += 1
            # end = last ON day end (use next day's ts if available, else last ts)
            end_ts = pd.Timestamp(ts[j]) if j < n else pd.Timestamp(ts[n - 1]) + pd.Timedelta(days=1)
            segs.append((pd.Timestamp(start, tz="UTC"), pd.Timestamp(end_ts, tz="UTC")))
            i = j
        else:
            i += 1
    return segs


# ----------------------------
# Plot
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="OHLCV parquet (minute or renko)")
    ap.add_argument("--events", required=True, help="run/events.parquet")
    ap.add_argument("--gate-csv", required=True, help="daily_gate.csv (ts + gate_on)")
    ap.add_argument("--gate-ts-col", default="ts")
    ap.add_argument("--gate-col", default="gate_on")
    ap.add_argument("--out", default="price_regime_equity_trades.png")
    ap.add_argument("--title", default="Price + Regime ON shading + Equity + Trades")
    ap.add_argument("--equity-initial", type=float, default=10_000.0)
    args = ap.parse_args()

    price = _load_ohlcv_parquet(args.parquet)
    ev = _load_events_parquet(args.events)
    gate = _load_gate_csv(args.gate_csv, ts_col=args.gate_ts_col, gate_col=args.gate_col)

    # constrain to price range
    t0 = price["ts"].iloc[0]
    t1 = price["ts"].iloc[-1]
    gate = gate[(gate["ts"] >= (t0.floor("D") - pd.Timedelta(days=2))) & (gate["ts"] <= (t1.ceil("D") + pd.Timedelta(days=2)))].copy()

    trades = _pair_trades(ev)
    eq = _equity_from_trades(trades, initial=args.equity_initial)

    # base series
    x = pd.DatetimeIndex(price["ts"])
    y = price["close"].astype(float).values

    fig = plt.figure(figsize=(24, 8))
    ax = plt.gca()

    # price
    ax.plot(x, y, linewidth=1.1, label="Price")

    # regime shading
    segs = _gate_segments(gate)
    for (a, b) in segs:
        ax.axvspan(a, b, alpha=0.12)

    # equity overlay scaled to price axis
    if len(eq) > 2:
        ex = pd.DatetimeIndex(eq["ts"])
        evv = eq["equity"].astype(float).values

        pmin, pmax = float(np.nanmin(y)), float(np.nanmax(y))
        emin, emax = float(np.nanmin(evv)), float(np.nanmax(evv))
        if np.isfinite(emin) and np.isfinite(emax) and (emax - emin) > 1e-12:
            eq_scaled = (evv - emin) / (emax - emin) * (pmax - pmin) + pmin
            ax.plot(ex, eq_scaled, linewidth=1.2, color="red", alpha=0.55, label="Equity (scaled)")

    # trades: entry/exit markers on price (as-of nearest price bar <= timestamp)
    idx = x
    def asof_loc(t: pd.Timestamp) -> Optional[int]:
        loc = idx.searchsorted(pd.Timestamp(t), side="right") - 1
        if loc < 0 or loc >= len(idx):
            return None
        return int(loc)

    if not trades.empty:
        for _, tr in trades.iterrows():
            s = int(tr["entry_side"])
            color = "green" if s > 0 else "red" if s < 0 else "black"

            e_loc = asof_loc(pd.to_datetime(tr["entry_ts"], utc=True))
            x_loc = asof_loc(pd.to_datetime(tr["exit_ts"], utc=True))
            if e_loc is None or x_loc is None:
                continue

            ex_t, ex_y = idx[e_loc], y[e_loc]
            xx_t, xx_y = idx[x_loc], y[x_loc]

            ax.scatter([ex_t], [ex_y], s=22, marker="o", color=color, zorder=5)
            ax.scatter([xx_t], [xx_y], s=28, marker="x", color=color, zorder=5)
            ax.plot([ex_t, xx_t], [ex_y, xx_y], linewidth=0.8, color=color, alpha=0.55, zorder=4)

    # title with headline stats
    if len(eq) > 0:
        final = float(eq["equity"].iloc[-1])
        total_pct = (final / float(args.equity_initial) - 1.0) * 100.0
        ax.set_title(f"{args.title} | trades={len(trades)} | total={total_pct:.2f}%")
    else:
        ax.set_title(f"{args.title} | trades={len(trades)} | total=NA")

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price (with scaled equity overlay)")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    plt.close(fig)

    print(f"WROTE: {args.out}")


if __name__ == "__main__":
    main()
