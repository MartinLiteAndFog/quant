#!/usr/bin/env python3
# scripts/plot_trades_window.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Utils
# ----------------------------

EXIT_EVENTS_FLIP = {"tp_exit", "signal_flip_exit"}
EXIT_EVENTS_FLAT = {"sl_exit", "be_exit"}


def _to_utc_ts(x) -> pd.Timestamp:
    return pd.to_datetime(x, utc=True, errors="coerce")


def _load_ohlcv_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts"})
        else:
            raise ValueError("parquet missing 'ts' column and not datetime-indexed")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = (
        df.dropna(subset=["ts"])
          .sort_values("ts")
          .drop_duplicates("ts", keep="last")
          .reset_index(drop=True)
    )

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

    # ensure seq exists for stable ordering
    if "seq" not in ev.columns:
        ev = ev.sort_values(["ts"], kind="mergesort").reset_index(drop=True)
        ev["seq"] = np.arange(len(ev), dtype=int)
    else:
        ev["seq"] = pd.to_numeric(ev["seq"], errors="coerce").fillna(0).astype(int)
        ev = ev.sort_values(["ts", "seq"], kind="mergesort").reset_index(drop=True)

    # normalize fields
    if "side" in ev.columns:
        ev["side"] = pd.to_numeric(ev["side"], errors="coerce").fillna(0).astype(int)
    if "price" in ev.columns:
        ev["price"] = pd.to_numeric(ev["price"], errors="coerce")
    if "pnl_pct" in ev.columns:
        ev["pnl_pct"] = pd.to_numeric(ev["pnl_pct"], errors="coerce")
    if "event" in ev.columns:
        ev["event"] = ev["event"].astype(str)

    return ev


def _load_signals_jsonl(path: str) -> pd.DataFrame:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return pd.DataFrame(columns=["ts", "signal"])

    df = pd.DataFrame(rows)
    if "ts" not in df.columns:
        raise ValueError("signals jsonl missing 'ts'")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce", format="mixed")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    if "signal" not in df.columns:
        if "position" in df.columns:
            df["signal"] = df["position"]
        else:
            raise ValueError("signals jsonl missing 'signal' (or 'position')")

    df["signal"] = pd.to_numeric(df["signal"], errors="coerce").fillna(0).astype(int)
    df["signal"] = np.sign(df["signal"]).astype(int)
    df = df[df["signal"] != 0].copy()
    return df[["ts", "signal"]]


def _asof_loc(bar_index: pd.DatetimeIndex, t: pd.Timestamp) -> Optional[int]:
    """
    Return location of last bar <= t (as-of). None if t before first bar.
    """
    if t is None or pd.isna(t):
        return None
    t = pd.to_datetime(t, utc=True)
    loc = bar_index.searchsorted(t, side="right") - 1
    if loc < 0 or loc >= len(bar_index):
        return None
    return int(loc)


def _pair_trades(events: pd.DataFrame) -> pd.DataFrame:
    """
    Pair entry -> next exit (tp_exit or sl_exit or signal_flip_exit/be_exit). Ignores open trade at end.

    Returns trades with:
      entry_ts, entry_side, entry_px, entry_note
      exit_ts, exit_event, exit_px, pnl_pct, exit_note
    """
    if events is None or len(events) == 0:
        return pd.DataFrame(columns=[
            "entry_ts", "entry_side", "entry_px", "entry_note",
            "exit_ts", "exit_event", "exit_px", "pnl_pct", "exit_note"
        ])

    ev = events.copy()
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True)
    ev = ev.sort_values(["ts", "seq"], kind="mergesort").reset_index(drop=True)

    exits = {"tp_exit", "sl_exit", "signal_flip_exit", "be_exit"}

    out = []
    open_entry = None

    for _, r in ev.iterrows():
        e = str(r.get("event", ""))
        if e == "entry":
            open_entry = r
            continue
        if e in exits and open_entry is not None:
            out.append({
                "entry_ts": open_entry["ts"],
                "entry_side": int(open_entry.get("side", 0)),
                "entry_px": float(open_entry.get("price", np.nan)),
                "entry_note": str(open_entry.get("note", "")),
                "exit_ts": r["ts"],
                "exit_event": e,
                "exit_px": float(r.get("price", np.nan)),
                "pnl_pct": float(r.get("pnl_pct", np.nan)),
                "exit_note": str(r.get("note", "")),
            })
            open_entry = None

    return pd.DataFrame(out)


def _compute_equity_from_trades(trades: pd.DataFrame, initial: float = 10_000.0) -> pd.DataFrame:
    """
    Equity curve sampled at exits.
    pnl_pct is assumed net (fees included) in your runner.
    """
    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=["ts", "equity"])

    t = trades.dropna(subset=["exit_ts", "pnl_pct"]).copy()
    t = t.sort_values("exit_ts")
    one_plus = 1.0 + t["pnl_pct"].astype(float).values
    equity = initial * np.cumprod(one_plus)
    return pd.DataFrame({"ts": t["exit_ts"].values, "equity": equity})


def _extract_entry_markers(events_window: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build two sets of "entries" to plot:

    1) REAL entries: event == "entry" (flat->pos)
    2) FLIP entries: for each flip-exit event (tp_exit/signal_flip_exit),
       we create a synthetic entry in the OPPOSITE direction.

       Example:
         exiting long (side=+1) via tp_exit => NEW entry short (side=-1)
         exiting short (side=-1) via tp_exit => NEW entry long  (side=+1)

    Returns: (real_entries_df, flip_entries_df) each with columns:
      ts, side, price, src_event
    """
    ev = events_window.copy()
    ev = ev.sort_values(["ts", "seq"], kind="mergesort").reset_index(drop=True)

    real = ev[ev["event"] == "entry"].copy()
    real = real[["ts", "side", "price"]].copy()
    real["src_event"] = "entry"

    flips = ev[ev["event"].isin(EXIT_EVENTS_FLIP)].copy()
    if len(flips):
        flips["side"] = -flips["side"].astype(int)  # new direction
        flips = flips[["ts", "side", "price", "event"]].rename(columns={"event": "src_event"})
    else:
        flips = pd.DataFrame(columns=["ts", "side", "price", "src_event"])

    # Clean
    for df in (real, flips):
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["side"] = pd.to_numeric(df["side"], errors="coerce").fillna(0).astype(int)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Only keep meaningful sides
    real = real[real["side"].isin([-1, 1])].copy()
    flips = flips[flips["side"].isin([-1, 1])].copy()

    return real, flips


# ----------------------------
# Plot
# ----------------------------

def plot_window(
    df: pd.DataFrame,
    events: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    signals: Optional[pd.DataFrame],
    out_path: str,
    title: str,
) -> None:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df[(df["ts"] >= start) & (df["ts"] <= end)].copy()
    if len(df) < 5:
        raise ValueError("window too small / no data in range")

    idx = pd.DatetimeIndex(df["ts"], name="ts")
    close = df["close"].astype(float).values

    ev = events.copy()
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True)
    ev = ev[(ev["ts"] >= start) & (ev["ts"] <= end)].copy()
    ev = ev.sort_values(["ts", "seq"], kind="mergesort").reset_index(drop=True)

    trades = _pair_trades(ev)
    eq = _compute_equity_from_trades(trades, initial=10_000.0)

    real_entries, flip_entries = _extract_entry_markers(ev)
    flat_exits = ev[ev["event"].isin(EXIT_EVENTS_FLAT)].copy()

    fig = plt.figure(figsize=(20, 9))
    ax = plt.gca()

    # price
    ax.plot(idx, close, linewidth=1.2)

    # equity curve overlay (scaled into price-axis lightly)
    if len(eq) > 1:
        eq_idx = pd.to_datetime(eq["ts"], utc=True)
        eq_vals = eq["equity"].astype(float).values

        pmin, pmax = np.nanmin(close), np.nanmax(close)
        emin, emax = np.nanmin(eq_vals), np.nanmax(eq_vals)
        if np.isfinite(emin) and np.isfinite(emax) and (emax - emin) > 1e-12:
            eq_scaled = (eq_vals - emin) / (emax - emin) * (pmax - pmin) + pmin
            ax.plot(eq_idx, eq_scaled, linewidth=1.1, alpha=0.35, color="red")

    # signals (optional)
    if signals is not None and len(signals) > 0:
        s = signals.copy()
        s["ts"] = pd.to_datetime(s["ts"], utc=True)
        s = s[(s["ts"] >= start) & (s["ts"] <= end)].copy()

        xs_long, ys_long, xs_short, ys_short = [], [], [], []
        for _, r in s.iterrows():
            loc = _asof_loc(idx, r["ts"])
            if loc is None:
                continue
            sig = int(r.get("signal", 0))
            if sig > 0:
                xs_long.append(idx[loc]); ys_long.append(close[loc])
            elif sig < 0:
                xs_short.append(idx[loc]); ys_short.append(close[loc])

        if xs_long:
            ax.scatter(xs_long, ys_long, s=70, marker="^", color="green", alpha=0.25, zorder=4)
        if xs_short:
            ax.scatter(xs_short, ys_short, s=70, marker="v", color="red", alpha=0.25, zorder=4)

    # marker offsets
    rng = float(np.nanmax(close) - np.nanmin(close))
    yoff = max(rng * 0.02, 0.5)  # ~2% of range, min 0.5

    def _draw_entry_arrow(ts: pd.Timestamp, side: int, price: float, z: int = 10) -> None:
        loc = _asof_loc(idx, ts)
        if loc is None:
            return
        x = idx[loc]
        y = close[loc] if np.isfinite(close[loc]) else price
        if side > 0:
            ax.annotate(
                "", xy=(x, y), xytext=(x, y - yoff),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="green"),
                zorder=z
            )
        else:
            ax.annotate(
                "", xy=(x, y), xytext=(x, y + yoff),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="red"),
                zorder=z
            )

    # REAL entries: arrows by side
    for _, r in real_entries.iterrows():
        _draw_entry_arrow(pd.Timestamp(r["ts"]), int(r["side"]), float(r.get("price", np.nan)), z=10)

    # FLIP entries: arrows by NEW side (computed as -exit_side)
    # Also add a small label so you can see these are flips.
    for _, r in flip_entries.iterrows():
        ts = pd.Timestamp(r["ts"])
        side = int(r["side"])
        _draw_entry_arrow(ts, side, float(r.get("price", np.nan)), z=11)
        # subtle marker at the flip timestamp
        loc = _asof_loc(idx, ts)
        if loc is not None:
            x = idx[loc]; y = close[loc]
            ax.scatter([x], [y], s=22, marker="o", color=("green" if side > 0 else "red"), alpha=0.55, zorder=11)

    # Flat exits (SL/BE): show as X
    if len(flat_exits):
        xs, ys = [], []
        for _, r in flat_exits.iterrows():
            loc = _asof_loc(idx, r["ts"])
            if loc is None:
                continue
            xs.append(idx[loc]); ys.append(close[loc])
        if xs:
            ax.scatter(xs, ys, s=90, marker="x", color="black", alpha=0.7, zorder=12)

    # Title with window PnL (from trades)
    if len(eq) > 0:
        final_equity = float(eq["equity"].iloc[-1])
        total_pct = (final_equity / 10_000.0 - 1.0) * 100.0
        title = f"{title} | window PnL: {total_pct:.2f}%"

    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="OHLCV parquet (1m or renko) with ts + close")
    ap.add_argument("--events", required=True, help="events.parquet from a run")
    ap.add_argument("--start", required=True, help="UTC timestamp or date, e.g. 2025-01-10 or 2025-01-10T00:00:00Z")
    ap.add_argument("--end", required=True, help="UTC timestamp or date")
    ap.add_argument("--signals-jsonl", default=None, help="optional signals jsonl for context markers")
    ap.add_argument("--out", default="trades.png")
    ap.add_argument("--title", default="Trades (flip entries as arrows)")

    args = ap.parse_args()

    start = _to_utc_ts(args.start)
    end = _to_utc_ts(args.end)
    if pd.isna(start) or pd.isna(end):
        raise ValueError("start/end could not be parsed")
    if end <= start:
        raise ValueError("end must be > start")

    df = _load_ohlcv_parquet(args.parquet)
    ev = _load_events_parquet(args.events)

    sig = None
    if args.signals_jsonl:
        sig = _load_signals_jsonl(args.signals_jsonl)

    plot_window(
        df=df,
        events=ev,
        start=start,
        end=end,
        signals=sig,
        out_path=args.out,
        title=args.title,
    )
    print(f"WROTE: {args.out}")


if __name__ == "__main__":
    main()
