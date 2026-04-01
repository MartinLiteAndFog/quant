"""
Repair entry_ts, entry_price, and pnl_pct for signal_flip_exit trades.

After a signal flip, _append_closed_trade previously read entry data from the
flip engine terminal which already reflected the NEW trade. This means:
  - entry_ts ≈ exit_ts (both at the flip point)
  - entry_price ≈ exit_price (both at the flip price)
  - pnl_pct ≈ 0 (because entry ≈ exit)

The correct entry data can be reconstructed: for a signal_flip_exit trade,
its real entry was at the exit of the PREVIOUS trade on the same venue/symbol.

Usage:
    python scripts/repair_flip_entry_data.py --dry-run   # preview changes
    python scripts/repair_flip_entry_data.py              # apply fixes
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, ".")
from quant.execution.event_store import get_conn


def _load_all_trades(venue: Optional[str] = None) -> pd.DataFrame:
    where = "where venue = %(venue)s" if venue else ""
    params: Dict[str, Any] = {}
    if venue:
        params["venue"] = venue
    sql = f"""
        select trade_id, venue, symbol, entry_ts, exit_ts, side, qty,
               entry_price, exit_price, pnl_pct, exit_event, strategy
        from closed_trades
        {where}
        order by exit_ts asc
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        rows,
        columns=[
            "trade_id", "venue", "symbol", "entry_ts", "exit_ts", "side", "qty",
            "entry_price", "exit_price", "pnl_pct", "exit_event", "strategy",
        ],
    )


def _find_prior_trade(df: pd.DataFrame, trade: pd.Series) -> Optional[pd.Series]:
    """Find the trade whose exit created this trade's entry (same venue+symbol, exit_ts <= this entry)."""
    mask = (
        (df["venue"] == trade["venue"])
        & (df["symbol"] == trade["symbol"])
        & (df["exit_ts"] <= trade["exit_ts"])
        & (df["trade_id"] != trade["trade_id"])
    )
    candidates = df[mask].sort_values("exit_ts")
    if candidates.empty:
        return None
    return candidates.iloc[-1]


def repair(dry_run: bool = True, venue: Optional[str] = None) -> None:
    df = _load_all_trades(venue=venue)
    if df.empty:
        print("No trades found.")
        return

    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True, errors="coerce")

    flips = df[df["exit_event"] == "signal_flip_exit"].copy()
    print(f"Total trades: {len(df)}, signal_flip_exit trades: {len(flips)}")

    # Detect broken flip trades: entry_ts very close to exit_ts (< 5 min apart)
    flips["duration_sec"] = (flips["exit_ts"] - flips["entry_ts"]).dt.total_seconds().abs()
    broken = flips[flips["duration_sec"] < 300].copy()
    print(f"Broken flip trades (entry≈exit, <5min duration): {len(broken)}")

    if broken.empty:
        print("Nothing to repair.")
        return

    fixes: List[Dict[str, Any]] = []
    for _, trade in broken.iterrows():
        prior = _find_prior_trade(df, trade)
        if prior is None:
            print(f"  SKIP {trade['trade_id']}: no prior trade found")
            continue

        # The real entry of this trade was when the prior trade exited (the flip point of the PRIOR trade)
        new_entry_ts = prior["exit_ts"]
        new_entry_price = prior["exit_price"]

        if new_entry_price is None or float(new_entry_price) <= 0:
            print(f"  SKIP {trade['trade_id']}: prior exit_price invalid ({new_entry_price})")
            continue

        exit_price = float(trade["exit_price"])
        entry_price = float(new_entry_price)
        side_mult = 1.0 if str(trade["side"]).strip().lower() in ("long", "l", "buy", "1") else -1.0
        new_pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0 * side_mult

        fixes.append({
            "trade_id": trade["trade_id"],
            "old_entry_ts": trade["entry_ts"],
            "new_entry_ts": new_entry_ts,
            "old_entry_price": trade["entry_price"],
            "new_entry_price": entry_price,
            "old_pnl_pct": trade["pnl_pct"],
            "new_pnl_pct": new_pnl_pct,
            "side": trade["side"],
        })

        label = "DRY-RUN" if dry_run else "FIX"
        print(
            f"  {label} {trade['trade_id'][:50]}...\n"
            f"    side={trade['side']} entry_ts: {trade['entry_ts']} -> {new_entry_ts}\n"
            f"    entry_px: {trade['entry_price']:.4f} -> {entry_price:.4f}\n"
            f"    pnl_pct: {trade['pnl_pct']:.4f} -> {new_pnl_pct:.4f}"
        )

    if not fixes:
        print("No fixes to apply.")
        return

    print(f"\n{'Would fix' if dry_run else 'Fixing'} {len(fixes)} trades.")

    if dry_run:
        print("\nRe-run without --dry-run to apply.")
        return

    sql = """
        update closed_trades
        set entry_ts = %(entry_ts)s,
            entry_price = %(entry_price)s,
            pnl_pct = %(pnl_pct)s
        where trade_id = %(trade_id)s
    """
    with get_conn() as conn, conn.cursor() as cur:
        for fix in fixes:
            cur.execute(sql, {
                "trade_id": fix["trade_id"],
                "entry_ts": fix["new_entry_ts"],
                "entry_price": fix["new_entry_price"],
                "pnl_pct": fix["new_pnl_pct"],
            })
    print(f"Done. {len(fixes)} trades repaired.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Repair signal_flip_exit entry data")
    p.add_argument("--dry-run", action="store_true", help="Preview without writing")
    p.add_argument("--venue", default=None, help="Filter by venue (e.g. kucoin, kraken)")
    args = p.parse_args()
    repair(dry_run=args.dry_run, venue=args.venue)
