"""Decision-based equity curve + performance aggregator.

This is the *single source of truth* for the dashboard's trade-mode equity
chart and the Performance card on the right sidebar. Both must show the same
numbers, computed from the same list, in the same order. See
:mod:`quant.execution.trade_counter` for the canonical definition of what a
"decision" is — only entries from flat and flips count, scale-ins and
partial closes do not.

Realized PnL per decision is attributed by joining the ``trade_decisions``
spine with the ``closed_trades`` table on a forward time match: every closed
KuCoin leg attaches to the earliest unmatched decision whose entry timestamp
sits at or just before the leg's close.

Open decisions (still in-position) keep their slot in the trade count but
carry ``open=True``, ``pnl_pct=None`` and do not contribute to wins / losses
or cumulative realized PnL.

Notes
-----
* Older trades that pre-date the ``action_events`` table cannot be
  reconstructed as decisions — that is a structural limit. Their data
  remains visible through the legacy ``closed_trades`` readers but is not
  the dashboard's source any more.
* If ``trade_decisions`` is empty (e.g. the Railway worker has not run the
  ``backfill_trade_decisions_from_action_events`` job yet), the builder
  returns an empty curve and a ``needs_backfill`` hint so the UI can ask
  the operator to run the backfill.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from quant.execution.dashboard_state import (
    _EPOCH_UTC,
    load_active_levels,
    load_closed_trades_from_postgres,
)
from quant.execution.trade_decisions_store import (
    list_recent_trade_decisions,
)


# Reject sentinel near-epoch timestamps that older NaT->0 writers left behind
# in ``closed_trades``. Anything before 2000-01-01 is treated as "no entry
# time recorded" rather than 1970 in the tooltip.
_MIN_VALID_EPOCH_SEC = 946_684_800  # 2000-01-01T00:00:00Z


def _ts_to_epoch_seconds(ts: Any) -> Optional[int]:
    """Convert any timestamp-like to epoch seconds or ``None``.

    Uses ``Timedelta``-based arithmetic so it stays correct regardless of the
    underlying ``datetime64`` precision (pandas 2.x reads parquet rows as
    ``us`` by default, which silently turned ``// 1_000_000_000`` into
    ``seconds // 1000`` historically). Any near-epoch sentinel is mapped to
    ``None`` so the frontend can render "—" instead of 1/1/1970.
    """

    if ts is None:
        return None
    try:
        if pd.isna(ts):
            return None
    except Exception:
        pass
    try:
        seconds = int((pd.Timestamp(ts) - _EPOCH_UTC) // pd.Timedelta(seconds=1))
    except Exception:
        return None
    if seconds < _MIN_VALID_EPOCH_SEC:
        return None
    return seconds


def _normalize_decisions(decisions: Optional[Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Project raw ``trade_decisions`` rows into the minimum fields we need,
    sorted ascending by timestamp."""

    rows: List[Dict[str, Any]] = []
    for d in (decisions or []):
        if not isinstance(d, dict):
            continue
        ts = pd.to_datetime(d.get("ts"), utc=True, errors="coerce")
        direction = str(
            d.get("direction")
            or (d.get("payload_json") or {}).get("direction")
            or ""
        ).lower()
        if pd.isna(ts) or direction not in ("long", "short"):
            continue
        rows.append(
            {
                "decision_id": str(d.get("decision_id") or ""),
                "ts": pd.Timestamp(ts),
                "direction": direction,
                "decision_kind": str(d.get("decision_kind") or "").lower(),
                "source_action_event_id": d.get("source_action_event_id"),
                "seq": d.get("seq"),
            }
        )
    rows.sort(key=lambda r: (r["ts"], r.get("seq") or 0))
    return rows


def _prepare_closed_trades_frame(closed_trades_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if closed_trades_df is None or not isinstance(closed_trades_df, pd.DataFrame) or closed_trades_df.empty:
        return pd.DataFrame()

    df = closed_trades_df.copy()
    df["entry_ts"] = pd.to_datetime(df.get("entry_ts"), utc=True, errors="coerce")
    df["exit_ts"] = pd.to_datetime(df.get("exit_ts"), utc=True, errors="coerce")
    df["pnl_pct"] = pd.to_numeric(df.get("pnl_pct"), errors="coerce")
    df["entry_price"] = pd.to_numeric(df.get("entry_price"), errors="coerce")
    df["exit_price"] = pd.to_numeric(df.get("exit_price"), errors="coerce")
    df["side"] = df.get("side", pd.Series([""] * len(df))).astype(str).str.lower()
    df = df.dropna(subset=["exit_ts", "pnl_pct"])
    df = df.sort_values("exit_ts").reset_index(drop=True)
    return df


_FLAT_TOKENS = frozenset({"", "flat", "0", "none"})


def _detect_open_side(open_side: Optional[str]) -> Optional[str]:
    """Normalize the live position side to ``"long"`` / ``"short"`` / ``None``.

    Distinguishes "we know the position is flat" from "we don't know":

    * Passing ``open_side=None`` triggers best-effort detection via
      ``load_active_levels``. Used by the production code path.
    * Passing a flat token (``""``, ``"flat"``, ``"0"``, ``"none"``)
      *explicitly* says "live position is flat" — the builder must NOT
      promote any unmatched decision to ``open=True`` because we have
      ground-truth evidence that no leg is currently running. Used by
      tests + by the chart endpoint when ``/api/position`` reports flat.
    * Passing ``"long"``/``"short"`` (or aliases) means the live position
      carries that side; the builder may flag the matching unmatched
      decision as open.
    """

    if open_side is not None:
        s = str(open_side).strip().lower()
        if s in _FLAT_TOKENS:
            return "flat"
        if s in ("long", "buy", "1"):
            return "long"
        if s in ("short", "sell", "-1"):
            return "short"
    try:
        levels = load_active_levels() or {}
    except Exception:
        levels = {}
    raw = str(levels.get("side") or (levels.get("terminal") or {}).get("side") or "").strip().lower()
    if raw in ("long", "buy", "1"):
        return "long"
    if raw in ("short", "sell", "-1"):
        return "short"
    if raw in _FLAT_TOKENS:
        return "flat"
    return None


def build_decision_equity_curve(
    *,
    symbol: str,
    venue: str = "kucoin",
    max_points: int = 500,
    decisions: Optional[Iterable[Dict[str, Any]]] = None,
    closed_trades_df: Optional[pd.DataFrame] = None,
    open_side: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the decision-attributed equity curve.

    Parameters
    ----------
    decisions, closed_trades_df, open_side
        Optional dependency-injection hooks for tests. When ``None`` the
        function reads from Postgres / the live execution state file.

    Returns
    -------
    dict
        ``{"points": [...], "source": "...", "needs_backfill": bool}``.
        Each point has ``decision_id``, ``side``, ``entry_time``,
        ``entry_price``, ``exit_time``, ``exit_price``, ``pnl_pct``,
        ``cum_pct``, ``open``, ``time``, ``source``. ``time`` mirrors the
        existing chart contract (chart x-axis); for closed decisions it is
        the exit time, for open decisions it is the entry time.
    """

    venue_eff = str(venue or "kucoin").lower()
    if venue_eff != "kucoin":
        return {"points": [], "source": "unsupported_venue", "needs_backfill": False}

    if decisions is None:
        try:
            decisions = list_recent_trade_decisions(
                venue=venue_eff,
                symbol=symbol,
                limit=int(max(500, max_points * 4)),
            )
        except Exception:
            decisions = []

    dec_rows = _normalize_decisions(decisions)

    if closed_trades_df is None:
        try:
            closed_trades_df = load_closed_trades_from_postgres(
                venue=venue_eff,
                symbol=symbol,
                max_points=int(max(5000, max_points * 10)),
            )
        except Exception:
            closed_trades_df = pd.DataFrame()

    ct = _prepare_closed_trades_frame(closed_trades_df)

    open_side_norm = _detect_open_side(open_side)

    # Greedy forward attribution. For each decision in chronological order,
    # walk closed_trades by exit_ts ascending and consume the first unmatched
    # row whose side matches and whose exit_ts is at or after the decision's
    # ts (with a small backfill tolerance for clock skew).
    used: set = set()
    matches: List[Optional[int]] = []
    tolerance = pd.Timedelta(seconds=120)
    for d in dec_rows:
        match_idx: Optional[int] = None
        for i in range(len(ct)):
            if i in used:
                continue
            row = ct.iloc[i]
            exit_ts_i = row["exit_ts"]
            side_i = str(row["side"]).lower()
            if pd.isna(exit_ts_i):
                continue
            # Skip closed_trades that end before this decision could have
            # opened — those belong to older decisions or legacy data.
            if exit_ts_i < d["ts"] - tolerance:
                continue
            if side_i and side_i != d["direction"]:
                continue
            match_idx = i
            break
        if match_idx is not None:
            used.add(match_idx)
        matches.append(match_idx)

    points: List[Dict[str, Any]] = []
    cum = 0.0
    last_idx = len(dec_rows) - 1
    src_label = "postgres:trade_decisions+closed_trades"
    for idx, d in enumerate(dec_rows):
        entry_time = _ts_to_epoch_seconds(d["ts"])
        match_idx = matches[idx]
        if match_idx is not None:
            row = ct.iloc[match_idx]
            pnl_pct = float(row["pnl_pct"])
            cum += pnl_pct
            exit_time = _ts_to_epoch_seconds(row["exit_ts"])
            entry_price = float(row["entry_price"]) if pd.notna(row["entry_price"]) else None
            exit_price = float(row["exit_price"]) if pd.notna(row["exit_price"]) else None
            points.append(
                {
                    "decision_id": d["decision_id"],
                    "side": d["direction"],
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": round(pnl_pct, 4),
                    "cum_pct": round(cum, 4),
                    "open": False,
                    "time": exit_time if exit_time is not None else entry_time,
                    "source": src_label,
                }
            )
        else:
            # An unmatched decision is "open" only when it's the most recent
            # one AND the live KuCoin position carries the same side. If we
            # explicitly know the position is flat, never flag as open
            # (ground-truth evidence > optimistic chart). If detection
            # returned ``None`` (couldn't determine), fall back to optimistic
            # is-open for the latest decision so the chart still surfaces
            # the live trade.
            if open_side_norm == "flat":
                is_open = False
            elif open_side_norm in ("long", "short"):
                is_open = (idx == last_idx) and (open_side_norm == d["direction"])
            else:  # detection returned None
                is_open = (idx == last_idx)
            points.append(
                {
                    "decision_id": d["decision_id"],
                    "side": d["direction"],
                    "entry_time": entry_time,
                    "exit_time": None,
                    "entry_price": None,
                    "exit_price": None,
                    "pnl_pct": None,
                    "cum_pct": round(cum, 4),
                    "open": bool(is_open),
                    "time": entry_time,
                    "source": src_label,
                }
            )

    points = points[-int(max(1, max_points)):]

    needs_backfill = False
    # Heuristic: the decision spine looks suspiciously empty next to the
    # closed_trades history. The UI can prompt "run backfill on Railway".
    if not dec_rows and not ct.empty:
        needs_backfill = True

    return {
        "points": points,
        "source": src_label if dec_rows else "none",
        "needs_backfill": needs_backfill,
    }


def compute_performance_from_decision_points(
    points: List[Dict[str, Any]],
    *,
    symbol: str,
    venue: str = "kucoin",
    now: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """Derive the Performance card numbers from the same decision points the
    equity chart plots. Every aggregate here can be reproduced by the
    frontend by walking the same list — there is no second source.

    PnL is summed (additive) so that the card's ``pnl_pct`` is exactly the
    chart's final ``cum_pct``.
    """

    if now is None:
        now = pd.Timestamp.now("UTC")

    closed = [p for p in points if not p.get("open")]
    open_decisions = [p for p in points if p.get("open")]

    winning = [p for p in closed if float(p.get("pnl_pct") or 0.0) > 0]
    losing = [p for p in closed if float(p.get("pnl_pct") or 0.0) < 0]

    pnl_total = sum(float(p.get("pnl_pct") or 0.0) for p in closed)
    avg_trade = (pnl_total / len(closed)) if closed else None
    cum_last = points[-1].get("cum_pct") if points else None

    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    monthly_growth_value: Optional[float] = None
    if closed:
        monthly_growth_value = 0.0
        for p in closed:
            xt = p.get("exit_time")
            if xt is None:
                continue
            ts = pd.to_datetime(int(xt), unit="s", utc=True)
            if ts >= month_start:
                monthly_growth_value += float(p.get("pnl_pct") or 0.0)

    winrate = None
    decided = len(winning) + len(losing)
    if decided > 0:
        winrate = 100.0 * len(winning) / decided

    return {
        "symbol": symbol,
        "venue": venue,
        "as_of": now.isoformat(),
        "window": "lifetime",
        "pnl_pct": round(float(pnl_total), 4) if closed else None,
        "winrate": round(float(winrate), 4) if winrate is not None else None,
        "monthly_growth": (
            round(float(monthly_growth_value), 4) if monthly_growth_value is not None else None
        ),
        "average_gain": round(float(avg_trade), 4) if avg_trade is not None else None,
        "trade_count": int(len(points)),
        "closed_decision_count": int(len(closed)),
        "winning_trade_count": int(len(winning)),
        "losing_trade_count": int(len(losing)),
        "open_decision_count": int(len(open_decisions)),
        "cum_pct": round(float(cum_last), 4) if cum_last is not None else None,
        "source": "postgres:trade_decisions+closed_trades",
    }


def build_decision_dashboard_payload(
    *,
    symbol: str,
    venue: str = "kucoin",
    max_points: int = 500,
    decisions: Optional[Iterable[Dict[str, Any]]] = None,
    closed_trades_df: Optional[pd.DataFrame] = None,
    open_side: Optional[str] = None,
    now: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """Convenience wrapper: build the equity curve AND derive the
    performance aggregates in one shot. The two outputs share the same
    underlying points list so the chart and card are guaranteed consistent.
    """

    curve = build_decision_equity_curve(
        symbol=symbol,
        venue=venue,
        max_points=max_points,
        decisions=decisions,
        closed_trades_df=closed_trades_df,
        open_side=open_side,
    )
    perf = compute_performance_from_decision_points(
        curve["points"],
        symbol=symbol,
        venue=venue,
        now=now,
    )
    return {
        "curve": curve,
        "performance": perf,
        "needs_backfill": bool(curve.get("needs_backfill")),
    }
