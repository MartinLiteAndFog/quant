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

import hashlib
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


def _uniform_downsample(items: List[Any], n: int) -> List[Any]:
    """Sample ``items`` down to ``n`` evenly-spaced entries.

    Always preserves the first and last item so the chart's leftmost x
    domain matches the oldest leg and the rightmost cumulative value
    matches the latest realised PnL. Never pads or duplicates — the
    output length is ``min(len(items), n)`` and trade counts therefore
    can only shrink, never inflate.
    """

    if n <= 0:
        return []
    if len(items) <= n:
        return list(items)
    if n == 1:
        return [items[-1]]
    step = (len(items) - 1) / (n - 1)
    indices = sorted({int(round(i * step)) for i in range(n)})
    return [items[i] for i in indices]


def build_decision_equity_curve(
    *,
    symbol: str,
    venue: str = "kucoin",
    max_points: int = 5000,
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
        ``{"points": [...], "merged_points": [...], "source": "...",
        "needs_backfill": bool, ...}``. ``merged_points`` is the full
        deduped list used for card aggregates; ``points`` is the same
        list, uniformly downsampled to ``max_points`` for chart display.
        Each point has ``decision_id``, ``side``, ``entry_time``,
        ``entry_price``, ``exit_time``, ``exit_price``, ``pnl_pct``,
        ``cum_pct``, ``open``, ``time``, ``source``. ``time`` is the
        chart x-axis position: for real decisions it is ``entry_time``
        (or the matched leg's ``exit_time`` when the decision's own
        timestamp is unusable); for synth points it is always the
        closed-trade ``exit_ts`` because synth ``entry_ts`` is the only
        field that may carry the 1970/NaT-zero sentinel.
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

    src_label = "postgres:trade_decisions+closed_trades"
    raw_points: List[Dict[str, Any]] = []
    last_idx = len(dec_rows) - 1
    for idx, d in enumerate(dec_rows):
        entry_time = _ts_to_epoch_seconds(d["ts"])
        match_idx = matches[idx]
        if match_idx is not None:
            row = ct.iloc[match_idx]
            pnl_pct = float(row["pnl_pct"])
            exit_time = _ts_to_epoch_seconds(row["exit_ts"])
            entry_price = float(row["entry_price"]) if pd.notna(row["entry_price"]) else None
            exit_price = float(row["exit_price"]) if pd.notna(row["exit_price"]) else None
            # Chart x position: prefer the decision's own entry_time so
            # the cumulative curve advances at the moment the trade was
            # taken. Fall back to the matched leg's exit_time only when
            # the decision ts is unusable (the bug that collapsed the
            # historical X domain to the last ~7 days).
            if entry_time is not None:
                plot_time = entry_time
            elif exit_time is not None:
                plot_time = exit_time
            else:
                # No usable timestamp at all — skip rather than plot at 0.
                continue
            raw_points.append(
                {
                    "decision_id": d["decision_id"],
                    "side": d["direction"],
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": round(pnl_pct, 4),
                    "open": False,
                    "time": plot_time,
                    "source": src_label,
                    "_sort_time": plot_time,
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
            if entry_time is None:
                # An unmatched decision has no closed-leg fallback, so
                # without a real entry_time we cannot place it on the
                # chart. Drop rather than corrupt the X domain.
                continue
            raw_points.append(
                {
                    "decision_id": d["decision_id"],
                    "side": d["direction"],
                    "entry_time": entry_time,
                    "exit_time": None,
                    "entry_price": None,
                    "exit_price": None,
                    "pnl_pct": None,
                    "open": bool(is_open),
                    "time": entry_time,
                    "source": src_label,
                    "_sort_time": entry_time,
                }
            )

    # Backfill the long tail in-memory. Any ``closed_trades`` row that was
    # not consumed above is a historical leg the persistent backfill
    # couldn't reconstruct (no valid entry_ts, no matching decision id,
    # etc.). Rather than silently truncating the chart to whatever the
    # spine happens to carry today, synthesize a decision row in-memory
    # so the user sees the full history they actually have. The
    # ``td_ct_synth_`` prefix marks provenance distinct from the
    # persisted ``td_ct_`` backfill ids.
    synth_label = "synth:closed_trades"
    synthesized_points: List[Dict[str, Any]] = []
    for i in range(len(ct)):
        if i in used:
            continue
        row = ct.iloc[i]
        exit_ts_i = row["exit_ts"]
        if pd.isna(exit_ts_i):
            continue
        try:
            pnl_pct = float(row["pnl_pct"])
        except Exception:
            continue
        side_i = str(row["side"]).lower() or "long"
        entry_ts_i = row.get("entry_ts")
        # ``entry_time`` is purely metadata for the tooltip here. It must
        # stay ``None`` when the persisted ``entry_ts`` is the legacy
        # NaT-zero sentinel, otherwise the UI would render 1/1/1970 or
        # mis-attribute the bar's timestamp.
        entry_time = _ts_to_epoch_seconds(entry_ts_i) if entry_ts_i is not None else None
        exit_time = _ts_to_epoch_seconds(exit_ts_i)
        if exit_time is None:
            # Without a usable ``exit_ts`` the synthesized point cannot
            # be placed on the chart; skip rather than collapse it to 0.
            continue
        entry_price = float(row["entry_price"]) if pd.notna(row["entry_price"]) else None
        exit_price = float(row["exit_price"]) if pd.notna(row["exit_price"]) else None
        trade_id_raw = row.get("trade_id") if "trade_id" in ct.columns else None
        synth_seed = "|".join(
            [
                str(venue_eff),
                str(symbol),
                str(side_i),
                str(entry_ts_i),
                str(exit_ts_i),
                str(trade_id_raw or ""),
            ]
        )
        synth_id = "td_ct_synth_" + hashlib.sha1(synth_seed.encode("utf-8")).hexdigest()[:16]
        # Synth points always plot at the closed leg's ``exit_ts``: it's
        # the only timestamp on the row that is reliably historic, so
        # using it preserves the full X-domain back to the earliest
        # closed trade. ``entry_time`` is reported separately as ``None``
        # when the source row's ``entry_ts`` was unusable.
        synthesized_points.append(
            {
                "decision_id": synth_id,
                "side": side_i,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_pct": round(pnl_pct, 4),
                "open": False,
                "time": exit_time,
                "source": synth_label,
                "_sort_time": exit_time,
            }
        )

    merged_points: List[Dict[str, Any]] = raw_points + synthesized_points
    # Sort by chart x position so synthesized historical legs interleave
    # before the action-event spine. Tie-break puts closed legs ahead of
    # open ones at the same instant so the open marker doesn't shadow a
    # simultaneously-closed flip.
    merged_points.sort(
        key=lambda p: (int(p.get("_sort_time") or 0), 1 if p.get("open") else 0)
    )

    # Dedupe by decision_id (keep first occurrence). Defensive guard
    # against the upstream spine returning a duplicated row after a
    # backfill retry — without this the Performance card could inflate
    # ``trade_count`` past the true number of decisions.
    seen_ids: set = set()
    deduped: List[Dict[str, Any]] = []
    for p in merged_points:
        did = str(p.get("decision_id") or "")
        if did and did in seen_ids:
            continue
        if did:
            seen_ids.add(did)
        deduped.append(p)
    merged_points = deduped

    # Recompute the cumulative PnL after the merge so synthesized rows
    # contribute to the chart's final ``cum_pct`` (and therefore the
    # card's ``pnl_pct``).
    cum = 0.0
    for p in merged_points:
        pnl = p.get("pnl_pct")
        if pnl is not None and not p.get("open"):
            cum += float(pnl)
        p["cum_pct"] = round(cum, 4)
        p.pop("_sort_time", None)

    # ``max_points`` is a downsampling guard — never a counter. When the
    # full merged list exceeds the cap we sample uniformly while
    # preserving the first and last entry, so the chart's start and end
    # x-positions (and the final ``cum_pct``) match what the card
    # aggregates over the full ``merged_points`` list.
    cap = int(max(1, max_points))
    if len(merged_points) > cap:
        points = _uniform_downsample(merged_points, cap)
    else:
        points = list(merged_points)

    # ``needs_backfill`` semantics, post-auto-backfill: true when the
    # persistent spine is still smaller than ``closed_trades`` (the chart
    # is being patched up by ``td_ct_synth_*`` rows that do not survive
    # a restart). The UI uses this to surface the explicit operator
    # ``?backfill=1`` button.
    needs_backfill = bool(synthesized_points) or (not dec_rows and not ct.empty)
    synthesized_count = len(synthesized_points)

    return {
        "points": points,
        "merged_points": merged_points,
        "source": src_label if dec_rows else (synth_label if synthesized_points else "none"),
        "needs_backfill": needs_backfill,
        "synthesized_count": synthesized_count,
        "decision_count": len(dec_rows),
        "closed_trade_count": int(len(ct)),
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

    Contract (matches the trade-mode equity chart):
        * ``trade_count`` = count of unique ``decision_id`` (open + closed).
        * ``open_decision_count`` = points with ``open == True``.
        * ``closed_decision_count`` = ``trade_count - open_decision_count``.
        * ``winning_trade_count`` / ``losing_trade_count`` = closed points
          with realised ``pnl_pct > 0`` / ``< 0``.
        * ``winrate`` = wins / (wins + losses) when the denominator > 0,
          else ``None`` — never 0/0 = 0%.
        * ``pnl_pct`` = the merged list's final ``cum_pct``, which by
          construction equals the chart's final cumulative line.
        * ``average_gain`` = mean of ``pnl_pct`` over closed points that
          carry a realised pnl; ``None`` when none do.
        * ``monthly_growth`` = sum of ``pnl_pct`` over closed points whose
          ``entry_time`` (or ``exit_time`` when entry_time is missing)
          falls in the current calendar month at ``now`` (UTC). It is
          NOT a copy of ``pnl_pct`` — those values only coincide when
          every closed trade in the spine happened this month.
    """

    if now is None:
        now = pd.Timestamp.now("UTC")

    # Defensive dedupe by ``decision_id`` so duplicated rows from the
    # spine cannot inflate ``trade_count`` past the real number of
    # decisions taken.
    seen_ids: set = set()
    unique_points: List[Dict[str, Any]] = []
    for p in points:
        did = str(p.get("decision_id") or "")
        if did and did in seen_ids:
            continue
        if did:
            seen_ids.add(did)
        unique_points.append(p)

    closed = [p for p in unique_points if not p.get("open")]
    open_decisions = [p for p in unique_points if p.get("open")]
    closed_with_pnl = [p for p in closed if p.get("pnl_pct") is not None]

    winning = [p for p in closed_with_pnl if float(p["pnl_pct"]) > 0]
    losing = [p for p in closed_with_pnl if float(p["pnl_pct"]) < 0]

    cum_last: Optional[float] = None
    if unique_points:
        last_cum = unique_points[-1].get("cum_pct")
        if last_cum is not None:
            cum_last = float(last_cum)

    # ``pnl_pct`` on the card must equal the chart's final ``cum_pct`` so
    # the two views can never disagree on the user's overall PnL.
    pnl_value: Optional[float] = cum_last if closed_with_pnl else None

    avg_trade: Optional[float] = None
    if closed_with_pnl:
        avg_trade = sum(float(p["pnl_pct"]) for p in closed_with_pnl) / len(closed_with_pnl)

    # Calendar-month bucket: anchor on the first of this month (UTC) at
    # ``now``, open-ended through the start of next month. Use
    # ``entry_time`` first so each trade is bucketed by when it was
    # taken; fall back to ``exit_time`` when entry_time is unusable —
    # this is the case for synthesized rows whose source ``entry_ts``
    # was the legacy 1970/NaT-zero sentinel.
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    next_month_start = month_start + pd.offsets.MonthBegin(1)
    monthly_growth_value: Optional[float] = None
    if closed_with_pnl:
        monthly_growth_value = 0.0
        for p in closed_with_pnl:
            t_raw = p.get("entry_time")
            if t_raw is None or int(t_raw) < _MIN_VALID_EPOCH_SEC:
                t_raw = p.get("exit_time")
            if t_raw is None:
                continue
            ts = pd.to_datetime(int(t_raw), unit="s", utc=True)
            if month_start <= ts < next_month_start:
                monthly_growth_value += float(p["pnl_pct"])

    winrate: Optional[float] = None
    decided = len(winning) + len(losing)
    if decided > 0:
        winrate = 100.0 * len(winning) / decided

    trade_count = len(unique_points)
    open_count = len(open_decisions)
    closed_count = trade_count - open_count

    return {
        "symbol": symbol,
        "venue": venue,
        "as_of": now.isoformat(),
        "window": "lifetime",
        "pnl_pct": round(float(pnl_value), 4) if pnl_value is not None else None,
        "winrate": round(float(winrate), 4) if winrate is not None else None,
        "monthly_growth": (
            round(float(monthly_growth_value), 4) if monthly_growth_value is not None else None
        ),
        "average_gain": round(float(avg_trade), 4) if avg_trade is not None else None,
        "trade_count": int(trade_count),
        "closed_decision_count": int(closed_count),
        "winning_trade_count": int(len(winning)),
        "losing_trade_count": int(len(losing)),
        "open_decision_count": int(open_count),
        "cum_pct": round(float(cum_last), 4) if cum_last is not None else None,
        "source": "postgres:trade_decisions+closed_trades",
    }


def build_decision_dashboard_payload(
    *,
    symbol: str,
    venue: str = "kucoin",
    max_points: int = 5000,
    decisions: Optional[Iterable[Dict[str, Any]]] = None,
    closed_trades_df: Optional[pd.DataFrame] = None,
    open_side: Optional[str] = None,
    now: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """Convenience wrapper: build the equity curve AND derive the
    performance aggregates in one shot. The two outputs share the same
    underlying points list so the chart and card are guaranteed consistent.

    Aggregates are computed off the full ``merged_points`` list rather
    than the (possibly) downsampled ``points`` list, so the card's counts
    and ``pnl_pct`` always reflect the true history even when the chart
    has been downsampled for display.
    """

    curve = build_decision_equity_curve(
        symbol=symbol,
        venue=venue,
        max_points=max_points,
        decisions=decisions,
        closed_trades_df=closed_trades_df,
        open_side=open_side,
    )
    perf_source = curve.get("merged_points") or curve.get("points") or []
    perf = compute_performance_from_decision_points(
        perf_source,
        symbol=symbol,
        venue=venue,
        now=now,
    )
    return {
        "curve": curve,
        "performance": perf,
        "needs_backfill": bool(curve.get("needs_backfill")),
        "synthesized_count": int(curve.get("synthesized_count") or 0),
        "decision_count": int(curve.get("decision_count") or 0),
        "closed_trade_count": int(curve.get("closed_trade_count") or 0),
    }
