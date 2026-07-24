"""Read-only signal/decision timeline diagnostics for fleet bots.

Why: the exchange-side view (kucoin_diag) shows what *filled*, but not *why* a
bot did or didn't act. A signal can vanish in several places the fill history
never reveals:

  * it never arrived (no TradingView alert / delivery failure),
  * it arrived but was addressed to a different bot (bot_mismatch),
  * it arrived while the executor was still warming up (not_ready / 503),
  * it arrived and was blocked by the CHOP/regime gate (block_all mode),
  * it arrived and executed (entry/exit/flip/tp/sl).

This router reconstructs the per-bot decision timeline from the durable
event tables written by tv_signal_executor + bot_webhook:

  * signal_events     -> inbound TradingView signals + disposition,
  * action_events     -> engine decisions incl. blocked + block_reason,
  * execution_events  -> orders sent to the exchange (fills side).

Mount it alongside kucoin_diag:

    from quant.execution.signal_diag import router as signal_diag_router
    app.include_router(signal_diag_router)

    GET /diag/timeline?token=<BOT_WEBHOOK_TOKEN>&hours=72   # merged, newest first
    GET /diag/actions?token=...&hours=72                    # action_events only
    GET /diag/signals?token=...&hours=72                    # inbound signals only

Guarded by the same BOT_WEBHOOK_TOKEN as /webhook and /kucoin/*. Read-only.
By default it scopes to this bot's own strategy_instance; pass
`instance=<strategy_instance>` to inspect another (the DB is shared fleet-wide).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/diag", tags=["signal-diag"])


def _require_token(token: Optional[str]) -> None:
    expected = (
        os.getenv("BOT_WEBHOOK_TOKEN")
        or os.getenv("TV_WEBHOOK_TOKEN")
        or os.getenv("WEBHOOK_TOKEN")
        or ""
    ).strip()
    if not expected:
        return
    if (token or "").strip() != expected:
        raise HTTPException(status_code=401, detail="invalid diag token")


def _this_instance() -> str:
    try:
        from quant.execution.bot_profiles import strategy_instance_id

        return str(strategy_instance_id())
    except Exception:
        return (os.getenv("BOT_INSTANCE_ID") or os.getenv("STRATEGY_INSTANCE") or "").strip()


def _iso(ts: Any) -> Optional[str]:
    if ts is None:
        return None
    return ts.isoformat() if hasattr(ts, "isoformat") else str(ts)


def _fnum(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _query(sql: str, params: Dict[str, Any]) -> List[tuple]:
    """Best-effort read; a missing table or DB error yields an empty list so one
    unavailable stream never sinks the whole timeline."""
    from quant.execution.event_store import get_conn

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return list(cur.fetchall() or [])


def _fetch_signals(instance: str, hours: float, limit: int) -> List[Dict[str, Any]]:
    sql = """
        select ts, symbol, signal, signal_side, source_type, payload_json
        from signal_events
        where strategy_instance = %(inst)s
          and ts >= now() - (%(hours)s || ' hours')::interval
        order by ts desc
        limit %(limit)s
    """
    try:
        rows = _query(sql, {"inst": instance, "hours": hours, "limit": limit})
    except Exception:
        return []
    out = []
    for r in rows:
        payload = r[5] if isinstance(r[5], dict) else {}
        out.append(
            {
                "kind": "signal",
                "ts": _iso(r[0]),
                "symbol": r[1],
                "signal": r[2],
                "side": r[3],
                "source_type": r[4],
                "disposition": payload.get("disposition"),
                "detail": payload.get("detail"),
            }
        )
    return out


def _fetch_actions(instance: str, hours: float, limit: int) -> List[Dict[str, Any]]:
    sql = """
        select ts, symbol, engine_action, action_side, reason_code,
               blocked, block_reason, position_before, position_after
        from action_events
        where strategy_instance = %(inst)s
          and ts >= now() - (%(hours)s || ' hours')::interval
        order by ts desc
        limit %(limit)s
    """
    try:
        rows = _query(sql, {"inst": instance, "hours": hours, "limit": limit})
    except Exception:
        return []
    out = []
    for r in rows:
        out.append(
            {
                "kind": "action",
                "ts": _iso(r[0]),
                "symbol": r[1],
                "engine_action": r[2],
                "side": r[3],
                "reason": r[4],
                "blocked": bool(r[5]) if r[5] is not None else False,
                "block_reason": r[6],
                "pos_before": r[7],
                "pos_after": r[8],
            }
        )
    return out


def _fetch_executions(instance: str, hours: float, limit: int) -> List[Dict[str, Any]]:
    sql = """
        select ts, symbol, execution_stage, side, qty, price, reduce_only, status, order_id
        from execution_events
        where strategy_instance = %(inst)s
          and ts >= now() - (%(hours)s || ' hours')::interval
        order by ts desc
        limit %(limit)s
    """
    try:
        rows = _query(sql, {"inst": instance, "hours": hours, "limit": limit})
    except Exception:
        return []
    out = []
    for r in rows:
        out.append(
            {
                "kind": "execution",
                "ts": _iso(r[0]),
                "symbol": r[1],
                "stage": r[2],
                "side": r[3],
                "qty": _fnum(r[4]),
                "price": _fnum(r[5]),
                "reduce_only": bool(r[6]) if r[6] is not None else None,
                "status": r[7],
                "order_id": r[8],
            }
        )
    return out


@router.get("/signals")
def diag_signals(
    token: Optional[str] = None, hours: float = 72.0, instance: Optional[str] = None, limit: int = 300
) -> Dict[str, Any]:
    _require_token(token)
    inst = (instance or _this_instance()).strip()
    return {"ok": True, "instance": inst, "hours": hours, "signals": _fetch_signals(inst, hours, limit)}


@router.get("/actions")
def diag_actions(
    token: Optional[str] = None, hours: float = 72.0, instance: Optional[str] = None, limit: int = 300
) -> Dict[str, Any]:
    _require_token(token)
    inst = (instance or _this_instance()).strip()
    return {"ok": True, "instance": inst, "hours": hours, "actions": _fetch_actions(inst, hours, limit)}


@router.get("/timeline")
def diag_timeline(
    token: Optional[str] = None, hours: float = 72.0, instance: Optional[str] = None, limit: int = 300
) -> Dict[str, Any]:
    """Merged, newest-first decision timeline: inbound signals, engine actions
    (incl. gate blocks), and exchange orders. This is the "where did the signal
    go / in what order did the bot act?" view."""
    _require_token(token)
    inst = (instance or _this_instance()).strip()
    events: List[Dict[str, Any]] = []
    events += _fetch_signals(inst, hours, limit)
    events += _fetch_actions(inst, hours, limit)
    events += _fetch_executions(inst, hours, limit)
    # Sort newest-first; None timestamps sink to the bottom.
    events.sort(key=lambda e: (e.get("ts") or ""), reverse=True)
    if limit and len(events) > limit:
        events = events[:limit]
    blocked = [e for e in events if e.get("kind") == "action" and e.get("blocked")]
    return {
        "ok": True,
        "instance": inst,
        "hours": hours,
        "counts": {
            "signals": sum(1 for e in events if e["kind"] == "signal"),
            "actions": sum(1 for e in events if e["kind"] == "action"),
            "executions": sum(1 for e in events if e["kind"] == "execution"),
            "blocked_actions": len(blocked),
        },
        "events": events,
    }
