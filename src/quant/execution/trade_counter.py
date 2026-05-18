"""Trade decision counter.

This module derives a *trade decision* stream from `action_events` rows.

A "trade decision" is the discrete event of opening (or re-opening) a directional
position with its own stop-loss / take-profit pair. Per the live-executor design,
every counted decision corresponds to a new SL/TP commitment that we want to
measure independently.

Classification rules (authoritative)
====================================

Inputs are dict-like rows mirroring the `action_events` schema. The relevant
fields are::

    engine_action     – the executor action label (case-insensitive)
    action_side       – optional 'long'/'short'/'flat' hint
    position_before   – int in {-1, 0, 1}
    position_after    – int in {-1, 0, 1}
    blocked           – bool; true means the action did not execute
    reason_code       – free-form, kept as metadata only

Counted as a trade decision (returns a :class:`TradeDecision`):

* ``enter_long``  – flat -> long  (kind=ENTRY, direction=long)
* ``enter_short`` – flat -> short (kind=ENTRY, direction=short)
* ``flip_to_long``  – short -> long (kind=FLIP, direction=long)
* ``flip_to_short`` – long  -> short (kind=FLIP, direction=short)

A flip is treated as one decision (the new opposite leg with its own SL/TP);
the close-half of the flip is implicit and is not double-counted.

NOT counted as a new decision (returns ``None``):

* ``scale_long`` / ``scale_short`` – same-direction add, no new SL/TP.
* ``tp1_partial`` and any partial close – reduces size, keeps SL.
* ``exit_long`` / ``exit_short`` – ends the existing trade lifecycle but does
  not start a new trade. The trade was already counted at entry/flip time.
* ``hold`` – no-op.
* Any unrecognised label.
* Any ``blocked=true`` action – the SL/TP was never committed.

Idempotency
===========

Every decision carries a deterministic :pyfunc:`deterministic_decision_id` so
that re-processing the same `action_events` rows always yields the same set of
decisions. Persistence layers (see :mod:`quant.execution.trade_decisions_store`)
upsert by ``decision_id``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


DECISION_ENTRY = "entry"
DECISION_FLIP = "flip"

# Action -> (decision_kind, direction).
_ENTRY_ACTIONS: Dict[str, str] = {
    "enter_long": "long",
    "enter_short": "short",
}
_FLIP_ACTIONS: Dict[str, str] = {
    "flip_to_long": "long",
    "flip_to_short": "short",
}

# Explicit non-decision actions kept here as documentation; anything not in
# _ENTRY_ACTIONS or _FLIP_ACTIONS is ignored anyway, but enumerating the known
# ignored cases makes the contract obvious for readers.
_IGNORED_ACTIONS = frozenset(
    {
        "hold",
        "exit_long",
        "exit_short",
        "scale_long",
        "scale_short",
        "tp1_partial",
    }
)


@dataclass(frozen=True)
class TradeDecision:
    """A single counted trade decision.

    ``decision_id`` is deterministic so that re-runs are idempotent.
    """

    decision_id: str
    ts: str
    venue: str
    symbol: str
    strategy: Optional[str]
    strategy_instance: Optional[str]
    decision_kind: str  # 'entry' | 'flip'
    direction: str      # 'long' | 'short'
    position_before: Optional[int]
    position_after: Optional[int]
    engine_action: str
    reason_code: Optional[str]
    source_action_event_id: Optional[str]
    seq: Optional[int]
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_db_row(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "ts": self.ts,
            "venue": self.venue,
            "symbol": self.symbol,
            "strategy": self.strategy,
            "strategy_instance": self.strategy_instance,
            "decision_kind": self.decision_kind,
            "direction": self.direction,
            "position_before": self.position_before,
            "position_after": self.position_after,
            "engine_action": self.engine_action,
            "reason_code": self.reason_code,
            "source_action_event_id": self.source_action_event_id,
            "seq": self.seq,
            "payload_json": dict(self.payload or {}),
        }


def _norm_action(v: Any) -> str:
    return str(v or "").strip().lower()


def _coerce_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return None


def _derive_direction(engine_action: str, action_side: Any, position_after: Any) -> Optional[str]:
    side = str(action_side or "").strip().lower()
    if side in ("long", "short"):
        return side
    if engine_action in _ENTRY_ACTIONS:
        return _ENTRY_ACTIONS[engine_action]
    if engine_action in _FLIP_ACTIONS:
        return _FLIP_ACTIONS[engine_action]
    pa = _coerce_int(position_after)
    if pa is None:
        return None
    if pa > 0:
        return "long"
    if pa < 0:
        return "short"
    return None


def deterministic_decision_id(
    *,
    venue: str,
    symbol: str,
    source_action_event_id: Optional[str],
    ts: str,
    seq: Optional[int],
    engine_action: str,
) -> str:
    """Build a stable, deterministic id for a trade decision.

    Prefers ``source_action_event_id`` (already unique per `action_events` row);
    falls back to a hash over (venue, symbol, ts, seq, action) so that we can
    still produce a stable id even for events that lack an event id.
    """

    if source_action_event_id:
        base = f"src:{source_action_event_id}"
    else:
        base = "|".join(
            [
                str(venue or ""),
                str(symbol or ""),
                str(ts or ""),
                str(seq if seq is not None else ""),
                _norm_action(engine_action),
            ]
        )
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    return f"td_{digest}"


def classify_action_event(row: Dict[str, Any]) -> Optional[TradeDecision]:
    """Classify a single `action_events` row.

    Returns a :class:`TradeDecision` if the row should be counted, else ``None``.
    """

    if not isinstance(row, dict):
        return None
    if bool(row.get("blocked", False)):
        return None

    action = _norm_action(row.get("engine_action"))
    if action in _ENTRY_ACTIONS:
        kind = DECISION_ENTRY
    elif action in _FLIP_ACTIONS:
        kind = DECISION_FLIP
    else:
        return None

    direction = _derive_direction(
        action,
        row.get("action_side"),
        row.get("position_after"),
    )
    if direction not in ("long", "short"):
        return None

    venue = str(row.get("venue") or "")
    symbol = str(row.get("symbol") or "")
    ts = str(row.get("ts") or "")
    seq_val = _coerce_int(row.get("seq"))
    source_event_id = row.get("event_id") or row.get("source_event_id")

    decision_id = deterministic_decision_id(
        venue=venue,
        symbol=symbol,
        source_action_event_id=str(source_event_id) if source_event_id else None,
        ts=ts,
        seq=seq_val,
        engine_action=action,
    )

    payload: Dict[str, Any] = {
        "engine_action": action,
        "action_side": row.get("action_side"),
        "position_before": row.get("position_before"),
        "position_after": row.get("position_after"),
        "reason_code": row.get("reason_code"),
        "kind": kind,
        "direction": direction,
    }

    return TradeDecision(
        decision_id=decision_id,
        ts=ts,
        venue=venue,
        symbol=symbol,
        strategy=row.get("strategy"),
        strategy_instance=row.get("strategy_instance"),
        decision_kind=kind,
        direction=direction,
        position_before=_coerce_int(row.get("position_before")),
        position_after=_coerce_int(row.get("position_after")),
        engine_action=action,
        reason_code=row.get("reason_code"),
        source_action_event_id=str(source_event_id) if source_event_id else None,
        seq=seq_val,
        payload=payload,
    )


def _sort_key(row: Dict[str, Any]) -> Any:
    # Sort by ts (string ISO) then seq for stable ordering. Both are optional.
    return (str(row.get("ts") or ""), _coerce_int(row.get("seq")) or 0)


def build_trade_decisions_from_action_events(
    events: Iterable[Dict[str, Any]],
) -> List[TradeDecision]:
    """Classify a batch of `action_events` rows into trade decisions.

    The output is sorted by (ts, seq) and deduplicated by ``decision_id`` so
    that running the builder repeatedly over the same input is idempotent.
    """

    rows = [r for r in events if isinstance(r, dict)]
    rows.sort(key=_sort_key)
    seen: Dict[str, TradeDecision] = {}
    out: List[TradeDecision] = []
    for r in rows:
        d = classify_action_event(r)
        if d is None:
            continue
        if d.decision_id in seen:
            continue
        seen[d.decision_id] = d
        out.append(d)
    return out
