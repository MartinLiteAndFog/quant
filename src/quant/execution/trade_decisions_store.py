"""Postgres persistence helpers for the trade decision counter.

Schema is owned by :file:`src/quant/sql/002_trade_decisions.sql`. To keep the
table self-bootstrapping (matches the pattern used by other event tables in
:mod:`quant.execution.event_store`), :func:`ensure_trade_decisions_schema`
issues an idempotent ``CREATE TABLE IF NOT EXISTS`` block.

All write paths upsert by ``decision_id`` so re-processing the same
``action_events`` rows never produces duplicates.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

from quant.execution.event_store import _payload, _normalize_symbol_token, get_conn
from quant.execution.trade_counter import (
    DECISION_ENTRY,
    DECISION_FLIP,
    TradeDecision,
    build_trade_decisions_from_action_events,
)


# Anything before 2000-01-01 is a NaT->0 sentinel left behind by the older
# fills reconstructor. Treat as "no valid entry timestamp" rather than
# materialising a 1970 row in ``trade_decisions``.
_MIN_VALID_EPOCH_SEC = 946_684_800  # 2000-01-01T00:00:00Z

# When two consecutive ``closed_trades`` rows are opposite-direction we treat
# the second as a flip only if their timing overlaps within this tolerance.
# Renko ticks aren't perfectly aligned so a small slack avoids classifying
# back-to-back independent re-entries as flips.
_FLIP_OVERLAP_TOLERANCE_SEC = 60

# Upper bound on rows read/upserted per backfill invocation so dashboard
# endpoints never scan the full historical tables in one request.
_DEFAULT_BACKFILL_BATCH_LIMIT = 500


_SCHEMA_SQL = """
create table if not exists trade_decisions (
  decision_id text primary key,
  ts timestamptz not null,
  venue text not null,
  symbol text not null,
  strategy text,
  strategy_instance text,
  decision_kind text not null check (decision_kind in ('entry', 'flip')),
  direction text not null check (direction in ('long', 'short')),
  position_before smallint,
  position_after smallint,
  engine_action text not null,
  reason_code text,
  source_action_event_id text,
  seq bigint,
  payload_json jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_trade_decisions_ts on trade_decisions (ts desc);
create index if not exists idx_trade_decisions_venue_symbol_ts
  on trade_decisions (venue, symbol, ts desc);
create index if not exists idx_trade_decisions_kind_ts
  on trade_decisions (decision_kind, ts desc);
create index if not exists idx_trade_decisions_source_action_event_id
  on trade_decisions (source_action_event_id);
"""


def ensure_trade_decisions_schema() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(_SCHEMA_SQL)


def upsert_trade_decision(row: Dict[str, Any]) -> None:
    """Upsert a single trade decision row by ``decision_id``."""

    ensure_trade_decisions_schema()
    sql = """
    insert into trade_decisions (
      decision_id, ts, venue, symbol, strategy, strategy_instance,
      decision_kind, direction, position_before, position_after,
      engine_action, reason_code, source_action_event_id, seq, payload_json
    ) values (
      %(decision_id)s, %(ts)s, %(venue)s, %(symbol)s, %(strategy)s, %(strategy_instance)s,
      %(decision_kind)s, %(direction)s, %(position_before)s, %(position_after)s,
      %(engine_action)s, %(reason_code)s, %(source_action_event_id)s, %(seq)s,
      %(payload_json)s::jsonb
    )
    on conflict (decision_id) do update set
      ts = excluded.ts,
      venue = excluded.venue,
      symbol = excluded.symbol,
      strategy = excluded.strategy,
      strategy_instance = excluded.strategy_instance,
      decision_kind = excluded.decision_kind,
      direction = excluded.direction,
      position_before = excluded.position_before,
      position_after = excluded.position_after,
      engine_action = excluded.engine_action,
      reason_code = excluded.reason_code,
      source_action_event_id = excluded.source_action_event_id,
      seq = excluded.seq,
      payload_json = excluded.payload_json
    """
    data = dict(row)
    data["payload_json"] = _payload(data.get("payload_json"))
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, data)


def upsert_trade_decisions(decisions: Iterable[TradeDecision]) -> int:
    """Upsert a batch of decisions. Returns the number written."""

    rows = [d.to_db_row() for d in decisions]
    if not rows:
        return 0
    ensure_trade_decisions_schema()
    sql = """
    insert into trade_decisions (
      decision_id, ts, venue, symbol, strategy, strategy_instance,
      decision_kind, direction, position_before, position_after,
      engine_action, reason_code, source_action_event_id, seq, payload_json
    ) values (
      %(decision_id)s, %(ts)s, %(venue)s, %(symbol)s, %(strategy)s, %(strategy_instance)s,
      %(decision_kind)s, %(direction)s, %(position_before)s, %(position_after)s,
      %(engine_action)s, %(reason_code)s, %(source_action_event_id)s, %(seq)s,
      %(payload_json)s::jsonb
    )
    on conflict (decision_id) do nothing
    """
    payload_rows = []
    for r in rows:
        rr = dict(r)
        rr["payload_json"] = _payload(rr.get("payload_json"))
        payload_rows.append(rr)
    with get_conn() as conn, conn.cursor() as cur:
        cur.executemany(sql, payload_rows)
    return len(payload_rows)


def count_trade_decisions(
    *,
    venue: Optional[str] = None,
    symbol: Optional[str] = None,
    decision_kind: Optional[str] = None,
    since_ts: Optional[str] = None,
) -> int:
    """Count trade decisions matching the given filters."""

    ensure_trade_decisions_schema()
    where: List[str] = []
    params: Dict[str, Any] = {}
    if venue:
        where.append("venue = %(venue)s")
        params["venue"] = str(venue)
    if symbol:
        where.append(
            "replace(replace(replace(upper(symbol), '-', ''), '_', ''), '/', '') = %(symbol_norm)s"
        )
        params["symbol_norm"] = _normalize_symbol_token(symbol)
    if decision_kind:
        where.append("decision_kind = %(decision_kind)s")
        params["decision_kind"] = decision_kind
    if since_ts:
        where.append("ts >= %(since_ts)s::timestamptz")
        params["since_ts"] = since_ts
    sql = "select count(*) from trade_decisions"
    if where:
        sql += " where " + " and ".join(where)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    if not row:
        return 0
    return int(row[0])


def list_recent_trade_decisions(
    *,
    venue: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Return the most recent decisions ordered newest-first."""

    ensure_trade_decisions_schema()
    where: List[str] = []
    params: Dict[str, Any] = {"limit": int(max(1, limit))}
    if venue:
        where.append("venue = %(venue)s")
        params["venue"] = str(venue)
    if symbol:
        where.append(
            "replace(replace(replace(upper(symbol), '-', ''), '_', ''), '/', '') = %(symbol_norm)s"
        )
        params["symbol_norm"] = _normalize_symbol_token(symbol)
    sql = """
    select decision_id, ts, venue, symbol, strategy, strategy_instance,
           decision_kind, direction, position_before, position_after,
           engine_action, reason_code, source_action_event_id, seq, payload_json
    from trade_decisions
    """
    if where:
        sql += " where " + " and ".join(where)
    sql += " order by ts desc, seq desc nulls last limit %(limit)s"
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall() or []
    columns = [
        "decision_id",
        "ts",
        "venue",
        "symbol",
        "strategy",
        "strategy_instance",
        "decision_kind",
        "direction",
        "position_before",
        "position_after",
        "engine_action",
        "reason_code",
        "source_action_event_id",
        "seq",
        "payload_json",
    ]
    out: List[Dict[str, Any]] = []
    for row in rows:
        record = dict(zip(columns, row))
        payload = record.get("payload_json")
        if isinstance(payload, (bytes, bytearray)):
            try:
                payload = json.loads(payload.decode("utf-8"))
            except Exception:
                payload = {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {}
        if not isinstance(payload, dict):
            payload = {}
        record["payload_json"] = payload
        ts_val = record.get("ts")
        if ts_val is not None and hasattr(ts_val, "isoformat"):
            record["ts"] = ts_val.isoformat()
        out.append(record)
    return out


def fetch_action_events_for_backfill(
    *,
    venue: Optional[str] = None,
    symbol: Optional[str] = None,
    since_ts: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Read raw rows from ``action_events`` for backfill / re-derivation."""

    where: List[str] = []
    params: Dict[str, Any] = {}
    if venue:
        where.append("venue = %(venue)s")
        params["venue"] = str(venue)
    if symbol:
        where.append(
            "replace(replace(replace(upper(symbol), '-', ''), '_', ''), '/', '') = %(symbol_norm)s"
        )
        params["symbol_norm"] = _normalize_symbol_token(symbol)
    if since_ts:
        where.append("ts >= %(since_ts)s::timestamptz")
        params["since_ts"] = since_ts
    sql = """
    select event_id, ts, seq, strategy, strategy_instance, symbol, venue,
           engine_action, action_side, position_before, position_after,
           reason_code, blocked, payload_json
    from action_events
    """
    if where:
        sql += " where " + " and ".join(where)
    sql += " order by ts asc, seq asc nulls last"
    if limit is not None:
        sql += " limit %(limit)s"
        params["limit"] = int(limit)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall() or []
    columns = [
        "event_id",
        "ts",
        "seq",
        "strategy",
        "strategy_instance",
        "symbol",
        "venue",
        "engine_action",
        "action_side",
        "position_before",
        "position_after",
        "reason_code",
        "blocked",
        "payload_json",
    ]
    out: List[Dict[str, Any]] = []
    for row in rows:
        record = dict(zip(columns, row))
        ts_val = record.get("ts")
        if ts_val is not None and hasattr(ts_val, "isoformat"):
            record["ts"] = ts_val.isoformat()
        out.append(record)
    return out


def backfill_trade_decisions_from_action_events(
    *,
    venue: Optional[str] = None,
    symbol: Optional[str] = None,
    since_ts: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Re-derive ``trade_decisions`` rows from existing ``action_events`` rows.

    Safe to run repeatedly: every decision id is deterministic and the upsert
    is a no-op on conflict. ``limit`` caps how many events are read per call
    (oldest-first after ``since_ts``) so dashboard backfill cannot scan the
    full table in one shot.
    """

    batch = int(limit) if limit is not None else _DEFAULT_BACKFILL_BATCH_LIMIT
    events = fetch_action_events_for_backfill(
        venue=venue, symbol=symbol, since_ts=since_ts, limit=max(1, batch)
    )
    decisions = build_trade_decisions_from_action_events(events)
    written = upsert_trade_decisions(decisions)
    return {
        "read_events": len(events),
        "decisions": len(decisions),
        "written": int(written),
        "venue": venue,
        "symbol": symbol,
        "batch_limit": batch,
        "since_ts": since_ts,
    }


# ---------------------------------------------------------------------------
# Backfill from ``closed_trades`` (long-tail historical spine).
# ---------------------------------------------------------------------------


def _ct_decision_id(*, venue: str, symbol: str, side: str, entry_ts_iso: str) -> str:
    """Deterministic, namespaced id for a decision derived from a
    ``closed_trades`` row.

    The ``td_ct_`` prefix keeps the id out of the action-event keyspace
    (``td_<sha>``) so a venue/symbol/side/entry_ts collision between the
    two backfills doesn't accidentally overwrite the authoritative
    action-event-derived decision.
    """

    base = "|".join([str(venue or ""), str(symbol or ""), str(side or ""), str(entry_ts_iso or "")])
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    return f"td_ct_{digest}"


def _coerce_iso(ts: Any) -> Optional[str]:
    """Best-effort ISO-8601 normaliser for the heterogeneous ts shapes we
    read out of Postgres / pandas."""

    if ts is None:
        return None
    if hasattr(ts, "isoformat"):
        try:
            return ts.isoformat()
        except Exception:
            pass
    s = str(ts).strip()
    return s or None


def _epoch_seconds_from_iso(ts: Any) -> Optional[int]:
    """Convert any timestamp-like to epoch seconds; return ``None`` for
    invalid / sentinel rows so the caller can skip them."""

    if ts is None:
        return None
    try:
        # Local import to avoid pulling pandas into the hot-import path of
        # event-emitting modules.
        import pandas as pd  # type: ignore

        t = pd.to_datetime(ts, utc=True, errors="coerce")
        if pd.isna(t):
            return None
        seconds = int(pd.Timestamp(t).timestamp())
    except Exception:
        return None
    if seconds < _MIN_VALID_EPOCH_SEC:
        return None
    return seconds


def _resolve_entry_ts_from_closed_trade(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    """Pick the real entry timestamp for a ``closed_trades`` row.

    Order of preference:

    1. ``entry_ts`` if it is a non-sentinel value (>= 2000-01-01Z).
    2. Any ``opened_at`` / ``created_at`` field present in ``payload_json``.
    3. The matched fill / bar timestamp from ``payload_json`` if available.

    Returns ``(iso_string, epoch_seconds)``. ``(None, None)`` means the row
    has no usable timestamp and should be skipped.
    """

    epoch = _epoch_seconds_from_iso(row.get("entry_ts"))
    if epoch is not None:
        iso = _coerce_iso(row.get("entry_ts"))
        if iso is not None:
            return iso, epoch
    payload = row.get("payload_json") or {}
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = json.loads(payload.decode("utf-8"))
        except Exception:
            payload = {}
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}
    if not isinstance(payload, dict):
        payload = {}
    for candidate_key in ("opened_at", "created_at", "entry_bar_ts", "bar_ts"):
        candidate = payload.get(candidate_key)
        epoch = _epoch_seconds_from_iso(candidate)
        if epoch is not None:
            iso = _coerce_iso(candidate)
            if iso is not None:
                return iso, epoch
    return None, None


def fetch_closed_trades_for_backfill(
    *,
    venue: Optional[str] = None,
    symbol: Optional[str] = None,
    after_entry_ts: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Read raw rows from ``closed_trades`` for the closed-trades backfill.

    Selects everything we need to reconstruct an entry-side decision and
    leaves filtering / normalisation to the caller so the SQL stays simple
    and easy to test.
    """

    where: List[str] = []
    params: Dict[str, Any] = {}
    if venue:
        where.append("venue = %(venue)s")
        params["venue"] = str(venue)
    if symbol:
        where.append(
            "replace(replace(replace(upper(symbol), '-', ''), '_', ''), '/', '') = %(symbol_norm)s"
        )
        params["symbol_norm"] = _normalize_symbol_token(symbol)
    if after_entry_ts:
        where.append("entry_ts > %(after_entry_ts)s::timestamptz")
        params["after_entry_ts"] = str(after_entry_ts)
    sql = """
    select trade_id, venue, symbol, side, entry_ts, exit_ts, strategy,
           strategy_instance, payload_json
    from closed_trades
    """
    if where:
        sql += " where " + " and ".join(where)
    # Order by exit_ts ascending so the flip detector sees adjacency in the
    # same direction the executor would have observed it.
    sql += " order by exit_ts asc nulls last, entry_ts asc nulls last"
    if limit is not None:
        sql += " limit %(limit)s"
        params["limit"] = int(max(1, limit))
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall() or []
    columns = [
        "trade_id",
        "venue",
        "symbol",
        "side",
        "entry_ts",
        "exit_ts",
        "strategy",
        "strategy_instance",
        "payload_json",
    ]
    out: List[Dict[str, Any]] = []
    for r in rows:
        record = dict(zip(columns, r))
        for k in ("entry_ts", "exit_ts"):
            v = record.get(k)
            if v is not None and hasattr(v, "isoformat"):
                record[k] = v.isoformat()
        out.append(record)
    return out


def build_trade_decisions_from_closed_trades(
    rows: Iterable[Dict[str, Any]],
) -> Tuple[List[TradeDecision], Dict[str, int]]:
    """Derive a list of :class:`TradeDecision` from raw ``closed_trades`` rows.

    Each closed leg becomes its own decision (``closed_trades`` semantics:
    every row is one closed leg). Two consecutive opposite-direction rows
    whose timing overlaps are recorded as ``(entry, flip)``. Rows with no
    usable entry timestamp (e.g. the historical 1970 sentinels) are
    skipped entirely — we'd rather under-count than write garbage.

    Returns ``(decisions, stats)`` where ``stats`` records skipped /
    flipped counters so callers (and tests) can assert on them.
    """

    materialised: List[Tuple[int, Dict[str, Any]]] = []
    skipped_invalid_ts = 0
    skipped_bad_side = 0
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        side = str(raw.get("side") or "").strip().lower()
        if side not in ("long", "short"):
            skipped_bad_side += 1
            continue
        entry_iso, entry_epoch = _resolve_entry_ts_from_closed_trade(raw)
        if entry_iso is None or entry_epoch is None:
            skipped_invalid_ts += 1
            continue
        materialised.append(
            (
                entry_epoch,
                {
                    **raw,
                    "_side": side,
                    "_entry_iso": entry_iso,
                    "_entry_epoch": entry_epoch,
                    "_exit_epoch": _epoch_seconds_from_iso(raw.get("exit_ts")) or entry_epoch,
                },
            )
        )

    # Process in chronological order of the *entry* event because that is
    # what defines the decision spine.
    materialised.sort(key=lambda pair: (pair[0], str(pair[1].get("trade_id") or "")))

    decisions: List[TradeDecision] = []
    flips = 0
    entries = 0
    prev_side: Optional[str] = None
    prev_exit_epoch: Optional[int] = None
    for _entry_epoch, row in materialised:
        side = row["_side"]
        venue = str(row.get("venue") or "")
        symbol = str(row.get("symbol") or "")
        entry_iso = row["_entry_iso"]
        entry_epoch = row["_entry_epoch"]
        kind = DECISION_ENTRY
        if (
            prev_side is not None
            and prev_exit_epoch is not None
            and prev_side != side
            and prev_exit_epoch >= entry_epoch - _FLIP_OVERLAP_TOLERANCE_SEC
        ):
            kind = DECISION_FLIP

        decision_id = _ct_decision_id(
            venue=venue, symbol=symbol, side=side, entry_ts_iso=entry_iso
        )
        engine_action = "enter_long" if side == "long" else "enter_short"
        if kind == DECISION_FLIP:
            engine_action = "flip_to_long" if side == "long" else "flip_to_short"

        payload = {
            "engine_action": engine_action,
            "action_side": side,
            "kind": kind,
            "direction": side,
            "source": "backfill:closed_trades",
            "closed_trade_id": row.get("trade_id"),
        }
        decisions.append(
            TradeDecision(
                decision_id=decision_id,
                ts=entry_iso,
                venue=venue,
                symbol=symbol,
                strategy=row.get("strategy"),
                strategy_instance=row.get("strategy_instance"),
                decision_kind=kind,
                direction=side,
                position_before=0 if kind == DECISION_ENTRY else (-1 if side == "long" else 1),
                position_after=1 if side == "long" else -1,
                engine_action=engine_action,
                reason_code="backfill:closed_trades",
                source_action_event_id=None,
                seq=None,
                payload=payload,
            )
        )
        if kind == DECISION_FLIP:
            flips += 1
        else:
            entries += 1
        prev_side = side
        prev_exit_epoch = row["_exit_epoch"]

    return decisions, {
        "entries": entries,
        "flips": flips,
        "skipped_invalid_ts": skipped_invalid_ts,
        "skipped_bad_side": skipped_bad_side,
    }


def backfill_trade_decisions_from_closed_trades(
    *,
    venue: Optional[str] = "kucoin",
    symbol: Optional[str] = None,
    after_entry_ts: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Backfill ``trade_decisions`` rows from the historical ``closed_trades``
    table.

    The ``action_events`` table was added later in the project lifecycle
    and therefore doesn't cover the full KuCoin trading history. Running
    the action-events backfill alone leaves a hole in the decision spine
    that is visible to the chart and the Performance card; this function
    fills that hole by emitting one decision per closed leg.

    Idempotent: ids are deterministic over ``(venue, symbol, side,
    entry_ts)`` and the upsert is a no-op on conflict, so re-running has
    no effect once the spine is populated.
    """

    batch = int(limit) if limit is not None else _DEFAULT_BACKFILL_BATCH_LIMIT
    rows = fetch_closed_trades_for_backfill(
        venue=venue,
        symbol=symbol,
        after_entry_ts=after_entry_ts,
        limit=max(1, batch),
    )
    decisions, stats = build_trade_decisions_from_closed_trades(rows)
    written = upsert_trade_decisions(decisions)
    return {
        "read_rows": len(rows),
        "decisions": len(decisions),
        "written": int(written),
        "venue": venue,
        "symbol": symbol,
        "batch_limit": batch,
        "after_entry_ts": after_entry_ts,
        **stats,
    }


def latest_decision_ts(
    *,
    venue: Optional[str] = None,
    symbol: Optional[str] = None,
) -> Optional[str]:
    """Return the most recent ``trade_decisions.ts`` for the filter, or
    ``None`` if the spine is empty. Used by the auto-backfill trigger to
    detect when ``closed_trades`` has moved on without us."""

    ensure_trade_decisions_schema()
    where: List[str] = []
    params: Dict[str, Any] = {}
    if venue:
        where.append("venue = %(venue)s")
        params["venue"] = str(venue)
    if symbol:
        where.append(
            "replace(replace(replace(upper(symbol), '-', ''), '_', ''), '/', '') = %(symbol_norm)s"
        )
        params["symbol_norm"] = _normalize_symbol_token(symbol)
    sql = "select max(ts) from trade_decisions"
    if where:
        sql += " where " + " and ".join(where)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    if not row or row[0] is None:
        return None
    v = row[0]
    if hasattr(v, "isoformat"):
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    return str(v)


def count_closed_trades(
    *,
    venue: Optional[str] = None,
    symbol: Optional[str] = None,
) -> int:
    """Cheap ``count(*)`` against ``closed_trades`` for the auto-backfill
    trigger heuristic. Keeps the SQL local to this module to avoid the
    dashboard_state in-process cache (we want a fresh count)."""

    where: List[str] = []
    params: Dict[str, Any] = {}
    if venue:
        where.append("venue = %(venue)s")
        params["venue"] = str(venue)
    if symbol:
        where.append(
            "replace(replace(replace(upper(symbol), '-', ''), '_', ''), '/', '') = %(symbol_norm)s"
        )
        params["symbol_norm"] = _normalize_symbol_token(symbol)
    sql = "select count(*) from closed_trades"
    if where:
        sql += " where " + " and ".join(where)
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
    except Exception:
        return 0
    if not row:
        return 0
    return int(row[0] or 0)


def latest_closed_trade_ts(
    *,
    venue: Optional[str] = None,
    symbol: Optional[str] = None,
) -> Optional[str]:
    """Return the most recent ``closed_trades.exit_ts`` for the filter."""

    where: List[str] = []
    params: Dict[str, Any] = {}
    if venue:
        where.append("venue = %(venue)s")
        params["venue"] = str(venue)
    if symbol:
        where.append(
            "replace(replace(replace(upper(symbol), '-', ''), '_', ''), '/', '') = %(symbol_norm)s"
        )
        params["symbol_norm"] = _normalize_symbol_token(symbol)
    sql = "select max(exit_ts) from closed_trades"
    if where:
        sql += " where " + " and ".join(where)
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
    except Exception:
        return None
    if not row or row[0] is None:
        return None
    v = row[0]
    if hasattr(v, "isoformat"):
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    return str(v)
