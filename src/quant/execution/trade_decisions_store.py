"""Postgres persistence helpers for the trade decision counter.

Schema is owned by :file:`src/quant/sql/002_trade_decisions.sql`. To keep the
table self-bootstrapping (matches the pattern used by other event tables in
:mod:`quant.execution.event_store`), :func:`ensure_trade_decisions_schema`
issues an idempotent ``CREATE TABLE IF NOT EXISTS`` block.

All write paths upsert by ``decision_id`` so re-processing the same
``action_events`` rows never produces duplicates.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from quant.execution.event_store import _payload, _normalize_symbol_token, get_conn
from quant.execution.trade_counter import (
    TradeDecision,
    build_trade_decisions_from_action_events,
)


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
) -> Dict[str, Any]:
    """Re-derive ``trade_decisions`` rows from existing ``action_events`` rows.

    Safe to run repeatedly: every decision id is deterministic and the upsert
    is a no-op on conflict.
    """

    events = fetch_action_events_for_backfill(
        venue=venue, symbol=symbol, since_ts=since_ts
    )
    decisions = build_trade_decisions_from_action_events(events)
    written = upsert_trade_decisions(decisions)
    return {
        "read_events": len(events),
        "decisions": len(decisions),
        "written": int(written),
        "venue": venue,
        "symbol": symbol,
    }
