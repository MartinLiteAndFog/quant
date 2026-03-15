from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Iterator, Optional

import psycopg


def _pg_dsn() -> str:
    dsn = (os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL") or "").strip()
    if not dsn:
        raise RuntimeError("POSTGRES_URL or DATABASE_URL is not set")
    return dsn


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(_pg_dsn(), autocommit=True)
    try:
        yield conn
    finally:
        conn.close()


def _to_jsonable(v: Any) -> Any:
    if is_dataclass(v):
        return {k: _to_jsonable(val) for k, val in asdict(v).items()}
    if isinstance(v, dict):
        return {str(k): _to_jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, datetime):
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc).isoformat()
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    return v


def _payload(v: Optional[Dict[str, Any]]) -> str:
    return json.dumps(_to_jsonable(v or {}), ensure_ascii=False, separators=(",", ":"))


def insert_signal_event(row: Dict[str, Any]) -> None:
    sql = """
    insert into signal_events (
      event_id, ts, seq, strategy, strategy_instance, config_hash, symbol, venue,
      signal, signal_side, signal_family, signal_kind,
      source_event_id, source_type,
      position_before, qty_before, engine_mode_before,
      regime_on, gate_name, payload_json
    ) values (
      %(event_id)s, %(ts)s, %(seq)s, %(strategy)s, %(strategy_instance)s, %(config_hash)s, %(symbol)s, %(venue)s,
      %(signal)s, %(signal_side)s, %(signal_family)s, %(signal_kind)s,
      %(source_event_id)s, %(source_type)s,
      %(position_before)s, %(qty_before)s, %(engine_mode_before)s,
      %(regime_on)s, %(gate_name)s, %(payload_json)s::jsonb
    )
    on conflict (event_id) do nothing
    """
    data = dict(row)
    data.setdefault("strategy_instance", None)
    data.setdefault("config_hash", data.get("strategy") or "unknown")
    data.setdefault("source_type", None)
    data.setdefault("qty_before", None)
    data.setdefault("regime_on", None)
    data.setdefault("gate_name", None)
    data["payload_json"] = _payload(data.get("payload_json"))
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, data)


def insert_action_event(row: Dict[str, Any]) -> None:
    sql = """
    insert into action_events (
      event_id, ts, seq, strategy, strategy_instance, config_hash, symbol, venue,
      source_signal_event_id, source_event_id,
      engine_action, action_side,
      position_before, position_after,
      qty_before, qty_after,
      engine_mode_before, engine_mode_after,
      reason_code, reason_detail,
      blocked, block_reason,
      regime_state, gate_name, payload_json
    ) values (
      %(event_id)s, %(ts)s, %(seq)s, %(strategy)s, %(strategy_instance)s, %(config_hash)s, %(symbol)s, %(venue)s,
      %(source_signal_event_id)s, %(source_event_id)s,
      %(engine_action)s, %(action_side)s,
      %(position_before)s, %(position_after)s,
      %(qty_before)s, %(qty_after)s,
      %(engine_mode_before)s, %(engine_mode_after)s,
      %(reason_code)s, %(reason_detail)s,
      %(blocked)s, %(block_reason)s,
      %(regime_state)s, %(gate_name)s, %(payload_json)s::jsonb
    )
    on conflict (event_id) do nothing
    """
    data = dict(row)
    data["payload_json"] = _payload(data.get("payload_json"))
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, data)


def insert_execution_event(row: Dict[str, Any]) -> None:
    sql = """
    insert into execution_events (
      event_id, ts, seq, symbol, venue, source_action_event_id,
      execution_stage, order_id, client_oid, side, qty, price,
      reduce_only, status, reject_reason, payload_json
    ) values (
      %(event_id)s, %(ts)s, %(seq)s, %(symbol)s, %(venue)s, %(source_action_event_id)s,
      %(execution_stage)s, %(order_id)s, %(client_oid)s, %(side)s, %(qty)s, %(price)s,
      %(reduce_only)s, %(status)s, %(reject_reason)s, %(payload_json)s::jsonb
    )
    on conflict (event_id) do nothing
    """
    data = dict(row)
    data["payload_json"] = _payload(data.get("payload_json"))
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, data)


def insert_equity_snapshot(row: Dict[str, Any]) -> None:
    sql = """
    insert into equity_snapshots (
      ts, venue, account, symbol, equity, currency, source, payload_json
    ) values (
      %(ts)s, %(venue)s, %(account)s, %(symbol)s, %(equity)s, %(currency)s, %(source)s, %(payload_json)s::jsonb
    )
    on conflict (venue, coalesce(account, ''), coalesce(symbol, ''), ts) do nothing
    """
    data = dict(row)
    data.setdefault("currency", "USD")
    data["payload_json"] = _payload(data.get("payload_json"))
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, data)


def upsert_closed_trade(row: Dict[str, Any]) -> None:
    sql = """
    insert into closed_trades (
      trade_id, venue, symbol, entry_ts, exit_ts, side, qty,
      entry_price, exit_price, pnl_pct, exit_event,
      strategy, strategy_instance, config_hash,
      source_action_event_id, payload_json
    ) values (
      %(trade_id)s, %(venue)s, %(symbol)s, %(entry_ts)s, %(exit_ts)s, %(side)s, %(qty)s,
      %(entry_price)s, %(exit_price)s, %(pnl_pct)s, %(exit_event)s,
      %(strategy)s, %(strategy_instance)s, %(config_hash)s,
      %(source_action_event_id)s, %(payload_json)s::jsonb
    )
    on conflict (trade_id) do update set
      venue = excluded.venue,
      symbol = excluded.symbol,
      entry_ts = excluded.entry_ts,
      exit_ts = excluded.exit_ts,
      side = excluded.side,
      qty = excluded.qty,
      entry_price = excluded.entry_price,
      exit_price = excluded.exit_price,
      pnl_pct = excluded.pnl_pct,
      exit_event = excluded.exit_event,
      strategy = excluded.strategy,
      strategy_instance = excluded.strategy_instance,
      config_hash = excluded.config_hash,
      source_action_event_id = excluded.source_action_event_id,
      payload_json = excluded.payload_json
    """
    data = dict(row)
    data["payload_json"] = _payload(data.get("payload_json"))
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, data)