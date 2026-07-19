from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Iterator, Optional

import pandas as pd
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


def _normalize_symbol_token(symbol: str) -> str:
    return "".join(ch for ch in str(symbol or "").upper() if ch.isalnum())


def ensure_daily_gate_history_schema() -> None:
    sql = """
    create table if not exists daily_gate_history (
      id bigserial primary key,
      ts timestamptz not null,
      symbol text not null,
      gate_on smallint not null check (gate_on in (0, 1)),
      gate_off smallint not null check (gate_off in (0, 1)),
      gate_countertrend_on smallint not null check (gate_countertrend_on in (0, 1)),
      gate_trend_on smallint not null check (gate_trend_on in (0, 1)),
      source text,
      payload_json jsonb not null default '{}'::jsonb,
      created_at timestamptz not null default now()
    );
    create unique index if not exists uq_daily_gate_history_symbol_ts
    on daily_gate_history (symbol, ts);
    create index if not exists idx_daily_gate_history_symbol_ts
    on daily_gate_history (symbol, ts desc);
    create index if not exists idx_daily_gate_history_ts
    on daily_gate_history (ts desc);
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)


def ensure_live_renko_bricks_schema() -> None:
    sql = """
    create table if not exists live_renko_bricks (
      id bigserial primary key,
      ts timestamptz not null,
      symbol text not null,
      open numeric not null,
      high numeric not null,
      low numeric not null,
      close numeric not null,
      source text,
      payload_json jsonb not null default '{}'::jsonb,
      created_at timestamptz not null default now()
    );
    create unique index if not exists uq_live_renko_bricks_symbol_ts
    on live_renko_bricks (symbol, ts);
    create index if not exists idx_live_renko_bricks_symbol_ts
    on live_renko_bricks (symbol, ts desc);
    create index if not exists idx_live_renko_bricks_ts
    on live_renko_bricks (ts desc);
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)


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
    # Optional columns default to NULL. Without these, a caller that omits one
    # key raises during binding and the whole event is silently lost to the
    # caller's except-and-warn handler.
    for key in (
        "strategy_instance", "venue", "source_signal_event_id", "source_event_id",
        "action_side", "position_before", "position_after", "qty_before", "qty_after",
        "engine_mode_before", "engine_mode_after", "reason_detail", "block_reason",
        "regime_state", "gate_name",
    ):
        data.setdefault(key, None)
    data.setdefault("config_hash", data.get("strategy") or "unknown")
    data.setdefault("blocked", False)
    data["payload_json"] = _payload(data.get("payload_json"))
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, data)


def insert_execution_event(row: Dict[str, Any]) -> None:
    sql = """
    insert into execution_events (
      event_id, ts, seq, symbol, venue, source_action_event_id,
      execution_stage, order_id, client_oid, side, qty, price,
      reduce_only, status, reject_reason, strategy_instance, config_hash, payload_json
    ) values (
      %(event_id)s, %(ts)s, %(seq)s, %(symbol)s, %(venue)s, %(source_action_event_id)s,
      %(execution_stage)s, %(order_id)s, %(client_oid)s, %(side)s, %(qty)s, %(price)s,
      %(reduce_only)s, %(status)s, %(reject_reason)s, %(strategy_instance)s,
      %(config_hash)s, %(payload_json)s::jsonb
    )
    on conflict (event_id) do nothing
    """
    data = dict(row)
    # `price` and `reject_reason` were referenced by this SQL but never supplied
    # by the TradingView executor, so every one of its execution events failed
    # to persist and was swallowed by the caller's warning handler.
    for key in (
        "seq", "venue", "source_action_event_id", "order_id", "client_oid",
        "side", "qty", "price", "reduce_only", "status", "reject_reason",
        "strategy_instance", "config_hash",
    ):
        data.setdefault(key, None)
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


def upsert_daily_gate_history(row: Dict[str, Any]) -> None:
    ensure_daily_gate_history_schema()
    sql = """
    insert into daily_gate_history (
      ts, symbol, gate_on, gate_off, gate_countertrend_on, gate_trend_on, source, payload_json
    ) values (
      %(ts)s, %(symbol)s, %(gate_on)s, %(gate_off)s, %(gate_countertrend_on)s, %(gate_trend_on)s, %(source)s, %(payload_json)s::jsonb
    )
    on conflict (symbol, ts) do update set
      gate_on = excluded.gate_on,
      gate_off = excluded.gate_off,
      gate_countertrend_on = excluded.gate_countertrend_on,
      gate_trend_on = excluded.gate_trend_on,
      source = excluded.source,
      payload_json = excluded.payload_json
    """
    data = dict(row)
    data["payload_json"] = _payload(data.get("payload_json"))
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, data)


def load_latest_daily_gate_from_postgres(
    *,
    symbol: str,
    now_ts: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    ensure_daily_gate_history_schema()
    sql = """
    select
      ts,
      symbol,
      gate_on,
      gate_off,
      gate_countertrend_on,
      gate_trend_on,
      source,
      payload_json
    from daily_gate_history
    where replace(replace(replace(upper(symbol), '-', ''), '_', ''), '/', '') = %(symbol_norm)s
      and (%(now_ts)s::timestamptz is null or ts <= %(now_ts)s::timestamptz)
    order by ts desc
    limit 1
    """
    params = {
        "symbol_norm": _normalize_symbol_token(symbol),
        "now_ts": now_ts,
    }
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    if not row:
        return None
    ts, out_symbol, gate_on, gate_off, gate_countertrend_on, gate_trend_on, source, payload_json = row
    payload = payload_json if isinstance(payload_json, dict) else {}
    return {
        "ts": ts,
        "symbol": out_symbol,
        "gate_on": int(gate_on),
        "gate_off": int(gate_off),
        "gate_countertrend_on": int(gate_countertrend_on),
        "gate_trend_on": int(gate_trend_on),
        "source": str(source or "postgres_daily_gate"),
        "payload_json": payload,
    }


def upsert_live_renko_bricks(
    *,
    symbol: str,
    renko: pd.DataFrame,
    source: Optional[str] = None,
    payload_json: Optional[Dict[str, Any]] = None,
) -> int:
    ensure_live_renko_bricks_schema()
    if renko is None or renko.empty:
        return 0
    work = renko.copy()
    need = {"ts", "open", "high", "low", "close"}
    missing = need - set(work.columns)
    if missing:
        raise ValueError(f"missing_renko_columns:{sorted(missing)}")
    work["ts"] = pd.to_datetime(work["ts"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    if work.empty:
        return 0
    sql = """
    insert into live_renko_bricks (
      ts, symbol, open, high, low, close, source, payload_json
    ) values (
      %(ts)s, %(symbol)s, %(open)s, %(high)s, %(low)s, %(close)s, %(source)s, %(payload_json)s::jsonb
    )
    on conflict (symbol, ts) do update set
      open = excluded.open,
      high = excluded.high,
      low = excluded.low,
      close = excluded.close,
      source = excluded.source,
      payload_json = excluded.payload_json
    """
    rows = []
    payload = _payload(payload_json)
    for row in work.to_dict("records"):
        rows.append(
            {
                "ts": row["ts"],
                "symbol": str(symbol),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "source": source,
                "payload_json": payload,
            }
        )
    with get_conn() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)


def load_live_renko_bricks_from_postgres(
    *,
    symbol: str,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    ensure_live_renko_bricks_schema()
    sql = """
    select ts, open, high, low, close
    from live_renko_bricks
    where replace(replace(replace(upper(symbol), '-', ''), '_', ''), '/', '') = %(symbol_norm)s
      and (%(start_ts)s::timestamptz is null or ts >= %(start_ts)s::timestamptz)
      and (%(end_ts)s::timestamptz is null or ts <= %(end_ts)s::timestamptz)
    order by ts desc
    """
    if limit is not None:
        sql += " limit %(limit)s"
    params = {
        "symbol_norm": _normalize_symbol_token(symbol),
        "start_ts": start_ts,
        "end_ts": end_ts,
    }
    if limit is not None:
        params["limit"] = int(limit)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close"])
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts").reset_index(drop=True)


def prune_live_renko_bricks_before(
    *,
    symbol: str,
    cutoff_ts: str,
) -> int:
    ensure_live_renko_bricks_schema()
    sql = """
    delete from live_renko_bricks
    where replace(replace(replace(upper(symbol), '-', ''), '_', ''), '/', '') = %(symbol_norm)s
      and ts < %(cutoff_ts)s::timestamptz
    """
    params = {
        "symbol_norm": _normalize_symbol_token(symbol),
        "cutoff_ts": cutoff_ts,
    }
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return int(cur.rowcount or 0)


def ensure_execution_calibration_schema() -> None:
    sql = """
    create table if not exists execution_calibration (
      telemetry_id text primary key,
      decision_ts timestamptz not null,
      submitted_ts timestamptz not null,
      acknowledged_ts timestamptz,
      filled_ts timestamptz,
      result_ts timestamptz not null,
      venue text not null,
      strategy text not null,
      symbol text not null,
      action text not null,
      exit_reason text,
      side text,
      reference_bid numeric,
      reference_ask numeric,
      reference_mid numeric,
      requested_qty numeric not null,
      filled_qty numeric,
      avg_fill_price numeric,
      order_id text,
      client_oid text,
      order_type text,
      liquidity text,
      fee numeric,
      fee_currency text,
      fee_bps numeric,
      requotes integer,
      fallback_used boolean not null default false,
      rejected boolean not null default false,
      reject_reason text,
      reduce_only boolean,
      status text not null,
      submit_to_ack_ms numeric,
      submit_to_fill_ms numeric,
      decision_to_result_ms numeric,
      slippage_bps numeric,
      timing_precision text not null,
      filled_qty_inferred boolean not null default false,
      fill_price_source text,
      fee_source text,
      created_at timestamptz not null default now()
    );
    create index if not exists idx_execution_calibration_symbol_decision
    on execution_calibration (symbol, decision_ts desc);
    create index if not exists idx_execution_calibration_order_id
    on execution_calibration (order_id);
    create index if not exists idx_execution_calibration_action_decision
    on execution_calibration (action, decision_ts desc);
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)


def insert_execution_calibration(row: Dict[str, Any]) -> None:
    ensure_execution_calibration_schema()
    sql = """
    insert into execution_calibration (
      telemetry_id, decision_ts, submitted_ts, acknowledged_ts, filled_ts, result_ts,
      venue, strategy, symbol, action, exit_reason, side,
      reference_bid, reference_ask, reference_mid, requested_qty, filled_qty, avg_fill_price,
      order_id, client_oid, order_type, liquidity, fee, fee_currency, fee_bps, requotes,
      fallback_used, rejected, reject_reason, reduce_only, status,
      submit_to_ack_ms, submit_to_fill_ms, decision_to_result_ms, slippage_bps,
      timing_precision, filled_qty_inferred, fill_price_source, fee_source
    ) values (
      %(telemetry_id)s, %(decision_ts)s, %(submitted_ts)s, %(acknowledged_ts)s, %(filled_ts)s, %(result_ts)s,
      %(venue)s, %(strategy)s, %(symbol)s, %(action)s, %(exit_reason)s, %(side)s,
      %(reference_bid)s, %(reference_ask)s, %(reference_mid)s, %(requested_qty)s, %(filled_qty)s, %(avg_fill_price)s,
      %(order_id)s, %(client_oid)s, %(order_type)s, %(liquidity)s, %(fee)s, %(fee_currency)s, %(fee_bps)s, %(requotes)s,
      %(fallback_used)s, %(rejected)s, %(reject_reason)s, %(reduce_only)s, %(status)s,
      %(submit_to_ack_ms)s, %(submit_to_fill_ms)s, %(decision_to_result_ms)s, %(slippage_bps)s,
      %(timing_precision)s, %(filled_qty_inferred)s, %(fill_price_source)s, %(fee_source)s
    )
    on conflict (telemetry_id) do nothing
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, dict(row))


def load_execution_calibration(
    *,
    symbol: Optional[str] = None,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
) -> list[Dict[str, Any]]:
    ensure_execution_calibration_schema()
    columns = [
        "telemetry_id", "decision_ts", "submitted_ts", "acknowledged_ts", "filled_ts", "result_ts",
        "venue", "strategy", "symbol", "action", "exit_reason", "side",
        "reference_bid", "reference_ask", "reference_mid", "requested_qty", "filled_qty", "avg_fill_price",
        "order_id", "client_oid", "order_type", "liquidity", "fee", "fee_currency", "fee_bps", "requotes",
        "fallback_used", "rejected", "reject_reason", "reduce_only", "status",
        "submit_to_ack_ms", "submit_to_fill_ms", "decision_to_result_ms", "slippage_bps",
        "timing_precision", "filled_qty_inferred", "fill_price_source", "fee_source",
    ]
    sql = f"""
    select {', '.join(columns)}
    from execution_calibration
    where (%(symbol)s is null or symbol = %(symbol)s)
      and (%(start_ts)s::timestamptz is null or decision_ts >= %(start_ts)s::timestamptz)
      and (%(end_ts)s::timestamptz is null or decision_ts <= %(end_ts)s::timestamptz)
    order by decision_ts, submitted_ts, telemetry_id
    """
    params = {"symbol": symbol, "start_ts": start_ts, "end_ts": end_ts}
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [dict(zip(columns, row)) for row in rows]
