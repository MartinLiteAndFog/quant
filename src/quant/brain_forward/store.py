"""Durable, idempotent ledger for the paper-only Brain observer."""

from __future__ import annotations

import json
from typing import Any, Iterable

import pandas as pd

from quant.brain_forward.evidence import ForwardProtocol


def _payload(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False, separators=(",", ":"), default=str)


def _get_conn() -> Any:
    # Keep psycopg out of the pure runtime/checkpoint import path. The dedicated
    # Railway image supplies it; offline parity and governance tests do not need it.
    from quant.execution.event_store import get_conn

    return get_conn()


def ensure_schema() -> None:
    sql = """
    create table if not exists brain_forward_minute_bars (
      ts timestamptz not null, symbol text not null, source text not null,
      open numeric not null, high numeric not null, low numeric not null, close numeric not null,
      volume numeric not null, taker_base numeric not null, created_at timestamptz not null default now(),
      primary key (symbol, source, ts)
    );
    create index if not exists idx_brain_forward_minute_bars_symbol_ts
      on brain_forward_minute_bars (symbol, ts desc);
    create table if not exists brain_forward_decisions (
      decision_id text primary key, event_ts timestamptz not null, symbol text not null, source text not null,
      expected_net_bps numeric not null, candle_range numeric not null, active_memories integer not null,
      payload_json jsonb not null default '{}'::jsonb, created_at timestamptz not null default now()
    );
    create index if not exists idx_brain_forward_decisions_symbol_ts
      on brain_forward_decisions (symbol, event_ts desc);
    create table if not exists brain_forward_trades (
      decision_id text primary key, event_ts timestamptz not null, symbol text not null, source text not null,
      entry_ts timestamptz not null, exit_ts timestamptz not null, entry_price numeric not null, exit_price numeric not null,
      target_price numeric not null, stop_price numeric not null, exit_reason text not null,
      gross_bps numeric not null, net_bps numeric not null, expected_net_bps numeric not null,
      created_at timestamptz not null default now(), updated_at timestamptz not null default now()
    );
    create index if not exists idx_brain_forward_trades_symbol_entry
      on brain_forward_trades (symbol, entry_ts desc);
    create table if not exists brain_forward_protocols (
      protocol_id text primary key, protocol_sha256 text not null unique,
      spec_json jsonb not null, created_at timestamptz not null default now()
    );
    alter table brain_forward_decisions add column if not exists protocol_id text;
    alter table brain_forward_decisions add column if not exists protocol_sha256 text;
    alter table brain_forward_decisions add column if not exists artifact_sha256 text;
    alter table brain_forward_decisions add column if not exists evidence_phase text;
    alter table brain_forward_trades add column if not exists protocol_id text;
    alter table brain_forward_trades add column if not exists protocol_sha256 text;
    alter table brain_forward_trades add column if not exists artifact_sha256 text;
    alter table brain_forward_trades add column if not exists evidence_phase text;
    """
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)


def register_protocol(protocol: ForwardProtocol) -> None:
    """Register a protocol once and reject any same-ID mutation."""

    ensure_schema()
    sql = """
    insert into brain_forward_protocols (protocol_id, protocol_sha256, spec_json)
    values (%(protocol_id)s, %(protocol_sha256)s, %(spec_json)s::jsonb)
    on conflict (protocol_id) do nothing
    """
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, {
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.protocol_sha256,
            "spec_json": _payload(protocol.raw),
        })
        cur.execute(
            "select protocol_sha256 from brain_forward_protocols where protocol_id=%s",
            (protocol.protocol_id,),
        )
        row = cur.fetchone()
    if not row or str(row[0]) != protocol.protocol_sha256:
        raise RuntimeError("stored brain-forward protocol differs from frozen local protocol")


def upsert_minute_bars(symbol: str, source: str, bars: pd.DataFrame) -> int:
    ensure_schema()
    required = {"ts", "open", "high", "low", "close", "volume", "taker_base"}
    if bars.empty or required.difference(bars.columns):
        return 0
    work = bars.loc[:, sorted(required)].copy().dropna().drop_duplicates("ts", keep="last")
    sql = """
    insert into brain_forward_minute_bars (ts, symbol, source, open, high, low, close, volume, taker_base)
    values (%(ts)s, %(symbol)s, %(source)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, %(taker_base)s)
    on conflict (symbol, source, ts) do update set
      open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close,
      volume=excluded.volume, taker_base=excluded.taker_base
    """
    rows = [{"symbol": symbol, "source": source, **record} for record in work.to_dict("records")]
    with _get_conn() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)


def upsert_decisions(
    symbol: str,
    source: str,
    decisions: Iterable[Any],
    *,
    artifact_sha256: str,
    protocol: ForwardProtocol,
) -> int:
    ensure_schema()
    sql = """
    insert into brain_forward_decisions (decision_id,event_ts,symbol,source,expected_net_bps,candle_range,active_memories,payload_json,protocol_id,protocol_sha256,artifact_sha256,evidence_phase)
    values (%(decision_id)s,%(event_ts)s,%(symbol)s,%(source)s,%(expected_net_bps)s,%(candle_range)s,%(active_memories)s,%(payload_json)s::jsonb,%(protocol_id)s,%(protocol_sha256)s,%(artifact_sha256)s,%(evidence_phase)s)
    on conflict (decision_id) do nothing
    """
    rows = []
    for decision in decisions:
        rows.append({
            "decision_id": f"{protocol.protocol_id}:brain-forward:{decision.event_ts.isoformat()}", "event_ts": decision.event_ts,
            "symbol": symbol, "source": source, "expected_net_bps": decision.expected_net_bps,
            "candle_range": decision.candle_range, "active_memories": decision.active_memories,
            "protocol_id": protocol.protocol_id, "protocol_sha256": protocol.protocol_sha256,
            "artifact_sha256": artifact_sha256,
            "evidence_phase": protocol.phase_at(decision.event_ts),
            "payload_json": _payload({"artifact_sha256": artifact_sha256, "protocol_id": protocol.protocol_id, "protocol_sha256": protocol.protocol_sha256, "shock_z": decision.shock_z, "close_position": decision.close_position, "volatility_ratio": decision.volatility_ratio, "flow_imbalance": decision.flow_imbalance}),
        })
    if not rows:
        return 0
    with _get_conn() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)


def upsert_trades(
    symbol: str,
    source: str,
    trades: Iterable[dict[str, object]],
    *,
    artifact_sha256: str,
    protocol: ForwardProtocol,
) -> int:
    ensure_schema()
    sql = """
    insert into brain_forward_trades (decision_id,event_ts,symbol,source,entry_ts,exit_ts,entry_price,exit_price,target_price,stop_price,exit_reason,gross_bps,net_bps,expected_net_bps,protocol_id,protocol_sha256,artifact_sha256,evidence_phase)
    values (%(decision_id)s,%(event_ts)s,%(symbol)s,%(source)s,%(entry_ts)s,%(exit_ts)s,%(entry_price)s,%(exit_price)s,%(target_price)s,%(stop_price)s,%(exit_reason)s,%(gross_bps)s,%(net_bps)s,%(expected_net_bps)s,%(protocol_id)s,%(protocol_sha256)s,%(artifact_sha256)s,%(evidence_phase)s)
    on conflict (decision_id) do update set
      exit_ts=excluded.exit_ts, exit_price=excluded.exit_price, exit_reason=excluded.exit_reason,
      gross_bps=excluded.gross_bps, net_bps=excluded.net_bps, updated_at=now()
    """
    rows = []
    for trade in trades:
        event_ts = trade["event_ts"]
        rows.append({
            "symbol": symbol,
            "source": source,
            **trade,
            "decision_id": f"{protocol.protocol_id}:{trade['decision_id']}",
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.protocol_sha256,
            "artifact_sha256": artifact_sha256,
            "evidence_phase": protocol.phase_at(event_ts),
        })
    if not rows:
        return 0
    with _get_conn() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)
