"""Durable comparison ledger for Brain Forward paper entry variants."""

from __future__ import annotations

from typing import Any, Iterable

from quant.brain_forward.evidence import ForwardProtocol
from quant.brain_forward.store import _get_conn, _payload


def ensure_variant_schema() -> None:
    sql = """
    create table if not exists brain_forward_variant_events (
      candidate_id text primary key, protocol_id text not null, protocol_sha256 text not null,
      artifact_sha256 text not null, evidence_phase text not null,
      variant_id text not null, event_ts timestamptz not null, status text not null,
      reason text not null, trigger_ts timestamptz, entry_ts timestamptz, entry_price numeric,
      payload_json jsonb not null default '{}'::jsonb,
      created_at timestamptz not null default now(), updated_at timestamptz not null default now()
    );
    create index if not exists idx_brain_forward_variant_events_variant_ts
      on brain_forward_variant_events (protocol_id, variant_id, event_ts desc);
    create table if not exists brain_forward_variant_trades (
      candidate_id text primary key, protocol_id text not null, protocol_sha256 text not null,
      artifact_sha256 text not null, evidence_phase text not null,
      variant_id text not null, event_ts timestamptz not null,
      entry_ts timestamptz not null, exit_ts timestamptz not null,
      entry_price numeric not null, exit_price numeric not null,
      target_price numeric not null, stop_price numeric not null, exit_reason text not null,
      gross_bps numeric not null, net_bps numeric not null, expected_net_bps numeric not null,
      created_at timestamptz not null default now(), updated_at timestamptz not null default now()
    );
    create index if not exists idx_brain_forward_variant_trades_variant_entry
      on brain_forward_variant_trades (protocol_id, variant_id, entry_ts desc);
    """
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)


def upsert_variant_events(
    events: Iterable[dict[str, Any]],
    *,
    artifact_sha256: str,
    protocol: ForwardProtocol,
) -> int:
    ensure_variant_schema()
    sql = """
    insert into brain_forward_variant_events
      (candidate_id,protocol_id,protocol_sha256,artifact_sha256,evidence_phase,
       variant_id,event_ts,status,reason,trigger_ts,entry_ts,entry_price,payload_json)
    values
      (%(candidate_id)s,%(protocol_id)s,%(protocol_sha256)s,%(artifact_sha256)s,%(evidence_phase)s,
       %(variant_id)s,%(event_ts)s,%(status)s,%(reason)s,%(trigger_ts)s,%(entry_ts)s,%(entry_price)s,%(payload_json)s::jsonb)
    on conflict (candidate_id) do update set
      status=excluded.status, reason=excluded.reason, trigger_ts=excluded.trigger_ts,
      entry_ts=excluded.entry_ts, entry_price=excluded.entry_price,
      payload_json=excluded.payload_json, updated_at=now()
    """
    rows = []
    for event in events:
        rows.append({
            **event,
            "candidate_id": f"{protocol.protocol_id}:{event['candidate_id']}",
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.protocol_sha256,
            "artifact_sha256": artifact_sha256,
            "evidence_phase": protocol.phase_at(event["event_ts"]),
            "payload_json": _payload(event.get("payload")),
        })
    if not rows:
        return 0
    with _get_conn() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)


def upsert_variant_trades(
    trades: Iterable[dict[str, Any]],
    *,
    artifact_sha256: str,
    protocol: ForwardProtocol,
) -> int:
    ensure_variant_schema()
    sql = """
    insert into brain_forward_variant_trades
      (candidate_id,protocol_id,protocol_sha256,artifact_sha256,evidence_phase,
       variant_id,event_ts,entry_ts,exit_ts,entry_price,exit_price,target_price,
       stop_price,exit_reason,gross_bps,net_bps,expected_net_bps)
    values
      (%(candidate_id)s,%(protocol_id)s,%(protocol_sha256)s,%(artifact_sha256)s,%(evidence_phase)s,
       %(variant_id)s,%(event_ts)s,%(entry_ts)s,%(exit_ts)s,%(entry_price)s,%(exit_price)s,%(target_price)s,
       %(stop_price)s,%(exit_reason)s,%(gross_bps)s,%(net_bps)s,%(expected_net_bps)s)
    on conflict (candidate_id) do update set
      exit_ts=excluded.exit_ts, exit_price=excluded.exit_price, exit_reason=excluded.exit_reason,
      gross_bps=excluded.gross_bps, net_bps=excluded.net_bps, updated_at=now()
    """
    rows = []
    for trade in trades:
        rows.append({
            **trade,
            "candidate_id": f"{protocol.protocol_id}:{trade['candidate_id']}",
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.protocol_sha256,
            "artifact_sha256": artifact_sha256,
            "evidence_phase": protocol.phase_at(trade["event_ts"]),
        })
    if not rows:
        return 0
    with _get_conn() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)
