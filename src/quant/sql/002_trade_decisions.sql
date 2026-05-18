-- Trade decisions: one row per discrete directional position-opening decision
-- (entry from flat or flip to opposite direction). See
-- quant/execution/trade_counter.py for the authoritative classification rules.
--
-- This table is intentionally derived from action_events. Rebuilding the
-- entire table from history MUST produce identical rows (decision_id is
-- deterministic), so re-running the backfill is always safe.
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
