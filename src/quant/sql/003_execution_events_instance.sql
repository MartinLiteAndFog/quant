-- Per-bot attribution + fill analysis on execution_events.
--
-- The four KuCoin pilot sub-accounts each write execution events. Without
-- strategy_instance on this table, fills from all of them are indistinguishable
-- and per-bot slippage/fill-quality analysis is impossible.

alter table execution_events
  add column if not exists strategy_instance text;

alter table execution_events
  add column if not exists config_hash text;

create index if not exists idx_execution_events_instance_ts
  on execution_events (strategy_instance, ts desc);

create index if not exists idx_execution_events_instance_stage
  on execution_events (strategy_instance, execution_stage, ts desc);

-- Fill-analysis helper: one row per order leg with the reference price the
-- decision was made at, so realised slippage can be computed against fills.
create or replace view execution_fill_analysis as
select
  e.strategy_instance,
  e.symbol,
  e.venue,
  e.execution_stage,
  e.side,
  e.reduce_only,
  e.qty,
  e.price                                as reference_price,
  (e.payload_json ->> 'mid_price')::numeric  as mid_price_at_decision,
  (e.payload_json ->> 'bid')::numeric        as bid_at_decision,
  (e.payload_json ->> 'ask')::numeric        as ask_at_decision,
  e.order_id,
  e.client_oid,
  e.status,
  e.reject_reason,
  e.ts
from execution_events e
where e.strategy_instance is not null;
