create table if not exists signal_events (
  event_id text primary key,
  ts timestamptz not null,
  seq bigint,
  strategy text not null,
  strategy_instance text,
  config_hash text not null,
  symbol text not null,
  venue text,
  signal smallint not null check (signal in (-1, 0, 1)),
  signal_side text not null check (signal_side in ('short', 'flat', 'long')),
  signal_family text not null,
  signal_kind text not null,
  source_event_id text,
  source_type text,
  position_before smallint,
  qty_before numeric,
  engine_mode_before text,
  regime_on boolean,
  gate_name text,
  payload_json jsonb not null default '{}'::jsonb
);

create index if not exists idx_signal_events_ts on signal_events (ts desc);
create index if not exists idx_signal_events_symbol_ts on signal_events (symbol, ts desc);
create index if not exists idx_signal_events_strategy_ts on signal_events (strategy, ts desc);


create table if not exists action_events (
  event_id text primary key,
  ts timestamptz not null,
  seq bigint,
  strategy text not null,
  strategy_instance text,
  config_hash text not null,
  symbol text not null,
  venue text,
  source_signal_event_id text references signal_events(event_id) on delete set null,
  source_event_id text,
  engine_action text not null,
  action_side text check (action_side in ('short', 'flat', 'long')),
  position_before smallint,
  position_after smallint,
  qty_before numeric,
  qty_after numeric,
  engine_mode_before text,
  engine_mode_after text,
  reason_code text not null,
  reason_detail text,
  blocked boolean not null default false,
  block_reason text,
  regime_state text,
  gate_name text,
  payload_json jsonb not null default '{}'::jsonb
);

create index if not exists idx_action_events_ts on action_events (ts desc);
create index if not exists idx_action_events_symbol_ts on action_events (symbol, ts desc);
create index if not exists idx_action_events_reason_ts on action_events (reason_code, ts desc);
create index if not exists idx_action_events_source_signal on action_events (source_signal_event_id);


create table if not exists execution_events (
  event_id text primary key,
  ts timestamptz not null,
  seq bigint,
  symbol text not null,
  venue text not null,
  source_action_event_id text references action_events(event_id) on delete set null,
  execution_stage text not null,
  order_id text,
  client_oid text,
  side text check (side in ('buy', 'sell')),
  qty numeric,
  price numeric,
  reduce_only boolean,
  status text,
  reject_reason text,
  payload_json jsonb not null default '{}'::jsonb
);

create index if not exists idx_execution_events_ts on execution_events (ts desc);
create index if not exists idx_execution_events_symbol_ts on execution_events (symbol, ts desc);
create index if not exists idx_execution_events_client_oid on execution_events (client_oid);
create index if not exists idx_execution_events_order_id on execution_events (order_id);


create table if not exists equity_snapshots (
  id bigserial primary key,
  ts timestamptz not null,
  venue text not null,
  account text,
  symbol text,
  equity numeric not null,
  currency text not null default 'USD',
  source text,
  payload_json jsonb not null default '{}'::jsonb,
  unique (venue, coalesce(account, ''), coalesce(symbol, ''), ts)
);

create index if not exists idx_equity_snapshots_ts on equity_snapshots (ts desc);
create index if not exists idx_equity_snapshots_venue_ts on equity_snapshots (venue, ts desc);


create table if not exists closed_trades (
  id bigserial primary key,
  trade_id text unique,
  venue text not null,
  symbol text not null,
  entry_ts timestamptz not null,
  exit_ts timestamptz not null,
  side text not null check (side in ('short', 'long')),
  qty numeric,
  entry_price numeric,
  exit_price numeric,
  pnl_pct numeric,
  exit_event text,
  strategy text,
  strategy_instance text,
  config_hash text,
  source_action_event_id text references action_events(event_id) on delete set null,
  payload_json jsonb not null default '{}'::jsonb
);

create index if not exists idx_closed_trades_exit_ts on closed_trades (exit_ts desc);
create index if not exists idx_closed_trades_symbol_exit_ts on closed_trades (symbol, exit_ts desc);