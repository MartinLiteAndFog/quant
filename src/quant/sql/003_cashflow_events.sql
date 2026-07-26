create table if not exists cashflow_events (
  event_id text primary key,
  ts timestamptz not null,
  venue text not null,
  account text not null,
  currency text not null,
  amount numeric not null,
  reporting_currency text not null default 'USD',
  reporting_amount numeric,
  fee numeric not null default 0,
  direction text not null check (direction in ('in', 'out')),
  flow_type text not null,
  status text not null,
  source_ref text,
  equity_after numeric,
  boundary_scope text not null default 'futures',
  payload_json jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_cashflow_events_account_ts
on cashflow_events (venue, account, ts);

create table if not exists cashflow_sync_state (
  venue text not null,
  account text not null,
  boundary_scope text not null default 'futures',
  coverage_start timestamptz,
  coverage_end timestamptz,
  last_success_at timestamptz,
  last_error text,
  source text not null,
  updated_at timestamptz not null default now(),
  primary key (venue, account)
);
