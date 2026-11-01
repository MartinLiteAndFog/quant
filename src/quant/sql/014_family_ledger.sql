-- Family order manager: separate from the existing KuCoin executors.
-- Create with Python ensure_family_ledger_schema() after deployment is reviewed.
-- Do not use these tables as an authorization to send live orders.

create table if not exists family_order_intents (
  family_id text not null,
  client_oid text not null,
  strategy_id text not null,
  symbol text not null,
  position_side text not null check(position_side in ('LONG', 'SHORT')),
  operation text not null check(operation in ('OPEN', 'CLOSE')),
  contracts bigint not null check (contracts > 0),
  contract_multiplier numeric not null check (contract_multiplier > 0),
  filled_contracts bigint not null default 0
    check (filled_contracts >= 0 and filled_contracts <= contracts),
  state text not null default 'prepared'
    check(state in ('prepared', 'submitted', 'partially_filled',
                    'filled', 'cancelled', 'rejected')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key(family_id, client_oid)
);
create index if not exists idx_family_order_intents_strategy
  on family_order_intents(family_id, strategy_id, symbol, state);

create table if not exists family_order_fills (
  family_id text not null,
  fill_id text not null,
  client_oid text not null,
  contracts bigint not null check(contracts > 0),
  price numeric not null check(price > 0),
  fee numeric not null check(fee >= 0),
  observed_at timestamptz not null default now(),
  primary key(family_id, fill_id),
  foreign key(family_id, client_oid)
    references family_order_intents(family_id, client_oid)
);
create index if not exists idx_family_order_fills_oid
  on family_order_fills(family_id, client_oid);

create table if not exists family_virtual_positions (
  family_id text not null,
  strategy_id text not null,
  symbol text not null,
  position_side text not null check(position_side in ('LONG', 'SHORT')),
  contracts bigint not null default 0 check(contracts >= 0),
  average_entry numeric not null default 0 check(average_entry >= 0),
  gross_realized numeric not null default 0,
  fees numeric not null default 0 check(fees >= 0),
  updated_at timestamptz not null default now(),
  primary key(family_id, strategy_id, symbol, position_side)
);
