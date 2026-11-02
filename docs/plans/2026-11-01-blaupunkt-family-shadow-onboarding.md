# Blaupunkt: add TradingView parent/child strategies to one Railway family

Status: DEVELOPMENT / SHADOW ONLY. None of the new family endpoints is
currently deployed on the LIVE Blaupunkt service. The old
/webhook/tv-execute path must keep working unchanged.

## Architecture

One KuCoin Futures subaccount per high-level strategy family ("Blaupunkt"),
with multiple independent parent strategy roots and nested daughters.
TradingView computes the actual signals. Railway manages identities,
independent positions, 90% equity allocation, and eventually one central
order writer. No child orders exist until live migration is explicitly
approved.

The legacy Blaupunkt TradingView alert keeps its current address:

  https://blaupunkt-production.up.railway.app/webhook/tv-execute

ALL 12 family strategy IDs eventually share ONE URL:

  https://blaupunkt-production.up.railway.app/webhook/family

Until the controlled migration, the existing Blaupunkt parent keeps its
old /webhook/tv-execute alarm. No per-strategy URL or per-strategy token.

Both paths are served from ONE ASGI process in the future app entrypoint:

  python -u -m quant.execution.blaupunkt_family_app

This command is only for a FUTURE controlled deployment; do not change the
production Railway start command as part of this draft PR.

A separate safe test app is also available:

  python -u -m quant.execution.family_shadow_app

It does NOT import a broker, doesn't start a legacy executor, has no KuCoin
credentials and never submits orders.

## Database and variables

The family registry/inbox creates its own tables on first authorized call.
A separate durable order/fill ledger is defined in:

  src/quant/sql/014_family_ledger.sql

Variables needed for an eventual STAGING exercise (not yet configured):
- DATABASE_URL: connection to shared Postgres, kept secret in Railway.
- FAMILY_ID=blaupunkt
- FAMILY_ROUTER_ENABLED=1
- FAMILY_ADMIN_TOKEN=<separate secret for admin registration API>
- FAMILY_WEBHOOK_TOKEN=<ONE shared secret for ALL 12 TradingView alerts>

Keep FAMILY_ADMIN_TOKEN and FAMILY_WEBHOOK_TOKEN separate. A strategy_id
is a routing identifier, NOT authentication. Put the shared webhook token
only into TradingView's private per-alert input/message and Railway env,
never into versioned source control. The endpoint rejects invalid tokens,
unknown strategy IDs and replayed/conflicting event IDs.

## 1. Bootstrap existing Blaupunkt parent

After the shadow API has been safely launched in staging:

  POST /api/family/bootstrap-legacy
  X-Family-Admin-Token: <ADMIN_TOKEN>

This creates a record for the existing Blaupunkt parent with an unusable
random shadow token. The existing TradingView alarm is NOT touched, no live
position is imported, and there is no new order.

## 2. Register a second parent

  POST /api/family/strategies
  X-Family-Admin-Token: <ADMIN_TOKEN>
  Content-Type: application/json

  {
    "strategy_id": "blaupunkt-parent-002",
    "display_name": "IMBA variant #2",
    "pool": "evolution",
    "configuration": {"imba_sensitivity": 18.2, "ttp_pct": 0.5}
  }

Server returns the strategy_id. NO token is generated per strategy.
All strategy IDs use the same FAMILY_WEBHOOK_TOKEN configured on Railway.
No subaccount API keys are generated per strategy.

For a daughter, add "parent_strategy_id":"blaupunkt" or the respective parent.
Use the same "pool" as its direct parent.

The registry tracks parent relationships and stores the configuration as
metadata. It DOES NOT change the TradingView strategy's sensitivity by API:
the operator must set up the TradingView input values for that instance.

## 3. TradingView setup

For new IMBA-family instances, copy tradingview/sol_imba_family.pine into
TradingView (after Pine compilation/visual verification). The strategy
calculation remains the same as sol_imba_strategy.pine; webhook metadata
adds strategy_id and event_id.

In the strategy's Railway webhook inputs:
- Strategy ID / Source = blaupunkt-parent-002
- Token = the SAME FAMILY_WEBHOOK_TOKEN for every strategy
- Symbol = SOL-USDT
- Enable the chosen alert mechanism (strategy order fills or alert() calls)
  consistently with the existing TV strategy; do not enable duplicates.

Configure the TradingView alert webhook URL:
  https://blaupunkt-production.up.railway.app/webhook/family

The Pine alert JSON includes the strategy_id, token, action, side,
symbol, source, reason, bar_time, bar_index, and event_id. The event_id
is based on source + bar timestamp + bar index + action + reason.
At most one identical such alert per bar is expected; if the strategy can
emit truly distinct identical events on one bar, this needs a stronger
sequence identifier BEFORE live use.

The URL won't accept signals on production until the family router is
mounted and explicitly enabled. For testing, point alerts to a standalone
staging shadow service with its own domain.

## 4. Bootstrap the 3x3 pilot and inspect budgets

After the shadow API has been enabled, authenticate with
X-Family-Admin-Token to:

- GET /api/family/pilot-roster : preview 3 parents + 9 daughters
- POST /api/family/bootstrap-pilot : register the legacy parent plus 11
  new IDs, one time (no trading, no new webhook tokens)
- POST /api/family/strategies : register later parents/daughters individually

The immutable starter IDs are:
- Parent 1 blaupunkt; daughters blaupunkt-d01/d02/d03
- Parent 2 blaupunkt-p02; daughters blaupunkt-p02-d01/d02/d03
- Parent 3 blaupunkt-p03; daughters blaupunkt-p03-d01/d02/d03

All 11 newly created TradingView alerts share the exact same URL and
FAMILY_WEBHOOK_TOKEN. Only strategy_id, event_id, action and side differ.
The existing Blaupunkt parent's alarm remains on the legacy path until
safe migration.

## 5. Inspect received alerts and budgets

Authorized read-only endpoints:
- GET /api/family/strategies
- GET /api/family/inbox?limit=50
- GET /api/family/template/blaupunkt-parent-002
- GET /api/family/allocation-preview?equity=1000

The allocation-preview accepts equity supplied by the caller. It is NOT
a real KuCoin balance yet. All values are simulated budgets; no order
is sent, no position is transferred.

At 1000 USDT with three root parents, each root family receives
300 USDT (90% split by 3); 100 USDT remain reserved. Each child initially
receives 2% of its DIRECT parent's allocated budget; descendants are nested.

Angel/Fishing bots remain outside this evolution-family 90% split and
need an explicitly separate budget before sharing a funded account.

## 6. Hard blockers to LIVE multi-bot execution

- Current Blaupunkt executor reads one signed net position per symbol and
  can cancel ALL stops/orders. This is INCOMPATIBLE with child execution.
  It must be migrated to a single hedge-aware writer; do not let two
  executors share this account.
- Real KuCoin position mode must be queried; 1=Hedge, 0=One-Way.
  Changing mode requires ALL positions closed and ALL orders cancelled,
  with a separately confirmed maintenance window. Never auto-flatten.
- The read-only preflight in quant.execution.family_hedge_preflight is
  now available for this check but not deployed to Blaupunkt.
- Durable ledger is tested against a THROWAWAY PostgreSQL database.
  It records CONFIRMED exchange fills, but is not yet connected to the
  exchange stream, stop/TP management or live order submission.
- No complete clientOid-based recovery/resubmission, funding-fee allocation,
  emergency-stop lifecycle or production risk manager exists yet.
- Current TradingView legacy parent signals are not yet imported into the
  durable family ledger. No live child order may be submitted before full
  reconciliation of the parent's real position.

## CI

- Blaupunkt Family Shadow Tests: unit/ASGI/preflight/tree/planner/parity tests.
- Blaupunkt Family PostgreSQL Integration: transaction/fill ownership/replay
  tests using temporary PostgreSQL 16.

Sources:
https://www.kucoin.com/docs-new/3475216e0
https://www.kucoin.com/docs-new/rest/futures-trading/orders/add-order
https://www.kucoin.com/docs-new/rest/futures-trading/positions/get-position-list
