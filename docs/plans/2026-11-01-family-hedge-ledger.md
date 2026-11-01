# Strategy family per KuCoin Futures subaccount — staged implementation

Date: 2026-11-01. Status: PROTOTYPE ONLY; NO LIVE ORDER ROUTING.

## Objective

Allow one KuCoin Futures subaccount to host many TradingView-driven bots
belonging to one strategy family. The family has a shared capital pool, but
every bot has its own virtual LONG/SHORT position, return history and risk
budget. Different families use different subaccounts / API credentials.

TradingView remains the source of trading signals until an independent
signal engine has demonstrated live parity.

## What the first implementation provides

- quant.execution.family_ledger.FamilyLedger: pure in-memory accounting,
  contract quantities (not SOL), weighted-average entry, realized PnL and
  fees; separate positions by bot, symbol and LONG/SHORT.
- BotIntent: explicit bot identity, unique clientOid, OPEN/CLOSE intent,
  KuCoin-style side and position-side fields. It does NOT place orders.
- Pending close reservations prevent two requests from closing one bot lot.
- Idempotent venue-fill processing, including partial fills.
- Strict reconciliation of virtual LONG/SHORT lots to the venue's two
  aggregate sides. A mismatch blocks new trading; never blindly flatten.
- Eight tests: PYTHONPATH=src python -m unittest discover -s tests -p test_family_ledger.py -v

This prototype has NO API calls, durable storage, automatic funding, stop
management, daughter creation, or Fleet UI yet. Do not enable it as an
executor. Actual exchange fills, not requests, are accounting truth.

## Exchange specifics

- Official KuCoin Futures Hedge Mode supports independent LONG and SHORT
  position sides per contract, not a physical position for each bot.
- Probe: GET /api/v2/position/getPositionMode
- Switch: POST /api/v2/position/switchPositionMode
- Switching requires NO positions or pending orders. Never switch a live
  account without operator confirmation and verified flat state.
- Orders need explicit positionSide, trading side and open/close intent.
  Confirm reduceOnly, stop and TP behavior on the actual account before live.
- The contract multiplier and minimum size must be sourced from the venue;
  0.1 SOL per SOLUSDTM contract in tests is an example.
- If a small bot allocation cannot meet the minimum size, it must wait.
  Never round its order above the funded risk budget.

## Minimum integration before live orders

1. Durable Postgres event log: families, bots, versioned configurations,
   intentions, venue orders, fills, per-bot positions and cash flows.
   Transactional locking, idempotency and replay across restarts required.
2. Single serialized writer per family account. Authenticate incoming
   TradingView webhooks and map bot ID, family ID and strategy version.
   Never let independent live executors trade the same venue account.
3. Hedge-aware KuCoin adapter. Probe and validate Hedge Mode at startup;
   send side-specific orders and stops. Existing signed net-position helpers
   cannot be reused without modification.
4. Reconcile both venue sides and all outstanding orders before trading.
   Lost fills, unknown manual positions, stale data or mismatched exposures
   must fail closed for NEW orders without harming unrelated positions.
5. Allocate budgets and margin per bot and cap family aggregate risk.
   Venue margin offsets do not mean bot risks offset perfectly.
6. Safe per-bot stops, TP, partial closes and flips under aggregated venue
   positions; no cross-bot cancellation or overclose.
7. Add family/parent/child registry and virtual position breakdown to Fleet
   Cockpit; preserve distinct Evolution and Angel/Fishing pools.
8. Mocked broker tests for failures, crashes, cancellation races and delayed
   fills, followed by a tiny authorized pilot on a SEPARATE subaccount.

## Acceptance scenario

- A and B hold SOL LONG simultaneously; C holds SOL SHORT.
- A closes; B and C remain untouched.
- Position-side venue totals match the sum of virtual positions.
- Every fill, fee, stop and funding allocation is attributed and replayable.
- A stale/mismatched condition prevents new orders, not forced flattening.
- A webhook for one family cannot route to another subaccount.
- TradingView remains the signal source throughout initial rollout.

References:
https://www.kucoin.com/docs-new/3475216e0
https://www.kucoin.com/docs-new/3475097e0
https://www.kucoin.com/docs-new/rest/futures-trading/orders/add-order
https://www.kucoin.com/support/48142946141325
