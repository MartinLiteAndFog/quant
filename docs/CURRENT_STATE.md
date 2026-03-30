# Quant – Current System State

This document captures the **current operational state** of the system.
It is meant to give a quick overview when starting a new development or debugging session.

For architecture details see:
- docs/ARCHITECTURE.md
- docs/event_schema_v1.md

---

# System Overview

Live trading infrastructure for:

- KuCoin Futures
- Kraken Futures

Main execution pipeline:

**Signal → Action → Execution → Closed Trade → Equity**

Primary architecture direction:

**Postgres-first forensic event system**

Runtime JSONL/state files remain as operational traces but are no longer intended as authoritative truth.

---

# Recently Fixed (commit f450c93)

| Bug | Fix |
|-----|-----|
| Position sizing qty=4 (was ~100x too small) | Now uses total equity × pos_pct × leverage / (mid × contract_multiplier). Contract multiplier fetched live from broker (0.1 for SOL-USDT), not hardcoded. |
| `write_execution_state()` TypeError | Call site now passes a single dict, matching the function signature. |
| Signal worker built independent Renko | Signal worker now reads shared `renko_latest.parquet` — same source as executor. |
| Signal snapping caused terminal_pos oscillation | Fuzzy `_snap_signals_to_bars` removed; exact timestamp matching used, same as backtest. |

---

# Recent Changes (uncommitted)

## live_executor_2.py — Stop-order-native Kraken executor (major extension)

This is the primary area of active development. Key additions:

### TTP re-entry handoff
After a TTP exit is detected externally (venue filled the stop), executor arms a `ttp_reenter_pending` handoff.
On next signal, it attempts a re-entry on the new side via a stop trigger entry order placed at the IMBA barrier.
Includes cooldown/dedup logic (`ttp_reenter_last_attempt_key`, `ttp_reenter_cooldown_until`) and expiry.

### Pending follow-entry
When a follow-entry is needed but market conditions aren't yet met, state stores `pending_follow_entry_*` fields.
This allows the executor to wait for the right moment to enter without immediately firing a market order.

### Stop order sync methods
`_sync_stop_order()` and `_sync_take_profit_order()` provide idempotent stop-order management:
- checks if the right order already exists at the right price
- cancels stale orders and re-places if price drift exceeds threshold
- tagged via `quant:<SYM>:<kind>:<ms>` client IDs for precise identification and cancellation

### `_record_ttp_external_exit()`
Records a synthetic execution event + closed trade when the TTP stop is filled by the venue
(detected via position reconciliation, not via explicit bot-placed order).

### New ExecutorState fields
`last_live_side`, full `pending_follow_entry_*` block, full `ttp_reenter_*` block.

## oms.py — Extended BrokerAPI and MakerFirstOMS

New abstract methods on `BrokerAPI`:
- `place_stop_market()`, `place_take_profit_market()`, `place_trigger_entry_market()`
- `cancel_order()`, `list_open_orders()`, `list_open_stop_orders()`

New methods on `MakerFirstOMS`:
- `arm_stop_entry()`, `arm_stop_exit()`, `arm_take_profit_exit()`, `arm_flip_close_stop()`
- `get_open_orders()`, `get_open_stop_orders()`
- `cancel_orders_by_kind()`, `find_stop_order_by_kind()`, `cancel_all_quant_orders()`

Client ID convention: `quant:<SYM>:<kind>:<ms>`

## kraken_futures.py — New stop order types

Added `place_take_profit_market()` (`orderType: take_profit`) and
`place_trigger_entry_market()` (`orderType: trigger_entry`) to `KrakenFuturesClient`.

## kucoin_futures.py — Order management additions

Added `cancel_order()`, `list_open_orders()`, `list_open_stop_orders()` to `KucoinFuturesBroker`.

## imba.py — New `get_latest_imba_barriers()`

Returns `{ts, long_barrier, short_barrier}` using the same rolling-window math as `compute_imba_signals()`.
Used by dashboard and live_executor_2 to compute IMBA entry barriers for stop order placement.

## dashboard_state.py — `build_fibo_levels()` refactored

Now delegates to `get_latest_imba_barriers()` instead of duplicating rolling-window logic.
Returns `latest: {long, mid, short, ts}` but no longer returns per-bar series arrays.

---

# Currently Running Components

## Renko cache updater

Worker: `quant.execution.renko_cache_updater`

**Single Renko authority** — fetches 1m candles, builds Renko, writes `data/live/renko_latest.parquet`.
Both signal worker and executor read from this file.

## Live signal generation

Worker: `quant.execution.live_signal_worker`

Responsibilities:
- reads `data/live/renko_latest.parquet` (shared Renko source)
- computes IMBA signals
- maintains countertrend and trendfollower streams
- routes active stream via gate
- writes signals to `SIGNALS_DIR/{SYM}/YYYYMMDD.jsonl`

Strategy routing:
- `gate_on = 1` → countertrend (IMBA flip strategy)
- `gate_on = 0` → trendfollower

## Live executor (KuCoin)

Worker: `quant.execution.live_executor`

Responsibilities:
- reads `data/live/renko_latest.parquet` (same shared Renko source)
- reads routed signal stream (exact timestamp match, no fuzzy snapping)
- reconstructs terminal state via `run_flip_state_machine`
- decides entry/exit/flip
- calls OMS / broker
- persists `action_events` and `execution_events`
- writes `execution_state.json`

Sizing: `qty = floor(equity × pos_pct × leverage / (mid × contract_multiplier))`

Key env vars:
- `LIVE_EXECUTOR_POS_PCT` (default: `0.90`)
- `LIVE_EXECUTOR_LEVERAGE` (set to desired multiple, e.g. `6`)
- `KUCOIN_FUTURES_ORDER_LEVERAGE` (sent to exchange, falls back to `LIVE_EXECUTOR_LEVERAGE`)
- `LIVE_RENKO_PATH` (default: `data/live/renko_latest.parquet`)

Key files:
- `src/quant/execution/live_executor.py`
- `src/quant/execution/oms.py`
- `src/quant/execution/kucoin_futures.py`

## Kraken execution bot (live_executor_2)

Worker: `quant.execution.live_executor_2`

Stop-order-native executor. Active development target.

Key files:
- `src/quant/execution/live_executor_2.py`
- `src/quant/execution/kraken_futures.py`

## Event Persistence

Primary durable storage: Postgres

Important tables:
- `action_events` ✓
- `execution_events` ✓
- `closed_trades` ✓
- `equity_snapshots` ✓
- `signal_events` (JSONL only — Postgres path incomplete)

## Dashboard

Served by: `quant.execution.webhook_server`

Main endpoint: `/dashboard`

Provides: Renko chart, gate shading, fib levels (via `get_latest_imba_barriers`), trade markers, entry→exit segments, SL/TTP/TP overlays

Main files:
- `src/quant/execution/dashboard_state.py`
- `src/quant/execution/webhook_server.py`

## Runtime Files

Operational state files:
- `/data/live/renko_latest.parquet` — shared Renko source (written by `renko_cache_updater`)
- `/data/live/execution_state.json`
- `/data/live/signals/*`
- `/data/live/live_executor_state.json`
- `/data/live/live_signal_state.json`

These are not authoritative forensic sources.

## Deployment

Current production platform: Railway

Docs:
- docs/RAILWAY.md
- docs/RAILWAY_RUNBOOK.md
- docs/LIVE_DEPLOY.md

---

# Current Known Issues

## 1. Signal-events Postgres insert is failing
Missing fields currently reported:
- `config_hash`
- `gate_name`
- `qty_before`
- `regime_on`
- `source_type`
- `strategy_instance`

## 2. Guard logic still active
The false-flat guard in `live_executor.py` fires at WARNING level (`GUARD FIRED`) with bar/signal counts.
After deploying the unified Renko fix, monitor logs — guard should no longer fire.
If it still fires, there is a remaining state-machine replay issue (D3).

## 3. OMS is not margin-aware
Current OMS behavior:
- uses the `qty` it receives
- does not compute max affordable qty
- does not shrink and retry on margin rejection
- only cancels working orders inside its normal reprice/fallback flow

## 4. Full replay instability (D3)
Executor re-runs `run_flip_state_machine` on all bars + signals every 5 seconds.
Now mitigated by unified Renko. Long-term fix: event-sourced incremental state.

## 5. TTP re-entry / pending follow-entry is Kraken-only
`live_executor_2.py` stop-order-native features (TTP re-entry handoff, pending follow-entry, stop sync)
are not yet ported to the KuCoin executor (`live_executor.py`).

## 6. `build_fibo_levels()` no longer returns per-bar series
The refactored dashboard function returns empty arrays for `long`, `mid`, `short` series.
Only `latest` values are populated. Any frontend that relied on per-bar series arrays will see empty data.

---

# Current Priorities

1. Monitor guard — confirm it stops firing after f450c93 deployment
2. Fix `signal_events` Postgres field parity
3. Incremental state machine (eliminate full replay)
4. OMS margin-awareness
5. Validate TTP re-entry and pending follow-entry in live_executor_2 under real market conditions
6. Continue separating strategy parity bugs from OMS/venue reality bugs

---

# Debugging Order

Preferred debugging chain:

1. Postgres
2. venue position
3. code path
4. executor sizing log (`executor sizing: equity=... -> qty=...`)
5. runtime files
6. logs

See: docs/DEBUGGING.md

---

# Session Start Reminder

When starting a new debugging or development session, read:

- docs/SESSION_CONTEXT.md
- docs/CURRENT_STATE.md
- docs/ARCHITECTURE.md
