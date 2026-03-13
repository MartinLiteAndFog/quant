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

## Kraken execution bot

Loop: `quant.execution.kraken_bot`

Responsibilities:
- maintain bot state
- execute strategy logic
- reconcile venue position
- persist execution events
- persist equity snapshots

Key file: `src/quant/execution/kraken_bot.py`

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

Provides: Renko chart, gate shading, fib levels, trade markers, entry→exit segments, SL/TTP/TP overlays

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

---

# Current Priorities

1. Monitor guard — confirm it stops firing after f450c93 deployment
2. Fix `signal_events` Postgres field parity
3. Incremental state machine (eliminate full replay)
4. OMS margin-awareness
5. Continue separating strategy parity bugs from OMS/venue reality bugs

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
