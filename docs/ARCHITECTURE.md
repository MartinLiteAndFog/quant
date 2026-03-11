# Quant Architecture

## Purpose

This document describes the current system architecture of the Quant stack at a practical level:

- live strategy execution
- event persistence
- dashboard data flow
- forensic reconstruction
- current temporary compromises

The main architectural direction is now:

**Postgres-first for forensic truth**, while keeping JSONL and legacy runtime files alive as transitional traces.

---

## Core goal

We want a queryable and durable chain from decision to outcome:

**Signal → Action → Execution → Closed Trade → Equity**

This should allow us to answer questions such as:

- which signal was generated?
- which engine decision followed?
- what did the OMS / broker actually try to do?
- what was actually executed at the venue?
- how did realized trades and equity evolve?

---

## Current high-level components

## 1. Strategy / signal layer

Main strategy families currently in use:

### Countertrend Flip Strategy (ON regime)

Core:
- IMBA impulses
- Renko bricks
- flip state machine

Entry:
- IMBA signal

Exit:
- trailing take profit
- stop loss
- opposite IMBA flip

Main files:
- `src/quant/strategies/flip_engine.py`
- `src/quant/backtest/renko_runner.py`

### ImbaTrend Strategy (OFF regime)

Trend-following variant.

Entry:
- IMBA signal

Exit:
- TP1 partial
- TP2 final
- SL clamp

Main runner:
- `renko_runner_tp2.py`

---

## 2. Live execution layer

The live execution layer translates strategy state into venue actions.

Main live components:

- `src/quant/execution/live_executor.py`
- `src/quant/execution/oms.py`
- `src/quant/execution/kucoin_futures.py`
- `src/quant/execution/kraken_bot.py`
- `src/quant/execution/kraken_futures.py`

Responsibilities:

### `live_executor.py`
- reads live signals
- aligns them with Renko/backtest state
- derives desired terminal state
- decides engine action (`hold`, `enter_*`, `flip_to_*`, `exit_*`, `scale_*`)
- writes `action_events`
- calls OMS / broker
- now also writes KuCoin `execution_events` on successful OMS execution paths

### `oms.py`
- maker-first execution abstraction
- entry ladder
- TP/flip exit logic
- SL fast exit
- flatten-first flip handling

### Venue adapters
- `kucoin_futures.py`
- `kraken_futures.py`

These handle the real exchange API interaction.

---

## 3. Persistence / forensic layer

Main persistence direction:

- JSONL remains as local append-only trace
- Postgres becomes the durable, queryable truth layer

Relevant files:

- `src/quant/execution/event_builders.py`
- `src/quant/execution/event_store.py`
- `src/quant/execution/event_types.py`

Current durable tables include at least:

- `action_events`
- `execution_events`
- `closed_trades`
- `equity_snapshots`

Planned / partial:
- `signal_events`

---

## Event chain

## Signal
Represents alpha / strategy statements.

Examples:
- IMBA long
- IMBA short
- trend flip
- flat / re-arm

Current status:
- schema direction exists
- not yet cleanly live-wired end-to-end into the production event chain

## Action
Represents engine decisions.

Examples:
- enter
- exit
- flip
- hold
- blocked
- scale

Current status:
- Kraken: persisted
- KuCoin/live_executor: persisted to JSONL + Postgres

## Execution
Represents venue / OMS facts.

Examples:
- fill
- submitted order
- cancel
- rejection
- fallback execution
- sync correction

Current status:
- Kraken: persisted
- KuCoin/live_executor: writer is now integrated for successful OMS execution paths; still needs live confirmation on a fresh real execution after deploy

## Closed trade
Represents realized trade-level outcome.

Current status:
- stored in Postgres
- dashboard already reads from Postgres-first

## Equity
Represents account / strategy equity snapshots.

Current status:
- Kraken and KuCoin flows already write to Postgres
- dashboard reads Postgres-first

---

## Postgres-first architecture

## Principle

Do not reconstruct truth from scattered runtime artifacts if avoidable.

Preferred source order:

1. Postgres
2. JSONL / runtime traces
3. Redis latest state
4. ad-hoc files / parquet / logs

## Why

Logs and runtime caches are useful operationally, but weak as forensic truth:

- incomplete
- overwritten
- hard to join
- hard to query historically

Postgres is intended to become the place where we can reconstruct a full timeline.

---

## Dashboard architecture

Main dashboard logic lives in:

- `src/quant/execution/dashboard_state.py`
- `src/quant/execution/webhook_server.py`

Current direction:
- dashboard reads Postgres first where available
- runtime files remain fallback / compatibility sources

Already moved to Postgres-first:
- real equity history
- Kraken equity history
- trade markers
- trading diary
- trade segments
- closed trades

Important detail:
- string sides such as `"long"` / `"short"` had to be normalized because older readers often assumed integer side encoding

---

## Current live persistence status

## Equity snapshots

### Kraken
- written by `kraken_bot.py`
- source in DB: `kraken_bot`

### KuCoin
- written via dashboard/equity path
- source in DB: `dashboard_state.load_real_equity_history`

## Closed trades
- persisted in Postgres
- already used by dashboard as source of truth for several views

## Action events

### Kraken
- JSONL + Postgres integrated

### KuCoin / live_executor
- JSONL + Postgres integrated
- verified in Postgres

## Execution events

### Kraken
- JSONL + Postgres integrated

### KuCoin / live_executor
- writer integrated into successful OMS result paths
- deploy completed
- still needs confirmation on the next real execution-triggered event

## Signal events
- target structure exists
- not yet fully live-wired
- foreign-key-level linking to `source_signal_event_id` should only be treated as authoritative once signal emission is truly live

---

## Current temporary compromises

These are deliberate transitional decisions, not final architecture.

### 1. JSONL is still emitted
Reason:
- local trace
- debugging convenience
- backward compatibility

Target:
- keep as secondary trace, not primary truth

### 2. `source_signal_event_id` is not yet fully authoritative
Reason:
- `signal_events` are not yet cleanly and consistently emitted live across producers

Consequence:
- some event records intentionally store `None`
- this is preferable to fake or broken foreign-key linkage

### 3. Execution events are currently coarse
Especially on KuCoin/live_executor:
- current implementation logs successful OMS execution outcomes
- it is not yet a full venue lifecycle model with submit / partial / cancel / reject / sync stages

That is acceptable for now because the goal is to first close the main forensic gap.

---

## Forensic debugging workflow

The architecture is being shaped around practical failure analysis.

Typical target questions:

- was there a signal?
- did the engine decide to act?
- was the action blocked or deduplicated?
- did OMS actually attempt venue execution?
- did the venue fill, reject, or ignore?
- did a closed trade get recorded?
- did equity reflect the event?

The intended debugging path is therefore:

1. query `signal_events`
2. query `action_events`
3. query `execution_events`
4. query `closed_trades`
5. query `equity_snapshots`

---

## Present priority order

## 1. Finish KuCoin event chain
- action events: done
- execution events: integrated, awaiting real-event confirmation
- signal events: still pending

## 2. Make linkage cleaner
- connect `source_action_event_id`
- connect `source_signal_event_id`
- standardize reason codes and execution stages

## 3. Improve documentation
- architecture document
- event schema documentation
- operator runbooks

## 4. Cleanup / hygiene
- remove obsolete backup files
- reduce duplicate code paths
- document authoritative readers/writers
- simplify legacy fallbacks

---

## Current architectural rule set

- Prefer incremental migration over big rewrites
- First add writers
- Then switch readers
- Then add links / foreign keys
- Then document
- Then clean up

And:

- do not rely on logs if durable events can answer the question
- do not hard-link upstream event ids until the upstream producer is truly live
- do not remove compatibility traces too early
- do not mix runtime convenience state with forensic source of truth

---

## Related documents

- `docs/event_schema_v1.md` — event family and field design
- `docs/DEBUGGING.md` — operational debugging notes
- `docs/LIVE_DEPLOY.md` — deployment details
- `docs/RAILWAY_RUNBOOK.md` — service/runbook details

---

## Open architecture tasks

- live-wire `signal_events`
- validate first real KuCoin `execution_event` after deploy
- connect `source_action_event_id` where possible
- connect `source_signal_event_id` once real signal persistence is live
- standardize `reason_code` values across Kraken and KuCoin
- add derived SQL views for:
  - event timeline reconstruction
  - action-to-execution attribution
  - trade/equity attribution
- clean up legacy runtime fallbacks once Postgres coverage is sufficient