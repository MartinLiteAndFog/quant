## Purpose

This document describes the current practical architecture of the Quant stack:

- live strategy execution
- event persistence
- dashboard data flow
- forensic reconstruction

Main direction:

**Postgres-first for forensic truth**, while keeping JSONL/runtime files as operational traces.

---

## Core goal

We want a durable, queryable chain from decision to outcome:

**Signal → Action → Execution → Closed Trade → Equity**

This should let us answer:

- which signal existed?
- which engine decision followed?
- what did the OMS / broker try to do?
- what was actually executed?
- how did realized trades and equity evolve?

---

## Current high-level components

## 1. Strategy / signal layer

### Countertrend Flip Strategy (ON regime)

Core:
- IMBA impulses
- Renko bricks
- flip state machine

Entry:
- IMBA signal

Exit:
- trailing take profit (TTP)
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

Main live components:

- `src/quant/execution/live_signal_worker.py`
- `src/quant/execution/live_executor.py`
- `src/quant/execution/oms.py`
- `src/quant/execution/kucoin_futures.py`
- `src/quant/execution/renko_cache_updater.py`
- `src/quant/execution/kraken_bot.py`
- `src/quant/execution/kraken_futures.py`

Responsibilities:

### `renko_cache_updater.py`
- **The single authority for live Renko data**
- fetches 1m candles from KuCoin
- builds Renko bricks via `renko_from_close`
- writes `renko_latest.parquet` to `DASHBOARD_RENKO_PARQUET` (default: `data/live/renko_latest.parquet`)
- also publishes to Redis if available
- polls on configurable interval (default: 60s)

### `live_signal_worker.py`
- **reads `renko_latest.parquet`** (the shared Renko authority — no independent Renko construction)
- computes IMBA signals via `compute_imba_signals`
- routes active stream via gate (countertrend or trendfollower)
- writes routed JSONL signal stream to `SIGNALS_DIR/{SYM}/YYYYMMDD.jsonl`
- writes strategy-specific substreams (`countertrend/`, `trendfollower/`)
- emits JSONL `signal_events`
- attempts Postgres `signal_events` persistence (partially working)

### `live_executor.py`
- reads live signals from JSONL
- **reads `renko_latest.parquet`** (same shared source as signal worker — identical Renko)
- reconstructs desired terminal state via `run_flip_state_machine`
- uses **exact timestamp matching** for signal/bar alignment (no fuzzy snapping)
- decides engine action (`hold`, `enter_*`, `flip_to_*`, `exit_*`)
- writes `action_events`
- calls OMS / broker
- writes KuCoin `execution_events`
- writes runtime execution state for dashboard / debugging
- regime gate controls exit strategy selection (TTP vs TP1/2), **never** force-flattens

### `oms.py`
- maker-first execution abstraction
- entry ladder
- TP/flip exit logic
- SL fast exit
- flatten-first flip handling (required for exchange safety)

Important current limitation:
- **OMS does not implement margin-aware resize/freeing logic**
- it uses the `qty` it receives
- it does not shrink and retry on margin rejection
- it only cancels working orders during reprice/fallback flow

### Venue adapters
- `kucoin_futures.py`
- `kraken_futures.py`

These handle real exchange API interaction.

### `kucoin_futures.py` — leverage note

`KUCOIN_FUTURES_ORDER_LEVERAGE` (fallback: `LIVE_EXECUTOR_LEVERAGE`, default: `1`) is sent to KuCoin
with every order. `LIVE_EXECUTOR_LEVERAGE` is used in the sizing formula only (local math, never
sent to the exchange directly unless `KUCOIN_FUTURES_ORDER_LEVERAGE` is not set).

---

## 3. Position sizing

Sizing formula (equity-based, restored in f450c93):

```
qty = floor(equity * pos_pct * leverage / (mid_price * contract_multiplier))
```

- `equity` — total account equity from `broker.get_account_balance()["equity"]`
- `pos_pct` — fraction of equity to deploy (`LIVE_EXECUTOR_POS_PCT`, default: `0.90`)
- `leverage` — from `LIVE_EXECUTOR_LEVERAGE` (default: `1`, set to desired multiple e.g. `6`)
- `contract_multiplier` — fetched live from `broker.get_contract_multiplier(symbol)` (e.g. 0.1 for SOL-USDT on KuCoin)

Previous bug (before f450c93): used `bal["available"]` (free margin) instead of `bal["equity"]`, and hardcoded `contract_multiplier=1.0` instead of fetching it from the broker. Combined effect: ~100x undersizing (qty=4 instead of ~400).

---

## 4. Persistence / forensic layer

Main direction:

- JSONL remains local append-only trace
- Postgres becomes durable, queryable truth

Relevant files:

- `src/quant/execution/event_builders.py`
- `src/quant/execution/event_store.py`
- `src/quant/execution/event_types.py`

Current durable tables:

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
- JSONL emission active in live worker
- Postgres insert path added but currently failing because builder/event schema/store are not yet aligned on required fields

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
- KuCoin/live_executor: persisted to Postgres and queryable

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
- KuCoin/live_executor: persisted

## Closed trade
Represents realized trade-level outcome.

Current status:
- stored in Postgres
- dashboard uses Postgres-first in several views

## Equity
Represents account / strategy equity snapshots.

Current status:
- Kraken and KuCoin flows write to Postgres
- dashboard reads Postgres-first where implemented

---

## Backtest vs live parity

Target principle:

- same Renko construction
- same IMBA logic
- same state machine semantics
- only real venue/OMS reality should differ

### Fixed divergences (commit f450c93)

| # | Issue | Status |
|---|-------|--------|
| D1 | Signal worker built its own Renko independently — brick timestamps could differ from executor | **FIXED** — signal worker now reads shared `renko_latest.parquet` |
| D2 | Signal snapping (`_snap_signals_to_bars`) with 5-min tolerance caused signals to intermittently vanish or shift | **FIXED** — exact timestamp matching, function removed |
| D8 | `write_execution_state()` called with kwargs, function signature expects a single dict → TypeError | **FIXED** |
| Sizing | Used `available` balance + hardcoded `contract_multiplier=1.0` → ~100x undersizing | **FIXED** — equity-based sizing, live contract multiplier fetch |

### Remaining divergence points

| # | Issue | Notes |
|---|-------|-------|
| D3 | Full replay of `run_flip_state_machine` on every poll | Now mitigated by unified Renko source; full fix would be event-sourced incremental state |
| D4 | Regime gate behavior | Gate selects exit strategy (TTP vs TP1/2), **never** flattens. Documented and enforced in executor |
| D5 | Guard logic (suppress false flat) | Keep until D3 fully resolved; fires at WARNING level with diagnostics |
| D10 | Flip handling (flatten-first) | Required for exchange safety; do not change |
| D11 | SL exit routing | Needs proper backtesting before changing |

Practical debugging rule:
- first ask whether live and backtest would choose the same terminal/action on the same bars/signals
- only then ask what OMS / venue did with that action

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
- trade markers
- trading diary
- trade segments
- closed trades

Still operationally relevant:
- active level/runtime execution state files
- signal JSONL
- live Renko parquet (`data/live/renko_latest.parquet`)

---

## Current live persistence status

### Equity snapshots
- Kraken: active
- KuCoin: active

### Closed trades
- persisted in Postgres

### Action events
- Kraken: active
- KuCoin/live_executor: active and verified

### Execution events
- Kraken: active
- KuCoin/live_executor: active and verified

### Signal events
- JSONL emission active
- Postgres insert currently broken by missing required fields:
  - `strategy_instance`
  - `config_hash`
  - `source_type`
  - `qty_before`
  - `regime_on`
  - `gate_name`

---

## Current temporary compromises

### 1. JSONL is still emitted
Reason:
- local trace
- debugging convenience
- backward compatibility

Target:
- keep as secondary trace, not primary truth

### 2. `source_signal_event_id` is not yet authoritative
Reason:
- `signal_events` are not yet cleanly persisted end-to-end

### 3. Execution events are still coarse
Especially on KuCoin:
- fill-level visibility exists
- not yet a full venue lifecycle model in every path

### 4. Guard logic still active
`_apply_live_ttp_guard` / false-flat guard in executor remains as a safety net.
Fires at WARNING level with `GUARD FIRED` prefix and bar/signal count.
Should be removed once D3 (full replay instability) is fully resolved.

---

## Forensic debugging workflow

Preferred reconstruction path:

1. `signal_events` / signal JSONL
2. `action_events`
3. `execution_events`
4. `closed_trades`
5. `equity_snapshots`

For live parity bugs, also verify:
- `LIVE_RENKO_PATH` env var (should be `data/live/renko_latest.parquet`)
- loaded bar count in executor log
- latest signal timestamp seen by executor
- latest reconstructed event
- terminal state
- executor sizing log line (`executor sizing: equity=... -> qty=...`)

---

## Present priority order

### 1. Finish signal-event parity
- align `SignalEvent` builder/type/store fields
- make Postgres signal persistence succeed

### 2. Incremental state machine (long term)
- replace full replay with event-sourced state update
- eliminates D3 and removes need for guard logic

### 3. OMS margin awareness
- detect margin rejection and retry with smaller qty

### 4. SL exit backtesting
- before changing D11 (SL exit routing), backtest first

---

## Current architectural rule set

- Prefer incremental migration over big rewrites
- First add writers
- Then switch readers
- Then add links
- Then document
- Then clean up

And:

- prefer Postgres over logs when available
- do not fake linkage
- do not treat runtime files as primary forensic truth
- keep strategy-parity questions separate from OMS/venue questions
