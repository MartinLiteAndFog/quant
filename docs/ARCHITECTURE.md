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

### KuCoin executor (`live_executor.py`)

Main live components:

- `src/quant/execution/live_signal_worker.py`
- `src/quant/execution/live_executor.py`
- `src/quant/execution/oms.py`
- `src/quant/execution/kucoin_futures.py`
- `src/quant/execution/renko_cache_updater.py`

### Kraken executor (`live_executor_2.py`)

Newer, more capable executor for Kraken. Implements stop-order-native architecture.

Main file:
- `src/quant/execution/live_executor_2.py`
- `src/quant/execution/kraken_futures.py`

---

## 3. OMS (`oms.py`)

Maker-first execution abstraction. Extended with native stop/take-profit order management.

### BrokerAPI abstract interface — new methods added:
- `place_stop_market()` — SL stop exit
- `place_take_profit_market()` — TP trigger exit
- `place_trigger_entry_market()` — stop-triggered entry
- `cancel_order()` — cancel by order_id or client_id
- `list_open_orders()` — active orders
- `list_open_stop_orders()` — active stop orders

### MakerFirstOMS — new methods added:
- `arm_stop_entry(side, qty, stop_price, kind)` — place a trigger-entry stop order
- `arm_stop_exit(side, qty, stop_price, kind)` — place a stop-loss exit order
- `arm_take_profit_exit(side, qty, stop_price, kind)` — place a take-profit exit order
- `arm_flip_close_stop(side, qty, stop_price, kind)` — place a stop to close and flip
- `get_open_orders()` / `get_open_stop_orders()` — fetch live venue order lists
- `cancel_orders_by_kind(symbol, kind_fragment)` — cancel all orders matching a `quant:<SYM>:<kind>:*` client ID pattern
- `find_stop_order_by_kind(symbol, kind_fragment)` — find an existing stop order by kind tag
- `cancel_all_quant_orders(symbol)` — cancel all quant-tagged orders

### Client ID convention:
All orders placed by OMS carry a `quant:<SYM>:<kind>:<ms>` client ID.
Kind tags include: `flat_entry_long`, `flat_entry_short`, `tp2_sl`, `tp2_tp1`, `tp2_tp2`, `ttp_exit`, `opposite_imba_long`, `opposite_imba_short`.

### Important current limitation:
- **OMS does not implement margin-aware resize/freeing logic**
- it uses the `qty` it receives
- it does not shrink and retry on margin rejection
- it only cancels working orders during reprice/fallback flow

---

## 4. Venue adapters

### `kucoin_futures.py` — new methods:
- `cancel_order(order_id, client_id)`
- `list_open_orders(symbol)` — active orders (normalized dicts)
- `list_open_stop_orders(symbol)` — active stop orders (normalized dicts)

### `kucoin_futures.py` — leverage note:
`KUCOIN_FUTURES_ORDER_LEVERAGE` (fallback: `LIVE_EXECUTOR_LEVERAGE`, default: `1`) is sent to KuCoin
with every order. `LIVE_EXECUTOR_LEVERAGE` is used in the sizing formula only (local math, never
sent to the exchange directly unless `KUCOIN_FUTURES_ORDER_LEVERAGE` is not set).

### `kraken_futures.py` — new methods:
- `place_take_profit_market(side, size, stop_price, ...)` — `orderType: take_profit`
- `place_trigger_entry_market(side, size, stop_price, ...)` — `orderType: trigger_entry`

---

## 5. `live_executor_2.py` — Kraken executor (major extension)

Stop-order-native execution engine for Kraken. Implements TTP re-entry and pending follow entry handoffs.

### New `KrakenOmsBroker` methods:
- `place_take_profit_market()`, `place_trigger_entry_market()`, `cancel_order()`, `list_open_orders()`, `list_open_stop_orders()`

### New `ExecutorState` fields:
```
last_live_side                  # last known live position side from venue
pending_follow_entry            # pending delayed entry
pending_follow_entry_side
pending_follow_entry_reason
pending_follow_entry_source_ts
pending_follow_entry_expires_at
ttp_reenter_pending             # TTP re-entry handoff pending
ttp_reenter_prior_side
ttp_reenter_target_side
ttp_reenter_source_ts
ttp_reenter_expires_at
ttp_reenter_exit_recorded
ttp_reenter_cooldown_until
ttp_reenter_last_attempt_key
```

### New helper functions:
- `_record_ttp_external_exit()` — records execution + closed trade when TTP exit detected externally (venue filled the stop)
- `_new_ttp_reenter_leg_id()` — generates unique leg ID for TTP re-entry legs
- `_clear_pending_follow_entry()` / `_arm_pending_follow_entry()` / `_pending_follow_entry_is_active()` — pending follow-entry lifecycle
- `_clear_ttp_reenter_handoff()` / `_arm_ttp_reenter_handoff()` / `_ttp_reenter_handoff_context()` — TTP re-entry handoff lifecycle
- `_ttp_reenter_attempt_allowed()` / `_mark_ttp_reenter_attempt()` — cooldown/dedup for TTP re-entry attempts
- `_ttp_reenter_handoff_action()` — resolves re-entry action from handoff context
- `_derive_action_event_fields()` — action event field derivation helper

### New `LiveExecutor2` methods:
- `_retag_stop_order()` — cancel old stop and re-arm with new kind tag when signal changes
- `_sync_stop_order(kind, side, stop_price, qty)` — idempotent stop-order sync (creates if missing, cancels if stale)
- `_sync_take_profit_order(kind, side, stop_price, qty)` — idempotent TP-order sync

### New IMBA barrier helper (`imba.py`):
- `get_latest_imba_barriers(df_ohlcv, params)` — returns `{ts, long_barrier, short_barrier}` using the same rolling-window math as `compute_imba_signals()`

---

## 6. Position sizing

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

## 6.5 Kraken trade tracking is gated and OFF by default

Kraken trade/equity/execution-event persistence in both
`src/quant/execution/live_executor_2.py` and the legacy
`src/quant/execution/kraken_bot.py` is now gated behind a single env var:

```
KRAKEN_TRADE_TRACKING_ENABLED=1   # default: 0 (disabled)
```

When unset (the default), the Kraken executors run their decision loops but
skip:

- equity snapshot inserts (`insert_equity_snapshot` / `_append_equity`)
- Kraken `action_events` / `execution_events` writes
- per-bot metrics JSON / equity CSV writes

This reflects the operational reality that KuCoin is the primary live target
and that Kraken-side persistence is currently opt-in.

---

## 7. Persistence / forensic layer

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
- `trade_decisions` — see `## 8. Trade decision counter` below; schema in
  `src/quant/sql/002_trade_decisions.sql`

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
- **KuCoin-focused**: chart, equity, performance, strategy and trade-count
  endpoints are produced from KuCoin live data; Kraken-only payload fields are
  kept for backwards compatibility but are now empty (see below)

> **In progress:** main dashboard performance/loading improvements. Some
> internal data flows may shift while that work lands; this section will be
> updated once it is verified.

### KuCoin-focused, Kraken removed from dashboard loaders

`dashboard_state.py` no longer contains Kraken loaders, and `webhook_server.py`
no longer calls `load_trade_segments()`. The `/api/dashboard/chart` payload
still contains the following Kraken/compatibility fields, but they are always
empty / inert:

- `equity_kraken: []` / `equity_kraken_source: "none"`
- `equity_combined: []` / `equity_combined_source: "none"`
- `equity_live: []` / `equity_live_source: "deprecated_use_equity_components"`
- `equity_realized: []` / `equity_realized_source: "deprecated_use_equity_components"`
- `kraken_metrics: {}`
- `segments: []` (trade-connector segments are no longer produced or rendered)

The `equity_components` field is the current per-source breakdown for the
KuCoin equity stack; `equity_total` mirrors KuCoin real equity.

### Trade-connector segments removed

The entry→exit line segments that used to be drawn on the price chart are no
longer produced by `dashboard_state.load_trade_segments()` (the function was
removed from the call graph) and are no longer rendered by:

- React: `frontend/src/components/charts/PriceChart.tsx`,
  `frontend/src/components/layout/Dashboard.tsx`
- Svelte: `dashboard/src/components/PriceChart.svelte`
- the inline HTML/JS dashboard served by `webhook_server.py`

The `segments: []` field is retained in the chart payload for one release as a
backwards-compatibility no-op.

### Fib/IMBA barrier display (refactored)

`build_fibo_levels()` in `dashboard_state.py` now delegates to `get_latest_imba_barriers()` from `imba.py`.
This ensures the dashboard uses exactly the same barrier math as the live strategy, not a separate rolling-window implementation.
The dashboard now returns `latest: {long, mid, short, ts}` but no longer returns per-bar series arrays
(previously `out_long`, `out_mid`, `out_short` were per-bar series — now empty lists are returned).

Already moved to Postgres-first:
- real equity history
- trade markers
- trading diary
- closed trades
- trade decision counts (see below)

Still operationally relevant:
- active level/runtime execution state files
- signal JSONL
- live Renko parquet (`data/live/renko_latest.parquet`)

---

## 8. Trade decision counter

A **trade decision** is a discrete directional position-opening event that
carries its own SL/TP commitment. Every entry from flat and every flip to the
opposite direction is one decision; scale-ins, partial closes and exits are
not new decisions.

The counter is derived from `action_events` so that it stays consistent with
the rest of the forensic chain.

### Files

- `src/quant/execution/trade_counter.py` — pure classifier
  (`classify_action_event`, `build_trade_decisions_from_action_events`,
  `deterministic_decision_id`)
- `src/quant/execution/trade_decisions_store.py` — Postgres helpers
  (`upsert_trade_decision`, `count_trade_decisions`,
  `list_recent_trade_decisions`, `backfill_trade_decisions_from_action_events`)
- `src/quant/sql/002_trade_decisions.sql` — idempotent `CREATE TABLE` (also
  re-asserted at runtime by `ensure_trade_decisions_schema()`)
- `scripts/backfill_trade_decisions.py` — one-shot CLI backfill

### Classification rules (authoritative)

Counted (one row per event, with `decision_kind` set as shown):

| `engine_action`   | `decision_kind` | Meaning                                  |
|-------------------|-----------------|------------------------------------------|
| `enter_long`      | `entry`         | flat → long (new SL/TP)                  |
| `enter_short`     | `entry`         | flat → short (new SL/TP)                 |
| `flip_to_long`    | `flip`          | short → long (new SL/TP on the new leg)  |
| `flip_to_short`   | `flip`          | long → short (new SL/TP on the new leg)  |

Not counted:

- `scale_long` / `scale_short` — same-direction add, no new SL/TP
- `tp1_partial` and any partial close — reduces size, keeps SL
- `exit_long` / `exit_short` — ends the existing trade lifecycle, but the
  trade was already counted at entry / flip time
- `hold` — no-op
- any unknown / unrecognised `engine_action`
- any row with `blocked = true` — the SL/TP was never committed

A flip is one decision (the new opposite-direction leg with its own SL/TP);
the close-half of the flip is implicit and is not double-counted.

### Idempotency

Each decision id is deterministic:

```
decision_id = "td_" + sha1(source_action_event_id)[:16]
```

(when there is no source event id, a stable fallback hash over
`(venue, symbol, ts, seq, engine_action)` is used instead). All write paths
upsert by `decision_id`, so re-running the live executor or the backfill
script over the same `action_events` history never produces duplicates.

### Write path (live)

`src/quant/execution/live_executor.py` calls `classify_action_event()` for
each `action_events` row it writes and, if the row is countable, upserts the
resulting `TradeDecision` via `upsert_trade_decision()`. Failures are logged
and swallowed so the trade counter can never break execution.

### Backfill

To re-derive `trade_decisions` from existing `action_events` history at any
time:

```bash
PYTHONPATH=src python3 scripts/backfill_trade_decisions.py \
  --venue kucoin --symbol SOL-USDT
```

Or one-shot via the dashboard API:

```
GET /api/dashboard/trade_count?symbol=SOL-USDT&venue=kucoin&backfill=1
```

Both paths are idempotent.

### API surface

- `GET /api/dashboard/trade_count?symbol=&venue=&recent_limit=&backfill=`
  → `{ total, entries, flips, recent: [...], backfill: ... }`
- `GET /api/dashboard/performance` → existing payload plus
  `trade_decision_count` (the same value as `trade_count.total` for the same
  venue/symbol)

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

### 5. TTP re-entry and pending follow-entry are new (live_executor_2 only)
These are Kraken-only at this stage. Not yet ported to KuCoin executor.

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
