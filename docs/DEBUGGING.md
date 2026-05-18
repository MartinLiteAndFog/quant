# Debugging Guide

This document describes the current preferred debugging approach for the Quant system.

The key rule is:

**Prefer durable evidence over log archaeology.**

When possible, debug in this order:

1. Postgres
2. current deployed code path
3. runtime JSONL / state files
4. logs

For broader system structure, see:
- `docs/ARCHITECTURE.md`
- `docs/event_schema_v1.md`

---

## 1. Preferred forensic chain

The intended reconstruction path is:

**Signal → Action → Execution → Closed Trade → Equity**

Typical debugging order:

1. Did a signal exist?
2. Did the engine produce an action?
3. Did the OMS / venue produce an execution?
4. Was a closed trade recorded?
5. Did equity reflect the outcome?

---

## 2. Main failure classes

## Signal storms
Typical source:
- IMBA signal generation
- live signal worker routing
- duplicate or unstable signal emission

Check:
- signal JSONL streams
- future `signal_events`
- dedupe / monotonicity logic
- gate/routing behavior

---

## Wrong position sizing
Typical source:
- equity source (available vs total)
- contract multiplier (hardcoded vs live)
- leverage env var not set

Check executor startup log:
```
executor sizing: equity=... pos_pct=... leverage=... mid=... mult=... -> qty=...
```

Verify env vars:
- `LIVE_EXECUTOR_POS_PCT` (default 0.90)
- `LIVE_EXECUTOR_LEVERAGE` (must be set explicitly, default is 1)
- `KUCOIN_FUTURES_ORDER_LEVERAGE` (sent to exchange; falls back to `LIVE_EXECUTOR_LEVERAGE`)

Formula: `qty = floor(equity × pos_pct × leverage / (mid × contract_multiplier))`

The contract multiplier is fetched live from the broker — it is **not** hardcoded.

---

## terminal_pos oscillation (false flatten)
Typical source:
- Renko mismatch between signal worker and executor (now fixed)
- signal snapping causing signals to appear/disappear (now fixed)
- D3: full replay instability (renko_latest.parquet updated mid-poll)

Check:
- `LIVE_RENKO_PATH` env var — must point to same file `renko_cache_updater` writes
- executor log for `GUARD FIRED` warnings — includes bar count and signal count
- whether guard keeps firing after unified Renko is deployed

---

## Wrong parity between backtest and live
Typical source:
- Renko source mismatch (now fixed — both use `renko_latest.parquet`)
- signal snapping (now removed — exact match only)
- state reconstruction mismatch
- regime gate incorrectly force-flattening

Check in this order:
1. `LIVE_RENKO_PATH` pointing to correct file
2. loaded Renko bar count in executor log
3. latest signal timestamp seen by executor
4. latest reconstructed event
5. terminal state
6. only then compare to backtest expectation

Important rule:
- first decide whether live and backtest would choose the same action on the same bars/signals
- only after that debug OMS / venue reality

Regime gate rule:
- gate selects exit strategy (TTP vs TP1/2) — it **never** force-flattens a live position
- `regime_forces_flat=False` is hardcoded in the live executor path

---

## Wrong equity
Typical source:
- backtest mapping
- fills mapping
- equity aggregation / sequencing
- stale runtime source vs durable source confusion

Check:
- `equity_snapshots`
- `closed_trades`
- dashboard equity readers
- backtest fill/equity construction
- whether you are looking at live equity, realized equity, or derived chart equity

---

## Wrong flip behavior
Typical source:
- flip state machine
- stale state
- wrong regime interpretation
- flatten-first / confirm / re-entry logic

Check:
- strategy state
- last signal
- last action event
- OMS result
- venue position after flip attempt

Main code areas:
- `flip_engine`
- `live_executor`
- `kraken_bot`
- `oms`

---

## Execution mismatch
Typical source:
- OMS logic
- live executor logic
- venue adapter behavior
- stale runtime state
- expected action != actual venue position

Check:
- `action_events`
- `execution_events`
- venue position
- fills
- OMS result payloads
- runtime state reconciliation

Main code areas:
- `oms.py`
- `live_executor.py`
- `kraken_bot.py`
- venue adapters

---

## Dashboard mismatch
Typical source:
- dashboard reading stale runtime files
- Postgres vs fallback-source divergence
- stale Renko cache
- stale active-level state
- trade markers not aligned to durable trade data

Check:
- whether dashboard reader is Postgres-first for that view
- `closed_trades`
- `equity_snapshots`
- `trade_decisions` (for `/api/dashboard/trade_count` and
  `trade_decision_count` on `/api/dashboard/performance`)
- runtime cache freshness
- `renko_health`
- active level files only as secondary evidence

Known intentional empties (do not treat as bugs):

- the chart payload always returns `segments: []` (trade-connector lines were
  removed from `dashboard_state.py` and from every frontend)
- the chart payload returns empty `equity_kraken`, `equity_combined`,
  `kraken_metrics`, `equity_live`, `equity_realized` — the dashboard is now
  KuCoin-focused; these fields are kept only for backwards compatibility
- `trade_decision_count` may be `null` on `/api/dashboard/performance` if the
  count query failed (the endpoint logs `trade decision count failed`); the
  chart and equity payloads are not affected

> Main dashboard performance/loading improvements are **in progress** and may
> change how some of these payloads are produced.

Main code areas:
- `dashboard_state.py`
- `webhook_server.py`

---

## Event-chain gaps
Typical source:
- missing writer integration
- action without execution
- execution without proper linkage
- signal layer not yet fully live-wired

Check:
- latest `action_events`
- latest `execution_events`
- whether `source_signal_event_id` / `source_action_event_id` are expected to be `None`
- whether the relevant writer is already deployed

---

## 3. Current practical debugging order

## For KuCoin / live_executor

1. Check current position
2. Check latest `action_events`
3. Check latest `execution_events`
4. Check `closed_trades`
5. Check `equity_snapshots`
6. Check executor startup + sizing log: `executor sizing: equity=... -> qty=...`
7. Only then inspect:
   - signal JSONL
   - local event JSONL
   - active levels JSON
   - logs (look for `GUARD FIRED` as a parity signal)

Current note:
- KuCoin `action_events` and `execution_events` are verified in Postgres
- guard firing (`GUARD FIRED` in logs) indicates terminal_pos=0 with live position — investigate Renko/signal alignment

---

## For Kraken

1. Check current venue position/state
2. Check latest Kraken `execution_events` (only present if tracking enabled)
3. Check equity snapshots (only present if tracking enabled)
4. Check bot state reconciliation
5. Only then inspect logs/runtime files

Current notes:
- Kraken trade tracking is **gated** behind `KRAKEN_TRADE_TRACKING_ENABLED=1`
  (default `0`). If you see no Kraken equity snapshots / action_events /
  execution_events, check that env var first — the decision loops in
  `live_executor_2.py` and the legacy `kraken_bot.py` run either way, but
  persistence is skipped when the gate is off.
- In `--once` mode, failures should now raise with full stacktrace instead of
  only logging a warning.

---

## 4. Typical questions to ask

### Was there a signal?
Look for:
- signal stream
- future `signal_events`
- routing/gate status

### Was there an action?
Look in:
- `action_events`

### Was the action actually executed?
Look in:
- `execution_events`
- venue position / fills

### Was the trade realized?
Look in:
- `closed_trades`

### Did account equity move accordingly?
Look in:
- `equity_snapshots`

---

## 5. Common anti-patterns

Avoid these unless necessary:

- relying on logs alone when Postgres already has the answer
- debugging from dashboard appearance only
- treating runtime files as authoritative when durable data exists
- hotpatching running containers instead of fixing repo → push → deploy
- assuming missing execution means missing action
- assuming missing linkage ids mean broken data; some are still intentionally `None`

---

## 6. Operational rules

- verify code in repo first
- push before live verification
- use `--once` modes where possible for targeted checks
- use full stacktraces, not warning-only error swallowing
- prefer minimal fixes with strong verification
- update docs when the debugging reality changes

---

## 7. Main files to inspect

Strategy / execution:
- `src/quant/strategies/flip_engine.py`
- `src/quant/strategies/imba.py`
- `src/quant/execution/renko_cache_updater.py` — single Renko authority
- `src/quant/execution/live_signal_worker.py`
- `src/quant/execution/live_executor.py`
- `src/quant/execution/live_executor_2.py` (Kraken; persistence gated by `KRAKEN_TRADE_TRACKING_ENABLED`)
- `src/quant/execution/kraken_bot.py` (legacy Kraken; same gate)
- `src/quant/execution/oms.py`
- `src/quant/execution/kucoin_futures.py`

Persistence / events:
- `src/quant/execution/event_builders.py`
- `src/quant/execution/event_store.py`
- `src/quant/execution/event_types.py`
- `src/quant/execution/trade_counter.py` — trade-decision classifier
- `src/quant/execution/trade_decisions_store.py` — `trade_decisions` table helpers
- `src/quant/sql/002_trade_decisions.sql`
- `scripts/backfill_trade_decisions.py`

Dashboard:
- `src/quant/execution/dashboard_state.py`
- `src/quant/execution/webhook_server.py`