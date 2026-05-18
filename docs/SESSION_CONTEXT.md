# Quant – Session Context

Use this document as entry point when starting a new session.

## Always read first

- docs/CURRENT_STATE.md
- docs/ARCHITECTURE.md

---

## Core execution pipeline

**Signal → Action → Execution → Closed Trade → Equity**

Renko source: `data/live/renko_latest.parquet` (written by `renko_cache_updater`, read by both signal worker and executor)

---

## Important components

### Execution (KuCoin)
- `src/quant/execution/live_executor.py` — reads shared Renko, exact signal match, equity-based sizing
- `src/quant/execution/live_signal_worker.py` — reads shared Renko, computes IMBA, writes signal JSONL
- `src/quant/execution/renko_cache_updater.py` — **single Renko authority**, writes `renko_latest.parquet`
- `src/quant/execution/oms.py` — maker-first order execution + stop/TP order management
- `src/quant/execution/kucoin_futures.py` — KuCoin API adapter

### Execution (Kraken)
- `src/quant/execution/live_executor_2.py` — stop-order-native executor with TTP re-entry and pending follow-entry. Trade/equity/event persistence is gated behind `KRAKEN_TRADE_TRACKING_ENABLED=1` (default OFF).
- `src/quant/execution/kraken_bot.py` — legacy executor, also gated behind `KRAKEN_TRADE_TRACKING_ENABLED=1`.
- `src/quant/execution/kraken_futures.py` — Kraken API adapter (now supports `place_take_profit_market`, `place_trigger_entry_market`)

### Strategies
- `src/quant/strategies/flip_engine.py` — countertrend state machine
- `src/quant/strategies/imba.py` — IMBA signal computation + `get_latest_imba_barriers()`
- `src/quant/backtest/renko_runner.py` — backtest runner

### Dashboard
- `src/quant/execution/webhook_server.py` — KuCoin-focused chart/equity/performance/strategy + new `/api/dashboard/trade_count`
- `src/quant/execution/dashboard_state.py` — `build_fibo_levels()` delegates to `get_latest_imba_barriers()`; Kraken loaders removed; `load_trade_segments()` no longer called by `webhook_server.py`

> Main dashboard performance/loading improvements are **in progress**; this
> section may shift while that work lands.

### Trade decision counter
- `src/quant/execution/trade_counter.py` — classifier (entries + flips count; scale/partial/exit/blocked do not)
- `src/quant/execution/trade_decisions_store.py` — Postgres upsert/count/list/backfill
- `src/quant/sql/002_trade_decisions.sql` — `trade_decisions` table
- `scripts/backfill_trade_decisions.py` — one-shot backfill CLI
- `live_executor.py` upserts a `trade_decisions` row on every classified `action_events` row
- API: `GET /api/dashboard/trade_count`; `GET /api/dashboard/performance` adds `trade_decision_count`

---

## Key env vars (KuCoin live)

```
LIVE_RENKO_PATH=data/live/renko_latest.parquet
LIVE_EXECUTOR_POS_PCT=0.90
LIVE_EXECUTOR_LEVERAGE=6
KUCOIN_FUTURES_ORDER_LEVERAGE=6
LIVE_TRADING_ENABLED=1
LIVE_EXECUTOR_DRY_RUN=0
```

## Key env vars (Kraken)

```
# 1 enables equity / action_events / execution_events / metrics persistence
# from live_executor_2.py and the legacy kraken_bot.py. Default 0 (disabled).
KRAKEN_TRADE_TRACKING_ENABLED=0
```

---

## OMS stop-order conventions

All stop/TP orders are tagged with client IDs: `quant:<SYM>:<kind>:<ms>`

Known kind tags:
- `flat_entry_long`, `flat_entry_short` — stop-triggered re-entry after flat
- `opposite_imba_long`, `opposite_imba_short` — stop exit on opposite IMBA signal
- `tp2_sl`, `tp2_tp1`, `tp2_tp2` — TP2 strategy exit orders
- `ttp_exit` — trailing take-profit exit

Use `oms.find_stop_order_by_kind(symbol, kind)` to find an order by kind, and
`oms.cancel_orders_by_kind(symbol, kind)` to cancel all matching orders.

---

## TTP re-entry flow (live_executor_2, Kraken)

1. Executor detects flat position after TTP (venue filled the stop externally)
2. `_record_ttp_external_exit()` records execution + closed trade
3. `_arm_ttp_reenter_handoff()` arms re-entry state on next signal
4. On next poll: `_ttp_reenter_handoff_action()` resolves re-entry action
5. Stop entry order placed at IMBA barrier via `oms.arm_stop_entry()`
6. Cooldown + expiry logic prevents repeated attempts

---

## Operations

- Deployment: docs/LIVE_DEPLOY.md, docs/RAILWAY_RUNBOOK.md
- Debugging: docs/DEBUGGING.md
- Glossary: docs/GLOSSARY.md

---

## Architecture direction

Postgres-first forensic system.

| Table | Status |
|-------|--------|
| `action_events` | ✓ active |
| `execution_events` | ✓ active |
| `closed_trades` | ✓ active |
| `equity_snapshots` | ✓ active |
| `trade_decisions` | ✓ active (derived from `action_events`) |
| `signal_events` | JSONL only — Postgres path incomplete |

---

## Current priorities

1. Monitor guard — confirm `GUARD FIRED` stops appearing in logs after f450c93
2. Fix `signal_events` Postgres field parity
3. Validate TTP re-entry + pending follow-entry in live_executor_2 under real market conditions
4. Incremental state machine to replace full replay
5. OMS margin-awareness
