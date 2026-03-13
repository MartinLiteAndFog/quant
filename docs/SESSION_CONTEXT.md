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
- `src/quant/execution/oms.py` — maker-first order execution
- `src/quant/execution/kucoin_futures.py` — KuCoin API adapter

### Execution (Kraken)
- `src/quant/execution/kraken_bot.py`
- `src/quant/execution/kraken_futures.py`

### Strategies
- `src/quant/strategies/flip_engine.py` — countertrend state machine
- `src/quant/strategies/imba.py` — IMBA signal computation
- `src/quant/backtest/renko_runner.py` — backtest runner

### Dashboard
- `src/quant/execution/webhook_server.py`
- `src/quant/execution/dashboard_state.py`

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
| `signal_events` | JSONL only — Postgres path incomplete |

---

## Current priorities

1. Monitor guard — confirm `GUARD FIRED` stops appearing in logs after f450c93
2. Fix `signal_events` Postgres field parity
3. Incremental state machine to replace full replay
4. OMS margin-awareness
