## Purpose

This document describes how the live Quant stack is currently deployed and operated.

It focuses on:

- live services
- environment variables
- runtime persistence
- dashboard/runtime flow

For system structure, see:

- `docs/ARCHITECTURE.md`
- `docs/event_schema_v1.md`
- `docs/RAILWAY_RUNBOOK.md`

---

## Current deployment model

The live stack consists of long-running services around:

- live Renko cache
- live signal generation
- live execution
- dashboard / API
- venue adapters
- Postgres-backed forensic persistence

Main execution venues currently in scope:

- KuCoin Futures
- Kraken Futures

---

## Core live services

### 1. Dashboard / API service

Main entrypoint:
- `quant.execution.webhook_server:app`

Responsibilities:
- dashboard UI
- dashboard APIs
- regime endpoints
- fills / trade views
- state-space and chart endpoints
- hosts live signal worker and live executor as background threads

Typical start command:

```bash
uvicorn quant.execution.webhook_server:app --host 0.0.0.0 --port $PORT
```

### 2. Renko cache updater

Worker: `quant.execution.renko_cache_updater`

**This is the single Renko authority.** Both the signal worker and the executor read from the
parquet file it writes. Never run independent Renko construction in parallel.

Output: `DASHBOARD_RENKO_PARQUET` (default: `data/live/renko_latest.parquet`)

Key env vars:
```
DASHBOARD_RENKO_PARQUET=data/live/renko_latest.parquet
DASHBOARD_RENKO_BOX=0.1
DASHBOARD_RENKO_DAYS_BACK=14
DASHBOARD_RENKO_POLL_SEC=60
```

### 3. Live signal worker

Worker: `quant.execution.live_signal_worker`

Responsibilities:
- read `renko_latest.parquet` (shared Renko — no independent Renko build)
- compute IMBA signals
- write active signal stream to `SIGNALS_DIR/{SYM}/YYYYMMDD.jsonl`
- maintain strategy substreams

Key env vars:
```
SIGNALS_DIR=/data/live/signals
LIVE_SYMBOL=SOL-USDT
LIVE_IMBA_LOOKBACK=250
LIVE_IMBA_SL_ABS=1.5
LIVE_SIGNAL_POLL_SEC=15
LIVE_RENKO_PATH=data/live/renko_latest.parquet
LIVE_DEFAULT_GATE_ON=1
```

### 4. Live executor

Worker: `quant.execution.live_executor`

Responsibilities:
- read `renko_latest.parquet` (same shared source)
- read signal JSONL (exact timestamp match — no fuzzy snapping)
- reconstruct terminal state
- call OMS / venue broker
- persist `action_events` and `execution_events`
- write `execution_state.json`

Key env vars:
```
LIVE_TRADING_ENABLED=1
LIVE_EXECUTOR_DRY_RUN=0
LIVE_SYMBOL=SOL-USDT
LIVE_RENKO_PATH=data/live/renko_latest.parquet
SIGNALS_DIR=/data/live/signals
LIVE_EXECUTOR_STATE=/data/live/live_executor_state.json

# Sizing
LIVE_EXECUTOR_POS_PCT=0.90          # fraction of equity to deploy (default 0.90 = 90%)
LIVE_EXECUTOR_LEVERAGE=6            # leverage multiplier for sizing math
KUCOIN_FUTURES_ORDER_LEVERAGE=6     # leverage sent to KuCoin per order (falls back to LIVE_EXECUTOR_LEVERAGE)

# Safety
LIVE_EXECUTOR_SYMBOL_ALLOWLIST=SOL-USDT
```

Sizing formula:
```
qty = floor(equity × LIVE_EXECUTOR_POS_PCT × LIVE_EXECUTOR_LEVERAGE
            / (mid_price × contract_multiplier_from_broker))
```

Contract multiplier is fetched live from the broker — do not hardcode it.

### 5. Kraken bot

Main entrypoint: `quant.execution.kraken_bot`

Responsibilities:
- live Kraken execution path
- event persistence
- equity persistence
- bot state reconciliation

Key file: `src/quant/execution/kraken_bot.py`

---

## Persistence model

Current practical rule:

- JSONL remains local append-only trace
- some runtime files remain operational state
- Postgres is the durable forensic source of truth

Already active in Postgres:
- `action_events` ✓
- `execution_events` ✓
- `closed_trades` ✓
- `equity_snapshots` ✓

Not yet fully wired:
- `signal_events`

---

## Runtime files

```
data/live/renko_latest.parquet      ← shared Renko source (written by renko_cache_updater)
data/live/execution_state.json      ← active strategy levels / last action
data/live/live_executor_state.json  ← executor loop state
data/live/live_signal_state.json    ← signal worker loop state
data/live/signals/                  ← JSONL signal streams
```

These are useful for operational inspection and local debugging but are **not** the preferred
forensic truth if Postgres already has the answer.

---

## Key environment variables

### KuCoin auth
```
KUCOIN_FUTURES_API_KEY
KUCOIN_FUTURES_API_SECRET
KUCOIN_FUTURES_PASSPHRASE
```

### Core runtime
```
PYTHONUNBUFFERED=1
SIGNALS_DIR=/data/live/signals
LIVE_RENKO_PATH=data/live/renko_latest.parquet
LIVE_EXECUTOR_STATE=/data/live/live_executor_state.json
LIVE_SIGNAL_STATE=/data/live/live_signal_state.json
```

### Safety switches
```
LIVE_TRADING_ENABLED=1          # must be 1 for live trading; defaults to 0
LIVE_EXECUTOR_DRY_RUN=0         # set to 1 to paper-trade without sending orders
```

### Sizing
```
LIVE_EXECUTOR_POS_PCT=0.90              # fraction of total equity
LIVE_EXECUTOR_LEVERAGE=6               # leverage multiplier (sizing math)
KUCOIN_FUTURES_ORDER_LEVERAGE=6        # leverage field sent to exchange per order
```

Note: `LIVE_EXECUTOR_LEVERAGE` and `KUCOIN_FUTURES_ORDER_LEVERAGE` are separate:
- `LIVE_EXECUTOR_LEVERAGE` → used only in local sizing formula
- `KUCOIN_FUTURES_ORDER_LEVERAGE` → sent to KuCoin in every order body; falls back to `LIVE_EXECUTOR_LEVERAGE` if unset
- KuCoin has their own max leverage per position size tier — if your position size exceeds a tier limit, KuCoin may silently cap it

---

## Operational verification pattern

Preferred pattern:

1. Verify code in repo
2. Push to Git
3. Deploy
4. Check executor startup log for sizing params: `executor sizing: equity=... -> qty=...`
5. Inspect Postgres tables
6. Inspect runtime logs/files only as secondary evidence

---

## Current deployment caveats

| Issue | Status |
|-------|--------|
| `write_execution_state()` signature mismatch | **FIXED** in f450c93 |
| Sizing undersized to qty=4 | **FIXED** in f450c93 — equity-based with live contract multiplier |
| Signal worker built independent Renko | **FIXED** in f450c93 — reads shared `renko_latest.parquet` |
| Signal snapping causing terminal_pos oscillation | **FIXED** in f450c93 — exact match only |
| `signal_events` Postgres insert failing | Still open — missing builder/store field alignment |
| OMS not margin-aware | Still open — uses qty as-is, no shrink-retry on rejection |

---

## Debugging order

When deployment/debugging instructions conflict, prefer:

1. current code behavior
2. ARCHITECTURE.md
3. current runbooks
4. older file-first notes
