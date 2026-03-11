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

Signal → Action → Execution → Closed Trade → Equity

Primary architecture direction:

**Postgres-first forensic event system**

Runtime JSONL/state files remain as operational traces but are no longer intended as authoritative truth.

---

# Currently Running Components

## Live signal generation

Worker:
quant.execution.live_signal_worker

Responsibilities:

compute IMBA signals

maintain countertrend and trendfollower streams

route active stream via gate

write signals to:

/data/live/signals/SOLUSDT/YYYYMMDD.jsonl

Strategy routing:

gate_on = 1 → countertrend (IMBA flip strategy)
gate_on = 0 → trendfollower
Live executor (KuCoin)

Worker:

quant.execution.live_executor

Responsibilities:

read routed signal stream

decide entry/exit/flip

call OMS

place KuCoin orders

persist action events

persist execution events

update execution_state.json

Key files:

src/quant/execution/live_executor.py
src/quant/execution/oms.py
src/quant/execution/kucoin_futures.py
Kraken execution bot

Loop:

quant.execution.kraken_bot

Responsibilities:

maintain bot state

execute strategy logic

reconcile venue position

persist execution events

persist equity snapshots

Key file:

src/quant/execution/kraken_bot.py
Event Persistence

Primary durable storage: Postgres

Important tables:

action_events
execution_events
closed_trades
equity_snapshots

Future:

signal_events

Execution events are now written by:

KuCoin live_executor

Kraken bot

Dashboard

Served by:

quant.execution.webhook_server

Main endpoint:

/dashboard

Provides:

Renko chart

gate shading

fib levels

trade markers

entry→exit segments

SL / TTP / TP overlays

Main files:

src/quant/execution/dashboard_state.py
src/quant/execution/webhook_server.py
Runtime Files

Operational state files:

/data/live/execution_state.json
/data/live/signals/*
/data/live/live_executor_state.json
/data/live/live_signal_state.json
/data/live/live_trailing_state.json

These are not authoritative forensic sources.

Deployment

Current production platform:

Railway

Services:

quant      → web + dashboard
Signal     → signal worker + live executor

Docs:

docs/RAILWAY.md
docs/RAILWAY_RUNBOOK.md
docs/LIVE_DEPLOY.md
Current Known Issues

Dashboard and worker may read different runtime files depending on container volume configuration.

Signal events are not yet persisted to Postgres.

Some event link fields remain None (expected during migration).

Execution state JSON is still used by dashboard for active levels.

Current Priorities

Verify KuCoin execution event persistence end-to-end

Add signal_events persistence

Complete Signal → Action → Execution linking

Reduce reliance on runtime files

Stabilize dashboard state reads

Debugging Order

Preferred debugging chain:

Postgres

venue position

code path

runtime files

logs

See:

docs/DEBUGGING.md
Session Start Reminder

When starting a new debugging or development session:

Read:

docs/SESSION_CONTEXT.md
docs/CURRENT_STATE.md
docs/ARCHITECTURE.md

before diving into code.


---

Danach einfach committen:

```bash
git add docs/CURRENT_STATE.md
git commit -m "Add CURRENT_STATE operational overview"
git push