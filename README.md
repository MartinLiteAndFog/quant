# quant

Quant is the working repository for live trading, research, execution, dashboarding, and forensic event persistence.

The current architecture direction is:

**Postgres-first for forensic truth**, with JSONL and runtime files kept as transitional operational traces.

The intended chain is:

**Signal → Action → Execution → Closed Trade → Equity**

---

## Core areas

### Live execution
Live trading and venue execution components for:
- KuCoin Futures
- Kraken Futures

Main files:
- `src/quant/execution/live_executor.py`
- `src/quant/execution/kraken_bot.py`
- `src/quant/execution/oms.py`
- `src/quant/execution/kucoin_futures.py`
- `src/quant/execution/kraken_futures.py`

### Strategy logic
Main live/backtest strategy logic includes:
- countertrend flip strategy
- ImbaTrend / trend-following regime
- IMBA-based signal generation
- Renko-based state and execution logic

Main files:
- `src/quant/strategies/flip_engine.py`
- `src/quant/backtest/renko_runner.py`

### Dashboard / API
Dashboard, state views, fills, markers, and runtime APIs.

Main files:
- `src/quant/execution/webhook_server.py`
- `src/quant/execution/dashboard_state.py`

### Event persistence
Durable event and forensic persistence.

Main files:
- `src/quant/execution/event_builders.py`
- `src/quant/execution/event_store.py`
- `src/quant/execution/event_types.py`

---

## Main documents

### Architecture and event model
- `docs/ARCHITECTURE.md`
- `docs/event_schema_v1.md`

### Deployment and operations
- `docs/LIVE_DEPLOY.md`
- `docs/RAILWAY.md`
- `docs/RAILWAY_RUNBOOK.md`

### Debugging and terminology
- `docs/DEBUGGING.md`
- `docs/GLOSSARY.md`

---

## Current persistence direction

The system is in an incremental migration phase.

### Already important in Postgres
- `action_events`
- `execution_events`
- `closed_trades`
- `equity_snapshots`

### Not yet fully live-wired end-to-end
- `signal_events`

### Practical rule
Use this source priority when debugging:

1. Postgres
2. current deployed code behavior
3. JSONL/runtime traces
4. logs

---

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

Install dependencies using the project’s current dependency method in the repo.
If package/import issues appear, prefer running commands with:

PYTHONPATH=src

Example:

PYTHONPATH=src python3 -m quant.execution.live_executor --once
Typical workflows
Local compile check
PYTHONPATH=src python3 -m py_compile src/quant/execution/live_executor.py
Run one live-executor cycle
PYTHONPATH=src python3 -m quant.execution.live_executor --once
Run one Kraken bot cycle
PYTHONPATH=src python3 -m quant.execution.kraken_bot --once
Working rules for live changes

Preferred workflow:

fix in repo

compile/check locally

commit

push

deploy

verify live

verify Postgres persistence

Avoid hotpatching running containers unless absolutely necessary.

State space

The repository also contains state-space / predictive-coding related research components.

For that area, see the corresponding scripts and docs under:

docs/plans/

docs/visual/

Older state-space work remains relevant, but the repo now also includes substantial live execution and forensic infrastructure beyond that earlier scope.