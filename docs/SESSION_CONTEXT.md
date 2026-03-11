# Quant – Session Context

Use this document as entry point when starting a new session.

## Always check first

docs/CURRENT_STATE.md
docs/ARCHITECTURE.md
docs/event_schema_v1.md

## Current state
docs/CURRENT_STATE.md

## Architecture
See:
- docs/ARCHITECTURE.md
- docs/event_schema_v1.md

Core execution pipeline:

Signal → Action → Execution → Closed Trade → Equity

## Important components

Execution
- src/quant/execution/live_executor.py
- src/quant/execution/kraken_bot.py
- src/quant/execution/oms.py

Strategies
- src/quant/strategies/flip_engine.py
- src/quant/backtest/renko_runner.py

Dashboard
- src/quant/execution/webhook_server.py

## Operations

Deployment
- docs/LIVE_DEPLOY.md
- docs/RAILWAY_RUNBOOK.md

Debugging
- docs/DEBUGGING.md

Glossary
- docs/GLOSSARY.md

## Current architecture direction

Postgres-first forensic system.

Important tables:
- signal_events (planned)
- action_events
- execution_events
- closed_trades
- equity_snapshots

## Current priorities

1. Finish KuCoin event persistence
2. Wire signal_events
3. Complete forensic chain
4. Dashboard stability