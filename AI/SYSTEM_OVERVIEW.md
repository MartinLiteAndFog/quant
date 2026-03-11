# Quant System Overview

This repository contains a live crypto trading system.

Core pipeline:

Signal → Action → Execution → Closed Trade → Equity

Execution venues:

- KuCoin Futures
- Kraken Futures

Primary architecture direction:

Postgres-first forensic event system.

Event tables:

- action_events
- execution_events
- closed_trades
- equity_snapshots
- signal_events (planned)

Core execution logic:

src/quant/execution/live_executor.py
src/quant/execution/kraken_bot.py
src/quant/execution/oms.py

Strategy logic:

src/quant/strategies/flip_engine.py

Dashboard:

src/quant/execution/webhook_server.py

Docs entry point:

docs/SESSION_CONTEXT.md