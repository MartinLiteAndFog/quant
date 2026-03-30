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
- signal_events (planned — Postgres path incomplete)

---

## Core execution logic

### KuCoin executor
src/quant/execution/live_executor.py

### Kraken executor (active development target)
src/quant/execution/live_executor_2.py

Stop-order-native executor. Implements TTP re-entry handoff, pending follow-entry,
stop/TP order sync via tagged client IDs (quant:<SYM>:<kind>:<ms>).

### OMS (Order Management System)
src/quant/execution/oms.py

Maker-first + stop-order management. New methods:
arm_stop_entry, arm_stop_exit, arm_take_profit_exit, arm_flip_close_stop,
find_stop_order_by_kind, cancel_orders_by_kind, cancel_all_quant_orders.

### Strategy logic
src/quant/strategies/flip_engine.py
src/quant/strategies/follow_tp2_engine.py
src/quant/strategies/imba.py  ← also provides get_latest_imba_barriers()

### Dashboard
src/quant/execution/webhook_server.py
src/quant/execution/dashboard_state.py

Docs entry point:

docs/SESSION_CONTEXT.md
