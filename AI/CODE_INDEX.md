# Code Index

Important components of the quant trading system.

## Execution

### KuCoin executor
src/quant/execution/live_executor.py

### Kraken executor (primary active development target)
src/quant/execution/live_executor_2.py
— Stop-order-native. TTP re-entry, pending follow-entry, stop/TP sync.
— Key helpers: _record_ttp_external_exit, _arm_ttp_reenter_handoff, _sync_stop_order, _sync_take_profit_order

### Signal worker
src/quant/execution/live_signal_worker.py

### Renko cache updater (single Renko authority)
src/quant/execution/renko_cache_updater.py

### OMS
src/quant/execution/oms.py
— BrokerAPI abstract base with stop/TP order methods
— MakerFirstOMS: arm_stop_entry, arm_stop_exit, arm_take_profit_exit, arm_flip_close_stop
— find_stop_order_by_kind, cancel_orders_by_kind, cancel_all_quant_orders

### Venue adapters
src/quant/execution/kucoin_futures.py
— cancel_order, list_open_orders, list_open_stop_orders

src/quant/execution/kraken_futures.py
— place_take_profit_market (orderType: take_profit)
— place_trigger_entry_market (orderType: trigger_entry)

## Strategy

### Flip engine (countertrend state machine)
src/quant/strategies/flip_engine.py

### Follow TP2 engine (trendfollower state machine)
src/quant/strategies/follow_tp2_engine.py

### IMBA signals + barriers
src/quant/strategies/imba.py
— compute_imba_signals() — full per-bar signal computation
— get_latest_imba_barriers() — returns {ts, long_barrier, short_barrier} for live use and dashboard

### Backtest runner
src/quant/backtest/renko_runner.py

## Event persistence

### Event builders
src/quant/execution/event_builders.py

### Event store
src/quant/execution/event_store.py

### Event types
src/quant/execution/event_types.py

## Dashboard

### Web server
src/quant/execution/webhook_server.py

### Dashboard state
src/quant/execution/dashboard_state.py
— build_fibo_levels() now delegates to get_latest_imba_barriers() (no duplicate rolling math)

## Runtime state

### Execution state
src/quant/execution/execution_state.py
