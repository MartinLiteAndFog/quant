# Code Index

Important components of the quant trading system.

## Execution

live_executor
src/quant/execution/live_executor.py

Kraken bot
src/quant/execution/kraken_bot.py

OMS
src/quant/execution/oms.py

Venue adapters
src/quant/execution/kucoin_futures.py
src/quant/execution/kraken_futures.py

## Strategy

Flip engine
src/quant/strategies/flip_engine.py

Backtest runner
src/quant/backtest/renko_runner.py

## Event persistence

Event builders
src/quant/execution/event_builders.py

Event store
src/quant/execution/event_store.py

Event types
src/quant/execution/event_types.py

## Dashboard

Web server
src/quant/execution/webhook_server.py

Dashboard state
src/quant/execution/dashboard_state.py

## Runtime state

Execution state
src/quant/execution/execution_state.py