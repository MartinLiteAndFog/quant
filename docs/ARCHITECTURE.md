# Strategies

## Countertrend Flip Strategy (ON regime)

Core:
- IMBA impulses
- Renko bricks
- flip state machine

Entry
- IMBA signal

Exit
- trailing take profit
- stop loss
- opposite IMBA flip

Files:
src/quant/strategies/flip_engine.py
src/quant/backtest/renko_runner.py

## ImbaTrend Strategy (OFF regime)

Trend following.

Entry:
IMBA signal

Exit:
TP1 partial
TP2 final
SL clamp

Runner:
renko_runner_tp2.py