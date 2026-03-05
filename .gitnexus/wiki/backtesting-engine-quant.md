# Backtesting Engine — quant

# Backtesting Engine Documentation

## Overview

The backtesting engine provides a framework for testing and evaluating trading strategies using historical price data. It supports multiple backtesting approaches:

- Basic position-based backtesting with fees
- Maker/taker fill modeling with price improvements
- Signal-following strategies with regime filters
- Two-target (TP1/TP2) strategies with dynamic stop losses

## Core Components

### Engine (`engine.py`)

The core backtesting logic that processes positions and calculates returns:

```python
def run_backtest(df: pd.DataFrame, pos: pd.Series, fee_bps: float = 4.0) -> BacktestResult:
    """
    Simple backtest using close-to-close returns and position signals.
    
    Args:
        df: OHLCV dataframe with 'ts' and 'close' columns
        pos: Position series (+1 long, -1 short, 0 flat)
        fee_bps: Fee in basis points per trade roundtrip
        
    Returns:
        BacktestResult with equity curve and summary statistics
    """
```

The engine:
- Calculates returns from close prices
- Applies position signals to returns
- Tracks turnover and fees
- Computes equity curve and key metrics

### Fill Model (`fill_model.py`) 

Models realistic trade execution with maker/taker dynamics:

```python
@dataclass
class FillModelParams:
    l1_bps: float = 0.0002      # L1 maker improvement
    l2_bps: float = 0.0005      # L2 maker improvement  
    entry_fallback_bps: float = 0.0003  # Taker slippage
    fee_bps_roundtrip: float = 0.0015   # Total fees
```

The fill model:
- Simulates limit order fills with price improvements
- Models taker fills with slippage
- Applies maker/taker fees based on fill type
- Supports probabilistic fill scenarios

### Runners

Multiple runner implementations for different strategy types:

- `follow_runner.py`: Basic signal following with regime filters
- `renko_runner.py`: Renko-based strategies with dynamic stops
- `renko_runner_tp2.py`: Two-target strategies with scaling out

The runners handle:
- Signal processing and position generation
- Entry/exit logic and risk management
- Regime filtering and trade management
- Performance tracking and reporting

### Analysis Tools

Utilities for analyzing backtest results:

- `compare_runs.py`: Compare metrics between backtest runs
- `signal_report.py`: Analyze signal quality and statistics
- `metrics.py`: Calculate performance metrics
- `sweep_confirm.py`: Parameter sweep optimization

## Usage Examples

Basic backtest:

```python
from quant.backtest.engine import run_backtest

# Run simple backtest
result = run_backtest(
    df=ohlcv_data,
    pos=signals, 
    fee_bps=4.0
)

# Access results
print(f"Return: {result.stats['total_return_pct']}%")
print(f"Max DD: {result.stats['max_drawdown_pct']}%")
```

Two-target strategy:

```python
from quant.backtest.renko_runner_tp2 import TP2Params, run_tp2_engine

# Configure strategy
params = TP2Params(
    tp1_pct=0.015,  # First target
    tp2_pct=0.030,  # Second target
    tp1_frac=0.5,   # Scale out 50% at TP1
    min_sl_pct=0.03 # Minimum stop loss
)

# Run backtest
events, legs = run_tp2_engine(
    bricks=renko_data,
    sig_event=signals,
    gate_on=regime,
    params=params
)
```

## Integration Points

The backtesting engine integrates with:

- Signal generation strategies (`quant.strategies`)
- Order management system (`quant.execution.oms`)
- Data processing pipeline (`quant.data`)
- Web API for live signals (`quant.api`)

## Best Practices

1. Always use UTC timestamps consistently
2. Handle missing data and edge cases gracefully
3. Validate input data quality before backtesting
4. Use realistic fees and fill assumptions
5. Compare results across multiple parameter sets
6. Analyze robustness through regime filters

## Limitations

- Uses close prices for simplicity (can be enhanced with OHLC)
- Does not model market depth/liquidity
- Simplified fee structure
- Perfect execution assumed unless fill model used
- No portfolio-level constraints

The backtesting engine provides a foundation for strategy development while making reasonable simplifications. Use the fill model and regime filters to add realism, and always validate strategies with out-of-sample data.