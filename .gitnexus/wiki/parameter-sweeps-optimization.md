# Parameter Sweeps & Optimization

# Parameter Sweeps & Optimization Module

This module provides tools for systematically exploring and optimizing trading strategy parameters through parameter sweeps, Monte Carlo simulation, and leverage optimization.

## Core Components

### Grid Search (sweep_grid.py)
Performs exhaustive grid search across multiple strategy parameters:
- Take-profit trailing percentage
- Stop-loss percentage 
- Lookback period
- Break-even trigger and offset levels

```python
# Example: Sweep across TTP and SL values
python -m scripts.sweep_grid \
  --parquet data/renko/btc.parquet \
  --signals-jsonl signals.jsonl \
  --ttp-grid "0.005,0.0075,0.01" \
  --sl-grid "0.007,0.01,0.013"
```

The script:
1. Generates all parameter combinations
2. Runs backtest for each combination
3. Collects results in CSV/Parquet files
4. Identifies top N performing parameter sets

### Regime Parameter Optimization (sweep_regime_er.py)
Optimizes regime filter parameters, particularly for Efficiency Ratio (ER) based filters:
- ER length
- ER threshold levels for entering/exiting regimes
- Can combine with other regime indicators (Chop, ADX)

### Renko Box Size Optimization
Two specialized sweepers for Renko chart parameters:

**sweep_renko_box.py** - Fixed box sizes:
- Sweeps across different fixed box sizes
- Compares equity curves across box sizes
- Helps identify optimal fixed box size

**sweep_renko_atr.py** - Adaptive ATR-based boxes:
- Tests ATR-based dynamic box sizing
- Compares fixed vs ATR-adaptive approaches
- Evaluates different ATR periods and multipliers

### Position Sizing Tools

**kelly_from_trades.py** - Kelly criterion optimization:
- Calculates optimal Kelly fraction from trade history
- Provides fractional Kelly recommendations (1/2, 1/4)
- Helps avoid over-leveraging

**leverage_sweep.py** - Leverage optimization:
- Simulates different leverage levels on historical trades
- Calculates key metrics (returns, drawdowns, Calmar ratio)
- Helps identify optimal leverage level

**monte_carlo_trades.py** - Monte Carlo simulation:
- Bootstrap resampling of historical trades
- Generates distribution of possible outcomes
- Provides confidence intervals for strategy performance

## Key Features

- **Comprehensive Coverage**: Tests all important strategy parameters
- **Efficient Storage**: Results saved in both CSV and Parquet formats
- **Top-N Analysis**: Automatically identifies best parameter combinations
- **Visualization**: Generates comparative equity curves
- **Risk Management**: Tools for position sizing and leverage optimization

## Usage Workflow

1. Start with broad grid search using `sweep_grid.py`
2. Optimize regime parameters with `sweep_regime_er.py`
3. Fine-tune Renko parameters using renko sweepers
4. Use Monte Carlo to validate robustness
5. Optimize position sizing using Kelly/leverage tools

## Integration Points

- Takes input from signal generation modules
- Uses backtest engine for evaluations
- Feeds optimal parameters to strategy execution
- Integrates with fill models for realistic simulation

## Best Practices

1. Start with wide parameter ranges, then narrow down
2. Use reasonable step sizes to manage computation time
3. Consider multiple metrics (returns, drawdown, Calmar)
4. Validate results with Monte Carlo simulation
5. Be conservative with leverage/position sizing

## Output Structure

```
data/sweeps/
  └── sweep_<timestamp>/
      ├── results.csv      # All results
      ├── results.parquet  # All results (binary)
      ├── top30.csv        # Best parameter sets
      ├── meta.json       # Sweep configuration
      └── logs/           # Individual run logs
```

This module is critical for finding robust parameter sets and avoiding overfitting through comprehensive testing and validation.