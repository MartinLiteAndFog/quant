# Other — sweeps

# Parameter Sweeps Documentation

## Overview
The sweeps module manages parameter optimization runs for trading strategies. It stores metadata and results from systematic exploration of different parameter combinations to help identify optimal trading configurations.

## Sweep Types

### Basic Parameter Sweeps
Basic sweeps explore combinations of core trading parameters:
- Take profit trail percentages (`ttp_trail_pct`)
- Stop loss cap percentages (`sl_cap_pct`) 
- Swing lookback periods (`swing_lookback`)

Each sweep run generates results for all combinations of parameters defined in the grid configuration.

### Efficiency Ratio (ER) Sweeps 
ER sweeps focus on optimizing efficiency ratio parameters:
- ER calculation length (`er_len`)
- Entry thresholds (`er_on_grid`) 
- Exit thresholds (`er_off_grid`)

These sweeps help tune the strategy's sensitivity to market efficiency conditions.

## Sweep Configuration

### Metadata Structure
Each sweep is identified by a unique ID (timestamp-based) and stores:
- Input data sources (parquet files, signal files)
- Fixed parameters (fees, box sizes)
- Parameter grids to explore
- Runtime information

Example metadata:
```json
{
  "sweep_id": "er_sweep_20260209T194415Z",
  "created_utc": "2026-02-09T19:56:10.546963+00:00",
  "regime": "er",
  "er_len": 40,
  "er_on_grid": [0.2, 0.25, 0.3, 0.35],
  "er_off_grid": [0.3, 0.35, 0.4, 0.45],
  "fixed_params": {
    "fee_bps": 3.0,
    "ttp_trail_pct": 0.00775,
    "sl_cap_pct": 0.011
    // ...
  }
}
```

### Results Format
Sweep results are stored as JSON files containing:
- Run ID for each parameter combination
- Performance metrics:
  - Total return percentage
  - Maximum drawdown
  - Turnover
  - Trade counts
- Time period covered
- Parameter values used

## Usage

1. Define sweep configuration in a metadata JSON file
2. Run the sweep using the specified module (e.g., `quant.backtest.renko_runner`)
3. Results are saved to timestamped JSON files for analysis
4. Compare performance across parameter combinations to identify optimal settings

## Integration
The sweeps module connects to:
- Backtesting engine for running parameter combinations
- Data cleaning pipeline for input data
- Signal generation systems for trade signals
- Analysis tools for evaluating results

## Best Practices

1. Use meaningful parameter ranges based on market characteristics
2. Include enough granularity in parameter grids without excessive combinations
3. Maintain consistent fee assumptions across related sweeps
4. Document sweep objectives and constraints in metadata
5. Version control sweep configurations alongside code

## Limitations

- Results are specific to the market conditions in the test data period
- Parameter stability across different market regimes not guaranteed
- Computational resources limit grid granularity
- Risk of overfitting with too many parameters