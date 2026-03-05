# Visualization & Plotting — scripts

# Visualization & Plotting Scripts

This module provides a collection of specialized plotting scripts for analyzing trading strategy performance, market data, and system behavior. Each script focuses on a specific visualization need while sharing common utilities for data loading and processing.

## Core Capabilities

### 1. Trade Analysis Plots

- **plot_trades_window.py**: Creates detailed trade visualizations showing entries, exits, and price action within a specified time window. Distinguishes between regular entries and "flip" entries (from position reversals).

- **plot_last5_trades.py**: Generates a focused view of the most recent 5 trades, useful for monitoring recent strategy behavior.

- **plot_monte_carlo.py**: Performs Monte Carlo analysis of trading results, visualizing:
  - Return distribution histograms
  - Maximum drawdown distributions 
  - Sharpe ratio distributions
  - Sample equity paths

### 2. Strategy Performance Visualization

- **plot_equity_gate_price.py**: Combines three key elements:
  - Price series
  - Equity curve
  - Trading gate windows (periods when strategy is active)

- **plot_merged_equity_solusdt.py**: Specialized plot for merged strategy performance against SOL-USDT price, with configurable Kelly leverage.

- **plot_equity_price_vol.py**: Three-panel visualization showing:
  - Price
  - Equity curve
  - Rolling volatility

### 3. Analysis Tools

- **build_density_images.py**: Pre-computes density heatmaps for dashboard state-space visualization, showing the distribution of strategy states across different parameter pairs.

## Key Implementation Patterns

### Data Loading & Preprocessing

The scripts share common patterns for robust data loading:

```python
def _read_parquet_ts(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    """Standardized timestamp handling for parquet data"""
    if ts_col not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": ts_col})
        else:
            raise ValueError(f"missing '{ts_col}'")
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    return df.dropna(subset=[ts_col]).sort_values(ts_col)
```

### Trade Processing

Most scripts that analyze trades implement similar logic for pairing entries with exits:

```python
def _pair_trades(events: pd.DataFrame) -> pd.DataFrame:
    """
    Pairs entry events with corresponding exits to form complete trades.
    Handles various exit types (TP, SL, signal flips).
    """
    exits = {"tp_exit", "sl_exit", "signal_flip_exit"}
    trades = []
    open_entry = None
    
    for _, event in events.iterrows():
        if event["event"] == "entry":
            open_entry = event
        elif event["event"] in exits and open_entry is not None:
            trades.append({
                "entry_ts": open_entry["ts"],
                "exit_ts": event["ts"],
                # ... other trade details
            })
            open_entry = None
    return pd.DataFrame(trades)
```

## Usage Examples

### Basic Trade Window Plot
```bash
python scripts/plot_trades_window.py \
  --parquet data/ohlcv.parquet \
  --events data/runs/strategy_v1/events.parquet \
  --start 2023-01-01 \
  --end 2023-01-31 \
  --out trades_jan.png
```

### Monte Carlo Analysis
```bash
python scripts/plot_monte_carlo.py \
  --trades data/runs/strategy_v1/trades.parquet \
  --paths 1000 \
  --maker-fill \
  --fee-bps 15 \
  --out monte_carlo.png
```

## Integration Points

- The scripts expect data in standardized formats (parquet files for OHLCV/trades, CSV for gates)
- Many scripts integrate with the broader trading system's output directory structure
- Results are typically saved as PNG files for dashboard integration or analysis

## Dependencies

- Core data processing: pandas, numpy
- Visualization: matplotlib
- File formats: parquet (pyarrow/fastparquet)

The module prioritizes robust error handling and data validation while maintaining flexibility in visualization parameters and output formats.