# Other — features

# Trading Gate Feature Configuration

This documentation covers the configuration metadata for a trading gate feature that controls market entry/exit based on market choppiness and trend indicators.

## Overview

The gate feature implements a trading filter that helps determine optimal market conditions for entering and exiting positions in the SOL-USDC trading pair. It uses a combination of the Choppiness Index (CHOP) and Average Directional Index (ADX) indicators to identify favorable trading conditions.

## Configuration Parameters

### Time Window Settings
- `freeze_start`: Trading gate activation date (2025-05-01)
- `dw`: Lookback window size of 3 days
- `pre_rows`: 490 periods of pre-analysis data
- `hold_rows`: 278 periods of hold analysis data

### Indicator Thresholds
- Choppiness Index (CHOP)
  - `chop_on`: 41.63 - Upper threshold for market choppiness
  - `chop_off`: 39.01 - Lower threshold for market choppiness
  - `chon_q`: 0.75 - Upper quantile for CHOP calibration
  - `choff_q`: 0.65 - Lower quantile for CHOP calibration

- Average Directional Index (ADX)
  - `adx_on`: 33.09 - Minimum ADX value for trend confirmation
  - `adx_q`: 0.50 - Median quantile for ADX calibration

### Performance Metrics
- `hold_on_rate`: 0.6115 (61.15%) - Success rate of hold signals

## Usage

This configuration file serves as a parameter set for the trading gate system. The gate uses these thresholds to:

1. Identify choppy market conditions using the CHOP indicator
2. Confirm trend strength using the ADX indicator
3. Generate trading signals based on the combined indicator readings

The system will begin applying these rules starting from the `freeze_start` date.

## Integration Notes

- This configuration is specific to the SOL-USDC trading pair
- Parameters were calibrated using 490 periods of historical data
- The hold analysis was conducted over 278 periods
- The file naming convention includes the trading pair, daily timeframe, and activation date

## File Format

The configuration is stored in JSON format with the naming pattern:
`{TRADING_PAIR}_daily_gate_FREEZE_{ACTIVATION_DATE}_dw{WINDOW_SIZE}.meta.json`