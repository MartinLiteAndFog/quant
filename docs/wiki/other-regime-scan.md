# Other — regime_scan

# Regime Scan Target Fingerprints

## Overview
This module contains target fingerprint data used for market regime classification and analysis. It defines statistical benchmarks for various technical indicators that help characterize different market conditions or "regimes."

## Data Structure
The data is organized in a JSON structure with two main components:
- `target_fp`: Target fingerprint values for each indicator
- `scales`: Scale factors for normalizing the indicators

### Tracked Indicators
The module tracks the following technical indicators:

| Indicator | Description | Metrics Captured |
|-----------|-------------|------------------|
| CHOP | Choppiness Index | mean, p25, p50, p75 |
| ADX | Average Directional Index | mean, p25, p50, p75 |
| ER | Efficiency Ratio | mean, p25, p50, p75 |
| ATR_PCT | Average True Range (%) | mean, p25, p50, p75 |
| RVOL_PCT | Relative Volume (%) | mean, p25, p50, p75 |
| RSI | Relative Strength Index | mean, p25, p50, p75 |
| TREND_SLOPE | Linear Regression Slope | mean, p25, p50, p75 |
| TREND_R2 | Trend R-squared | mean, p25, p50, p75 |

## Usage
This fingerprint data serves as a reference point for comparing current market conditions against historical patterns. The values represent statistical distributions of each indicator as of January 2026.

### Example Interpretation
```python
# CHOP Index interpretation
chop_median = 43.22  # From target_fp["CHOP:p50"]
chop_iqr = 50.44 - 35.88  # p75 - p25

# Values significantly above median suggest choppy/ranging markets
# Values below median suggest trending markets
```

## Scale Factors
The `scales` object provides normalization factors for each indicator. These values should be used to standardize raw indicator values before comparing them to the target fingerprints:

```python
normalized_value = (raw_value - target_fp[indicator]) / scales[indicator]
```

## Notes for Contributors
- When updating fingerprint values, maintain the same statistical metrics (mean, p25, p50, p75)
- Scale factors should be updated whenever the target fingerprints are recalculated
- The filename format (`jan2026_target_fp.json`) indicates this is a point-in-time snapshot

## Related Components
This module is typically used in conjunction with:
- Market regime classification systems
- Trading strategy adjustments based on regime detection
- Risk management systems that adapt to market conditions