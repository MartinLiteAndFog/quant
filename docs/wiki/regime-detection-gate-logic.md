# Regime Detection & Gate Logic

# Regime Detection & Gate Logic

This module provides a framework for detecting market regimes and managing trading gates based on technical indicators and state transitions. It consists of three main components:

## Core Components

### 1. Gate Confidence Engine
The `GateConfidenceLiveEngine` provides real-time confidence scores for trading decisions by:

- Analyzing market state using voxel maps and basin transitions
- Computing confidence intervals using Wilson scoring
- Providing forward-looking probabilities across multiple time horizons
- Caching results with configurable refresh intervals

```python
confidence = get_live_gate_confidence()
# Returns confidence scores, regime state, and probabilities
```

### 2. Regime Store
`RegimeStore` provides persistent storage and retrieval of regime states and transitions using SQLite:

- Stores regime states with timestamps and confidence levels
- Tracks regime transitions with metadata
- Maintains threshold snapshots and data quality metrics
- Provides historical querying capabilities

### 3. Regime Service
`RegimeService` acts as the high-level interface for regime management:

- Handles regime state updates and transitions
- Computes confidence scores
- Manages gate logic and debouncing
- Provides a clean API for other system components

## Key Features

### Indicator-Based Detection
The system uses multiple technical indicators to detect regimes:

- Choppiness Index (CHOP)
- Average Directional Index (ADX) 
- Efficiency Ratio (ER)
- Relative Volume (RVOL)
- Trend metrics (slope, R²)

### Confidence Scoring
Confidence scores are computed using:

- Historical gate statistics
- Wilson confidence intervals
- Forward transition probabilities
- Basin-level aggregation

### State Management
The system maintains:

- Current regime state (trend/countertrend)
- Gate status (on/off)
- Transition history
- Data quality metrics

## Configuration

Key environment variables:

```bash
GATE_CONF_CACHE_SEC=30        # Cache duration
GATE_CONF_HORIZONS_MINUTES=5,30,120,240  # Forward-looking horizons
GATE_DAILY_PATH=              # Path to daily gate data
GATE_CONF_ARTIFACT_DIR=       # Path to model artifacts
```

## Usage Examples

### Basic Usage
```python
from quant.regime import RegimeService, RegimeStore

# Initialize
store = RegimeStore()
service = RegimeService(store)

# Update regime state
service.upsert_decision(RegimeDecision(
    ts="2024-01-01T00:00:00Z",
    symbol="BTC-USD",
    gate_on=1,
    regime_state="trend",
    regime_score=0.75,
    confidence=0.85,
    reason_code="indicator_threshold"
))

# Get latest state
latest = store.get_latest_state("BTC-USD")
```

### Live Confidence
```python
from quant.regime import get_live_gate_confidence

conf = get_live_gate_confidence()
print(f"Current confidence: {conf['selected_p_trend']:.2f}")
print(f"Regime state: {'trend' if conf['now_gate_on_rate_voxel'] > 0.5 else 'countertrend'}")
```

## Integration Points

The module integrates with:

- Live signal generation systems
- Trading execution engines
- Dashboard/monitoring systems
- Historical analysis tools

## Utilities

The module includes scripts for:

- Building walkforward gates
- Analyzing regime windows
- Plotting diagnostics
- Scanning for similar regimes

## Best Practices

1. Always use UTC timestamps
2. Handle missing data gracefully
3. Use appropriate warmup periods for indicators
4. Monitor data quality metrics
5. Regularly backup the regime store

## Dependencies

- pandas
- numpy
- sqlite3
- matplotlib (for diagnostics)

The module is designed to be thread-safe and suitable for both live trading and historical analysis.