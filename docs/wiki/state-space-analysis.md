# State Space Analysis

# State Space Analysis Module

## Overview

The State Space Analysis module provides a framework for analyzing financial market behavior in a 3-dimensional state space. It maps market conditions to three key axes:

- **X-axis (Drift/Trend)**: Captures directional price movement and momentum
- **Y-axis (Elasticity)**: Measures mean-reversion and oscillatory behavior
- **Z-axis (Instability)**: Quantifies market turbulence and structural breaks

The module transforms raw OHLCV data into this normalized state space representation, enabling systematic analysis of market regimes and behavioral patterns.

## Core Components

### 1. Pipeline (`pipeline.py`)

The main entry point is `compute_state_space()`, which orchestrates the full computation pipeline:

```python
def compute_state_space(df: pd.DataFrame, cfg: StateSpaceConfig = None) -> pd.DataFrame:
    # 1. Compute base features
    feat = compute_features(df, cfg)
    
    # 2. Calculate sensor readings for each axis
    sx = compute_sensors_x(feat, cfg)  # Drift sensors
    sy = compute_sensors_y(feat, cfg)  # Elasticity sensors 
    sz = compute_sensors_z(feat, cfg)  # Instability sensors
    
    # 3. Aggregate sensors into axis values
    x_axis = aggregate_axis(sx, weights=cfg.sensor_weights_x)
    y_axis = aggregate_axis(sy, weights=cfg.sensor_weights_y)
    z_axis = aggregate_axis(sz, weights=cfg.sensor_weights_z)
    
    # 4. Residualize Y and Z axes against X
    res = residualize_axes(x_axis["signal"], y_axis["signal"], z_axis["signal"])
```

### 2. Sensor Systems

Each axis has multiple specialized sensors that measure different aspects of market behavior:

**X-axis (Drift) Sensors:**
- Slope across multiple timeframes
- Efficiency ratios
- Trend alignment metrics

**Y-axis (Elasticity) Sensors:**
- Deviation from equilibrium price
- Ornstein-Uhlenbeck mean reversion
- Autocorrelation measures

**Z-axis (Instability) Sensors:**
- Jump detection rates
- RV/BV jumpiness metrics
- Entropy measures
- Price noise indicators

### 3. Basin Analysis

The module includes functionality to identify and analyze stable regions ("basins") in the state space:

```python
basins, labels = voxel_basins(
    x=df["X_raw"], 
    y=df["Y_res"],
    z=df["Z_res"],
    bins=cfg.basin_bins,
    top_k=cfg.basin_top_k
)
```

This helps identify common market regimes and transition patterns.

## Configuration

The module is configured via `StateSpaceConfig`, which controls:

- Window sizes for various calculations
- Sensor weights for each axis
- Basin detection parameters
- Signal normalization settings

Example:
```python
cfg = StateSpaceConfig(
    window_W=240,          # Base window size
    horizons=(5,15,60,240),# Timeframes for multi-scale analysis
    z_max=5.0,            # Max z-score for normalization
    basin_bins=40,        # Resolution for basin detection
)
```

## Usage

Basic usage:
```python
from quant.state_space import StateSpaceConfig, compute_state_space

# Load OHLCV data into df
cfg = StateSpaceConfig()
state = compute_state_space(df, cfg)

# Access state variables
print(state[["X_raw", "Y_res", "Z_res"]])  # Core state coordinates
print(state.attrs["basins"])                # Basin analysis results
```

## Visualization

The module includes a plotting script (`plot_state_space_v01.py`) that generates comprehensive visualizations of:

- 3D state space surface
- 2D projections of all axes
- Current market position
- Basin locations
- State variable distributions

## Dependencies

- Core dependencies: pandas, numpy
- Visualization: matplotlib (optional)
- Input data must include: timestamp, open, high, low, close
- Volume is optional but recommended

## Notes

- All signals are normalized to [-1, 1] range
- Y and Z axes are residualized against X to reduce correlation
- Basin detection uses adaptive density-based clustering
- The module is designed for both real-time and historical analysis