# Predictive Coding System

# Predictive Coding System Documentation

## Overview

The Predictive Coding System is a machine learning framework for time series prediction that combines temporal dynamics modeling with online learning. It's specifically designed for financial market prediction, using a latent state space model with both forward prediction and reconstruction objectives.

## Core Components

### 1. Temporal PC Model

The `TemporalPCModel` class implements the core predictive coding algorithm with these key features:

- Maintains a latent state vector that captures market dynamics
- Uses a temporal transition matrix (A) to predict state evolution
- Learns observation mappings (C) for feature reconstruction
- Maintains horizon-specific prediction weights (W_h) for different forecast horizons
- Tracks prediction uncertainties through variance estimates

```python
# Example initialization
model = TemporalPCModel(PCConfig(
    d_latent=32,  # latent state dimension
    n_obs=5,      # number of input features
    horizons=[1, 5, 15, 60]  # prediction horizons
))
```

### 2. Inference Process

The model performs causal inference at each timestep through:

1. **Prior Computation**: Uses transition matrix A to predict next state
2. **Observation Integration**: Updates state estimate using current features
3. **Variance Tracking**: Maintains uncertainty estimates for predictions
4. **Online Learning**: Updates model parameters using realized outcomes

Key aspects:
- No lookahead bias - uses only past/current data for inference
- Delayed supervised learning - updates prediction weights only when targets become available
- Robust estimation through variance-weighted updates and residual clipping

### 3. Trading Logic

The `TradeDecisionLayer` implements a trading policy based on model predictions:

- Uses agreement gates across multiple horizons (e.g., 5m and 15m)
- Incorporates transaction costs and minimum edge requirements
- Implements position management with stop-loss and take-profit rules
- Includes cooldown periods to prevent overtrading

## Configuration

The `PCConfig` class centralizes all hyperparameters:

```python
@dataclass
class PCConfig:
    # Model parameters
    d_latent: int = 32        # latent dimension
    n_obs: int = 5            # input features
    horizons: List[int]       # prediction horizons
    
    # Learning rates
    lr_x: float = 0.05       # state inference
    lr_A: float = 1e-4       # transition matrix
    lr_W: float = 1e-4       # prediction weights
    
    # Trading parameters
    fee_bps: float = 7.0     # transaction costs
    margin: float = 0.02     # required edge
    sl_pct: float = 0.015    # stop-loss
```

## Feature Engineering

The system expects specific input features computed by `build_obs_features()`:

- Short-term returns (r_1, r_5, r_15)
- Realized volatility estimates (rv_20, rv_60)
- Features are normalized and designed to be stationary

## Usage Example

```python
# Initialize
config = PCConfig(d_latent=32, horizons=[1, 5, 15])
model = TemporalPCModel(config)
trader = TradeDecisionLayer(config)

# Process each bar
obs = build_obs_features(prices)
predictions = model.step(price, obs)
signal, events = trader.update(predictions, price)
```

## Calibration Tools

The module includes tools for model diagnostics:

- Probability calibration analysis
- Regime detection for trading gates
- Performance attribution by horizon

## Key Considerations

1. **Warmup Period**: The model requires an initial warmup period (default 200 bars) before learning parameters

2. **Risk Management**: Multiple layers of risk controls:
   - Variance-based uncertainty estimates
   - Position sizing based on prediction strength
   - Stop-loss and take-profit rules
   - Cooldown periods after trades

3. **Computational Efficiency**: 
   - Vectorized operations for core computations
   - Ring buffers for managing delayed targets
   - Bounded state updates through clipping

4. **Robustness Features**:
   - Variance tracking for adaptive learning
   - Residual clipping for outlier handling
   - Multi-horizon agreement requirements
   - Shrinkage regularization on transition matrix

## Dependencies

- NumPy for numerical computations
- Pandas for data handling (calibration tools only)
- Pure Python implementation of core algorithm

The system is designed to be self-contained and efficient, suitable for both research and production deployment.