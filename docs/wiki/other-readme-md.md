# Other — README.md

# Quant System Documentation

This document provides an overview of the quantitative trading system and its key components.

## System Overview

The Quant system is a production trading platform with two main components:
- A live futures execution engine (currently operational on KuCoin)
- A dashboard/worker architecture for trade monitoring and execution

## Core Components

### 1. State Space Engine
The system implements a 3-dimensional market state analysis pipeline that processes OHLCV data to identify significant market states:

- **Input**: OHLCV data in parquet/csv format
- **Processing**: Computes three axes (`X_raw`, `Y_res`, `Z_res`) to characterize market state
- **Output**: 
  - State space data with axis values and diagnostics
  - Basin analysis identifying dense state clusters

```python
# Example usage
from quant.state_space.pipeline import compute_state_space
from quant.state_space.config import StateSpaceConfig

state_space = compute_state_space(ohlcv_data, StateSpaceConfig())
basins = state_space.attrs["basins"]
```

### 2. Live Trading Infrastructure

The system uses a distributed architecture:
- `quant` service: Core trading engine
- `Signal` service: Signal generation/processing
- Dashboard: Trade monitoring and control interface

Current features include:
- Futures execution (verified on KuCoin SOLUSDTM)
- Gate routing
- Fibonacci overlay visualization
- Trade markers
- Stop-loss/Take-profit status tracking

## Development Setup

1. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/activate
```

2. Install dependencies (requirements.txt to be added)

## Operational Documentation

Key operational documents are located in the `docs/` directory:
- `RAILWAY_RUNBOOK.md`: Primary operations guide
- `RAILWAY.md`: Deployment procedures
- `LIVE_DEPLOY.md`: Production deployment notes

## Known Issues & Roadmap

Current priority: Unify state storage between `quant` and `Signal` services to ensure dashboard consistency with worker execution state.

## State Space CLI Reference

Process market data through the state space pipeline:
```bash
PYTHONPATH=src python3 scripts/run_state_space_v01.py \
  --input /path/to/market_data.parquet \
  --output data/state_space/output.parquet
```

The pipeline requires:
- Mandatory: `ts`, `open`, `high`, `low`, `close`
- Optional: `volume` (enables VWAP-based equilibrium calculations)