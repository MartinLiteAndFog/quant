# Other — quant.egg-info

# Quant Package Documentation

## Overview
The `quant` package is a Python-based cryptocurrency futures quantitative research and execution framework. This documentation covers the package configuration and structure as defined in the package metadata.

## Package Requirements

### Python Version
- Requires Python 3.9 or higher

### Core Dependencies
```
pandas >= 2.0
numpy >= 1.24
pyarrow >= 12.0
pydantic >= 2.0
python-dotenv >= 1.0
loguru >= 0.7
fastapi >= 0.110
uvicorn >= 0.27
```

## Package Structure
The package is organized into several key modules:

- **backtest**: Backtesting engine and analysis tools
- **data**: Data fetching and storage utilities
- **execution**: Trading execution and order management
- **features**: Feature engineering components
- **strategies**: Trading strategy implementations
- **utils**: Utility functions and logging

## Command Line Tools
The package provides two command-line entry points:

1. `quant-backtest`: Entry point for running backtests
   ```bash
   # Launches the Renko runner backtesting tool
   quant-backtest
   ```

2. `quant-webhook`: Entry point for the webhook server
   ```bash
   # Starts the webhook server for trade execution
   quant-webhook
   ```

## Key Components

### Backtesting Module
Contains tools for:
- Strategy performance comparison (`compare_runs.py`)
- Backtesting engine (`engine.py`)
- Signal reporting and analysis (`signal_report.py`)
- Specialized runners for different strategies (`renko_runner.py`, `follow_runner.py`)

### Execution Module
Provides:
- KuCoin exchange integration (`kucoin.py`)
- Order Management System (`oms.py`)
- Webhook server for trade execution (`webhook_server.py`)

### Strategies
Implements various trading strategies including:
- Baseline flip strategy
- Renko-based strategies
- Imbalance trading strategy

## Development Notes
- The package uses modern Python packaging standards with `pyproject.toml`
- Logging is handled through the `loguru` package
- API functionality is built using FastAPI
- Data processing relies heavily on pandas and numpy

For developers looking to contribute or modify the package, start by familiarizing yourself with the module structure and ensure all dependencies are properly installed. The package follows a modular design pattern, making it straightforward to add new strategies or modify existing ones.