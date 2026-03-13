# Other — quant

# Quant Module Documentation

## Overview
The quant module provides a collection of tools and utilities for quantitative trading, with functionality spanning backtesting, execution, regime detection, and state space modeling. The module is organized into several key subpackages, each handling distinct aspects of the trading system.

## Core Components

### Backtesting (`quant.backtest`)
The backtesting engine, centered around `renko_runner_tp2.py`, implements a sophisticated trading strategy backtester with the following key features:

- Take-profit targets (TP1 and TP2) with partial position scaling
- Dynamic stop-loss based on swing high/low levels
- Break-even (BE) protection after TP1 is hit
- Regime filtering support
- Real fill price mapping for execution analysis

### Execution (`quant.execution`) 
Handles order execution and position management through:

- Order Management System (OMS) with maker-first routing
- Exchange-specific implementations (KuCoin, Kraken)
- Live monitoring and signal processing
- Webhook server for external integrations

### Regime Detection (`quant.regime`)
Provides regime classification and confidence scoring:

```python
from quant.regime import RegimeService, RegimeDecision

# Example usage
service = RegimeService()
decision = service.get_regime_decision(confidence_score=0.75)
```

### State Space Transitions (`quant.state_space_transitions`)
Models market state transitions in a 3-dimensional space:

- Voxelized state representation
- Sparse transition matrices
- Density basin detection
- JSON/Parquet artifact outputs

### Utilities (`quant.utils`)
Common utilities including the logging system with throttling support:

```python
from quant.utils.log import get_logger, log_throttled

logger = get_logger("my_component")
log_throttled(logger, logging.INFO, "key", 60.0, "Message") # Max once per minute
```

## Configuration
The module uses environment variables for configuration, managed through `quant.config`:

```python
from quant.config import Config

config = Config(
    exchange="kucoin",
    market_type="spot",
    symbol="BTC/USDT",
    timeframe="1m"
)
```

## Data Flow
1. Raw market data ingestion through exchange APIs
2. Feature computation and state space mapping
3. Regime classification and confidence scoring
4. Strategy signal generation
5. Order execution through OMS
6. Performance monitoring and analysis

## Integration Points
- Exchange APIs via CCXT
- External signal providers through webhooks
- Monitoring dashboards
- Database storage for regime states
- File-based artifacts for analysis

## Error Handling and Logging
The module implements a robust logging system with:

- Rich console output when available
- Message throttling to prevent log spam
- Thread-safe logging state management
- Configurable log levels per component

## Development Guidelines
1. Use pure functions where possible
2. Avoid global state
3. Implement proper error handling
4. Write tests for critical components
5. Document public interfaces
6. Use type hints consistently

## Dependencies
- pandas: Data manipulation
- numpy: Numerical operations
- ccxt: Exchange connectivity
- rich: Enhanced console output (optional)
- dotenv: Configuration management

The module is designed to be modular and extensible, allowing for easy addition of new strategies, execution venues, and analysis tools while maintaining a consistent interface across components.