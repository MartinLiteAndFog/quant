# Other — tests

# Test Suite Documentation

This module contains the test suite for the quantitative trading system. The tests cover multiple key components including state space modeling, regime detection, dashboard functionality, and trade execution logic.

## Overview

The test suite uses Python's unittest framework and follows standard testing practices with setUp/tearDown fixtures and clear test case organization. The tests extensively use temporary directories and environment variables to isolate test execution.

## Key Test Areas

### Dashboard & API Tests
- Tests the web dashboard's chart rendering, API endpoints, and data refresh mechanisms
- Verifies proper handling of trading state, position data, and regime information
- Validates caching behavior and environment variable configurations
- Key files: `test_webhook_dashboard_api.py`, `test_dashboard_state.py`

### State Space & Regime Tests  
- Validates state space computation pipeline and transformations
- Tests regime detection and state transitions
- Verifies proper handling of time series data and statistical calculations
- Key files: `test_state_space_v01.py`, `test_state_space_transitions.py`, `test_regime_store.py`

### Trading Logic Tests
- Tests trade execution logic including entries, exits, and position management
- Validates stop loss, take profit, and position flipping behavior
- Tests manual order handling and safety checks
- Key files: `test_kraken_bot_statemachine.py`, `test_live_executor.py`, `test_manual_orders.py`

### Data Processing Tests
- Tests data cleaning, transformation, and storage
- Validates proper handling of timestamps and time zones
- Tests parquet file operations and data format conversions
- Key files: `test_trim_statespace_parquet.py`, `test_renko_cache_updater.py`

## Testing Patterns

### Mock Objects
The tests make extensive use of mock objects to simulate external dependencies:

```python
class _DummyBroker:
    def __init__(self, pos: float, bid: float, ask: float):
        self._pos = float(pos)
        self._bid = float(bid)
        self._ask = float(ask)
        
    def get_best_bid_ask(self, symbol: str):
        return (self._bid, self._ask)
```

### Temporary Test Data
Tests create isolated test environments using temporary directories:

```python
def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.root = Path(self.tmp.name)
    # Create test data files
    
def tearDown(self):
    self.tmp.cleanup()
```

### Environment Configuration
Tests manage environment variables to control system behavior:

```python
os.environ["DASHBOARD_RENKO_PARQUET"] = str(self.root / "renko.parquet")
os.environ["REGIME_DB_PATH"] = str(self.root / "regime.db")
```

## Best Practices

1. **Isolation**: Each test creates its own isolated environment with temporary files and directories

2. **Clear Setup/Teardown**: Tests properly initialize and clean up resources

3. **Comprehensive Coverage**: Tests cover both happy paths and edge cases

4. **Mock External Dependencies**: External services are properly mocked

5. **Data Generation**: Tests include utilities for generating realistic test data

## Running the Tests

The tests can be run using Python's unittest framework:

```bash
python -m unittest discover tests/
```

Individual test files can also be run directly:

```bash 
python -m unittest tests/test_webhook_dashboard_api.py
```

## Adding New Tests

When adding new tests:

1. Follow the existing pattern of test organization
2. Create proper setUp/tearDown methods
3. Use temporary directories for file operations
4. Mock external dependencies
5. Test both success and failure cases
6. Include clear assertions and error messages

## Common Test Utilities

The test suite includes several shared utilities for:

- Creating temporary test environments
- Generating synthetic price data
- Mocking broker interfaces
- Managing environment variables
- Creating test regime data

These utilities should be reused when adding new tests to maintain consistency.