# Other — tools

# Kraken Futures Authentication Tools

This module provides diagnostic tools for verifying and troubleshooting Kraken Futures API authentication. It consists of two utilities that help developers validate their API credentials and environment configuration.

## Tools Overview

### 1. Environment Checker (`check_kraken_env.py`)

A lightweight utility that validates the presence and format of required Kraken Futures environment variables:

- Verifies `KRAKEN_FUTURES_BASE_URL` and `KRAKEN_GATE_URL` settings
- Validates the format of `KRAKEN_FUTURES_SECRET` (checks base64 encoding)
- Tests connectivity to the gate service
- Optionally performs a live API test if `RUN_KRAKEN_PRIVATE_TEST=1`

Usage:
```bash
python3 data/tools/check_kraken_env.py
```

### 2. Manual Authentication Tester (`check_kraken_manual.py`)

A comprehensive authentication debugging tool that:

- Tests API credentials with multiple encoding approaches
- Provides detailed diagnostic output
- Helps identify common authentication issues

Usage:
1. Edit the script to add your credentials:
```python
KRAKEN_KEY    = "your_key_here"
KRAKEN_SECRET = "your_secret_here"
```

2. Run the script:
```bash
python3 data/tools/check_kraken_manual.py
```

## Common Authentication Issues

The tools help diagnose several frequent authentication problems:

1. **Invalid Secret Format**
   - Base64 decoding failures
   - Whitespace or quote characters in credentials
   
2. **Wrong API Type**
   - Using Spot API keys instead of Futures API keys
   - Missing Futures trading account access

3. **Environment Configuration**
   - Missing environment variables
   - Incorrectly formatted secrets
   - Gate service connectivity issues

## Integration with Main System

These tools are standalone utilities that support the main trading system by:

- Validating credentials before deployment
- Troubleshooting authentication issues in production
- Verifying environment configuration
- Testing connectivity to required services

They interact with the same APIs used by `KrakenFuturesClient` in the main application but are intentionally isolated to provide independent verification.

## Security Notes

- The manual checker (`check_kraken_manual.py`) requires temporary insertion of API credentials
- Always remove credentials after testing
- Both tools support the standard Kraken Futures authentication flow using:
  - HMAC-SHA512 signatures
  - Base64-encoded secrets
  - Nonce-based request signing

## Best Practices

1. Run `check_kraken_env.py` after environment changes
2. Use `check_kraken_manual.py` for detailed authentication debugging
3. Never commit files containing API credentials
4. Verify credentials work in these tools before deploying to production

The tools are designed to be simple to use while providing comprehensive diagnostics for authentication and configuration issues.