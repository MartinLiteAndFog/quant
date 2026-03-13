# Other — requirements.txt

# Requirements Documentation

## Overview
This `requirements.txt` file specifies the core Python dependencies needed to run the webhook and dashboard components of the application. It ensures consistent dependency versions across different deployment environments.

## Core Dependencies

### Web Framework Components
- **FastAPI (≥0.110)**: Modern web framework for building APIs
- **Uvicorn (≥0.27)**: ASGI server implementation for running FastAPI applications

### Data Processing
- **Pandas (≥2.0)**: Data manipulation and analysis library
- **NumPy (≥1.24)**: Numerical computing foundation

### Configuration & Utilities
- **python-dotenv (≥1.0)**: Environment variable management from `.env` files
- **Pydantic (≥2.0)**: Data validation using Python type annotations
- **Rich (≥13.0)**: Terminal formatting and display utilities

### Local Package Installation
The file includes a `.` entry which installs the local package in editable mode, ensuring the `quant.execution.webhook_server` module is importable.

## Usage

### Development Setup
```bash
pip install -r requirements.txt
```

### Deployment Notes
- This requirements file is specifically referenced by cloud platforms like Railway during deployment
- Version constraints use `>=` to allow compatible updates while preventing breaking changes
- All major dependencies are pinned to major versions (e.g., Pandas ≥2.0) to maintain stability

## Best Practices
1. Keep version constraints as permissive as possible while ensuring compatibility
2. Update dependencies periodically to incorporate security fixes
3. Test the application when upgrading major versions of any dependency

## Related Components
- The dependencies primarily support the webhook server and dashboard functionality
- FastAPI and Uvicorn form the web server infrastructure
- Pandas and NumPy support data processing operations
- Pydantic handles data validation for API endpoints