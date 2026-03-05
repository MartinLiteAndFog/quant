# Other — Dockerfile

# Dockerfile Documentation

## Overview
This Dockerfile defines the container environment for a Python-based quantitative trading application. It creates a single-stage build that serves as both the build and runtime environment, specifically designed to run a FastAPI webhook server.

## Key Components

### Base Image
```dockerfile
FROM python:3.12-slim
```
Uses the official Python 3.12 slim image to minimize container size while providing a complete Python environment.

### Application Structure
The container is configured with:
- Working directory: `/app`
- Source code: Copied from local `src` directory
- Dependencies: Defined in `pyproject.toml`
- Trading data: Includes regime data file `pc_3axis_gate_latest.csv`

### Server Configuration
- Default port: 8080 (configurable via environment variable)
- Entry point: `quant.execution.webhook_server:app`
- Host: `0.0.0.0` (accepts connections from any IP)

## Usage

### Building the Container
```bash
docker build -t quant-trading-app .
```

### Running the Container
```bash
docker run -p 8080:8080 quant-trading-app
```

With custom port:
```bash
docker run -e PORT=3000 -p 3000:3000 quant-trading-app
```

## Design Considerations

### Single-Stage Build
The Dockerfile uses a single-stage build pattern because:
- Ensures runtime environment matches build environment
- Prevents "No module named uvicorn" errors on platforms like Railway
- Simplifies deployment and reduces potential version mismatches

### Environment Variables
- `PORT`: Configurable at runtime, defaults to 8080
- Designed to work seamlessly with Railway's dynamic port allocation

### File Selection
Only essential files are copied into the container:
```dockerfile
COPY pyproject.toml .
COPY src ./src
COPY data/regimes/pc_3axis_gate_latest.csv ./data/regimes/pc_3axis_gate_latest.csv
```
This minimizes container size and reduces rebuild frequency when unrelated files change.

## Integration Points

### Webhook Server
The container runs a FastAPI webhook server (`quant.execution.webhook_server:app`) that:
- Handles incoming trading-related webhooks
- Processes quantitative trading signals
- Manages trading execution logic

### Data Files
The regime data file (`pc_3axis_gate_latest.csv`) is included for trading strategy configuration and must be present for proper operation.

## Best Practices
- Uses `--no-cache-dir` with pip to reduce image size
- Avoids shell expansion in CMD for better security
- Exposes port explicitly for documentation purposes
- Uses slim base image to minimize container size

## Common Issues and Solutions

### Port Conflicts
If the default port (8080) is in use:
1. Set a different port using the `PORT` environment variable
2. Update the port mapping in your `docker run` command

### Missing Dependencies
If new dependencies are added:
1. Update `pyproject.toml`
2. Rebuild the container