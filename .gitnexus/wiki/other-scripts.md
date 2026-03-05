# Other — scripts

# Railway Deployment Scripts

This module contains deployment configuration scripts specifically for the Railway platform deployment environment.

## Overview

The main component is `railway_entrypoint.sh`, which serves as the entry point script for deploying the webhook server application on Railway's hosting platform.

## Entry Point Script

### Purpose
The `railway_entrypoint.sh` script handles the initialization and startup of the webhook server with the correct port configuration for Railway's environment.

### Key Features

- **Dynamic Port Configuration**: 
  - Uses Railway's `PORT` environment variable
  - Falls back to port 8080 if `PORT` is not set
  - Ensures the application binds to the correct port assigned by Railway

- **Server Configuration**:
  - Binds to `0.0.0.0` to accept external connections
  - Launches the webhook server using Uvicorn
  - Runs the FastAPI application defined in `quant.execution.webhook_server`

### Usage

The script is automatically executed by Railway's deployment process. It doesn't require manual intervention, but understanding its operation is important for debugging deployment issues.

```bash
# Example of how Railway executes the script
./railway_entrypoint.sh
```

## Integration Points

- **Webhook Server**: The script launches the FastAPI application defined in `quant.execution.webhook_server`
- **Railway Platform**: Integrates with Railway's environment variables and deployment workflow
- **Uvicorn**: Uses Uvicorn as the ASGI server to run the FastAPI application

## Deployment Considerations

- Ensure the script has executable permissions (`chmod +x railway_entrypoint.sh`)
- The script must be referenced in Railway's configuration
- The `quant.execution.webhook_server` module must be available in the deployed environment

## Error Handling

The script includes basic error handling through:
- Port fallback mechanism
- Logging of startup information
- Use of `exec` to properly handle process signals