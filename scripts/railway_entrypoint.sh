#!/bin/sh
# Railway setzt PORT; Fallback 8080. So hört die App garantiert auf dem richtigen Port.
export PORT="${PORT:-8080}"
echo "Starting webhook server on 0.0.0.0:${PORT}"
exec python -m uvicorn quant.execution.webhook_server:app --host 0.0.0.0 --port "$PORT"
