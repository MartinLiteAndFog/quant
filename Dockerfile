# Stage 1 — Build React frontend
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit
COPY frontend/ ./
RUN npm run build

# Stage 2 — Python runtime
FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y curl jq && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y procps && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src ./src
COPY scripts ./scripts
COPY data/regimes/pc_3axis_gate_latest.csv ./data/regimes/pc_3axis_gate_latest.csv
RUN pip install --no-cache-dir .

COPY --from=frontend /app/frontend/dist /app/frontend/dist

EXPOSE 8080
CMD ["python", "-c", "import os, uvicorn; port = int(os.environ.get('PORT', '8080')); print('Starting on 0.0.0.0:' + str(port)); uvicorn.run('quant.execution.webhook_server:app', host='0.0.0.0', port=port)"]