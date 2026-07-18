# Stage 1 — Build React frontend
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit
COPY frontend/ ./
RUN npm run build

# Stage 1b — Build Svelte dashboard
FROM node:20-slim AS dashboard
WORKDIR /app/dashboard
COPY dashboard/package.json dashboard/package-lock.json* ./
RUN npm ci --no-audit
COPY dashboard/ ./
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
COPY --from=dashboard /app/dashboard/dist /app/dashboard/dist

ENV DASHBOARD2_DIST=/app/dashboard/dist

EXPOSE 8080
CMD ["python", "-m", "quant.execution.railway_entrypoint"]
