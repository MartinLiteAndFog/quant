# Build und Laufzeit in einer Umgebung – behebt "No module named uvicorn" bei Railway
FROM python:3.12-slim

WORKDIR /app

# Abhängigkeiten + Paket installieren (dasselbe Python wie zur Laufzeit)
COPY pyproject.toml .
COPY src ./src
COPY data/runs/visual_v02_seed/transitions/voxel_map.parquet ./data/runs/visual_v02_seed/transitions/voxel_map.parquet
COPY data/runs/visual_v02_seed/transitions/voxel_stats.parquet ./data/runs/visual_v02_seed/transitions/voxel_stats.parquet
COPY data/runs/visual_v02_seed/transitions/transitions_topk.parquet ./data/runs/visual_v02_seed/transitions/transitions_topk.parquet
COPY data/runs/visual_v02_seed/transitions/basins_v02_components.parquet ./data/runs/visual_v02_seed/transitions/basins_v02_components.parquet
RUN pip install --no-cache-dir .

# PORT aus der Umgebung lesen (Python), keine Shell-Expansion – Railway setzt PORT
EXPOSE 8080
CMD ["python", "-c", "import os, uvicorn; port = int(os.environ.get('PORT', '8080')); print('Starting on 0.0.0.0:' + str(port)); uvicorn.run('quant.execution.webhook_server:app', host='0.0.0.0', port=port)"]
