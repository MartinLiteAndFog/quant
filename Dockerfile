# Build und Laufzeit in einer Umgebung – behebt "No module named uvicorn" bei Railway
FROM python:3.12-slim

WORKDIR /app

# Abhängigkeiten + Paket installieren (dasselbe Python wie zur Laufzeit)
COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir .

# PORT aus der Umgebung lesen (Python), keine Shell-Expansion – Railway setzt PORT
EXPOSE 8080
CMD ["python", "-c", "import os, uvicorn; port = int(os.environ.get('PORT', '8080')); print('Starting on 0.0.0.0:' + str(port)); uvicorn.run('quant.execution.webhook_server:app', host='0.0.0.0', port=port)"]
