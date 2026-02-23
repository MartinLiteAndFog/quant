# Build und Laufzeit in einer Umgebung – behebt "No module named uvicorn" bei Railway
FROM python:3.12-slim

WORKDIR /app

# Abhängigkeiten + Paket installieren (dasselbe Python wie zur Laufzeit)
COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir .

# PORT setzt Railway beim Start
EXPOSE 8080
CMD ["/bin/sh", "-c", "exec python -m uvicorn quant.execution.webhook_server:app --host 0.0.0.0 --port ${PORT:-8080}"]
