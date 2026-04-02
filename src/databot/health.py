"""Minimal health/status API for Railway monitoring."""
from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import FastAPI

from databot.config import DatabotConfig
from databot.renko_pipeline import get_last_refresh

app = FastAPI(title="databot-health", version="0.1.0")

_start_time = time.time()


@app.get("/health")
def health() -> Dict[str, Any]:
    cfg = DatabotConfig()
    symbols = cfg.symbols

    pipelines: Dict[str, Any] = {}
    all_ok = True
    for sym in symbols:
        last = get_last_refresh(sym)
        if last is None:
            pipelines[sym] = {"status": "pending", "last_refresh": None}
            all_ok = False
        elif last.get("ok"):
            pipelines[sym] = {
                "status": "ok",
                "bricks": last.get("bricks"),
                "last_close": last.get("last_close"),
                "ts": last.get("ts"),
                "redis_ok": (last.get("redis") or {}).get("ok", False),
                "postgres_ok": (last.get("postgres") or {}).get("ok", False),
            }
        else:
            pipelines[sym] = {
                "status": "error",
                "reason": last.get("reason"),
                "ts": last.get("ts"),
            }
            all_ok = False

    return {
        "service": "databot",
        "status": "ok" if all_ok else "degraded",
        "uptime_sec": int(time.time() - _start_time),
        "pipelines": pipelines,
    }


@app.get("/")
def root() -> Dict[str, str]:
    return {"service": "databot", "status": "running"}
