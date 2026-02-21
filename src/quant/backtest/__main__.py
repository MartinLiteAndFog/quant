from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..execution.oms import normalize_payload

app = FastAPI(title="quant-webhook", version="0.1.0")


@app.get("/health")
def health():
    return {"ok": True, "ts": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")}


def _signals_day_path(symbol: str, ts_utc: pd.Timestamp) -> Path:
    safe_symbol = symbol.replace("/", "-").replace(":", "-")
    day = ts_utc.strftime("%Y%m%d")
    base = Path("data/signals") / safe_symbol
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{day}.jsonl"


@app.post("/webhook/tradingview")
async def webhook_tradingview(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"ok": False, "error": "invalid_json"})

    try:
        sig = normalize_payload(payload)

        out = {
            "server_ts": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "ts": sig.ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbol": sig.symbol,
            "action": sig.action,
            "signal": int(sig.signal),
        }

        path = _signals_day_path(sig.symbol, sig.ts)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

        return {"ok": True, "saved_to": str(path), "symbol": sig.symbol}

    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})
