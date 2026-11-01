"""Shadow-only Blaupunkt family API with NO trading executor.

Useful for a separate Railway development service: does not import the live
bot_webhook module, initialize broker credentials or run any signal workers.

Run: python -u -m quant.execution.family_shadow_app
Requires DATABASE_URL, FAMILY_ROUTER_ENABLED=1, FAMILY_ADMIN_TOKEN=...
"""
from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI

from quant.execution.family_ingress import router as family_ingress_router
from quant.execution.family_control_plane import router as family_control_router
from quant.execution.family_bootstrap_pilot import router as family_roster_router

app = FastAPI(title="blaupunkt-family-shadow", version="0.1.0")
app.include_router(family_ingress_router)
app.include_router(family_control_router)
app.include_router(family_roster_router)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "mode": "shadow", "orders_enabled": False}


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))


if __name__ == "__main__":
    main()
