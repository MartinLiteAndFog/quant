"""Future same-container Blaupunkt API entrypoint (NOT enabled in production).

Loads exactly one existing Blaupunkt bot_webhook app; adds disabled-by-default
family registry and shadow-only signal inbox endpoints alongside the unchanged
legacy /webhook/tv-execute. No new orders can be placed by the family router.

For an eventual planned switch of the existing Railway start command from
  python -u -m quant.execution.bot_webhook
to
  python -u -m quant.execution.blaupunkt_family_app
only after the new endpoints pass staging tests, with no duplicate service
using the same KuCoin credentials.
"""
from __future__ import annotations

import os

import uvicorn

from quant.execution.bot_webhook import app as app
from quant.execution.family_ingress import router as family_ingress_router
from quant.execution.family_control_plane import router as family_control_router
from quant.execution.family_bootstrap_pilot import router as family_roster_router

app.include_router(family_ingress_router)
app.include_router(family_control_router)
app.include_router(family_roster_router)


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))


if __name__ == "__main__":
    main()
