from __future__ import annotations

import os


def fleet_api_only() -> bool:
    """Return whether this deployment should run only Fleet API/data services."""

    return str(os.getenv("FLEET_API_ONLY", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
