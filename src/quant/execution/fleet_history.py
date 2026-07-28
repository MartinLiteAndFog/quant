"""Shared Fleet history boundary for every bot and read model."""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd

DEFAULT_FLEET_HISTORY_START = "2026-07-22"


def fleet_history_start() -> Optional[pd.Timestamp]:
    """Return the global inclusive Fleet history floor in UTC.

    Invalid values fail closed to the audited default rather than silently
    exposing older experimental history.  Explicit ``off``-style values are
    still supported when an all-history view is intentionally requested.
    """
    raw = str(
        os.getenv("FLEET_HISTORY_START") or DEFAULT_FLEET_HISTORY_START
    ).strip()
    if raw.lower() in {"", "0", "off", "none", "all"}:
        return None
    try:
        parsed = pd.Timestamp(raw)
    except Exception:
        parsed = pd.Timestamp(DEFAULT_FLEET_HISTORY_START)
    return (
        parsed.tz_localize("UTC")
        if parsed.tzinfo is None
        else parsed.tz_convert("UTC")
    )
