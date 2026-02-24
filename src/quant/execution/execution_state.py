from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def execution_state_path() -> Path:
    return Path(os.getenv("DASHBOARD_LEVELS_JSON", "data/live/execution_state.json"))


def write_execution_state(state: Dict[str, Any]) -> Path:
    p = execution_state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload.setdefault("updated_at", pd.Timestamp.now("UTC").isoformat())
    p.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str), encoding="utf-8")
    return p
