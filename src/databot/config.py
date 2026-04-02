from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List


def _parse_symbol_config() -> Dict[str, Any]:
    raw = os.getenv("DATABOT_SYMBOL_CONFIG", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


@dataclass
class DatabotConfig:
    """All DATABOT configuration from environment variables."""

    symbols: List[str] = field(default_factory=lambda: _parse_symbols())
    symbol_config: Dict[str, Any] = field(default_factory=lambda: _parse_symbol_config())
    renko_box: float = field(default_factory=lambda: float(os.getenv("DATABOT_RENKO_BOX", os.getenv("DASHBOARD_RENKO_BOX", "0.1"))))
    renko_days_back: int = field(default_factory=lambda: int(os.getenv("DATABOT_RENKO_DAYS_BACK", os.getenv("DASHBOARD_RENKO_DAYS_BACK", "14"))))
    renko_step_hours: int = field(default_factory=lambda: int(os.getenv("DATABOT_RENKO_STEP_HOURS", os.getenv("DASHBOARD_RENKO_STEP_HOURS", "6"))))
    poll_sec: float = field(default_factory=lambda: float(os.getenv("DATABOT_POLL_SEC", os.getenv("DASHBOARD_RENKO_POLL_SEC", "60"))))
    redis_bars: int = field(default_factory=lambda: int(os.getenv("DATABOT_REDIS_BARS", os.getenv("RENKO_REDIS_BARS", "500"))))
    retention_days: int = field(default_factory=lambda: int(os.getenv("DATABOT_RETENTION_DAYS", os.getenv("LIVE_RENKO_RETENTION_DAYS", "30"))))
    health_port: int = field(default_factory=lambda: int(os.getenv("PORT", os.getenv("DATABOT_HEALTH_PORT", "8080"))))
    log_throttle_sec: float = field(default_factory=lambda: float(os.getenv("DATABOT_LOG_THROTTLE_SEC", os.getenv("DASHBOARD_LOG_THROTTLE_SEC", "60"))))

    def box_for_symbol(self, symbol: str) -> float:
        key = str(symbol).strip()
        entry = self.symbol_config.get(key)
        if entry is None:
            entry = self.symbol_config.get(key.upper())
        if isinstance(entry, dict):
            box = entry.get("box")
            if box is not None:
                return float(box)
        return float(self.renko_box)


def _parse_symbols() -> List[str]:
    # If DATABOT_SYMBOLS is set but empty on Railway, getenv returns "" — treat as unset.
    raw = os.getenv("DATABOT_SYMBOLS", "").strip()
    if not raw:
        raw = os.getenv("DASHBOARD_SYMBOL", "SOL-USDT").strip()
    if not raw:
        return ["SOL-USDT"]
    return [s.strip() for s in raw.split(",") if s.strip()]
