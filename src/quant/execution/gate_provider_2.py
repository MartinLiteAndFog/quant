from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict

from quant.utils.log import get_logger

log = get_logger("quant.gate_provider_2")


def _fetch_json(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "quant-gate-provider-2/1"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_gate(url: str, symbol: str = "SOL-USDT") -> Dict[str, Any]:
    """
    Kraken gate read path copied from the old kraken bot:
    1) RegimeStore latest state
    2) HTTP fallback
    3) default gate_off on failure
    """
    try:
        from quant.regime.store import RegimeStore

        store = RegimeStore()
        latest = store.get_latest_state(symbol)
        if latest and "gate_on" in latest:
            return {
                "gate_on": int(latest["gate_on"]),
                "ts": str(latest.get("ts", "")),
                "source": "store",
            }
    except Exception as e:
        log.warning("gate store fetch failed, falling back to HTTP: %s", e)

    try:
        obj = _fetch_json(url)
        return {
            "gate_on": int(obj.get("gate_on", 0) or 0),
            "ts": str(obj.get("ts", "")),
            "source": "http",
        }
    except Exception as e:
        log.warning("gate fetch failed: %s", e)
        return {"gate_on": 0, "ts": "", "source": "error", "error": str(e)}
