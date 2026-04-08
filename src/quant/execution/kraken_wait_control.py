from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from quant.utils.log import get_logger

log = get_logger("quant.kraken_wait_control")


def _canon_symbol(symbol: str) -> str:
    return "".join(ch for ch in str(symbol or "").upper() if ch.isalnum())


def wait_mode_pin_key(symbol: str) -> str:
    return f"kraken:wait_mode_pin:{_canon_symbol(symbol)}"


def _normalize_side(side: Optional[str]) -> Optional[str]:
    s = str(side or "").strip().lower()
    if not s:
        return None
    if s in ("buy", "long"):
        return "long"
    if s in ("sell", "short"):
        return "short"
    raise ValueError(f"invalid side: {side!r}")


def _maybe_redis_client():
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        return None
    import redis as redis_lib

    return redis_lib.from_url(redis_url, decode_responses=True)


def _require_redis_client():
    client = _maybe_redis_client()
    if client is None:
        raise RuntimeError("REDIS_URL is not set")
    return client


def get_wait_mode_pin(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        client = _maybe_redis_client()
        if client is None:
            return None
        raw = client.get(wait_mode_pin_key(symbol))
        if not raw:
            return None
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except Exception as e:
        log.warning("kraken wait pin read failed symbol=%s err=%s", symbol, e)
        return None


def set_wait_mode_pin(
    symbol: str,
    *,
    side: Optional[str] = None,
    reason: Optional[str] = None,
    actor: str = "api",
    ttl_sec: Optional[int] = None,
) -> Dict[str, Any]:
    client = _require_redis_client()
    ttl = int(ttl_sec) if ttl_sec is not None else None
    if ttl is not None and ttl <= 0:
        ttl = None

    payload: Dict[str, Any] = {
        "enabled": True,
        "symbol": str(symbol),
        "side": _normalize_side(side),
        "reason": str(reason or "manual_wait_mode"),
        "actor": str(actor or "api"),
        "ts": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "ttl_sec": ttl,
        "key": wait_mode_pin_key(symbol),
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if ttl is not None:
        client.set(payload["key"], raw, ex=ttl)
    else:
        client.set(payload["key"], raw)
    return payload


def clear_wait_mode_pin(symbol: str) -> Dict[str, Any]:
    client = _require_redis_client()
    key = wait_mode_pin_key(symbol)
    deleted = int(client.delete(key) or 0)
    return {
        "symbol": str(symbol),
        "key": key,
        "cleared": bool(deleted),
        "deleted": deleted,
    }


def reconcile_wait_mode_pin(symbol: str, current_side: Optional[str]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    payload = get_wait_mode_pin(symbol)
    if not isinstance(payload, dict):
        return False, None

    live_side = _normalize_side(current_side)
    pin_side = _normalize_side(payload.get("side"))
    if live_side is None:
        try:
            clear_wait_mode_pin(symbol)
        except Exception as e:
            log.warning("kraken wait pin clear failed symbol=%s err=%s", symbol, e)
        return False, payload

    if pin_side is not None and pin_side != live_side:
        try:
            clear_wait_mode_pin(symbol)
        except Exception as e:
            log.warning("kraken wait pin clear mismatch failed symbol=%s err=%s", symbol, e)
        return False, payload

    return True, payload
