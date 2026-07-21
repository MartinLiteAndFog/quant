"""Live venue account snapshots for health / fleet capitalization.

Each Railway bot owns its own API credentials. Fleet aggregation must therefore
pull equity from each bot's `/health` (or a dedicated equity endpoint), not from
the dashboard service's single KuCoin key.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional

from quant.utils.log import get_logger

log = get_logger("quant.live_account")

_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {"ts": 0.0, "payload": None}


def _truthy(v: Optional[str]) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "on"}


def _cache_ttl_sec() -> float:
    try:
        return max(5.0, float(os.getenv("LIVE_ACCOUNT_CACHE_SEC", "15")))
    except Exception:
        return 15.0


def trading_mode_from_env() -> Dict[str, Any]:
    """Env-derived trading flags shared by webhook_server + bot_webhook."""
    live = _truthy(os.getenv("LIVE_TRADING_ENABLED"))
    dry = (
        _truthy(os.getenv("TV_EXEC_DRY_RUN"))
        or _truthy(os.getenv("LIVE_EXECUTOR_DRY_RUN"))
        or _truthy(os.getenv("KRAKEN_DRY_RUN"))
    )
    return {
        "live_trading_enabled": live,
        "dry_run": dry and not live,
    }


def _maybe_persist_kucoin_snapshot(
    *,
    equity: float,
    currency: str,
    account: str,
    payload: Dict[str, Any],
) -> None:
    if equity <= 0 or not account:
        return
    if not _truthy(os.getenv("FLEET_PERSIST_LIVE_EQUITY", "1")):
        return
    try:
        import pandas as pd
        from quant.execution.event_store import insert_equity_snapshot

        insert_equity_snapshot(
            {
                "ts": pd.Timestamp.now("UTC"),
                "venue": "kucoin",
                "account": account,
                "symbol": None,
                "equity": float(equity),
                "currency": currency,
                "source": "live_account.bot_health",
                "payload_json": payload,
            }
        )
    except Exception as e:
        log.debug("equity snapshot persist skipped: %s", e)


def fetch_kucoin_account(*, persist_account: Optional[str] = None) -> Dict[str, Any]:
    key = (os.getenv("KUCOIN_FUTURES_API_KEY") or "").strip()
    if not key:
        return {"ok": False, "error": "kucoin_credentials_missing", "source": "kucoin_live"}
    try:
        from quant.execution.kucoin_futures import KucoinFuturesBroker

        currency = (os.getenv("LIVE_EQUITY_CCY") or os.getenv("DASHBOARD_EQUITY_CCY") or "USDT").strip()
        bal = KucoinFuturesBroker().get_account_balance(currency=currency)
        equity = float(bal.get("equity", 0) or 0)
        available = float(bal.get("available", 0) or 0)
        margin = float(bal.get("margin", 0) or 0)
        upnl = float(bal.get("unrealised_pnl", 0) or 0)
        out = {
            "ok": True,
            "source": "kucoin_live",
            "currency": currency,
            "equity": equity,
            "available": available,
            "margin": margin,
            "unrealised_pnl": upnl,
            "ts": time.time(),
        }
        if persist_account:
            _maybe_persist_kucoin_snapshot(
                equity=equity,
                currency=currency,
                account=persist_account,
                payload=out,
            )
        return out
    except Exception as e:
        log.warning("kucoin live account fetch failed: %s", e)
        return {"ok": False, "error": str(e), "source": "kucoin_live"}


def _maybe_persist_kraken_snapshot(*, equity: float, payload: Dict[str, Any]) -> None:
    if equity <= 0:
        return
    if not _truthy(os.getenv("FLEET_PERSIST_LIVE_EQUITY", "1")):
        return
    try:
        import pandas as pd
        from quant.execution.event_store import insert_equity_snapshot

        insert_equity_snapshot(
            {
                "ts": pd.Timestamp.now("UTC"),
                "venue": "kraken",
                "account": "main",
                "symbol": None,
                "equity": float(equity),
                "currency": "USD",
                "source": "live_account.kraken_health",
                "payload_json": payload,
            }
        )
    except Exception as e:
        log.debug("kraken equity snapshot persist skipped: %s", e)


def fetch_kraken_account(*, persist: bool = True) -> Dict[str, Any]:
    key = (os.getenv("KRAKEN_FUTURES_KEY") or "").strip()
    if not key:
        return {"ok": False, "error": "kraken_credentials_missing", "source": "kraken_live"}
    try:
        from quant.execution.kraken_futures import KrakenFuturesClient

        eq = KrakenFuturesClient().get_account_equity()
        out = {
            "ok": True,
            "source": "kraken_live",
            "currency": "USD",
            "equity": float(eq.get("equity_usd", 0) or 0),
            "available": float(eq.get("available_usd", 0) or 0),
            "wallet": float(eq.get("wallet_usd", 0) or 0),
            "unrealised_pnl": float(eq.get("upnl_usd", 0) or 0),
            "ts": time.time(),
        }
        if persist:
            _maybe_persist_kraken_snapshot(equity=float(out["equity"]), payload=out)
        return out
    except Exception as e:
        log.warning("kraken live account fetch failed: %s", e)
        return {"ok": False, "error": str(e), "source": "kraken_live"}


def live_account_snapshot(
    *,
    prefer: Optional[str] = None,
    persist_account: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Cached live account overview for the credentials on this process."""
    now = time.time()
    with _LOCK:
        if use_cache and _CACHE["payload"] is not None and (now - float(_CACHE["ts"])) < _cache_ttl_sec():
            return dict(_CACHE["payload"])

    venue = (prefer or "").strip().lower()
    if not venue:
        if (os.getenv("KRAKEN_FUTURES_KEY") or "").strip():
            venue = "kraken"
        elif (os.getenv("KUCOIN_FUTURES_API_KEY") or "").strip():
            venue = "kucoin"
        else:
            return {"ok": False, "error": "no_venue_credentials", "source": "none"}

    if venue == "kraken":
        payload = fetch_kraken_account(persist=True)
    else:
        payload = fetch_kucoin_account(persist_account=persist_account)

    with _LOCK:
        _CACHE["ts"] = now
        _CACHE["payload"] = dict(payload)
    return payload