"""Read-only KuCoin Futures diagnostics endpoints for fleet bots.

Why: the fleet only ever saw KuCoin *equity* (via /health -> account-overview).
Positions, live fills and order history were invisible, so "did the order fill?"
questions could not be answered without direct exchange access. This router
exposes that data read-only, signed with the SAME per-bot keys already deposited
in each bot's Railway env (KUCOIN_FUTURES_API_KEY / _API_SECRET / _PASSPHRASE).

Mount it in bot_webhook (and webhook_server for the KuCoin main account):

    from quant.execution.kucoin_diag import router as kucoin_diag_router
    app.include_router(kucoin_diag_router)

Then, per bot host:
    GET /kucoin/diag?token=<BOT_WEBHOOK_TOKEN>       # account + position(s) + recent fills
    GET /kucoin/account?token=...                    # account-overview
    GET /kucoin/positions?token=...                  # all open positions
    GET /kucoin/fills?token=...&symbol=SOL-USDT      # recent fills (trade history)
    GET /kucoin/orders?token=...&status=done         # recent orders

Guarded by BOT_WEBHOOK_TOKEN (same secret as /webhook). Read-only: it never
places or cancels anything, so it is safe to iterate against a live account —
worst case a wrong signature yields a KuCoin 401, never an order.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/kucoin", tags=["kucoin-diag"])

_BASE = (os.getenv("KUCOIN_FUTURES_BASE_URL") or "https://api-futures.kucoin.com").rstrip("/")


def _creds() -> tuple[str, str, str]:
    key = (os.getenv("KUCOIN_FUTURES_API_KEY") or "").strip()
    secret = (os.getenv("KUCOIN_FUTURES_API_SECRET") or "").strip()
    passphrase = (
        os.getenv("KUCOIN_FUTURES_PASSPHRASE")
        or os.getenv("KUCOIN_FUTURES_API_PASSPHRASE")
        or ""
    ).strip()
    return key, secret, passphrase


def _require_token(token: Optional[str]) -> None:
    """Reuse the bot's webhook token so these endpoints aren't public."""
    expected = (
        os.getenv("BOT_WEBHOOK_TOKEN")
        or os.getenv("TV_WEBHOOK_TOKEN")
        or os.getenv("WEBHOOK_TOKEN")
        or ""
    ).strip()
    if not expected:
        return  # no token configured on this host -> leave open like /health
    if (token or "").strip() != expected:
        raise HTTPException(status_code=401, detail="invalid diag token")


def _symbol_to_contract(symbol: str) -> str:
    """SOL-USDT -> SOLUSDTM (KuCoin perpetual contract code)."""
    s = (symbol or "SOL-USDT").upper().replace("-", "").replace("_", "").replace("/", "")
    return s if s.endswith("M") else s + "M"


def _signed_get(endpoint: str, *, timeout: float = 10.0) -> Any:
    """GET a KuCoin Futures private endpoint (v2 key auth). Read-only."""
    key, secret, passphrase = _creds()
    if not key or not secret or not passphrase:
        raise HTTPException(status_code=503, detail="kucoin_credentials_missing")

    ts = str(int(time.time() * 1000))
    method = "GET"
    str_to_sign = ts + method + endpoint  # endpoint must include any query string
    sign = base64.b64encode(
        hmac.new(secret.encode(), str_to_sign.encode(), hashlib.sha256).digest()
    ).decode()
    pp = base64.b64encode(
        hmac.new(secret.encode(), passphrase.encode(), hashlib.sha256).digest()
    ).decode()
    headers = {
        "KC-API-KEY": key,
        "KC-API-SIGN": sign,
        "KC-API-TIMESTAMP": ts,
        "KC-API-PASSPHRASE": pp,
        "KC-API-KEY-VERSION": "2",
        "Content-Type": "application/json",
    }
    req = Request(_BASE + endpoint, method=method, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:300] if hasattr(e, "read") else str(e.reason)
        raise HTTPException(status_code=502, detail=f"kucoin_http_{e.code}: {detail}")
    except URLError as e:
        raise HTTPException(status_code=502, detail=f"kucoin_unreachable: {e.reason}")
    if isinstance(body, dict) and str(body.get("code")) not in ("200000", "None", ""):
        # KuCoin wraps success as code 200000; anything else is an API error.
        if body.get("code") and str(body.get("code")) != "200000":
            raise HTTPException(status_code=502, detail=f"kucoin_api_error: {body}")
    return body.get("data") if isinstance(body, dict) and "data" in body else body


@router.get("/account")
def kucoin_account(token: Optional[str] = None, currency: str = "USDT") -> Dict[str, Any]:
    _require_token(token)
    return {"ok": True, "account": _signed_get(f"/api/v1/account-overview?currency={currency}")}


@router.get("/positions")
def kucoin_positions(token: Optional[str] = None) -> Dict[str, Any]:
    _require_token(token)
    data = _signed_get("/api/v1/positions")
    # Trim to the fields that actually matter for "is there a position?".
    slim = []
    for p in data or []:
        if not isinstance(p, dict):
            continue
        qty = p.get("currentQty")
        if not qty:
            continue  # skip flat contracts
        slim.append(
            {
                "symbol": p.get("symbol"),
                "side": "long" if float(qty) > 0 else "short",
                "qty": qty,
                "avg_entry": p.get("avgEntryPrice"),
                "mark_price": p.get("markPrice"),
                "liq_price": p.get("liquidationPrice"),
                "unrealised_pnl": p.get("unrealisedPnl"),
                "realised_pnl": p.get("realisedPnl"),
                "leverage": p.get("realLeverage"),
                "margin": p.get("posMargin"),
            }
        )
    return {"ok": True, "open_positions": slim, "raw_count": len(data or [])}


@router.get("/fills")
def kucoin_fills(token: Optional[str] = None, symbol: Optional[str] = None) -> Dict[str, Any]:
    _require_token(token)
    ep = "/api/v1/recentFills"
    if symbol:
        ep = f"/api/v1/fills?symbol={_symbol_to_contract(symbol)}"
    data = _signed_get(ep)
    items = data.get("items") if isinstance(data, dict) else data
    return {"ok": True, "fills": items}


@router.get("/orders")
def kucoin_orders(
    token: Optional[str] = None, status: str = "done", symbol: Optional[str] = None
) -> Dict[str, Any]:
    _require_token(token)
    q = f"/api/v1/orders?status={status}"
    if symbol:
        q += f"&symbol={_symbol_to_contract(symbol)}"
    data = _signed_get(q)
    items = data.get("items") if isinstance(data, dict) else data
    return {"ok": True, "orders": items}


@router.get("/diag")
def kucoin_diag(token: Optional[str] = None, symbol: str = "SOL-USDT") -> Dict[str, Any]:
    """One-shot: account + open positions + recent fills — the "what's really
    happening on KuCoin right now" view."""
    _require_token(token)
    out: Dict[str, Any] = {"ok": True, "ts": int(time.time())}
    try:
        out["account"] = _signed_get("/api/v1/account-overview?currency=USDT")
    except HTTPException as e:
        out["account_error"] = e.detail
    try:
        out["positions"] = kucoin_positions(token=token)["open_positions"]
    except HTTPException as e:
        out["positions_error"] = e.detail
    try:
        out["recent_fills"] = kucoin_fills(token=token, symbol=symbol)["fills"]
    except HTTPException as e:
        out["fills_error"] = e.detail
    return out
