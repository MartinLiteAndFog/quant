# src/quant/execution/kucoin_futures.py
"""
KuCoin Futures (perpetuals) REST client implementing BrokerAPI for live execution.

Uses env: KUCOIN_FUTURES_API_KEY, KUCOIN_FUTURES_API_SECRET, KUCOIN_FUTURES_PASSPHRASE.
Base URL: https://api-futures.kucoin.com (classic futures).
Symbol: pass "SOL-USDT" or "SOLUSDT"; we map to KuCoin contract symbol (e.g. SOLUSDTM).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from quant.execution.oms import BrokerAPI
from quant.utils.log import get_logger

log = get_logger("quant.kucoin_futures")

BASE_URL = "https://api-futures.kucoin.com"
KC_API_KEY_VERSION = "2"


def _sanitize_client_oid(v: str) -> str:
    """
    KuCoin allows only [A-Za-z0-9_-] in clientOid.
    """
    s = str(v or "").strip()
    s = re.sub(r"[^A-Za-z0-9_-]", "-", s)
    if not s:
        s = f"oid-{int(time.time() * 1000)}"
    return s[:40]


def _symbol_to_contract(symbol: str) -> str:
    """SOL-USDT / SOLUSDT -> SOLUSDTM (KuCoin perpetual contract)."""
    s = symbol.strip().upper().replace("-", "").replace("_", "")
    if not s.endswith("USDT"):
        s = s + "USDT"
    return s + "M"  # M = perpetual


def _sign(secret: str, timestamp: str, method: str, path: str, body: str = "") -> str:
    to_sign = timestamp + method.upper() + path + body
    sig = hmac.new(
        secret.encode("utf-8"),
        to_sign.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return base64.b64encode(sig).decode("ascii")


def _passphrase_encoded(secret: str, passphrase: str) -> str:
    return base64.b64encode(
        hmac.new(
            secret.encode("utf-8"),
            passphrase.encode("utf-8"),
            hashlib.sha256,
        ).digest()
    ).decode("ascii")


def _request(
    method: str,
    path: str,
    *,
    body: Optional[Dict[str, Any]] = None,
    api_key: str = "",
    api_secret: str = "",
    passphrase: str = "",
) -> Dict[str, Any]:
    url = BASE_URL + path
    body_str = json.dumps(body, separators=(",", ":")) if body else ""
    ts = str(int(time.time() * 1000))
    sign = _sign(api_secret, ts, method, path, body_str)
    pp_enc = _passphrase_encoded(api_secret, passphrase) if KC_API_KEY_VERSION == "2" else passphrase

    req = Request(
        url,
        data=body_str.encode("utf-8") if body_str else None,
        method=method,
        headers={
            "Content-Type": "application/json",
            "KC-API-KEY": api_key,
            "KC-API-SIGN": sign,
            "KC-API-TIMESTAMP": ts,
            "KC-API-PASSPHRASE": pp_enc,
            "KC-API-KEY-VERSION": KC_API_KEY_VERSION,
        },
    )
    try:
        with urlopen(req, timeout=30) as resp:
            out = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        try:
            out = json.loads(e.read().decode("utf-8"))
        except Exception:
            out = {"code": str(e.code), "msg": str(e.reason)}
        log.warning("kucoin_futures request failed path=%s code=%s body=%s", path, e.code, out)
        raise
    except URLError as e:
        log.warning("kucoin_futures request error path=%s err=%s", path, e.reason)
        raise
    if not out.get("code", "").startswith("2"):
        raise RuntimeError(f"KuCoin API error: {out.get('msg', out)}")
    return out.get("data", out)


class KucoinFuturesBroker(BrokerAPI):
    """
    KuCoin Futures REST client implementing BrokerAPI.
    Credentials from env: KUCOIN_FUTURES_API_KEY, KUCOIN_FUTURES_API_SECRET, KUCOIN_FUTURES_PASSPHRASE.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        passphrase: Optional[str] = None,
    ):
        self._key = (api_key or os.getenv("KUCOIN_FUTURES_API_KEY", "")).strip()
        self._secret = (api_secret or os.getenv("KUCOIN_FUTURES_API_SECRET", "")).strip()
        self._pass = (passphrase or os.getenv("KUCOIN_FUTURES_PASSPHRASE", "")).strip()
        self._order_leverage = float(os.getenv("KUCOIN_FUTURES_ORDER_LEVERAGE", os.getenv("LIVE_EXECUTOR_LEVERAGE", "1")))
        self._margin_mode = (os.getenv("KUCOIN_FUTURES_MARGIN_MODE", "") or "").strip().lower()
        if not self._key or not self._secret or not self._pass:
            log.warning("KuCoin Futures credentials missing; set KUCOIN_FUTURES_* env vars")

    def _req(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return _request(
            method, path,
            body=body,
            api_key=self._key,
            api_secret=self._secret,
            passphrase=self._pass,
        )

    def get_best_bid_ask(self, symbol: str) -> Tuple[float, float]:
        """Best bid, best ask from ticker."""
        contract = _symbol_to_contract(symbol)
        path = f"/api/v1/ticker?symbol={contract}"
        data = self._req("GET", path)
        # Response: bestBid, bestAsk or similar; check KuCoin docs for exact keys
        bid = float(data.get("bestBidPrice", data.get("bestBid", 0)) or 0)
        ask = float(data.get("bestAskPrice", data.get("bestAsk", 0)) or 0)
        return (bid, ask)

    def get_1m_range_pct_proxy(self, symbol: str) -> Optional[float]:
        """(high - low) / close for latest 1m candle."""
        contract = _symbol_to_contract(symbol)
        now_ms = int(time.time() * 1000)
        from_ms = now_ms - (5 * 60 * 1000)
        path = f"/api/v1/kline/query?symbol={contract}&granularity=1&from={from_ms}&to={now_ms}"
        try:
            data = self._req("GET", path)
        except Exception:
            return None
        # data may be list of [time, open, high, low, close, ...]
        if not data or not isinstance(data, list):
            return None
        arr = data[0] if data else []
        if len(arr) < 5:
            return None
        try:
            high, low, close = float(arr[2]), float(arr[3]), float(arr[4])
        except (IndexError, TypeError, ValueError):
            return None
        if not close or close <= 0:
            return None
        return (high - low) / close

    def get_position(self, symbol: str) -> float:
        """Signed position size (contracts): >0 long, <0 short."""
        contract = _symbol_to_contract(symbol)
        # KuCoin Futures expects symbol for single-position lookup.
        path = f"/api/v1/position?symbol={contract}"
        data = self._req("GET", path)

        if not data:
            return 0.0

        # Typical response is a single object for the requested symbol.
        if isinstance(data, dict):
            size = float(data.get("currentQty", data.get("size", 0)) or 0)
            side = (data.get("side") or "").lower()
            if side == "short":
                return -abs(size)
            if side == "long":
                return abs(size)
            # Fallback: if side is missing, keep sign from size if present.
            return float(size)

        # Defensive fallback if API shape changes to list.
        if isinstance(data, list):
            for p in data:
                if p.get("symbol") == contract:
                    size = float(p.get("currentQty", p.get("size", 0)) or 0)
                    side = (p.get("side") or "").lower()
                    if side == "short":
                        return -abs(size)
                    if side == "long":
                        return abs(size)
                    return float(size)

        return 0.0

    def cancel_all(self, symbol: str) -> None:
        contract = _symbol_to_contract(symbol)
        path = f"/api/v1/orders/cancelAll?symbol={contract}"
        try:
            self._req("DELETE", path)
        except Exception as e:
            log.warning("cancel_all %s failed: %s", contract, e)

    def place_limit(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        post_only: bool,
        reduce_only: bool,
        client_id: str,
    ) -> str:
        contract = _symbol_to_contract(symbol)
        body = {
            "clientOid": _sanitize_client_oid(client_id),
            "symbol": contract,
            "side": side.lower(),
            "type": "limit",
            "price": str(round(price, 8)),
            "size": int(qty),
            "reduceOnly": reduce_only,
            "postOnly": post_only,
            "leverage": str(max(1.0, float(self._order_leverage))),
        }
        if self._margin_mode in ("isolated", "cross"):
            body["marginMode"] = self._margin_mode
        data = self._req("POST", "/api/v1/orders", body=body)
        return str(data.get("orderId", data.get("order_id", "")))

    def place_marketable_limit(
        self,
        symbol: str,
        side: str,
        qty: float,
        limit_price: float,
        reduce_only: bool,
        client_id: str,
    ) -> str:
        """Limit that crosses spread (taker with cap). KuCoin: limit order with aggressive price."""
        return self.place_limit(
            symbol=symbol,
            side=side,
            qty=qty,
            price=limit_price,
            post_only=False,
            reduce_only=reduce_only,
            client_id=client_id,
        )

    def wait_filled(self, symbol: str, order_id: str, timeout_s: int) -> bool:
        contract = _symbol_to_contract(symbol)
        path = f"/api/v1/orders/{order_id}"
        t0 = time.time()
        while (time.time() - t0) < timeout_s:
            try:
                data = self._req("GET", path)
            except Exception:
                time.sleep(0.5)
                continue
            status = (data.get("status") or data.get("orderStatus", "") or "").lower()
            if status in ("done", "filled", "match"):
                return True
            if status in ("cancel", "cancelled", "reject"):
                return False
            time.sleep(0.3)
        return False


def list_fills(
    api_key: str = "",
    api_secret: str = "",
    passphrase: str = "",
    symbol: str = "",
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Fetch recent fills (trade history) for monitoring."""
    key = (api_key or os.getenv("KUCOIN_FUTURES_API_KEY", "")).strip()
    secret = (api_secret or os.getenv("KUCOIN_FUTURES_API_SECRET", "")).strip()
    pp = (passphrase or os.getenv("KUCOIN_FUTURES_PASSPHRASE", "")).strip()
    contract = _symbol_to_contract(symbol) if symbol else ""
    q = []
    if contract:
        q.append(f"symbol={contract}")
    if start_ts:
        q.append(f"from={start_ts}")
    if end_ts:
        q.append(f"to={end_ts}")
    q.append(f"pageSize={limit}")
    path = "/api/v1/fills?" + "&".join(q)
    try:
        data = _request("GET", path, api_key=key, api_secret=secret, passphrase=pp)
    except Exception:
        return []
    items = data if isinstance(data, list) else (data.get("items", data.get("data", [])) or [])
    return list(items)
