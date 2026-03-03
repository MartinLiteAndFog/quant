from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from quant.utils.log import get_logger

log = get_logger("quant.kraken_futures")


class KrakenFuturesClient:
    """Minimal Kraken Futures REST client for account + order lifecycle."""

    def __init__(self) -> None:
        self.key = (os.getenv("KRAKEN_FUTURES_KEY", "") or "").strip()
        self.secret = (os.getenv("KRAKEN_FUTURES_SECRET", "") or "").strip()
        self.base = (os.getenv("KRAKEN_FUTURES_BASE_URL", "https://futures.kraken.com") or "").rstrip("/")
        self.symbol = os.getenv("KRAKEN_FUTURES_SYMBOL", "PF_SOLUSD")
        self.timeout_s = int(os.getenv("KRAKEN_FUTURES_TIMEOUT_SEC", "10"))

    def _signed_headers(self, endpoint_path: str, body: bytes, nonce: str) -> Dict[str, str]:
        if not self.key or not self.secret:
            return {"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "quant-kraken/1"}
        post_data = body.decode("utf-8") if body else ""
        sign_path = endpoint_path
        if sign_path.startswith("/derivatives"):
            sign_path = sign_path[len("/derivatives"):]
        message = post_data + nonce + sign_path
        digest = hashlib.sha256(message.encode("utf-8")).digest()
        secret_decoded = base64.b64decode(self.secret)
        sig = hmac.new(secret_decoded, digest, hashlib.sha512).digest()
        authent = base64.b64encode(sig).decode("utf-8")
        return {
            "Content-Type": "application/x-www-form-urlencoded",
            "APIKey": self.key,
            "Nonce": nonce,
            "Authent": authent,
            "User-Agent": "quant-kraken/1",
        }

    def _req(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, private: bool = False) -> Dict[str, Any]:
        params = params or {}
        endpoint_path = path
        url = self.base + path
        body = b""
        headers = {"User-Agent": "quant-kraken/1"}

        if method.upper() == "GET" and params:
            qs = urllib.parse.urlencode(params)
            url = url + ("?" + qs)
        elif params:
            body = urllib.parse.urlencode(params).encode("utf-8")

        if private:
            nonce = str(int(time.time() * 1000))
            headers.update(self._signed_headers(endpoint_path, body, nonce))
        req = urllib.request.Request(url, data=(body if method.upper() != "GET" else None), method=method.upper(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                out = r.read().decode("utf-8")
            data = json.loads(out)
        except Exception as e:
            raise RuntimeError(f"kraken request failed path={path} err={e}") from e
        if isinstance(data, dict) and (data.get("result") == "error" or data.get("error")):
            raise RuntimeError(f"kraken api error path={path} data={data}")
        return data if isinstance(data, dict) else {"result": "success", "data": data}

    def get_mark_price(self, symbol: Optional[str] = None) -> float:
        sym = symbol or self.symbol
        data = self._req("GET", "/derivatives/api/v3/tickers")
        tickers = data.get("tickers", []) if isinstance(data, dict) else []
        for t in tickers:
            if str(t.get("symbol")) == sym:
                return float(t.get("markPrice", t.get("last", 0)) or 0)
        return 0.0

    def get_position(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        sym = symbol or self.symbol
        data = self._req("GET", "/derivatives/api/v3/openpositions", private=True)
        ps = data.get("openPositions", []) if isinstance(data, dict) else []
        for p in ps:
            if str(p.get("symbol")) == sym:
                size = float(p.get("size", 0) or 0)
                side = "long" if size > 0 else ("short" if size < 0 else "flat")
                return {
                    "side": side,
                    "size": abs(size),
                    "size_signed": size,
                    "entry_price": float(p.get("price", p.get("entryPrice", 0)) or 0),
                    "leverage": float(p.get("effectiveLeverage", p.get("leverage", 0)) or 0),
                    "raw": p,
                }
        return {"side": "flat", "size": 0.0, "size_signed": 0.0, "entry_price": 0.0, "leverage": None, "raw": None}

    def get_account_equity(self) -> Dict[str, float]:
        data = self._req("GET", "/derivatives/api/v3/accounts", private=True)
        accts = data.get("accounts", {}) if isinstance(data, dict) else {}
        # Kraken keys can vary; try robust extraction.
        flex = accts.get("flex", accts.get("fi_xbtusd", {})) if isinstance(accts, dict) else {}
        wallet = float(flex.get("balanceValue", flex.get("balance", 0)) or 0)
        upnl = float(flex.get("unrealizedFunding", 0) or 0) + float(flex.get("unrealizedPnl", 0) or 0)
        equity = float(flex.get("portfolioValue", 0) or 0)
        if equity <= 0:
            equity = wallet + upnl
        return {"wallet_usd": wallet, "upnl_usd": upnl, "equity_usd": equity}

    def place_market(self, side: str, size: float, symbol: Optional[str] = None, reduce_only: bool = False) -> Dict[str, Any]:
        sym = symbol or self.symbol
        size = max(0.0, float(size))
        if size <= 0:
            return {"ok": False, "reason": "size_zero"}
        params = {
            "symbol": sym,
            "side": "buy" if str(side).lower() == "buy" else "sell",
            "size": f"{size:.8f}",
            "orderType": "mkt",
            "reduceOnly": "true" if reduce_only else "false",
        }
        data = self._req("POST", "/derivatives/api/v3/sendorder", params=params, private=True)
        return {"ok": True, "data": data}

    def close_position(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        p = self.get_position(symbol=symbol)
        side = p.get("side")
        size = float(p.get("size", 0) or 0)
        if size <= 0 or side == "flat":
            return {"ok": True, "reason": "already_flat"}
        close_side = "sell" if side == "long" else "buy"
        return self.place_market(close_side, size=size, symbol=symbol, reduce_only=True)
