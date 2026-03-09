from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

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
        self.default_trigger_signal = (os.getenv("KRAKEN_FUTURES_TRIGGER_SIGNAL", "mark") or "mark").strip().lower()

    # ------------------------------------------------------------------
    # Core HTTP
    # ------------------------------------------------------------------

    def _signed_headers(self, endpoint_path: str, body: bytes, nonce: str) -> Dict[str, str]:
        if not self.key or not self.secret:
            return {
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "quant-kraken/1",
            }

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

    def _req(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        private: bool = False,
    ) -> Dict[str, Any]:
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

        req = urllib.request.Request(
            url,
            data=(body if method.upper() != "GET" else None),
            method=method.upper(),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                out = r.read().decode("utf-8")
            data = json.loads(out)
        except Exception as e:
            raise RuntimeError(f"kraken request failed path={path} err={e}") from e

        if isinstance(data, dict) and (data.get("result") == "error" or data.get("error")):
            raise RuntimeError(f"kraken api error path={path} data={data}")

        return data if isinstance(data, dict) else {"result": "success", "data": data}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _norm_symbol(self, symbol: Optional[str]) -> str:
        return symbol or self.symbol

    def _norm_side(self, side: str) -> str:
        s = str(side or "").strip().lower()
        if s not in ("buy", "sell"):
            raise ValueError(f"invalid side: {side!r}")
        return s

    def _norm_size_str(self, size: float) -> str:
        s = max(0.0, float(size))
        if s <= 0:
            raise ValueError("size must be > 0")
        return f"{s:.8f}"

    def _norm_trigger_signal(self, trigger_signal: Optional[str]) -> str:
        sig = (trigger_signal or self.default_trigger_signal or "mark").strip().lower()
        if sig not in ("mark", "last", "spot"):
            raise ValueError(f"invalid trigger_signal: {trigger_signal!r}")
        return sig

    def _extract_open_orders(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(data, dict):
            return []
        for key in ("openOrders", "orders", "open_orders"):
            v = data.get(key)
            if isinstance(v, list):
                return v
        return []

    # ------------------------------------------------------------------
    # Market/account
    # ------------------------------------------------------------------

    def get_mark_price(self, symbol: Optional[str] = None) -> float:
        sym = self._norm_symbol(symbol)
        data = self._req("GET", "/derivatives/api/v3/tickers")
        tickers = data.get("tickers", []) if isinstance(data, dict) else []
        for t in tickers:
            if str(t.get("symbol")) == sym:
                return float(t.get("markPrice", t.get("last", 0)) or 0)
        return 0.0

    def get_position(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        sym = self._norm_symbol(symbol)
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
        return {
            "side": "flat",
            "size": 0.0,
            "size_signed": 0.0,
            "entry_price": 0.0,
            "leverage": None,
            "raw": None,
        }

    def get_account_equity(self) -> Dict[str, float]:
        data = self._req("GET", "/derivatives/api/v3/accounts", private=True)
        accts = data.get("accounts", {}) if isinstance(data, dict) else {}
        flex = accts.get("flex", accts.get("fi_xbtusd", {})) if isinstance(accts, dict) else {}
        wallet = float(flex.get("balanceValue", flex.get("balance", 0)) or 0)
        upnl = float(flex.get("unrealizedFunding", 0) or 0) + float(flex.get("unrealizedPnl", 0) or 0)
        equity = float(flex.get("portfolioValue", 0) or 0)
        if equity <= 0:
            equity = wallet + upnl
        return {
            "wallet_usd": wallet,
            "upnl_usd": upnl,
            "equity_usd": equity,
        }

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        sym = self._norm_symbol(symbol)
        data = self._req("GET", "/derivatives/api/v3/openorders", private=True)
        rows = self._extract_open_orders(data)
        out: List[Dict[str, Any]] = []
        for o in rows:
            if sym and str(o.get("symbol", "")) != sym:
                continue
            out.append(
                {
                    "order_id": o.get("order_id") or o.get("orderId") or o.get("id"),
                    "cli_ord_id": o.get("cliOrdId") or o.get("clientOrderId"),
                    "symbol": o.get("symbol"),
                    "side": o.get("side"),
                    "size": float(o.get("size", 0) or 0),
                    "filled": float(o.get("filled", 0) or 0),
                    "reduce_only": bool(o.get("reduceOnly", False)),
                    "order_type": o.get("orderType"),
                    "stop_price": o.get("stopPrice"),
                    "limit_price": o.get("limitPrice"),
                    "trigger_signal": o.get("triggerSignal"),
                    "raw": o,
                }
            )
        return out

    def place_market(
        self,
        side: str,
        size: float,
        symbol: Optional[str] = None,
        reduce_only: bool = False,
        cli_ord_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        sym = self._norm_symbol(symbol)
        side_n = self._norm_side(side)
        size_s = self._norm_size_str(size)

        params: Dict[str, Any] = {
            "symbol": sym,
            "side": side_n,
            "size": size_s,
            "orderType": "mkt",
            "reduceOnly": "true" if reduce_only else "false",
        }
        if cli_ord_id:
            params["cliOrdId"] = str(cli_ord_id)

        data = self._req("POST", "/derivatives/api/v3/sendorder", params=params, private=True)
        return {"ok": True, "data": data}

    def place_stop_market(
        self,
        side: str,
        size: float,
        stop_price: float,
        symbol: Optional[str] = None,
        reduce_only: bool = True,
        trigger_signal: Optional[str] = None,
        cli_ord_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Place a stop-market style trigger order.

        Intended for protective stop-loss usage:
        - orderType='stp'
        - stopPrice used as trigger price
        - reduceOnly defaults to True
        """
        sym = self._norm_symbol(symbol)
        side_n = self._norm_side(side)
        size_s = self._norm_size_str(size)
        stop_f = float(stop_price)
        if stop_f <= 0:
            raise ValueError("stop_price must be > 0")

        params: Dict[str, Any] = {
            "symbol": sym,
            "side": side_n,
            "size": size_s,
            "orderType": "stp",
            "stopPrice": f"{stop_f:.8f}",
            "triggerSignal": self._norm_trigger_signal(trigger_signal),
            "reduceOnly": "true" if reduce_only else "false",
        }
        if cli_ord_id:
            params["cliOrdId"] = str(cli_ord_id)

        data = self._req("POST", "/derivatives/api/v3/sendorder", params=params, private=True)
        return {"ok": True, "data": data}

    def cancel_order(
        self,
        order_id: Optional[str] = None,
        cli_ord_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not order_id and not cli_ord_id:
            return {"ok": False, "reason": "missing_order_id"}

        params: Dict[str, Any] = {}
        if order_id:
            params["order_id"] = str(order_id)
        if cli_ord_id:
            params["cliOrdId"] = str(cli_ord_id)

        data = self._req("POST", "/derivatives/api/v3/cancelorder", params=params, private=True)
        return {"ok": True, "data": data}

    def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        sym = self._norm_symbol(symbol)
        params: Dict[str, Any] = {}
        if sym:
            params["symbol"] = sym
        data = self._req("POST", "/derivatives/api/v3/cancelallorders", params=params, private=True)
        return {"ok": True, "data": data}

    def cancel_all_reduce_only_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Conservative helper:
        - fetch open orders
        - cancel only reduce-only orders for the symbol
        """
        rows = self.get_open_orders(symbol=symbol)
        cancelled: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        for o in rows:
            if not bool(o.get("reduce_only")):
                continue
            order_id = o.get("order_id")
            cli_ord_id = o.get("cli_ord_id")
            try:
                res = self.cancel_order(
                    order_id=str(order_id) if order_id else None,
                    cli_ord_id=str(cli_ord_id) if cli_ord_id else None,
                )
                cancelled.append(
                    {
                        "order_id": order_id,
                        "cli_ord_id": cli_ord_id,
                        "result": res,
                    }
                )
            except Exception as e:
                errors.append(
                    {
                        "order_id": order_id,
                        "cli_ord_id": cli_ord_id,
                        "error": str(e),
                    }
                )

        return {
            "ok": len(errors) == 0,
            "cancelled": cancelled,
            "errors": errors,
        }

    def close_position(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        p = self.get_position(symbol=symbol)
        side = p.get("side")
        size = float(p.get("size", 0) or 0)
        if size <= 0 or side == "flat":
            return {"ok": True, "reason": "already_flat"}
        close_side = "sell" if side == "long" else "buy"
        return self.place_market(close_side, size=size, symbol=symbol, reduce_only=True)