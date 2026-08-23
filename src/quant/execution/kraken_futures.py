from __future__ import annotations

import base64
from decimal import Decimal, ROUND_HALF_UP
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
        self._tick_size_cache: Dict[str, float] = {}

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
            err_body = None
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = None
            raise RuntimeError(f"kraken request failed path={path} err={e} body={err_body}") from e
            
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

    def _tick_size(self, symbol: Optional[str] = None) -> Optional[float]:
        sym = self._norm_symbol(symbol)
        cached = self._tick_size_cache.get(sym)
        if cached is not None and cached > 0:
            return cached

        try:
            data = self._req("GET", "/derivatives/api/v3/instruments")
        except Exception:
            return None

        rows = data.get("instruments", []) if isinstance(data, dict) else []
        for row in rows:
            if str(row.get("symbol")) != sym:
                continue
            try:
                tick = float(row.get("tickSize") or 0.0)
            except Exception:
                tick = 0.0
            if tick > 0:
                self._tick_size_cache[sym] = tick
                return tick
        return None

    def _norm_price_str(self, price: float, symbol: Optional[str] = None) -> str:
        price_f = float(price)
        tick = self._tick_size(symbol)
        if tick is not None and tick > 0:
            price_dec = Decimal(str(price_f))
            tick_dec = Decimal(str(tick))
            steps = (price_dec / tick_dec).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
            price_f = float(steps * tick_dec)
        return f"{price_f:.8f}"

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
                raw_size = float(p.get("size", 0) or 0)
                abs_size = abs(raw_size)

                raw_side = str(p.get("side", "") or "").strip().lower()
                if raw_side in ("short", "sell"):
                    side = "short"
                    size_signed = -abs_size
                elif raw_side in ("long", "buy"):
                    side = "long"
                    size_signed = abs_size
                else:
                    if raw_size > 0:
                        side = "long"
                        size_signed = abs_size
                    elif raw_size < 0:
                        side = "short"
                        size_signed = -abs_size
                    else:
                        side = "flat"
                        size_signed = 0.0

                return {
                    "side": side,
                    "size": abs_size,
                    "size_signed": size_signed,
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
        if not isinstance(flex, dict):
            flex = {}

        def _f(*keys: str) -> float:
            for k in keys:
                if k not in flex or flex.get(k) is None:
                    continue
                try:
                    v = float(flex.get(k) or 0)
                except Exception:
                    continue
                if v == v:  # finite
                    return v
            return 0.0

        # Do NOT treat availableMargin as equity — that understates capital when
        # margin is locked in open positions.
        wallet = _f("balanceValue", "balance", "walletBalance")
        available = _f("availableMargin", "available", "availableBalance")
        portfolio = _f(
            "portfolioValue",
            "portfolio_value",
            "marginEquity",
            "collateralValue",
            "equity",
        )
        # Flex accounts report open-position PnL as "totalUnrealized"; the old
        # "unrealizedPnl" keys are legacy per-market payloads. Missing this key
        # made equity ignore the running trade (showed margin only).
        upnl = _f("totalUnrealized", "unrealizedPnl", "unrealizedPNL") + _f(
            "unrealizedFunding"
        )
        # Some payload variants report portfolioValue WITHOUT open-position PnL
        # (it then tracks wallet). True equity = capital + running trade.
        equity = portfolio
        if equity <= 0:
            equity = wallet + upnl if (wallet > 0 or upnl != 0) else 0.0
        elif upnl != 0 and wallet > 0 and abs(equity - wallet) < abs(upnl) * 0.5:
            # portfolio ≈ wallet although a position is open → PnL missing.
            equity = wallet + upnl
        # Last resort only: some flex payloads omit portfolio/wallet.
        if equity <= 0 and available > 0:
            equity = available
        return {
            "wallet_usd": wallet,
            "upnl_usd": upnl,
            "equity_usd": equity,
            "available_usd": available,
            "portfolio_usd": portfolio,
        }

    def get_position_events(
        self,
        *,
        symbol: Optional[str] = None,
        since_ms: Optional[int] = None,
        before_ms: Optional[int] = None,
        limit: int = 500,
        include_funding: bool = False,
    ) -> List[Dict[str, Any]]:
        """Read authenticated position events from Kraken Futures.

        The history API uses the Futures ``APIKey``/``Authent`` scheme and
        signs the URL-encoded GET query as ``postData``. This method is
        deliberately read-only and never calls an order route.

        ``include_funding`` is deliberately opt-in so legacy callers retain a
        trades-only result.  The Fleet activity read model opts in to expose
        exchange-reported funding and settlement events beside real fills.
        """
        if not self.key or not self.secret:
            raise RuntimeError("Kraken Futures credentials are not configured")

        endpoint_path = "/api/history/v3/positions"
        # ``count`` is still capped at 100 per exchange request; this bound is
        # the number of rows accumulated across continuation pages.
        wanted = max(1, min(int(limit), 10_000))
        continuation: Optional[str] = None
        out: List[Dict[str, Any]] = []

        while len(out) < wanted:
            params: Dict[str, Any] = {
                "sort": "desc",
                "count": min(100, wanted - len(out)),
                "trades": "true",
                "tradeable": self._norm_symbol(symbol),
            }
            if include_funding:
                params["funding_realization"] = "true"
                params["settlement"] = "true"
            if since_ms is not None:
                params["since"] = int(since_ms)
            if before_ms is not None:
                params["before"] = int(before_ms)
            if continuation:
                params["continuation_token"] = continuation

            query = urllib.parse.urlencode(params)
            nonce = str(int(time.time() * 1000))
            headers = {
                "Accept": "application/json",
                **self._signed_headers(endpoint_path, query.encode("utf-8"), nonce),
            }
            req = urllib.request.Request(
                f"{self.base}{endpoint_path}?{query}",
                method="GET",
                headers=headers,
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except Exception as exc:
                err_body = None
                try:
                    err_body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    err_body = None
                raise RuntimeError(
                    f"kraken position history failed err={exc} body={err_body}"
                ) from exc

            if not isinstance(payload, dict):
                raise RuntimeError("kraken position history returned invalid payload")
            if payload.get("error") or payload.get("errors"):
                raise RuntimeError(f"kraken position history error data={payload}")

            rows = payload.get("elements")
            if not isinstance(rows, list):
                rows = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                # Current Futures history wraps every PositionUpdate in an
                # envelope: {timestamp, uid, event: {PositionUpdate: {...}}}.
                # Normalise that transport shape at the client boundary so
                # every read-only Fleet consumer sees the documented position
                # fields, while retaining compatibility with the older flat
                # response used by existing callers and tests.
                event = row.get("event")
                position_update = (
                    event.get("PositionUpdate")
                    if isinstance(event, dict)
                    else None
                )
                if isinstance(position_update, dict):
                    normalised = dict(position_update)
                    normalised.setdefault("timestamp", row.get("timestamp"))
                    normalised.setdefault("historyUid", row.get("uid"))
                    out.append(normalised)
                else:
                    out.append(row)

            next_token = payload.get("continuationToken")
            if not next_token or not rows:
                break
            continuation = str(next_token)

        return out[:wanted]

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
            "stopPrice": self._norm_price_str(stop_f, symbol=sym),
            "triggerSignal": self._norm_trigger_signal(trigger_signal),
            "reduceOnly": "true" if reduce_only else "false",
        }
        if cli_ord_id:
            params["cliOrdId"] = str(cli_ord_id)

        data = self._req("POST", "/derivatives/api/v3/sendorder", params=params, private=True)
        return {"ok": True, "data": data}


    def place_take_profit_market(
        self,
        side: str,
        size: float,
        stop_price: float,
        symbol: Optional[str] = None,
        reduce_only: bool = True,
        trigger_signal: Optional[str] = None,
        cli_ord_id: Optional[str] = None,
    ) -> Dict[str, Any]:
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
            "orderType": "take_profit",
            "stopPrice": self._norm_price_str(stop_f, symbol=sym),
            "triggerSignal": self._norm_trigger_signal(trigger_signal),
            "reduceOnly": "true" if reduce_only else "false",
        }
        if cli_ord_id:
            params["cliOrdId"] = str(cli_ord_id)

        data = self._req("POST", "/derivatives/api/v3/sendorder", params=params, private=True)
        return {"ok": True, "data": data}

    def place_trigger_entry_market(
        self,
        side: str,
        size: float,
        stop_price: float,
        symbol: Optional[str] = None,
        reduce_only: bool = False,
        trigger_signal: Optional[str] = None,
        cli_ord_id: Optional[str] = None,
    ) -> Dict[str, Any]:
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
            "stopPrice": self._norm_price_str(stop_f, symbol=sym),
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
