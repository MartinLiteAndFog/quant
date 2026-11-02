"""Blaupunkt multi-strategy hedge executor.

One KuCoin Futures sub-account; N independently attributed strategy positions.
KuCoin keeps one LONG and one SHORT position per symbol, so the bot-owned
quantities are a durable virtual ledger, reconciled against BOTH exchange legs.

This module is opt-in. It must never call the legacy, net-position TV executor.
Do not run multiple replicas against the same exchange account/state file.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from quant.execution.kucoin_futures import KucoinFuturesBroker, _symbol_to_contract
from quant.utils.log import get_logger

log = get_logger("quant.blaupunkt_hedge")
_LOCK = threading.RLock()
_FLEET = None
_VALID_BOT = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def _truthy(s: Any) -> bool:
    return str(s or "").strip().lower() in {"1", "true", "yes", "on"}


def _canonical(s: str) -> str:
    return s.strip().upper().replace("-", "").replace("_", "")


def _qty(value: Any) -> int:
    return int(abs(float(value or 0)))


def _oid(bot: str, kind: str) -> str:
    return "bp-" + bot[:12].lower() + "-" + kind[:5] + "-" + uuid.uuid4().hex[:12]


class HedgeFleet:
    def __init__(self, broker: Optional[KucoinFuturesBroker] = None, path: Optional[Path] = None):
        self.broker = broker or KucoinFuturesBroker()
        self.path = Path(path or os.getenv("HEDGE_STATE_FILE", "/data/blaupunkt-hedge.json"))
        self.symbol = os.getenv("HEDGE_SYMBOL", "XBT-USDT").strip().upper()
        self.contract = _symbol_to_contract(self.symbol)

    def _read(self) -> Dict[str, Any]:
        if self.path.exists():
            state = json.loads(self.path.read_text(encoding="utf8"))
            if state.get("version") != 1 or state.get("symbol") != self.contract:
                raise RuntimeError("hedge ledger version/symbol mismatch; refusing orders")
            return state
        names = os.getenv("HEDGE_BOT_IDS", "Blaupunkt,ElliotP1").split(",")
        bots = {}
        for name in names:
            name = name.strip()
            if name:
                if not _VALID_BOT.fullmatch(name):
                    raise ValueError("invalid bot in HEDGE_BOT_IDS")
                bots[name.lower()] = self._new_bot(name)
        if not bots:
            raise RuntimeError("no hedge bots configured")
        state = {"version": 1, "symbol": self.contract, "bots": bots, "pending": None}
        self._save(state)
        return state

    @staticmethod
    def _new_bot(name: str) -> Dict[str, Any]:
        return {"name": name, "side": "flat", "qty": 0, "stop_id": None,
                "stop_fill": 0, "stop_price": None, "unprotected": False}

    def _save(self, state: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("w", encoding="utf8") as out:
            json.dump(state, out, separators=(",", ":"), sort_keys=True)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, self.path)

    def status(self) -> Dict[str, Any]:
        with _LOCK:
            s = self._read()
            n = len(s["bots"])
            return {
                "enabled": True, "symbol": self.symbol, "bot_count": n,
                "allocation_pct": (100.0 / n if n else 0.0),
                "pending_order": bool(s.get("pending")),
                "bots": [{"id": b["name"], "side": b["side"], "qty": b["qty"],
                          "unprotected": b.get("unprotected", False)}
                         for b in s["bots"].values()],
            }

    def _mode(self) -> int:
        data = self.broker._req("GET", "/api/v2/position/getPositionMode")
        if not isinstance(data, dict) or "positionMode" not in data:
            raise RuntimeError("KuCoin position mode unreadable; refusing orders")
        return int(data["positionMode"])

    def _positions(self) -> Dict[str, int]:
        data = self.broker._req("GET", "/api/v2/position?symbol=" + self.contract)
        rows = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
        result = {"long": 0, "short": 0}
        for row in rows:
            if row.get("symbol") != self.contract:
                continue
            q = _qty(row.get("currentQty"))
            if not q:
                continue
            side = str(row.get("positionSide") or "").strip().lower()
            if side not in result:
                raise RuntimeError("unrecognized exchange positionSide: " + side)
            result[side] += q
        return result

    def _check_positions(self, s: Dict[str, Any]) -> None:
        real = self._positions()
        ledger = {"long": 0, "short": 0}
        for b in s["bots"].values():
            if b["side"] in ledger:
                ledger[b["side"]] += int(b["qty"])
        if real != ledger:
            raise RuntimeError("hedge reconciliation mismatch (exchange=%s ledger=%s); "
                               "orders blocked, manual attribution required" % (real, ledger))

    def _order_details(self, order_id: str) -> Dict[str, Any]:
        data = self.broker.get_order(order_id)
        if not isinstance(data, dict):
            raise RuntimeError("KuCoin order details unavailable")
        return data

    def _apply_fill(self, b: Dict[str, Any], position_side: str, delta: int) -> None:
        if delta > 0:
            if b["side"] not in ("flat", position_side):
                raise RuntimeError("bot already has opposite side")
            b["side"] = position_side
            b["qty"] += delta
        elif delta < 0:
            if b["side"] != position_side or b["qty"] < -delta:
                raise RuntimeError("bot close would cross virtual ownership")
            b["qty"] += delta
            if b["qty"] == 0:
                b["side"] = "flat"
        if b["qty"] == 0:
            b["stop_id"] = None
            b["stop_price"] = None
            b["stop_fill"] = 0
            b["unprotected"] = False

    def _poll_order(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pending = state.get("pending")
        if not pending:
            return {"complete": True, "filled": 0}
        oid = pending.get("order_id")
        if not oid:
            raise RuntimeError("unknown order acceptance after request; blocked pending reconciliation")
        for attempt in range(12):
            order = self._order_details(oid)
            if "filledSize" not in order:
                raise RuntimeError("order has no filledSize; reconciliation blocked")
            filled = _qty(order.get("filledSize"))
            requested = int(pending["qty"])
            if filled > requested or filled < int(pending["booked"]):
                raise RuntimeError("exchange fill count inconsistent")
            difference = filled - int(pending["booked"])
            if difference:
                bot = state["bots"][pending["bot"]]
                self._apply_fill(bot, pending["side"], difference * int(pending["direction"]))
                pending["booked"] = filled
                self._save(state)
            done = (filled >= requested
                    or str(order.get("status") or "").lower() in {"done", "cancelled", "canceled"}
                    or order.get("isActive") is False)
            if done:
                state["pending"] = None
                self._save(state)
                return {"complete": True, "filled": filled}
            if attempt < 11:
                time.sleep(0.25)
        return {"complete": False, "filled": int(pending["booked"])}

    def _sync_stops(self, s: Dict[str, Any]) -> None:
        for b in s["bots"].values():
            oid = b.get("stop_id")
            if not oid:
                continue
            order = self._order_details(oid)
            filled = _qty(order.get("filledSize"))
            previous = int(b.get("stop_fill") or 0)
            if filled < previous or filled - previous > b["qty"]:
                raise RuntimeError("stop fill attribution mismatch")
            if filled > previous:
                self._apply_fill(b, b["side"], previous - filled)
                b["stop_fill"] = filled
                if b["qty"] > 0:
                    b["unprotected"] = True
                self._save(s)
            if b["qty"] > 0 and (str(order.get("status") or "").lower() == "done"
                                or order.get("isActive") is False):
                b["unprotected"] = True
                self._save(s)

    def _cancel_own_stop(self, s: Dict[str, Any], b: Dict[str, Any]) -> None:
        oid = b.get("stop_id")
        if not oid:
            return
        self.broker.cancel_order(order_id=oid)
        for _ in range(10):
            order = self._order_details(oid)
            filled = _qty(order.get("filledSize"))
            previous = int(b.get("stop_fill") or 0)
            if filled > previous:
                self._apply_fill(b, b["side"], previous - filled)
                if b["qty"] > 0:
                    b["stop_fill"] = filled
                self._save(s)
            if order.get("isActive") is False or str(order.get("status") or "").lower() in {"done", "cancelled", "canceled"}:
                b["stop_id"] = None
                b["stop_fill"] = 0
                b["stop_price"] = None
                b["unprotected"] = b["qty"] > 0
                self._save(s)
                return
            time.sleep(0.2)
        raise RuntimeError("stop cancellation not confirmed; refusing position change")

    def _margin_mode(self) -> str:
        mode = os.getenv("KUCOIN_FUTURES_MARGIN_MODE", "").strip().upper()
        if mode not in {"ISOLATED", "CROSS"}:
            raise RuntimeError("hedge requires explicit KUCOIN_FUTURES_MARGIN_MODE")
        return mode

    def _send_market(self, s: Dict[str, Any], key: str, position_side: str,
                     direction: int, qty: int) -> int:
        if qty <= 0:
            return 0
        side = ("buy" if position_side == "long" else "sell") if direction > 0 else (
            "sell" if position_side == "long" else "buy")
        client_id = _oid(key, "open" if direction > 0 else "close")
        s["pending"] = {"bot": key, "side": position_side, "direction": direction,
                        "qty": qty, "booked": 0, "client_id": client_id,
                        "order_id": None}
        self._save(s)  # Durable intent BEFORE sending order.
        body = {
            "clientOid": client_id, "symbol": self.contract, "positionSide": position_side.upper(),
            "side": side, "type": "market", "size": int(qty),
            "marginMode": self._margin_mode(),
            "leverage": str(float(os.getenv("KUCOIN_FUTURES_ORDER_LEVERAGE", "1"))),
            "reduceOnly": direction < 0,
        }
        data = self.broker._req("POST", "/api/v1/orders", body=body)
        oid = str(data.get("orderId") or "")
        if not oid:
            raise RuntimeError("KuCoin accepted order without an ID; pending reconciliation required")
        s["pending"]["order_id"] = oid
        self._save(s)
        result = self._poll_order(s)
        if not result["complete"]:
            raise RuntimeError("market order still active; no further orders until settled")
        self._check_positions(s)
        return result["filled"]

    def _price_and_qty(self, n: int) -> tuple[float, int]:
        balance = self.broker.get_account_balance("USDT")
        equity = float(balance.get("equity") or 0)
        available = float(balance.get("available") or 0)
        bid, ask = self.broker.get_best_bid_ask(self.symbol)
        price = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
        multiplier = float(self.broker.get_contract_multiplier(self.symbol))
        leverage = float(os.getenv("KUCOIN_FUTURES_ORDER_LEVERAGE", "1"))
        # The shared trading pool is 90% of total equity, *then* divided
        # equally between registered strategies. The other 10% is not
        # distributable, regardless of N or current free margin.
        use = float(os.getenv("HEDGE_BOT_CAPITAL_USAGE", "0.90"))
        if not (0 < use <= 0.90) or not (0 < leverage <= float(os.getenv("HEDGE_MAX_LEVERAGE", "20"))):
            raise RuntimeError("invalid hedge capital usage (>90%) or leverage")
        if not (equity > 0 and available > 0 and price > 0 and multiplier > 0):
            return price, 0
        per_bot_margin = equity * use / n
        # Deduct the shared reserve ONCE from free balance. Multiplying
        # remaining free margin by 0.90 on every entry would compound a
        # new 10% haircut per bot, incorrectly starving the later bots.
        reserve_margin = equity * (1.0 - use)
        affordable_margin = max(0.0, available - reserve_margin)
        qty = math.floor(min(per_bot_margin, affordable_margin) * leverage / (price * multiplier))
        max_contracts = int(os.getenv("LIVE_EXECUTOR_MAX_CONTRACTS", "0") or 0)
        max_margin = float(os.getenv("LIVE_EXECUTOR_MAX_MARGIN_USDT", "0") or 0)
        if max_margin > 0:
            qty = min(qty, math.floor(max_margin * leverage / (price * multiplier)))
        if max_contracts > 0:
            qty = min(qty, max_contracts)
        return price, max(0, qty)

    def _protect(self, s: Dict[str, Any], b: Dict[str, Any], price: float,
                 requested_stop: Optional[float] = None) -> None:
        if b["qty"] <= 0:
            return
        if price <= 0:
            b["unprotected"] = True
            self._save(s)
            raise RuntimeError("cannot protect without a valid reference price")
        side = b["side"]
        pct = float(os.getenv("HEDGE_EMERGENCY_SL_PCT", "0.025"))
        if not 0 < pct < 0.5:
            raise RuntimeError("invalid HEDGE_EMERGENCY_SL_PCT")
        stop_price = price * (1 - pct if side == "long" else 1 + pct)
        if requested_stop is not None and requested_stop > 0:
            if (side == "long" and requested_stop < price) or (side == "short" and requested_stop > price):
                stop_price = requested_stop
        contract_info = self.broker._req("GET", "/api/v1/contracts/" + self.contract)
        tick = float(contract_info.get("tickSize") or 0.1)
        stop_price = round(stop_price / tick) * tick
        if stop_price <= 0:
            raise RuntimeError("invalid hedge stop trigger")
        cid = _oid(b["name"], "stop")
        body = {
            "clientOid": cid, "symbol": self.contract,
            "positionSide": side.upper(),
            "side": "sell" if side == "long" else "buy",
            "type": "market", "size": int(b["qty"]),
            "stop": "down" if side == "long" else "up",
            "stopPriceType": "TP", "stopPrice": str(stop_price),
            "marginMode": self._margin_mode(),
            "leverage": str(float(os.getenv("KUCOIN_FUTURES_ORDER_LEVERAGE", "1"))),
            "reduceOnly": True,
        }
        b["unprotected"] = True
        self._save(s)
        data = self.broker._req("POST", "/api/v1/orders", body=body)
        oid = str(data.get("orderId") or "")
        if not oid:
            raise RuntimeError("emergency stop returned no ID")
        b["stop_id"] = oid
        b["stop_price"] = stop_price
        b["stop_fill"] = 0
        self._save(s)
        order = self._order_details(oid)
        if str(order.get("stop") or "").lower() not in {"up", "down"} or str(order.get("status") or "").lower() == "done":
            raise RuntimeError("emergency stop did not register as resting conditional order")
        b["unprotected"] = False
        self._save(s)

    def execute(self, bot_id: str, action: str, side: str, symbol: str,
                sl_price: Optional[float], tp1_pct: float, dry_run: bool) -> Dict[str, Any]:
        with _LOCK:
            if not _VALID_BOT.fullmatch(bot_id or ""):
                raise ValueError("a valid bot ID is required")
            if _canonical(symbol) != _canonical(self.symbol):
                raise ValueError("wrong symbol for Blaupunkt hedge fleet")
            s = self._read()
            key = bot_id.lower()
            if key not in s["bots"]:
                if not _truthy(os.getenv("HEDGE_AUTO_REGISTER", "1")):
                    raise ValueError("unregistered bot ID")
                s["bots"][key] = self._new_bot(bot_id)
                self._save(s)
            b = s["bots"][key]
            n = len(s["bots"])
            summary = {"bot": b["name"], "bot_count": n,
                       "allocation_pct": round(100.0 / n, 8), "action": action}
            if dry_run or not _truthy(os.getenv("LIVE_TRADING_ENABLED")):
                return {**summary, "ok": True, "dry_run": True,
                        "side": side, "current_side": b["side"]}
            if self._mode() != 1:
                raise RuntimeError("KuCoin is not in Hedge Mode (positionMode=1); no orders placed")
            if s.get("pending"):
                result = self._poll_order(s)
                if not result["complete"]:
                    raise RuntimeError("previous order not settled")
            self._sync_stops(s)
            self._check_positions(s)
            if any(x.get("unprotected") for k, x in s["bots"].items() if k != key):
                raise RuntimeError("other bot has unprotected position; no further orders")
            desired = ("long" if side == "buy" else "short") if action in {"entry", "flip"} else None
            if action in {"entry", "flip"} and desired == b["side"]:
                return {**summary, "ok": True, "reason": "duplicate_same_side"}
            if action == "entry" and b["side"] != "flat":
                raise RuntimeError("entry would overwrite own position; send flip")
            if action in {"exit", "sl", "tp2", "tp1"} and b["qty"] == 0:
                return {**summary, "ok": True, "reason": "already_flat"}
            if action in {"exit", "sl", "tp2", "tp1"} or (action == "flip" and b["qty"]):
                self._cancel_own_stop(s, b)
                if b["qty"]:
                    close_qty = max(1, int(b["qty"] * tp1_pct)) if action == "tp1" else int(b["qty"])
                    filled = self._send_market(s, key, b["side"], -1, close_qty)
                    if filled != close_qty:
                        raise RuntimeError("partial close: bot ledger preserved; further action blocked")
                if b["qty"] and action == "tp1":
                    price, _ = self._price_and_qty(n)
                    self._protect(s, b, price, sl_price)
                    return {**summary, "ok": True, "closed_qty": close_qty, "remaining": b["qty"]}
                if action != "flip":
                    return {**summary, "ok": True, "closed_qty": close_qty, "remaining": b["qty"]}
            if action in {"flip", "entry"}:
                if b["qty"] != 0:
                    raise RuntimeError("bot is not flat; refusing new hedge leg")
                price, qty = self._price_and_qty(n)
                if qty < 1:
                    return {**summary, "ok": False, "reason": "insufficient_allocated_margin"}
                filled = self._send_market(s, key, desired, +1, qty)
                if filled:
                    self._protect(s, b, price, sl_price)
                if filled != qty:
                    raise RuntimeError("entry partial or not filled; ledger reconciled")
                return {**summary, "ok": True, "opened_side": desired, "qty": filled,
                        "stop_id": b["stop_id"]}
            raise ValueError("unsupported action")


def fleet() -> HedgeFleet:
    global _FLEET
    with _LOCK:
        if _FLEET is None:
            _FLEET = HedgeFleet()
        return _FLEET
