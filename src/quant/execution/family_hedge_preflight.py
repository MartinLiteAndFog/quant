"""Read-only KuCoin Hedge Mode preflight for family migration.

This module NEVER submits/cancels orders or changes account position mode.
It must run before a dedicated family's single-writer executor is activated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class HedgePreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class HedgePreflight:
    mode: str
    symbol: str
    long_contracts: int
    short_contracts: int
    open_regular_orders: int
    open_conditional_orders: int

    @property
    def flat_and_order_free(self) -> bool:
        return (
            self.long_contracts == 0
            and self.short_contracts == 0
            and self.open_regular_orders == 0
            and self.open_conditional_orders == 0
        )

    @property
    def hedge_ready(self) -> bool:
        return self.mode == "HEDGE"


def _mode(raw: Any) -> str:
    if isinstance(raw, dict):
        raw = raw.get("positionMode")
    if str(raw) == "1":
        return "HEDGE"
    if str(raw) == "0":
        return "ONE_WAY"
    raise HedgePreflightError("position mode unknown; do not trade")


def _items(raw: Any, field: str) -> list[dict]:
    if isinstance(raw, dict):
        raw = raw.get("items", raw.get("data"))
    if raw is None:
        raise HedgePreflightError(f"{field} response unavailable")
    if not isinstance(raw, list) or any(not isinstance(x, dict) for x in raw):
        raise HedgePreflightError(f"{field} response has unexpected shape")
    return raw


def inspect_hedge_account(broker: Any, symbol: str = "SOLUSDTM") -> HedgePreflight:
    """Inspect live mode, both side quantities and unfilled orders (GET only).

    Existing broker methods may internally call GET endpoints; no write method
    is referenced. A network or schema failure must propagate as a hard error.
    """
    if symbol != "SOLUSDTM":
        raise HedgePreflightError("first family pilot only supports SOLUSDTM")
    try:
        current_mode = _mode(broker._req("GET", "/api/v2/position/getPositionMode"))
        raw_positions = _items(broker._req("GET", "/api/v1/positions"), "positions")
        regular = broker.list_open_orders("SOL-USDT")
        conditional = broker.list_open_stop_orders("SOL-USDT")
    except Exception as exc:
        raise HedgePreflightError(f"read-only preflight failed: {exc}") from exc
    if not isinstance(regular, list) or not isinstance(conditional, list):
        raise HedgePreflightError("open order inspection unavailable")

    long_qty = short_qty = 0
    for pos in raw_positions:
        if pos.get("symbol") != symbol:
            # A family account with an unrelated open symbol is NOT dedicated.
            try:
                other_qty = float(pos.get("currentQty") or 0)
            except (TypeError, ValueError, OverflowError):
                raise HedgePreflightError("unrecognized other-symbol quantity") from None
            if other_qty:
                raise HedgePreflightError("unrelated open position on family account")
            continue
        try:
            qty_raw = pos.get("currentQty")
            if qty_raw is None or isinstance(qty_raw, bool):
                raise ValueError("currentQty missing")
            qty = float(qty_raw)
            if not qty.is_integer() or not (-10**12 < qty < 10**12):
                raise ValueError("invalid contract count")
            quantity = abs(int(qty))
        except (ValueError, TypeError, OverflowError):
            raise HedgePreflightError("invalid exchange contract quantity") from None
        if quantity == 0:
            continue
        pos_side = str(pos.get("positionSide") or "BOTH").upper()
        if pos_side == "LONG":
            long_qty += quantity
        elif pos_side == "SHORT":
            short_qty += quantity
        else:
            raise HedgePreflightError("non-flat BOTH/unknown side position")
    return HedgePreflight(
        mode=current_mode, symbol=symbol,
        long_contracts=long_qty, short_contracts=short_qty,
        open_regular_orders=len(regular),
        open_conditional_orders=len(conditional),
    )


def require_hedge_ready(state: HedgePreflight) -> None:
    if not state.hedge_ready:
        raise HedgePreflightError("account is not in Hedge Mode")
