"""Pure TradingView signal -> one bot-owned order-intent planner.

Plans ONLY. Never sends exchange orders, mutates the FamilyLedger, or assumes
an intent is an actual fill. A separate durable single-writer executor must
validate/reconcile immediately before live submission.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from hashlib import sha256
from typing import Optional

from quant.execution.family_ledger import BotIntent, FamilyLedger, FamilyLedgerError

SUPPORTED = frozenset({"entry", "exit", "flip", "tp1", "tp2", "sl"})


class PlanningError(ValueError):
    pass


def _num(value: object, name: str, *, positive: bool = False) -> Decimal:
    try:
        result = Decimal(str(value))
    except (ValueError, TypeError, InvalidOperation):
        raise PlanningError(f"invalid {name}") from None
    if not result.is_finite() or (result <= 0 if positive else result < 0):
        raise PlanningError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return result


def _client_oid(family: str, bot: str, event: str, phase: str) -> str:
    if not family or not bot or not event:
        raise PlanningError("family, bot and event_id are required")
    token = "|".join((family, bot, event, phase))
    return "fam-" + sha256(token.encode("utf-8")).hexdigest()[:28]


@dataclass(frozen=True)
class OrderPlan:
    strategy_id: str
    event_id: str
    intent: Optional[BotIntent]
    status: str
    deferred_side: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "event_id": self.event_id,
            "status": self.status,
            "order_fields": None if self.intent is None else self.intent.order_fields(),
            "deferred_side": self.deferred_side,
            "order_sent": False,
        }


def plan_signal(
    *,
    ledger: FamilyLedger,
    strategy_id: str,
    event_id: str,
    symbol: str,
    action: str,
    side: str = "",
    bot_margin_budget: object,
    free_margin: object,
    reserve_floor: object,
    mark_price: object,
    leverage: object,
    contract_multiplier: object,
    tp1_close_share: object = "0.50",
) -> OrderPlan:
    """Plan just the NEXT action based on a confirmed virtual book.

    Margin of new entry is bounded by the bot's budget AND genuinely usable
    free account margin after an explicit cash reserve. Prior open position
    reservations and outstanding orders must ALSO be deducted by the live
    manager; the planner is not a comprehensive live risk engine.

    For an opposite-side FLIP, return ONLY a close. The requested new side
    is deferred until the actual close fill and reconciliation are confirmed.
    """
    action, side = str(action).lower(), str(side).lower()
    if action not in SUPPORTED:
        raise PlanningError("unsupported TradingView action")
    if side not in ("", "buy", "sell"):
        raise PlanningError("side must be buy or sell")
    if action in ("entry", "flip") and side not in ("buy", "sell"):
        raise PlanningError("entry/flip requires a buy/sell side")
    if not strategy_id or not event_id or not symbol:
        raise PlanningError("strategy_id, event_id, symbol are mandatory")

    long_pos = ledger.position(strategy_id, symbol, "LONG").contracts
    short_pos = ledger.position(strategy_id, symbol, "SHORT").contracts
    if long_pos and short_pos:
        raise PlanningError("bot has two open sides; manual reconciliation required")
    current_side = "LONG" if long_pos else ("SHORT" if short_pos else None)
    current_qty = long_pos or short_pos
    wanted_side = "LONG" if side == "buy" else "SHORT"
    margin_budget = _num(bot_margin_budget, "bot_margin_budget")
    available = _num(free_margin, "free_margin")
    reserve = _num(reserve_floor, "reserve_floor")
    px = _num(mark_price, "mark_price", positive=True)
    lev = _num(leverage, "leverage", positive=True)
    mult = _num(contract_multiplier, "contract_multiplier", positive=True)
    part = _num(tp1_close_share, "tp1_close_share")
    if part > 1:
        raise PlanningError("tp1_close_share must be <= 1")
    if mult != ledger.contract_multiplier:
        raise PlanningError("contract_multiplier differs from ledger")

    def make_intent(operation: str, pos_side: str, contracts: int, phase: str) -> OrderPlan:
        if contracts <= 0:
            return OrderPlan(strategy_id, event_id, None, "below_venue_contract_minimum")
        intent = BotIntent(
            client_oid=_client_oid(ledger.family_id, strategy_id, event_id, phase),
            bot_id=strategy_id,
            symbol=symbol,
            position_side=pos_side,
            operation=operation,
            contracts=contracts,
        )
        return OrderPlan(strategy_id, event_id, intent, "planned_no_execution")

    def new_entry(phase: str) -> OrderPlan:
        capacity = min(margin_budget, max(Decimal(0), available - reserve))
        contract_margin = px * mult / lev
        qty = int((capacity / contract_margin).to_integral_value(rounding=ROUND_DOWN))
        return make_intent("OPEN", wanted_side, qty, phase)

    if action in ("entry", "flip"):
        if current_side == wanted_side:
            return OrderPlan(strategy_id, event_id, None, "already_on_requested_side")
        if current_side is None:
            return new_entry("open")
        if action == "entry":
            return OrderPlan(strategy_id, event_id, None, "entry_cannot_reverse_without_flip")
        close = make_intent("CLOSE", current_side, current_qty, "flip-close")
        return OrderPlan(
            strategy_id, event_id, close.intent,
            "flip_close_only_wait_for_confirmed_fill",
            deferred_side=wanted_side,
        )

    if current_side is None:
        return OrderPlan(strategy_id, event_id, None, "already_flat")
    if action == "tp1":
        close_qty = max(1, int((Decimal(current_qty) * part).to_integral_value(rounding=ROUND_DOWN)))
    else:
        close_qty = current_qty
    return make_intent("CLOSE", current_side, close_qty, action)
