"""Pure, fail-closed virtual position ledger for one KuCoin strategy family.

NO API calls or live order execution. One authoritative owner per subaccount is
required before integrating this with a venue executor. Venue fills, not intent
creation, change positions. Amounts are contract counts, not SOL quantities.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import Dict, Tuple


class FamilyLedgerError(RuntimeError):
    """Reject ambiguous, unsafe, or inconsistent accounting events."""


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FamilyLedgerError(f"{label} must be a positive integer contract count")
    return value


def _amount(value: object, label: str, *, allow_zero: bool = False) -> Decimal:
    try:
        out = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise FamilyLedgerError(f"invalid {label}") from None
    if not out.is_finite() or (out < 0 if allow_zero else out <= 0):
        raise FamilyLedgerError(f"{label} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return out


@dataclass(frozen=True)
class BotIntent:
    client_oid: str
    bot_id: str
    symbol: str
    position_side: str  # LONG or SHORT (Hedge Mode, never BOTH)
    operation: str  # OPEN or CLOSE
    contracts: int

    def __post_init__(self) -> None:
        if not self.client_oid or not self.bot_id or not self.symbol:
            raise FamilyLedgerError("client_oid, bot_id and symbol are mandatory")
        if self.position_side not in ("LONG", "SHORT"):
            raise FamilyLedgerError("position_side must be LONG or SHORT")
        if self.operation not in ("OPEN", "CLOSE"):
            raise FamilyLedgerError("operation must be OPEN or CLOSE")
        _positive_int(self.contracts, "contracts")

    def order_fields(self) -> dict:
        """Order identifiers, not a full broker request (no credentials)."""
        buy = (self.position_side == "LONG") == (self.operation == "OPEN")
        return {
            "clientOid": self.client_oid,
            "symbol": self.symbol,
            "positionSide": self.position_side,
            "side": "buy" if buy else "sell",
            "size": self.contracts,
            "reduceOnly": self.operation == "CLOSE",
        }


@dataclass
class BotPosition:
    contracts: int = 0
    average_entry: Decimal = Decimal("0")
    gross_realized: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")

    @property
    def net_realized(self) -> Decimal:
        return self.gross_realized - self.fees


@dataclass
class _Pending:
    intent: BotIntent
    filled: int = 0
    cancel_confirmed: bool = False


class FamilyLedger:
    """Independent virtual LONG/SHORT books under one physical subaccount.

    Caller must serialize mutations and persist/replay intents and fills in a
    transactional durable store before enabling real orders. This in-memory
    implementation is strictly a research/test building block.
    """

    def __init__(self, family_id: str, contract_multiplier: object) -> None:
        if not family_id:
            raise FamilyLedgerError("family_id is required")
        self.family_id = family_id
        self.contract_multiplier = _amount(contract_multiplier, "contract_multiplier")
        self._positions: Dict[Tuple[str, str, str], BotPosition] = {}
        self._intents: Dict[str, _Pending] = {}
        self._fills: Dict[str, tuple] = {}

    def position(self, bot_id: str, symbol: str, side: str) -> BotPosition:
        if side not in ("LONG", "SHORT"):
            raise FamilyLedgerError("side must be LONG or SHORT")
        return replace(self._positions.get((bot_id, symbol, side), BotPosition()))

    def register(self, intent: BotIntent) -> dict:
        prior = self._intents.get(intent.client_oid)
        if prior is not None:
            if prior.intent != intent:
                raise FamilyLedgerError("clientOid reused for a different intent")
            return intent.order_fields()  # idempotent registration, NOT a resend instruction
        if intent.operation == "CLOSE":
            pos = self.position(intent.bot_id, intent.symbol, intent.position_side)
            reserved = sum(
                p.intent.contracts - p.filled
                for p in self._intents.values()
                if p.intent.operation == "CLOSE" and not p.cancel_confirmed
                and p.intent.bot_id == intent.bot_id
                and p.intent.symbol == intent.symbol
                and p.intent.position_side == intent.position_side
            )
            if intent.contracts > pos.contracts - reserved:
                raise FamilyLedgerError("close exceeds bot-owned unreserved contracts")
        self._intents[intent.client_oid] = _Pending(intent=intent)
        return intent.order_fields()

    def confirm_cancel(self, client_oid: str) -> None:
        """Call ONLY after venue confirms cancellation and all fills are settled."""
        pending = self._intents.get(client_oid)
        if pending is None:
            raise FamilyLedgerError("unknown intent")
        pending.cancel_confirmed = True

    def apply_fill(
        self, client_oid: str, fill_id: str, contracts: int,
        price: object, fee: object = "0",
    ) -> bool:
        """Apply actual executed contracts. Returns False on exact replay."""
        _positive_int(contracts, "filled contracts")
        px = _amount(price, "fill price")
        cost = _amount(fee, "fill fee", allow_zero=True)
        if not fill_id:
            raise FamilyLedgerError("fill_id is required")
        fingerprint = (client_oid, contracts, px, cost)
        if fill_id in self._fills:
            if self._fills[fill_id] != fingerprint:
                raise FamilyLedgerError("fill_id reused with different values")
            return False
        pending = self._intents.get(client_oid)
        if pending is None:
            raise FamilyLedgerError("fill for unknown intent")
        if pending.cancel_confirmed:
            raise FamilyLedgerError("fill after confirmed cancellation; manual reconciliation required")
        if pending.filled + contracts > pending.intent.contracts:
            raise FamilyLedgerError("fill exceeds intended contracts")
        req = pending.intent
        key = (req.bot_id, req.symbol, req.position_side)
        pos = self._positions.get(key, BotPosition())
        next_pos = replace(pos)
        if req.operation == "OPEN":
            qty = next_pos.contracts + contracts
            next_pos.average_entry = (
                next_pos.average_entry * next_pos.contracts + px * contracts
            ) / qty
            next_pos.contracts = qty
        else:
            if contracts > next_pos.contracts:
                raise FamilyLedgerError("fill would close another bot's contracts")
            direction = Decimal("1") if req.position_side == "LONG" else Decimal("-1")
            next_pos.gross_realized += (
                px - next_pos.average_entry
            ) * direction * contracts * self.contract_multiplier
            next_pos.contracts -= contracts
            if next_pos.contracts == 0:
                next_pos.average_entry = Decimal("0")
        next_pos.fees += cost
        self._positions[key] = next_pos
        pending.filled += contracts
        self._fills[fill_id] = fingerprint
        return True

    def reconcile(self, symbol: str, *, venue_long: int, venue_short: int) -> None:
        """Block on any discrepancy, including manual/external venue positions."""
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (venue_long, venue_short)):
            raise FamilyLedgerError("venue totals must be nonnegative contract integers")
        for side, actual in (("LONG", venue_long), ("SHORT", venue_short)):
            expected = sum(
                p.contracts for (_, sym, s), p in self._positions.items()
                if sym == symbol and s == side
            )
            if actual != expected:
                raise FamilyLedgerError(
                    f"{self.family_id} {symbol} {side} mismatch: virtual={expected}, venue={actual}"
                )
