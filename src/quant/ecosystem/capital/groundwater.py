"""Paper-only Groundwater allocator: all NAV is water; 90% budgets, 10% reserve.

No venue access, payments, position closing, or currency conversion.
Supports conservative per-account isolation and exact Decimal arithmetic.
Allocation is a *future-order entitlement*, not occupied margin/exposure.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import Iterable, Sequence


RATIO = Decimal("0.90")
ZERO = Decimal("0")


class GroundwaterError(ValueError):
    """Rejected or unreconciled allocation input."""


def dec(value: str | Decimal, name: str) -> Decimal:
    if not isinstance(value, (str, Decimal)) or isinstance(value, bool):
        raise GroundwaterError(f"{name} must be a decimal string/Decimal")
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError):
        raise GroundwaterError(f"{name} is not a decimal") from None
    if not result.is_finite():
        raise GroundwaterError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class Account:
    account_id: str
    currency: str
    equity: Decimal
    free_margin: Decimal

    def __post_init__(self):
        if not self.account_id.strip() or not self.currency.strip():
            raise GroundwaterError("account and currency are mandatory")
        eq, free = dec(self.equity, "account equity"), dec(self.free_margin, "free margin")
        if free < ZERO or free > max(ZERO, eq):
            raise GroundwaterError("free margin must be within positive account NAV")
        object.__setattr__(self, "equity", eq)
        object.__setattr__(self, "free_margin", free)


@dataclass(frozen=True)
class Demand:
    strategy_id: str
    account_id: str
    weight: Decimal
    max_entitlement_fraction: Decimal = Decimal("1")
    used_margin: Decimal = ZERO
    reserved_order_margin: Decimal = ZERO
    eligible: bool = True

    def __post_init__(self):
        if not self.strategy_id.strip() or not self.account_id.strip():
            raise GroundwaterError("strategy and account IDs are mandatory")
        values = {
            "weight": self.weight,
            "max_entitlement_fraction": self.max_entitlement_fraction,
            "used_margin": self.used_margin,
            "reserved_order_margin": self.reserved_order_margin,
        }
        for name, value in values.items():
            number = dec(value, name)
            if number < ZERO:
                raise GroundwaterError(f"{name} cannot be negative")
            object.__setattr__(self, name, number)
        if self.max_entitlement_fraction > Decimal("1"):
            raise GroundwaterError("strategy max_entitlement_fraction > 1")


@dataclass(frozen=True)
class Entitlement:
    strategy_id: str
    account_id: str
    amount: Decimal
    used_margin: Decimal
    reserved_order_margin: Decimal
    entry_capacity: Decimal

    @property
    def locked(self) -> Decimal:
        return self.used_margin + self.reserved_order_margin


@dataclass(frozen=True)
class GroundwaterPlan:
    equity: Decimal
    reserve: Decimal
    distributable: Decimal
    currency: str
    entitlements: tuple[Entitlement, ...]
    unallocated: Decimal
    allocations_frozen: bool
    risk_breach: bool

    def by_strategy(self, strategy_id: str) -> Entitlement:
        for entitlement in self.entitlements:
            if entitlement.strategy_id == strategy_id:
                return entitlement
        raise GroundwaterError(f"unknown strategy {strategy_id}")


def _waterfill(
    budget: Decimal, claims: Sequence[Demand]
) -> dict[str, Decimal]:
    """Fill mandatory locked margin, then weight residual without exceeding caps.

    In a drawdown, locked positions are preserved as observations; a breach
    is flagged by the caller rather than pretending existing orders vanished.
    """
    balances = {d.strategy_id: d.used_margin + d.reserved_order_margin for d in claims}
    locked = sum(balances.values(), ZERO)
    if locked >= budget:
        # This account is frozen on a shortfall. Entitlement figures are
        # pro-rata *limits*, not instructions to flatten locked positions.
        if locked:
            rem = budget
            for d in claims[:-1]:
                balances[d.strategy_id] = budget * balances[d.strategy_id] / locked
                rem -= balances[d.strategy_id]
            balances[claims[-1].strategy_id] = rem
        return balances

    remaining = budget - locked
    active = [
        d for d in claims
        if d.eligible and d.weight > ZERO
        and balances[d.strategy_id] < budget * d.max_entitlement_fraction
    ]
    # Iterative water-filling: unused capacity of capped plants goes to others.
    while remaining > ZERO and active:
        total_weight = sum((d.weight for d in active), ZERO)
        round_budget = remaining
        unspent = remaining
        for d in active:
            headroom = max(ZERO, budget * d.max_entitlement_fraction - balances[d.strategy_id])
            # Decimal rounding may make the independent pro-rata shares sum
            # to slightly more than the available budget. Never overspend.
            share = min(headroom, round_budget * d.weight / total_weight, unspent)
            balances[d.strategy_id] += share
            unspent -= share
        if unspent >= remaining:
            break
        remaining = unspent
        active = [
            d for d in active
            if balances[d.strategy_id] < budget * d.max_entitlement_fraction
        ]
        # Decimal division may leave tiny precision residuals; do not loop forever.
        if remaining < Decimal("1e-20"):
            break
    return balances


def _with_entry_capacities(
    entitlements: Sequence[Entitlement], accounts: Sequence[Account], frozen: bool
) -> tuple[Entitlement, ...]:
    """Distribute observed *free margin*, never count it once per bot."""
    free_by_account = {a.account_id: a.free_margin for a in accounts}
    result: list[Entitlement] = []
    for account in sorted(accounts, key=lambda a: a.account_id):
        entries = [e for e in entitlements if e.account_id == account.account_id]
        headrooms = {e.strategy_id: max(ZERO, e.amount - e.locked) for e in entries}
        aggregate = sum(headrooms.values(), ZERO)
        total_free = ZERO if frozen else min(aggregate, free_by_account[account.account_id])
        remaining = total_free
        eligible = [e for e in entries if headrooms[e.strategy_id] > ZERO]
        for i, e in enumerate(eligible):
            allowance = (
                remaining if i == len(eligible) - 1
                else total_free * headrooms[e.strategy_id] / aggregate
            )
            allowance = min(allowance, headrooms[e.strategy_id], remaining)
            remaining -= allowance
            result.append(replace(e, entry_capacity=allowance))
        result.extend(replace(e, entry_capacity=ZERO) for e in entries if headrooms[e.strategy_id] == ZERO)
    return tuple(sorted(result, key=lambda e: e.strategy_id))


def allocate_groundwater(
    accounts: Iterable[Account], demands: Iterable[Demand]
) -> GroundwaterPlan:
    """Propose *only* virtual budgets. No order intent is generated."""
    accts = tuple(sorted(accounts, key=lambda a: a.account_id))
    desires = tuple(sorted(demands, key=lambda d: d.strategy_id))
    if not accts:
        raise GroundwaterError("at least one account required")
    if len({a.account_id for a in accts}) != len(accts):
        raise GroundwaterError("duplicate account, would count NAV twice")
    if len({d.strategy_id for d in desires}) != len(desires):
        raise GroundwaterError("same strategy cannot have two capital entitlements")
    currencies = {a.currency for a in accts}
    if len(currencies) != 1:
        raise GroundwaterError("unreconciled currencies: FX conversion required")
    account_ids = {a.account_id for a in accts}
    if any(d.account_id not in account_ids for d in desires):
        raise GroundwaterError("demand references unknown account")

    equity = sum((a.equity for a in accts), ZERO)
    reserve = max(ZERO, equity) * (Decimal("1") - RATIO)
    available = max(ZERO, equity) * RATIO
    risk_breach = any(a.equity < ZERO for a in accts)
    # Distressed/negative accounts reduce the portfolio's net NAV. Do not
    # allocate 90% of gross positive NAV and accidentally spend the reserve.
    gross_positive_budget = sum((max(ZERO, a.equity) * RATIO for a in accts), ZERO)
    scale = min(Decimal("1"), available / gross_positive_budget) if gross_positive_budget else ZERO
    claims: list[Entitlement] = []
    for a in accts:
        own = [d for d in desires if d.account_id == a.account_id]
        budget = max(ZERO, a.equity) * RATIO * scale
        if not own:
            continue
        balances = _waterfill(budget, own)
        for d in own:
            balance = balances[d.strategy_id]
            locked = d.used_margin + d.reserved_order_margin
            if locked > balance or locked > budget * d.max_entitlement_fraction:
                risk_breach = True
            claims.append(Entitlement(d.strategy_id, d.account_id, balance,
                                      d.used_margin, d.reserved_order_margin, ZERO))
    allocated = sum((e.amount for e in claims), ZERO)
    if allocated > available:
        # A Decimal context's finite precision can still round the final
        # per-account additions upwards by a tiny ULP. Trim only demonstrably
        # microscopic excess from an uncommitted entitlement. Any material
        # overspend remains a hard fail.
        excess = allocated - available
        tolerance = max(Decimal("1e-22"), available * Decimal("1e-22"))
        if excess > tolerance:
            raise GroundwaterError("allocator bug: reserve would be spent")
        for index in range(len(claims)-1, -1, -1):
            claim = claims[index]
            if claim.amount - claim.locked >= excess:
                claims[index] = replace(claim, amount=claim.amount-excess)
                break
        allocated = sum((e.amount for e in claims), ZERO)
        if allocated > available:
            raise GroundwaterError("allocator bug: reserve would be spent")
    result = _with_entry_capacities(claims, accts, frozen=risk_breach)
    return GroundwaterPlan(
        equity, reserve, available, next(iter(currencies)), result,
        available - allocated, risk_breach, risk_breach,
    )


def split_seed_budget(
    plan: GroundwaterPlan, parent_id: str, child_id: str, share: str | Decimal,
    account_free_margin: str | Decimal,
) -> GroundwaterPlan:
    """Fund one *simulated* child from currently uncommitted parent entitlement.

    The caller must use an immutable unique child_id to make this idempotent
    at the event-log level; duplicate births are always rejected here.
    """
    fraction = dec(share, "reproduction share")
    if not ZERO < fraction <= Decimal("1"):
        raise GroundwaterError("seed share must be in (0,1]")
    if plan.allocations_frozen:
        raise GroundwaterError("no new seeds while allocation risk-frozen")
    if any(e.strategy_id == child_id for e in plan.entitlements):
        raise GroundwaterError("duplicate child / repeated seed event")
    parent = plan.by_strategy(parent_id)
    newborn = parent.amount * fraction
    if newborn > parent.amount - parent.locked:
        raise GroundwaterError("cannot take occupied margin to fund a seed")
    modified = [
        replace(e, amount=e.amount - newborn)
        if e.strategy_id == parent_id else e for e in plan.entitlements
    ]
    modified.append(Entitlement(child_id, parent.account_id, newborn, ZERO, ZERO, ZERO))
    free = dec(account_free_margin, "account free margin")
    if free < ZERO:
        raise GroundwaterError("negative free margin")
    # Other accounts: retain their published capacity; only parent account
    # recalculates current free-margin shares. This remains proposal-only.
    parent_entries = [e for e in modified if e.account_id == parent.account_id]
    on_account = _with_entry_capacities(
        parent_entries,
        (Account(parent.account_id, plan.currency,
                 max(plan.equity, free), free),),
        frozen=False,
    )
    by_id = {e.strategy_id: e for e in on_account}
    rest = tuple(sorted((by_id.get(e.strategy_id, e) for e in modified),
                        key=lambda e: e.strategy_id))
    return replace(plan, entitlements=rest)
