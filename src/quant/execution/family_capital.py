"""Deterministic two-stage capital sizing for strategy families.

No order creation or venue interactions. The 90% utilization is an upper
limit on the combined margin budget, NOT on net notional exposure.
Callers must separately constrain leverage, available margin, open order
reserves, fees, liquidation buffer and contract minimums.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Dict, Mapping, Sequence


class CapitalAllocationError(ValueError):
    pass


def _dec(value: object, field: str) -> Decimal:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise CapitalAllocationError(f"{field} is not numeric") from None
    if not amount.is_finite():
        raise CapitalAllocationError(f"{field} must be finite")
    return amount


@dataclass(frozen=True)
class FamilyAllocation:
    parent_id: str
    family_budget: Decimal
    parent_budget: Decimal
    daughter_budgets: Dict[str, Decimal]

    @property
    def total_allocated(self) -> Decimal:
        return self.parent_budget + sum(self.daughter_budgets.values(), Decimal(0))


@dataclass(frozen=True)
class AccountAllocation:
    equity: Decimal
    tradable_equity: Decimal
    reserve: Decimal
    families: Dict[str, FamilyAllocation]

    @property
    def allocated(self) -> Decimal:
        return sum((x.total_allocated for x in self.families.values()), Decimal(0))


def allocate_account(
    equity: object,
    active_parents: Sequence[str],
    daughter_shares: Mapping[str, Mapping[str, object]] | None = None,
    *,
    utilization: object = "0.90",
) -> AccountAllocation:
    """Allocate 90% equally across active parent families.

    Daughter shares are FRACTIONS OF THEIR FAMILY budget. Example:
      {"blaupunkt": {"daughter-a": "0.02", "daughter-b": "0.02"}}
    assigns 2% of that parent's family budget to each daughter. The parent
    retains the remaining budget. This is NOT a global, rating-based split.

    Returns full Decimal precision; order size quantization must round DOWN
    to the venue's contract minimum, never round capital allocations up.
    If no active parents exist, all equity is retained as reserve.
    """
    eq = _dec(equity, "equity")
    util = _dec(utilization, "utilization")
    if eq < 0:
        raise CapitalAllocationError("equity must be nonnegative")
    if not Decimal(0) <= util <= Decimal(1):
        raise CapitalAllocationError("utilization must be between 0 and 1")

    parents = list(active_parents)
    if len(parents) != len(set(parents)) or any(
        not isinstance(x, str) or not x.strip() for x in parents
    ):
        raise CapitalAllocationError("active parent identifiers must be unique nonblank strings")
    shares_by_parent = daughter_shares or {}
    if any(p not in parents for p in shares_by_parent):
        raise CapitalAllocationError("daughter allocation refers to inactive parent")

    if not parents:
        return AccountAllocation(eq, Decimal(0), eq, {})

    tradable = eq * util
    per_family = tradable / Decimal(len(parents))
    families: Dict[str, FamilyAllocation] = {}
    for parent_id in parents:
        children = shares_by_parent.get(parent_id, {})
        if parent_id in children:
            raise CapitalAllocationError("parent cannot be its own daughter")
        fractions: Dict[str, Decimal] = {}
        for child_id, amount in children.items():
            if not isinstance(child_id, str) or not child_id.strip():
                raise CapitalAllocationError("daughter ID must be nonblank")
            value = _dec(amount, f"share of {child_id}")
            if not Decimal(0) <= value <= Decimal(1):
                raise CapitalAllocationError("daughter share must be between 0 and 1")
            fractions[child_id] = value
        summed = sum(fractions.values(), Decimal(0))
        if summed > 1:
            raise CapitalAllocationError("daughters cannot claim >100% of their family")
        daughter_budgets = {k: per_family * v for k, v in fractions.items()}
        parent_budget = per_family - sum(daughter_budgets.values(), Decimal(0))
        families[parent_id] = FamilyAllocation(
            parent_id=parent_id,
            family_budget=per_family,
            parent_budget=parent_budget,
            daughter_budgets=daughter_budgets,
        )
    reserve = eq - tradable
    return AccountAllocation(eq, tradable, reserve, families)
