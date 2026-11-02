"""Hierarchical equity allocation for a KuCoin strategy-family subaccount.

Pure calculation: never sends orders, transfers funds, or changes positions.
Every active ROOT parent receives the same share of 90% of account equity.
Every active child begins with 2% of its *direct* parent's budget; the parent
retains the remaining 98% per child. Descendants do not increase root count.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Dict, Iterable, Optional, Tuple


class AllocationError(ValueError):
    pass


@dataclass(frozen=True)
class StrategyNode:
    strategy_id: str
    parent_strategy_id: Optional[str] = None
    pool: str = "evolution"
    status: str = "shadow"


@dataclass(frozen=True)
class AllocationResult:
    equity: Decimal
    usable_budget: Decimal
    reserve: Decimal
    family_budgets: Dict[str, Decimal]
    bot_budgets: Dict[str, Decimal]

    def audit(self) -> bool:
        return sum(self.bot_budgets.values(), Decimal(0)) + self.reserve == self.equity


def _decimal(value: object, name: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (ValueError, TypeError, InvalidOperation):
        raise AllocationError(f"invalid {name}") from None
    if not number.is_finite():
        raise AllocationError(f"{name} must be finite")
    return number


def allocate_tree(
    equity: object,
    nodes: Iterable[StrategyNode],
    *,
    utilization: object = "0.90",
    child_share: object = "0.02",
) -> AllocationResult:
    eq, util, share = (
        _decimal(equity, "equity"),
        _decimal(utilization, "utilization"),
        _decimal(child_share, "child_share"),
    )
    if eq < 0 or not 0 <= util <= 1 or not 0 <= share <= 1:
        raise AllocationError("equity must be nonnegative; shares in [0, 1]")

    active: Dict[str, StrategyNode] = {}
    for node in nodes:
        if not node.strategy_id or not node.strategy_id.strip():
            raise AllocationError("blank strategy_id")
        if node.strategy_id in active:
            raise AllocationError(f"duplicate strategy_id {node.strategy_id}")
        if node.pool not in ("evolution", "angel") or node.status not in ("shadow", "disabled"):
            raise AllocationError(f"invalid pool or status for {node.strategy_id}")
        active[node.strategy_id] = node

    # Exclude angel/fishing bots completely from the evolution capital pool.
    evolution = {
        k: n for k, n in active.items()
        if n.pool == "evolution" and n.status == "shadow"
    }
    children: Dict[str, list[str]] = {id: [] for id in evolution}
    roots: list[str] = []
    for sid, node in evolution.items():
        parent = node.parent_strategy_id
        if parent is None:
            roots.append(sid)
        elif parent not in evolution:
            raise AllocationError(f"missing or inactive parent for {sid}: {parent}")
        else:
            children[parent].append(sid)

    # No node can hide in a loop and disappear from the allocation.
    visited: set[str] = set()
    visiting: set[str] = set()

    def validate(node: str) -> None:
        if node in visiting:
            raise AllocationError(f"cycle detected at {node}")
        if node in visited:
            return
        visiting.add(node)
        for child in children[node]:
            validate(child)
        visiting.remove(node)
        visited.add(node)

    for sid in evolution:
        validate(sid)
    for parent, daughters in children.items():
        if Decimal(len(daughters)) * share > 1:
            raise AllocationError(f"children of {parent} claim more than 100%")

    if not roots:
        if evolution:
            raise AllocationError("evolution bots without an active root")
        return AllocationResult(eq, Decimal(0), eq, {}, {})

    total = eq * util
    each_family = total / Decimal(len(roots))
    family_budgets = {sid: each_family for sid in roots}
    budgets: Dict[str, Decimal] = {}

    def allocate_branch(sid: str, assigned: Decimal) -> None:
        direct_daughters = children[sid]
        each_child = assigned * share
        budgets[sid] = assigned - each_child * len(direct_daughters)
        for child in direct_daughters:
            allocate_branch(child, each_child)

    for root in roots:
        allocate_branch(root, each_family)

    # The ideal theoretical allocation can be 1/3 cent. Do not drop or
    # round parts: Decimal precision is preserved, order sizing rounds DOWN.
    unassigned_rounding = eq - sum(budgets.values(), Decimal(0))
    # Decimal / 3 can generate tiny truncation. Put that amount in reserve.
    reserve = unassigned_rounding
    result = AllocationResult(eq, total, reserve, family_budgets, budgets)
    if not result.audit():
        raise AllocationError("capital conservation check failed")
    return result
