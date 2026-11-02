"""Admin-only monthly rating/reallocation PREVIEW for Blaupunkt.

Caller-supplied mark-to-market, cashflow-adjusted monthly data only. This
does NOT collect real KuCoin NAV/fills, submit orders, or persist allocations.
The real monthly ingestion/settlement worker must be added before live use.
"""
from __future__ import annotations

from decimal import Decimal
import re
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from quant.execution.family_control_plane import _read_nodes
from quant.execution.family_ingress import _admin, _family_id
from quant.execution.family_monthly_evaluation import (
    EvaluationError,
    MonthlyMetrics,
    propose_family_rebalance,
    rate_month,
)
from quant.execution.family_tree_allocation import AllocationError, allocate_tree

router = APIRouter(tags=["family-monthly-preview"])


class MonthlyData(BaseModel):
    strategy_id: str
    monthly_net_twr: str
    max_drawdown: str
    closed_trades: int = Field(ge=0)
    observed_days: int = Field(ge=0)
    cashflows_adjusted: bool
    marked_to_market: bool
    fees_and_funding_included: bool


class MonthlyPreviewRequest(BaseModel):
    period: str
    equity: str
    metrics: list[MonthlyData] = Field(default_factory=list)


@router.post("/api/family/monthly-preview")
def monthly_preview(
    data: MonthlyPreviewRequest,
    x_family_admin_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _admin(x_family_admin_token)
    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", data.period):
        raise HTTPException(status_code=422, detail="period must be YYYY-MM")
    family = _family_id()
    nodes = _read_nodes(family)
    try:
        allocation = allocate_tree(data.equity, nodes)
        active = {n.strategy_id: n for n in nodes if n.pool == "evolution" and n.status == "shadow"}
        if len({m.strategy_id for m in data.metrics}) != len(data.metrics):
            raise EvaluationError("duplicate monthly strategy ID")
        if any(m.strategy_id not in active for m in data.metrics):
            raise EvaluationError("monthly metrics must belong to active evolution bots")
        ratings = {
            m.strategy_id: rate_month(
                MonthlyMetrics(
                    strategy_id=m.strategy_id,
                    monthly_net_twr=m.monthly_net_twr,
                    max_drawdown=m.max_drawdown,
                    closed_trades=m.closed_trades,
                    observed_days=m.observed_days,
                    cashflows_adjusted=m.cashflows_adjusted,
                    marked_to_market=m.marked_to_market,
                    fees_and_funding_included=m.fees_and_funding_included,
                )
            ) for m in data.metrics
        }
        def root_for(bot_id: str) -> str:
            visited: set[str] = set()
            cur = bot_id
            while active[cur].parent_strategy_id is not None:
                if cur in visited:
                    raise EvaluationError("cycle in family registry")
                visited.add(cur)
                cur = active[cur].parent_strategy_id
            return cur
        roots = {bot_id: root_for(bot_id) for bot_id in active}
        results = []
        for root, family_budget in allocation.family_budgets.items():
            members = [b for b in allocation.bot_budgets if roots[b] == root]
            preview = propose_family_rebalance(
                family_id=root,
                current_budgets={bot: allocation.bot_budgets[bot] for bot in members},
                ratings={bot: ratings[bot] for bot in members if bot in ratings},
            )
            if not preview.conserves_capital:
                raise EvaluationError("capital conservation failed")
            results.append({
                "parent_strategy_id": root,
                "family_budget": str(family_budget),
                "bot_targets": {bot: str(val) for bot, val in preview.proposed_budgets.items()},
                "bot_budget_deltas": {bot: str(val) for bot, val in preview.deltas.items()},
                "ratings": {
                    bot: {
                        "score": str(ratings[bot].score),
                        "credibility": str(ratings[bot].credibility),
                        "eligible": ratings[bot].eligible_for_reallocation,
                    } for bot in members if bot in ratings
                },
            })
    except (AllocationError, EvaluationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "ok": True,
        "family_id": family,
        "period": data.period,
        "source": "USER_SUPPLIED_METRICS_NOT_VENUE_VERIFIED",
        "account_equity": str(allocation.equity),
        "reserved_equity": str(allocation.reserve),
        "parent_family_count": len(allocation.family_budgets),
        "families": results,
        "orders_sent": False,
        "real_capital_transfers": False,
        "persisted": False,
        "warning": "No actual monthly settlement. Implement verified NAV/cashflow ingestion and durable entitlements before automatic live rebalance.",
    }
