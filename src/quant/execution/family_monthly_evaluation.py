"""Monthly family rating + internal capital-entitlement proposal.

NO venue API calls. Input returns MUST be mark-to-market, cashflow-adjusted,
net of trading fees AND funding. Missing or unverified inputs fail closed.
Changes in entitlements do not transfer money: all bots share one subaccount.
A later single-writer executor applies budgets ONLY to future entries after
reconciliation and sufficient available margin; it never auto-closes positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from math import exp, isfinite
from typing import Mapping, Optional


class EvaluationError(ValueError):
    pass


def _d(raw: object, name: str) -> Decimal:
    try:
        val = Decimal(str(raw))
    except (ValueError, TypeError, InvalidOperation):
        raise EvaluationError(f"{name} is invalid") from None
    if not val.is_finite():
        raise EvaluationError(f"{name} must be finite")
    return val


@dataclass(frozen=True)
class MonthlyMetrics:
    strategy_id: str
    # Example: 0.05 is a +5% return for the month, -0.1 a -10% month.
    monthly_net_twr: object
    max_drawdown: object
    closed_trades: int
    observed_days: int
    cashflows_adjusted: bool = True
    marked_to_market: bool = True
    fees_and_funding_included: bool = True


@dataclass(frozen=True)
class Rating:
    strategy_id: str
    score: Decimal
    credibility: Decimal
    eligible_for_reallocation: bool


@dataclass(frozen=True)
class MonthlyRebalanceProposal:
    family_id: str
    family_budget: Decimal
    ratings: dict[str, Rating]
    current_budgets: dict[str, Decimal]
    proposed_budgets: dict[str, Decimal]
    deltas: dict[str, Decimal]
    execution: str = "PREVIEW_ONLY_NO_ORDERS_NO_TRANSFERS"

    @property
    def conserves_capital(self) -> bool:
        return (
            sum(self.current_budgets.values(), Decimal(0))
            == self.family_budget
            == sum(self.proposed_budgets.values(), Decimal(0))
        )


def rate_month(
    m: MonthlyMetrics,
    *,
    min_trades: int = 15,
    min_days: int = 30,
    sample_prior_trades: int = 20,
    drawdown_penalty: object = "0.35",
) -> Rating:
    """Score in percentage points: net TWR is primary, drawdown penalized.

    Credibility shrinks the return component toward zero while the bot has
    few trades. Ratings are NOT statistical significance or a profit promise.
    """
    r = _d(m.monthly_net_twr, "monthly net TWR")
    dd = _d(m.max_drawdown, "max drawdown")
    penalty = _d(drawdown_penalty, "drawdown penalty")
    if not m.strategy_id or not m.strategy_id.strip():
        raise EvaluationError("strategy ID required")
    if r < -1 or not 0 <= dd <= 1 or not 0 <= penalty <= 10:
        raise EvaluationError("monthly return, drawdown or penalty out of range")
    for name, v in (("closed_trades", m.closed_trades),
                    ("observed_days", m.observed_days)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise EvaluationError(f"{name} must be a nonnegative integer")
    if min_trades < 1 or min_days < 1 or sample_prior_trades < 1:
        raise EvaluationError("rating sample parameters must be positive")
    if not (m.cashflows_adjusted and m.marked_to_market and m.fees_and_funding_included):
        raise EvaluationError(
            "monthly TWR must include external cashflows, open PnL, fees and funding"
        )
    credibility = Decimal(m.closed_trades) / Decimal(m.closed_trades + sample_prior_trades)
    # Cap the influence of a single extreme month without declaring it false.
    clipped_return = min(max(r, Decimal("-1")), Decimal("2"))
    score = Decimal(100) * (clipped_return * credibility - penalty * dd)
    eligible = m.closed_trades >= min_trades and m.observed_days >= min_days
    return Rating(m.strategy_id, score, credibility, eligible)


def propose_family_rebalance(
    *,
    family_id: str,
    current_budgets: Mapping[str, object],
    ratings: Mapping[str, Rating],
    temperature_points: object = "8",
    monthly_adjustment_fraction: object = "0.25",
) -> MonthlyRebalanceProposal:
    """Reallocate ONLY within one root family, never among root families.

    Unrated/ineligible bots retain their existing budget (e.g. the initial
    2% daughter seed). Eligible bots share the rest by softmax score, with a
    capped monthly adjustment step to avoid a sharp switch after one month.

    This is an AUDITABLE proposal only, not an order/funding instruction.
    """
    if not family_id:
        raise EvaluationError("family_id required")
    temp = _d(temperature_points, "temperature")
    step = _d(monthly_adjustment_fraction, "monthly adjustment")
    if temp <= 0 or not 0 <= step <= 1:
        raise EvaluationError("temperature must be positive; adjustment in [0,1]")
    initial: dict[str, Decimal] = {}
    for bot, raw in current_budgets.items():
        if not bot or bot in initial:
            raise EvaluationError("strategy IDs must be unique and nonblank")
        budget = _d(raw, "bot budget")
        if budget < 0:
            raise EvaluationError("bot budget must not be negative")
        initial[bot] = budget
    if not initial:
        raise EvaluationError("family must include at least one bot")
    if any(id not in initial or id != rating.strategy_id for id, rating in ratings.items()):
        raise EvaluationError("rating must belong to a registered family strategy")
    total = sum(initial.values(), Decimal(0))
    target = dict(initial)
    eligible_ids = [bot for bot in initial if bot in ratings and ratings[bot].eligible_for_reallocation]
    if len(eligible_ids) > 1 and step > 0:
        eligible_capital = sum((initial[id] for id in eligible_ids), Decimal(0))
        adjusted = {id: float(ratings[id].score / temp) for id in eligible_ids}
        mx = max(adjusted.values())
        weights = {id: exp(adjusted[id] - mx) for id in eligible_ids}
        divisor = sum(weights.values())
        if divisor <= 0 or not isfinite(divisor):
            raise EvaluationError("unusable rating weights")
        # Assign the last eligible bot as residual to preserve every cent.
        remainder = eligible_capital
        for id in eligible_ids[:-1]:
            desired = eligible_capital * Decimal(str(weights[id] / divisor))
            target[id] = initial[id] * (Decimal(1) - step) + desired * step
            remainder -= target[id]
        target[eligible_ids[-1]] = remainder
    deltas = {id: target[id] - initial[id] for id in initial}
    out = MonthlyRebalanceProposal(family_id, total, dict(ratings), initial, target, deltas)
    if not out.conserves_capital or any(value < 0 for value in target.values()):
        raise EvaluationError("proposed budget violates conservation/nonnegative constraints")
    return out
