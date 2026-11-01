"""Generative 'evolution meadow' mutation planner, DRY / PROPOSAL ONLY.

Generate 5-15 new parameter candidates for each credible profitable bot.
More robust excess performance produces more seeds and a broader cone.
Positions are sampled uniformly over a 2-D disk via sqrt(U), with
stratified angular sectors to avoid directional clumping. Parameter
axes are normalized independently so TTP and sensitivity are equally
represented despite their different units.

NO TradingView, KuCoin, Railway API, deployment, or order calls.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Collection

from quant.execution.family_monthly_evaluation import MonthlyMetrics, rate_month


class MutationError(ValueError):
    pass


@dataclass(frozen=True)
class MutationBounds:
    sensitivity_min: float = 12.0
    sensitivity_max: float = 25.0
    sensitivity_step: float = 0.1
    ttp_min_pct: float = 0.25
    ttp_max_pct: float = 1.65
    ttp_step_pct: float = 0.05


@dataclass(frozen=True)
class CandidateSeed:
    strategy_id: str
    parent_strategy_id: str
    month: str
    sensitivity: float
    ttp_pct: float
    generation_proposal: bool = True
    signal_source: str = "TRADINGVIEW_MANUAL_ALERT_REQUIRED"
    execution_enabled: bool = False


@dataclass(frozen=True)
class SowingPlan:
    parent_strategy_id: str
    month: str
    score: Decimal
    seed_count_requested: int
    radius_normalized: float
    proposals: tuple[CandidateSeed, ...]
    status: str = "PROPOSAL_ONLY"


def _quantized(value: float, minimum: float, maximum: float, step: float) -> float:
    increments = Decimal(str((value - minimum) / step)).to_integral_value(rounding=ROUND_HALF_UP)
    q = float(Decimal(str(minimum)) + increments * Decimal(str(step)))
    return max(minimum, min(maximum, q))


def _valid_bounds(bounds: MutationBounds) -> None:
    nums = (
        bounds.sensitivity_min, bounds.sensitivity_max, bounds.sensitivity_step,
        bounds.ttp_min_pct, bounds.ttp_max_pct, bounds.ttp_step_pct,
    )
    if any(not math.isfinite(x) for x in nums):
        raise MutationError("bounds must be finite")
    if (bounds.sensitivity_min >= bounds.sensitivity_max or
        bounds.ttp_min_pct >= bounds.ttp_max_pct or
        bounds.sensitivity_step <= 0 or bounds.ttp_step_pct <= 0):
        raise MutationError("invalid parameter bounds")


def propose_sowing(
    parent_strategy_id: str,
    *,
    sensitivity: float,
    ttp_pct: float,
    metrics: MonthlyMetrics,
    month: str,
    known_configs: Collection[tuple[float, float]] = (),
    bounds: MutationBounds = MutationBounds(),
) -> SowingPlan:
    """Candidates only; parent live positions and capital remain unchanged.

    A bot must have >5% VERIFIED monthly net return, at least 15 closed
    trades and 30 observed days to reproduce. The prior size=6 policy is
    deliberately not modified in production; this is the new suggested
    variable 5–15 seed policy for future user approval.
    """
    _valid_bounds(bounds)
    if not parent_strategy_id or metrics.strategy_id != parent_strategy_id:
        raise MutationError("metrics identity must equal parent strategy ID")
    if len(month) != 7 or month[4] != "-" or not month[:4].isdigit() or not month[5:].isdigit() or not 1 <= int(month[5:]) <= 12:
        raise MutationError("month must be YYYY-MM")
    if not math.isfinite(sensitivity) or not math.isfinite(ttp_pct):
        raise MutationError("strategy parameters must be finite")
    if not (bounds.sensitivity_min <= sensitivity <= bounds.sensitivity_max and
            bounds.ttp_min_pct <= ttp_pct <= bounds.ttp_max_pct):
        raise MutationError("parent strategy parameters outside allowed rectangle")
    rating = rate_month(metrics)
    net = Decimal(str(metrics.monthly_net_twr))
    if not rating.eligible_for_reallocation or net <= Decimal("0.05") or rating.score <= 0:
        return SowingPlan(parent_strategy_id, month, rating.score, 0, 0.0, ())
    strength = max(0.0, min(1.0, (float(rating.score) - 1) / 20))
    count = max(5, min(15, 5 + round(10 * strength)))
    radius = 0.036 + 0.115 * strength

    digest = hashlib.sha256(
        (parent_strategy_id + "|" + month + "|v1").encode("utf-8")
    ).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    existing = {
        (_quantized(x, bounds.sensitivity_min, bounds.sensitivity_max, bounds.sensitivity_step),
         _quantized(y, bounds.ttp_min_pct, bounds.ttp_max_pct, bounds.ttp_step_pct))
        for x, y in known_configs
    }
    existing.add((
        _quantized(sensitivity, bounds.sensitivity_min, bounds.sensitivity_max, bounds.sensitivity_step),
        _quantized(ttp_pct, bounds.ttp_min_pct, bounds.ttp_max_pct, bounds.ttp_step_pct),
    ))
    seeds: list[CandidateSeed] = []
    for i in range(count):
        # Several retries in the SAME angular sector are allowed to avoid
        # overlapping/duplicate quantized parameter combinations.
        for _attempt in range(24):
            theta = 2 * math.pi * (i + 0.5 + (rng.random() - 0.5) * 0.65) / count
            r = radius * math.sqrt(rng.random())
            sx = sensitivity + math.cos(theta) * r * (bounds.sensitivity_max - bounds.sensitivity_min)
            ty = ttp_pct + math.sin(theta) * r * (bounds.ttp_max_pct - bounds.ttp_min_pct)
            x = _quantized(sx, bounds.sensitivity_min, bounds.sensitivity_max, bounds.sensitivity_step)
            y = _quantized(ty, bounds.ttp_min_pct, bounds.ttp_max_pct, bounds.ttp_step_pct)
            if (x, y) not in existing:
                existing.add((x, y))
                suffix = hashlib.sha256((parent_strategy_id + "|" + month).encode()).hexdigest()[:6]
                strategy_id = parent_strategy_id[:32] + "-g" + month.replace("-", "") + "-" + suffix + "-" + str(i + 1).zfill(2)
                seeds.append(CandidateSeed(strategy_id, parent_strategy_id, month, x, y))
                break
    return SowingPlan(
        parent_strategy_id, month, rating.score, count, radius, tuple(seeds),
    )
