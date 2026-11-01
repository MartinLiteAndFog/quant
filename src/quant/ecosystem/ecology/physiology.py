"""Paper-only daily plant physiology: groundwater, roots, branches, leaves and cactus rind.

Groundwater is global NAV, not an extra wallet. This module computes *proposed*
virtual utilization and exposure; it never trades, creates money, or closes positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, time, timezone
from decimal import Decimal, InvalidOperation
from math import exp, isfinite, sqrt
from typing import Sequence

ZERO = Decimal("0")
ONE = Decimal("1")


class EcologyError(ValueError):
    pass


def _unit(value: float, name: str) -> float:
    if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
        raise EcologyError(f"{name} must be a finite fraction in [0,1]")
    return float(value)


def _positive(value: float, name: str) -> float:
    if type(value) not in (int, float) or not isfinite(value) or value <= 0:
        raise EcologyError(f"{name} must be positive/finite")
    return float(value)


def _money(value: str | Decimal, name: str, *, signed: bool = False) -> Decimal:
    if not isinstance(value, (str, Decimal)):
        raise EcologyError(f"{name} must be a decimal string/Decimal")
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError):
        raise EcologyError(f"{name} invalid") from None
    if not result.is_finite() or (result < ZERO and not signed):
        raise EcologyError(f"{name} negative/nonfinite")
    return result


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class PlantGenome:
    root_growth: float
    root_depth: float
    root_spread: float
    branching: float
    leaf_count: int
    leaf_size: float
    bark_strength: float
    dormancy: float
    photosynthetic_rind: bool
    rind_efficiency: float
    reproduction_budget: float
    protective_stop: bool
    max_leverage: Decimal
    max_margin_fraction: float

    def __post_init__(self):
        for name in ("root_growth", "root_depth", "root_spread", "branching",
                     "leaf_size", "bark_strength", "dormancy", "rind_efficiency",
                     "reproduction_budget", "max_margin_fraction"):
            _unit(getattr(self, name), name)
        if type(self.leaf_count) is not int or not 1 <= self.leaf_count <= 64:
            raise EcologyError("leaf_count must be 1..64")
        if type(self.photosynthetic_rind) is not bool or self.protective_stop is not True:
            raise EcologyError("photosynthetic_rind bool and mandatory protective_stop required")
        leverage = _money(self.max_leverage, "max leverage")
        if leverage < ONE:
            raise EcologyError("max_leverage below one")
        object.__setattr__(self, "max_leverage", leverage)


@dataclass(frozen=True)
class Plant:
    strategy_id: str
    age_days: int
    biomass: float
    root_depth: float
    root_spread: float
    leaf_area: float
    vitality: float
    water_access: bool = False

    def __post_init__(self):
        if not self.strategy_id or type(self.age_days) is not int or self.age_days < 0:
            raise EcologyError("invalid plant ID/age")
        for name in ("biomass", "root_depth", "root_spread", "leaf_area"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not isfinite(value) or value < 0:
                raise EcologyError(f"invalid plant {name}")
        if self.biomass == 0:
            raise EcologyError("living plants need positive biomass")
        _unit(self.vitality, "vitality")


@dataclass(frozen=True)
class PlantInput:
    plant: Plant
    genome: PlantGenome
    daily_net_return: float
    entitlement: Decimal
    bonus_qualified: bool
    shadow: float = 0

    def __post_init__(self):
        if type(self.daily_net_return) not in (float, int) or not isfinite(self.daily_net_return) or not -1 <= self.daily_net_return <= 10:
            raise EcologyError("invalid daily net return")
        _unit(self.shadow, "shadow")
        if type(self.bonus_qualified) is not bool:
            raise EcologyError("bonus_qualified must be boolean")
        object.__setattr__(self, "entitlement", _money(self.entitlement, "entitlement"))


@dataclass(frozen=True)
class Weather:
    observed_at: datetime
    known_at: datetime
    sun: float
    synthetic: bool

    def __post_init__(self):
        if type(self.observed_at) is not datetime or type(self.known_at) is not datetime:
            raise EcologyError("weather needs timestamps")
        if self.observed_at.tzinfo is None or self.known_at.tzinfo is None:
            raise EcologyError("weather must be time-zone aware")
        if self.observed_at.utcoffset() is None or self.known_at.utcoffset() is None:
            raise EcologyError("weather needs valid UTC offsets")
        if self.known_at < self.observed_at:
            raise EcologyError("weather known before observation")
        if type(self.synthetic) is not bool:
            raise EcologyError("weather synthetic marker required")
        _unit(self.sun, "sun")


@dataclass(frozen=True)
class EcologyConfig:
    reference_equity: Decimal = Decimal("100")
    min_water_depth: float = 0.02
    max_water_depth: float = 8.0
    water_sensitivity: float = 0.6
    rise_speed: float = 0.05
    fall_speed: float = 0.10
    hysteresis_depth: float = 0.15
    root_energy_rate: float = 0.08
    root_opportunity_cost: float = 0.50
    daily_growth_cap: float = 0.03
    daily_shrink_cap: float = 0.04
    global_leverage_cap: Decimal = Decimal("10")
    global_margin_fraction_cap: float = 1.0
    max_water_access_fraction: float | None = None

    def __post_init__(self):
        reference = _money(self.reference_equity, "reference_equity")
        leverage = _money(self.global_leverage_cap, "global_leverage_cap")
        if reference <= ZERO or leverage < ONE:
            raise EcologyError("reference equity must be >0 and leverage >=1")
        object.__setattr__(self, "reference_equity", reference)
        object.__setattr__(self, "global_leverage_cap", leverage)
        _positive(self.min_water_depth, "min_water_depth")
        _positive(self.max_water_depth, "max_water_depth")
        if self.min_water_depth >= self.max_water_depth:
            raise EcologyError("min_water_depth must be lower than max_water_depth")
        for name in ("water_sensitivity", "hysteresis_depth", "root_energy_rate"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not isfinite(value) or value < 0:
                raise EcologyError(f"invalid {name}")
        for name in ("rise_speed", "fall_speed", "root_opportunity_cost",
                     "daily_growth_cap", "daily_shrink_cap", "global_margin_fraction_cap"):
            _unit(getattr(self, name), name)
        if self.max_water_access_fraction is not None:
            _unit(self.max_water_access_fraction, "max_water_access_fraction")


@dataclass(frozen=True)
class GroundwaterLine:
    day: date
    equity: Decimal
    depth: float
    target_depth: float

    def __post_init__(self):
        _money(self.equity, "equity", signed=True)
        _positive(self.depth, "depth")
        _positive(self.target_depth, "target depth")


@dataclass(frozen=True)
class PlantOutput:
    plant: Plant
    water_eligible: bool
    root_effort: float
    effective_leaf_area: float
    own_utilization: float
    intended_margin: Decimal
    effective_leverage: Decimal
    intended_notional: Decimal


@dataclass(frozen=True)
class DailyEcology:
    groundwater: GroundwaterLine
    plants: tuple[PlantOutput, ...]
    reserve: Decimal
    distributable: Decimal
    unassigned_budget: Decimal


def water_target(equity: str | Decimal, cfg: EcologyConfig) -> float:
    """Equity increases => depth decreases, always stays strictly below soil."""
    nav = max(ZERO, _money(equity, "NAV", signed=True))
    ratio = min(nav / cfg.reference_equity, Decimal("10000"))
    return cfg.min_water_depth + (cfg.max_water_depth - cfg.min_water_depth) * exp(
        -cfg.water_sensitivity * float(ratio)
    )


def initial_groundwater(first_day: date, equity: str | Decimal, cfg: EcologyConfig) -> GroundwaterLine:
    if type(first_day) is not date:
        raise EcologyError("first_day must be a date")
    nav = _money(equity, "NAV", signed=True)
    target = water_target(nav, cfg)
    return GroundwaterLine(first_day - timedelta(days=1), nav, target, target)


def step_groundwater(previous: GroundwaterLine, day: date, equity: str | Decimal, cfg: EcologyConfig) -> GroundwaterLine:
    if type(day) is not date or day != previous.day + timedelta(days=1):
        raise EcologyError("groundwater must advance exactly one UTC day")
    nav = _money(equity, "NAV", signed=True)
    target = water_target(nav, cfg)
    speed = cfg.rise_speed if target < previous.depth else cfg.fall_speed
    depth = previous.depth + speed * (target - previous.depth)
    depth = _clip(depth, cfg.min_water_depth, cfg.max_water_depth)
    return GroundwaterLine(day, nav, depth, target)


def _plant_step(inp: PlantInput, ground: GroundwaterLine, weather: Weather, cfg: EcologyConfig
                ) -> PlantOutput:
    p, g = inp.plant, inp.genome
    # Both deep and spreading roots can have high genes, then compete more
    # strongly with above-ground leaves for *growth effort*. No NAV consumed.
    root_effort = g.root_growth * (g.root_depth + g.root_spread) / 2.0
    energy = cfg.root_energy_rate * sqrt(p.biomass) * (0.2 + 0.8 * p.vitality)
    depth_gain = energy * g.root_growth * g.root_depth
    spread_gain = energy * g.root_growth * g.root_spread
    roots_depth = p.root_depth + depth_gain
    roots_spread = p.root_spread + spread_gain

    growth = _clip(inp.daily_net_return * (1 - 0.5 * root_effort),
                   -cfg.daily_shrink_cap, cfg.daily_growth_cap)
    new_biomass = max(0.001, p.biomass * (1 + growth))
    # Functional phenotype: more branches and more leaves increase total
    # projected foliage, but cannot bypass hard global exposure constraints.
    stem_leaf_area = (new_biomass ** (2.0/3.0)) * g.leaf_size * (
        1 + 2 * g.branching
    ) * g.leaf_count / 64.0
    # Cactus genetic rule: leaf_count >= 1. At exactly one leaf, a green
    # photosynthetic rind/stem may serve as leaf area instead of a crown.
    # Armor strength and green photosynthesis are DISTINCT functions.
    rind_area = ((new_biomass ** (2.0/3.0)) * g.rind_efficiency * 0.22
                 if g.photosynthetic_rind and g.leaf_count == 1 else 0.0)
    effective_area = max(stem_leaf_area, rind_area)
    foliage = p.leaf_area * 0.75 + effective_area * 0.25
    updated = Plant(
        p.strategy_id, p.age_days + 1, new_biomass,
        roots_depth, roots_spread, foliage, p.vitality,
        p.water_access,
    )
    # Hysteresis keeps already submerged roots alive through small oscillations.
    # Qualification is independent of rootedness, and only verified data can
    # qualify elsewhere. A new seed begins with shallow roots and no access.
    threshold = (ground.depth - cfg.hysteresis_depth if p.water_access
                 else ground.depth + cfg.hysteresis_depth)
    access = bool(inp.bonus_qualified and roots_depth >= max(0, threshold))
    updated = Plant(updated.strategy_id, updated.age_days, updated.biomass,
                    updated.root_depth, updated.root_spread, updated.leaf_area,
                    updated.vitality, access)

    # Foliage and sun determine potential margin utilization. Roots impose
    # an explicit opportunity cost. Dormancy makes inactivity an actual trait.
    exposed_area = foliage * (1 - inp.shadow)
    leaf_factor = exposed_area / (1.0 + exposed_area)
    exposure_intensity = _clip(leaf_factor * weather.sun * (1.0 - g.dormancy)
                               * (1.0 - cfg.root_opportunity_cost * root_effort), 0, 1)
    own_util = min(exposure_intensity, g.max_margin_fraction,
                   cfg.global_margin_fraction_cap)
    intended_margin = inp.entitlement * Decimal(str(own_util))
    cap = min(g.max_leverage, cfg.global_leverage_cap)
    # Sun+foliage also modulate leverage, which can be toggled/ablation-tested
    # later; this is double-effect on notional and must be measured empirically.
    effective_lev = ONE + (cap - ONE) * Decimal(str(leaf_factor * weather.sun))
    intended_notional = intended_margin * effective_lev
    return PlantOutput(updated, access, root_effort, foliage, own_util,
                       intended_margin, effective_lev, intended_notional)


def ecology_day(
    previous: GroundwaterLine,
    day: date,
    equity: str | Decimal,
    plants: Sequence[PlantInput],
    weather: Weather,
    cfg: EcologyConfig = EcologyConfig(),
) -> DailyEcology:
    """Compute one simulated biology day, no venue, no account writes."""
    if len({p.plant.strategy_id for p in plants}) != len(plants):
        raise EcologyError("duplicate strategy IDs would duplicate entitlements")
    if type(day) is not date:
        raise EcologyError("day must be date")
    end = datetime.combine(day, time.max, timezone.utc)
    if weather.known_at > end or weather.observed_at > end:
        raise EcologyError("weather not available by day cutoff; no lookahead")
    nav = _money(equity, "global NAV", signed=True)
    available = max(ZERO, nav) * Decimal("0.90")
    reserve = max(ZERO, nav) - available
    budgeted = sum((p.entitlement for p in plants), ZERO)
    if budgeted > available:
        raise EcologyError("strategy entitlements exceed 90% global NAV")
    ground = step_groundwater(previous, day, nav, cfg)
    results = tuple(_plant_step(p, ground, weather, cfg) for p in plants)
    if cfg.max_water_access_fraction is not None and results:
        # Optional scarcity regulator: only select best eligible roots, never
        # forcibly qualify a strategy below fitness gate.
        max_count = int(len(results) * cfg.max_water_access_fraction)
        eligible = [r for r in results if r.water_eligible]
        winners = {r.plant.strategy_id for r in sorted(
            eligible, key=lambda r: (-r.plant.root_depth, r.plant.strategy_id)
        )[:max_count]}
        results = tuple(r if r.plant.strategy_id in winners else
                        PlantOutput(
                            Plant(r.plant.strategy_id, r.plant.age_days, r.plant.biomass,
                                  r.plant.root_depth, r.plant.root_spread,
                                  r.plant.leaf_area, r.plant.vitality, False),
                            False, r.root_effort, r.effective_leaf_area,
                            r.own_utilization, r.intended_margin,
                            r.effective_leverage, r.intended_notional,
                        ) for r in results)
    return DailyEcology(ground, results, reserve, available, available-budgeted)
