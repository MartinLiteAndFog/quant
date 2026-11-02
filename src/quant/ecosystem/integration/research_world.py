"""ECO-16: deterministic paper-world adapter for clock + capital + plant ecology.

This is an end-of-day research harness, NOT a signal backtester or live system.
The only PnL comes from explicitly supplied incremental, time-known events.
All account entitlements are inside one virtual, single-currency paper account.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date
from decimal import Decimal
from hashlib import sha256
import json
from typing import Sequence

from quant.ecosystem.capital.groundwater import (
    Account, Demand, GroundwaterPlan, allocate_groundwater, split_seed_budget,
)
from quant.ecosystem.ecology.physiology import (
    EcologyConfig, GroundwaterLine, Plant, PlantGenome, PlantInput, Weather,
    ecology_day, initial_groundwater,
)
from quant.ecosystem.sim_clock.clock import (
    DayResult, IntradayEvent, SimulationState, advance_day, new_simulation,
)


class IntegrationError(ValueError):
    """Fail closed when a paper-world input is inconsistent."""


@dataclass(frozen=True)
class ResearchPlant:
    strategy_id: str
    genome: PlantGenome
    initial_plant: Plant
    parent_id: str | None = None
    initial_seed_share: Decimal = Decimal("0.02")
    root_weight: Decimal = Decimal("1")
    evidence_qualified: bool = False

    def __post_init__(self):
        if not self.strategy_id or self.initial_plant.strategy_id != self.strategy_id:
            raise IntegrationError("plant identity mismatch")
        if self.root_weight <= 0 or not self.root_weight.is_finite():
            raise IntegrationError("root weight must be positive finite")
        if self.parent_id is not None and (
            self.initial_seed_share <= 0 or self.initial_seed_share > 1
            or not self.initial_seed_share.is_finite()
        ):
            raise IntegrationError("seed share must be in (0, 1]")
        if type(self.evidence_qualified) is not bool:
            raise IntegrationError("evidence qualification must be explicit")


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _manifest(plants: Sequence[ResearchPlant], cfg: EcologyConfig) -> str:
    def canon(p: ResearchPlant) -> dict:
        result = asdict(p)
        result["genome"]["max_leverage"] = str(p.genome.max_leverage)
        result["initial_seed_share"] = str(p.initial_seed_share)
        result["root_weight"] = str(p.root_weight)
        return result
    config = asdict(cfg)
    for key in ("reference_equity", "global_leverage_cap"):
        config[key] = str(config[key])
    return sha256(_json({"plants": [canon(p) for p in plants],
                        "ecology_config": config, "version": "1"}).encode()).hexdigest()


def _check_plants(plants: Sequence[ResearchPlant]) -> None:
    ids = [p.strategy_id for p in plants]
    if not ids or len(ids) != len(set(ids)):
        raise IntegrationError("population must be nonempty and have unique IDs")
    roots = {p.strategy_id for p in plants if p.parent_id is None}
    if not roots:
        raise IntegrationError("at least one initial parent is required")
    if any(p.parent_id not in roots for p in plants if p.parent_id is not None):
        raise IntegrationError("initial children must be funded by a registered root")
    # Canonical root-first order makes deterministic child capital splitting
    # independent of the caller's tuple ordering.
    if list(plants) != sorted(plants, key=lambda p: (p.parent_id is not None, p.strategy_id)):
        raise IntegrationError("pass population in sorted root-first order")


def start_world(
    simulation_id: str, start_day: date, equity: str, population: Sequence[ResearchPlant],
    *, seed: int = 1, config: EcologyConfig = EcologyConfig(),
) -> SimulationState:
    """Create a 90/10 conservation-correct fixture, including funded children."""
    _check_plants(population)
    initial = new_simulation(simulation_id, equity, start_day, seed=seed)
    nav = initial.equity
    account = Account("paper", "USDT", nav, nav)
    roots = [p for p in population if p.parent_id is None]
    capital = allocate_groundwater(
        [account], [Demand(p.strategy_id, "paper", p.root_weight) for p in roots]
    )
    for seedling in (p for p in population if p.parent_id is not None):
        capital = split_seed_budget(
            capital, seedling.parent_id, seedling.strategy_id,
            seedling.initial_seed_share, str(nav),
        )
    claims = {e.strategy_id: str(e.amount) for e in capital.entitlements}
    if set(claims) != {p.strategy_id for p in population}:
        raise IntegrationError("unfunded or duplicated initial research plant")
    initial_water = initial_groundwater(start_day, str(nav), config)
    state = {
        "schema": "research-world-1", "manifest": _manifest(population, config),
        "initial_equity": str(nav), "entitlements": claims,
        "plants": {p.strategy_id: asdict(p.initial_plant) for p in population},
        "groundwater": {
            "day": initial_water.day.isoformat(), "equity": str(initial_water.equity),
            "depth": initial_water.depth, "target_depth": initial_water.target_depth,
        },
        "risk_frozen": False, "intents": {},
        "synthetic": True,
    }
    created = replace(initial, state_json=_json(state))
    # Check canonical/complete checkpoint before making it available.
    return SimulationState.from_json(created.to_json())


def _water_from_dict(state: dict) -> GroundwaterLine:
    data = state["groundwater"]
    return GroundwaterLine(
        date.fromisoformat(data["day"]), Decimal(data["equity"]),
        float(data["depth"]), float(data["target_depth"]),
    )


def _pnl_by_plant(events: Sequence[IntradayEvent], population: Sequence[ResearchPlant]
                  ) -> dict[str, Decimal]:
    ids = {p.strategy_id for p in population}
    result = {id_: Decimal(0) for id_ in ids}
    for event in events:
        if event.kind in ("pnl_delta", "fee", "funding"):
            if event.strategy_id not in ids:
                raise IntegrationError("P&L event must be owned by one registered plant")
            result[event.strategy_id] += event.amount
        # External deposits/withdrawals change account NAV but are *never*
        # considered strategy alpha or a plant's own daily return.
    return result


def step_world(
    previous: SimulationState, day: date, events: Sequence[IntradayEvent],
    population: Sequence[ResearchPlant], weather: Weather,
    *, free_margin: str | None = None,
    used_margin: dict[str, Decimal] | None = None,
    config: EcologyConfig = EcologyConfig(),
) -> DayResult:
    """Advance exactly one day with account budgeting before plant physiology.

    Returns the clock result; all extra game state lives in its canonical
    state_json. The caller, not this module, supplies causal execution events.
    """
    _check_plants(population)
    current = previous.ecology
    if current.get("schema") != "research-world-1":
        raise IntegrationError("unrecognized research-world snapshot")
    if current.get("manifest") != _manifest(population, config):
        raise IntegrationError("population/genome/config changed since checkpoint")
    if set(current["entitlements"]) != {p.strategy_id for p in population}:
        raise IntegrationError("missing or duplicate plant entitlements")
    if set(current["plants"]) != {p.strategy_id for p in population}:
        raise IntegrationError("missing or duplicate biological states")
    returns = _pnl_by_plant(events, population)
    locks = used_margin or {}
    if set(locks) - {p.strategy_id for p in population}:
        raise IntegrationError("unknown existing margin owner")

    def capital_reducer(snapshot: dict, _rng) -> dict:
        state = snapshot["state"]
        nav = Decimal(snapshot["equity"])
        free = max(Decimal(0), nav) if free_margin is None else Decimal(free_margin)
        account = Account("paper", "USDT", nav, free)
        demands = []
        for p in population:
            old_amount = Decimal(state["entitlements"][p.strategy_id])
            # Growth of root access influences budget *priority*, not money.
            old_plant = Plant(**state["plants"][p.strategy_id])
            priority = (old_amount if old_amount > 0 else p.root_weight / 1000)
            if old_plant.water_access and p.evidence_qualified:
                priority *= Decimal("1.10")
            locked = locks.get(p.strategy_id, Decimal(0))
            demands.append(Demand(
                p.strategy_id, "paper", priority, used_margin=locked,
                eligible=True,
            ))
        plan = allocate_groundwater([account], demands)
        state["entitlements"] = {e.strategy_id: str(e.amount) for e in plan.entitlements}
        state["entry_capacities"] = {
            e.strategy_id: str(e.entry_capacity) for e in plan.entitlements
        }
        state["risk_frozen"] = plan.allocations_frozen
        state["unallocated"] = str(plan.unallocated)
        return state

    def ecology_reducer(snapshot: dict, _rng) -> dict:
        state = snapshot["state"]
        inputs = []
        for p in population:
            old_budget = Decimal(current["entitlements"][p.strategy_id])
            own_delta = returns[p.strategy_id]
            if old_budget <= 0 and own_delta:
                raise IntegrationError("nonzero plant PnL with no opening capital")
            own_return = float(own_delta / old_budget) if old_budget > 0 else 0.0
            inputs.append(PlantInput(
                Plant(**state["plants"][p.strategy_id]), p.genome,
                own_return, Decimal(state["entitlements"][p.strategy_id]),
                p.evidence_qualified,
            ))
        result = ecology_day(
            _water_from_dict(state), day, snapshot["equity"], inputs, weather, config
        )
        intents: dict[str, dict] = {}
        for output in result.plants:
            id_ = output.plant.strategy_id
            entry_limit = Decimal(state["entry_capacities"][id_])
            # Proposed additional margin never exceeds owned budget or actual
            # account-free margin, even for a sun-drenched high-leverage plant.
            effective = min(output.intended_margin, entry_limit)
            intents[id_] = {
                "entry_margin": str(effective),
                "notional": str(effective * output.effective_leverage),
                "leverage": str(output.effective_leverage),
                "water_eligible": output.water_eligible,
                "light_area": output.effective_leaf_area,
                "root_effort": output.root_effort,
            }
            state["plants"][id_] = asdict(output.plant)
        state["groundwater"] = {
            "day": result.groundwater.day.isoformat(),
            "equity": str(result.groundwater.equity),
            "depth": result.groundwater.depth,
            "target_depth": result.groundwater.target_depth,
        }
        state["intents"] = intents
        state["synthetic"] = True
        return state

    return advance_day(
        previous, day, events,
        {"capital": capital_reducer, "ecology": ecology_reducer},
    )


def simple_demo_population() -> tuple[ResearchPlant, ...]:
    """Explicitly SYNTHETIC 3-parent/20-child starter, not live bot configs."""
    def genome(kind: str) -> PlantGenome:
        root = kind == "cactus"
        return PlantGenome(
            root_growth=0.88 if root else 0.3,
            root_depth=0.95 if root else 0.28,
            root_spread=0.25 if root else 0.62,
            branching=0.08 if root else 0.75,
            leaf_count=1 if root else 35,
            leaf_size=0.08 if root else 0.72,
            bark_strength=0.9 if root else 0.2,
            dormancy=0.8 if root else 0.05,
            photosynthetic_rind=root, rind_efficiency=0.8,
            reproduction_budget=0.10, protective_stop=True,
            max_leverage=Decimal("6"), max_margin_fraction=0.8,
        )
    parents = [
        ResearchPlant(f"parent-{j:02d}", genome("cactus" if j == 2 else "tree"),
                      Plant(f"parent-{j:02d}", 20, 1.0, 0.2, 0.3, 0.2, 0.8))
        for j in range(3)
    ]
    children = [
        ResearchPlant(
            f"seed-{i:02d}", genome("cactus" if i % 5 == 0 else "tree"),
            Plant(f"seed-{i:02d}", 0, 0.18, 0.01, 0.02, 0.05, 0.5),
            parent_id=f"parent-{i%3:02d}",
            initial_seed_share=Decimal("0.02"),
        )
        for i in range(20)
    ]
    return tuple(sorted(parents + children, key=lambda p: (p.parent_id is not None, p.strategy_id)))
