"""ECO-09: deterministic paper-only predation, dormancy and archival.

Predators do NOT flatten positions, cancel orders, or transfer funds. An attacked
plant moves to 'closing' (a non-executable request). Only externally reconciled
flat positions and orders may release a prior capital entitlement for reallocation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from math import isfinite
import random
from typing import Literal, Sequence

Status = Literal["alive", "dormant", "closing"]
HunterKind = Literal["weakest", "weighted_random", "age"]
ZERO = Decimal("0")


class LifecycleError(ValueError):
    """Invalid lifecycle, evidence or external settlement input."""


def fraction(value: float, name: str) -> float:
    if type(value) not in (float, int) or not isfinite(value) or not 0 <= value <= 1:
        raise LifecycleError(f"{name} must be finite and between 0 and 1")
    return float(value)


def money(value: Decimal | str, name: str) -> Decimal:
    if not isinstance(value, (Decimal, str)):
        raise LifecycleError(f"{name} must be Decimal or a string")
    try:
        x = Decimal(value)
    except (InvalidOperation, ValueError) as e:
        raise LifecycleError(f"invalid {name}") from e
    if not x.is_finite() or x < ZERO:
        raise LifecycleError(f"{name} must be finite and nonnegative")
    return x


@dataclass(frozen=True)
class Candidate:
    strategy_id: str
    account_id: str
    tile_id: str
    age_days: int
    vitality: float
    credibility: float
    bark_strength: float
    bark_evidence: float
    effective_trades: float
    recent_eligible_opportunities: int
    dormancy_days: int
    recent_mark_to_market_loss: bool
    entitlement: Decimal
    used_margin: Decimal = ZERO
    reserved_margin: Decimal = ZERO
    status: Status = "alive"

    def __post_init__(self):
        if not all(isinstance(s, str) and s.strip() for s in (
            self.strategy_id, self.account_id, self.tile_id
        )):
            raise LifecycleError("stable IDs required")
        for name in ("age_days", "recent_eligible_opportunities", "dormancy_days"):
            v = getattr(self, name)
            if type(v) is not int or v < 0:
                raise LifecycleError(f"{name} must be a nonnegative integer")
        if type(self.effective_trades) not in (int, float) or not isfinite(
            self.effective_trades
        ) or self.effective_trades < 0:
            raise LifecycleError("invalid effective trades")
        for name in ("vitality", "credibility", "bark_strength", "bark_evidence"):
            fraction(getattr(self, name), name)
        for name in ("entitlement", "used_margin", "reserved_margin"):
            object.__setattr__(self, name, money(getattr(self, name), name))
        if type(self.recent_mark_to_market_loss) is not bool:
            raise LifecycleError("recent_mark_to_market_loss must be boolean")
        if self.status not in ("alive", "dormant", "closing"):
            raise LifecycleError("invalid lifecycle status")
        if self.used_margin + self.reserved_margin > self.entitlement:
            # Existing locked commitments can exceed the new budget after an
            # account drawdown; that is a safety alert, NOT predation permission.
            # Keep candidate observable; funding allocator freezes separately.
            pass


@dataclass(frozen=True)
class PredatorConfig:
    hunt_threshold: float = 0.70
    protected_threshold: float = 0.80
    seedling_grace_days: int = 30
    evidence_grace_days: int = 90
    min_effective_trades: float = 8
    dormancy_grace_days: int = 365
    dormant_min_evidence: float = 0.15
    base_attack_chance: float = 0.95
    max_bark_shield: float = 0.93
    attacks_per_day: int = 2

    def __post_init__(self):
        for name in ("hunt_threshold", "protected_threshold", "dormant_min_evidence",
                     "base_attack_chance", "max_bark_shield"):
            fraction(getattr(self, name), name)
        if not 0 < self.hunt_threshold < self.protected_threshold <= 1:
            raise LifecycleError("hunt/protected thresholds reversed")
        for name in ("seedling_grace_days", "evidence_grace_days",
                     "dormancy_grace_days", "attacks_per_day"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise LifecycleError(f"invalid {name}")
        if type(self.min_effective_trades) not in (int, float) or not isfinite(
            self.min_effective_trades
        ) or self.min_effective_trades < 0:
            raise LifecycleError("invalid minimum evidence")


@dataclass(frozen=True)
class PredationEvent:
    event_id: str
    day: date
    strategy_id: str
    kind: Literal["closing_requested", "archived"]
    reason: str
    account_id: str
    tile_id: str
    previous_entitlement: Decimal
    reusable_entitlement: Decimal


@dataclass(frozen=True)
class ArchiveRecord:
    strategy_id: str
    account_id: str
    tile_id: str
    archived_on: date
    genome_version: str
    research_ref: str
    prior_entitlement: Decimal
    last_vitality: float
    # Historical genome/results live in the data archive; no visual fungus.


@dataclass(frozen=True)
class Reconciliation:
    strategy_id: str
    positions_flat: bool
    orders_cancelled: bool
    fills_reconciled: bool
    free_margin_confirmed: bool
    genome_version: str
    research_ref: str

    def __post_init__(self):
        if not self.strategy_id or not self.genome_version or not self.research_ref:
            raise LifecycleError("reconciliation identity and version required")
        if any(type(getattr(self, field)) is not bool for field in (
            "positions_flat", "orders_cancelled", "fills_reconciled",
            "free_margin_confirmed"
        )):
            raise LifecycleError("explicit reconciliation booleans required")

    @property
    def confirmed(self) -> bool:
        return all((self.positions_flat, self.orders_cancelled,
                    self.fills_reconciled, self.free_margin_confirmed))


def bark_shield(candidate: Candidate, config: PredatorConfig) -> float:
    """Bark protects only to the extent long-run evidence independently supports."""
    return min(config.max_bark_shield,
               candidate.bark_strength * candidate.bark_evidence * candidate.credibility)


def huntable(candidate: Candidate, config: PredatorConfig) -> bool:
    if candidate.status == "closing":
        return False
    if candidate.vitality >= config.hunt_threshold:
        return False   # 70–80 and >80 both safe; never hunt healthy plants
    if candidate.age_days < config.seedling_grace_days:
        return False
    if candidate.age_days < config.evidence_grace_days and (
        candidate.effective_trades < config.min_effective_trades
    ):
        return False
    if (candidate.status == "dormant" and
        candidate.recent_eligible_opportunities == 0 and
        not candidate.recent_mark_to_market_loss and
        candidate.dormancy_days <= config.dormancy_grace_days and
        candidate.bark_evidence >= config.dormant_min_evidence):
        return False
    return True


def attack_probability(candidate: Candidate, config: PredatorConfig) -> float:
    if not huntable(candidate, config):
        return 0.0
    # Low vitality means more predation pressure. Rind never grants immunity
    # from independent account-level risk/stop policies.
    weakness = (config.hunt_threshold - candidate.vitality) / config.hunt_threshold
    return min(1.0, config.base_attack_chance * (0.4 + 0.6 * weakness)
               * (1.0 - bark_shield(candidate, config)))


def _rng(simulation_id: str, seed: int, day: date, kind: HunterKind) -> random.Random:
    if not simulation_id or type(seed) is not int or type(day) is not date:
        raise LifecycleError("stable simulation seed and UTC date required")
    hashed = sha256(
        f"predation-v1|{simulation_id}|{seed}|{day.isoformat()}|{kind}".encode()
    ).digest()
    return random.Random(int.from_bytes(hashed, "big"))


def select_predators(
    simulation_id: str, seed: int, day: date,
    population: Sequence[Candidate],
    *, hunter: HunterKind = "weighted_random",
    config: PredatorConfig = PredatorConfig(),
) -> tuple[tuple[Candidate, ...], tuple[PredationEvent, ...]]:
    """Schedule possible predation without executing closing or releasing margin.

    Identical inputs produce identical choices. The caller must advance the
    calendar day and persist event IDs to avoid replaying a predation event.
    """
    if hunter not in ("weakest", "weighted_random", "age"):
        raise LifecycleError("unknown hunter type")
    if len({p.strategy_id for p in population}) != len(population):
        raise LifecycleError("duplicate strategy ID in population")
    rng = _rng(simulation_id, seed, day, hunter)
    # Canonical ordering prevents callers from changing weighted RNG draws.
    remaining = sorted((p for p in population if huntable(p, config)),
                       key=lambda p: p.strategy_id)
    updated = {p.strategy_id: p for p in population}
    events: list[PredationEvent] = []
    for attempt in range(min(config.attacks_per_day, len(remaining))):
        if hunter == "weakest":
            chosen = min(remaining, key=lambda p: (p.vitality, p.strategy_id))
        elif hunter == "age":
            chosen = max(remaining, key=lambda p: (p.age_days, -p.vitality, p.strategy_id))
        else:
            # Weighted random draw without replacement; weight based only on
            # current evidence and vulnerability, not on future outcomes.
            weights = [max(0.00001, (config.hunt_threshold - p.vitality) ** 2)
                       for p in remaining]
            choice = rng.random() * sum(weights)
            chosen = remaining[-1]
            for candidate, weight in zip(remaining, weights):
                choice -= weight
                if choice <= 0:
                    chosen = candidate
                    break
        remaining.remove(chosen)
        if rng.random() > attack_probability(chosen, config):
            continue
        updated[chosen.strategy_id] = replace(chosen, status="closing")
        events.append(PredationEvent(
            event_id=f"{simulation_id}:{day.isoformat()}:predator:{hunter}:{chosen.strategy_id}",
            day=day, strategy_id=chosen.strategy_id,
            kind="closing_requested", reason=hunter,
            account_id=chosen.account_id, tile_id=chosen.tile_id,
            previous_entitlement=chosen.entitlement, reusable_entitlement=ZERO,
        ))
    return tuple(updated[p.strategy_id] for p in population), tuple(events)


def finalize_archive(
    simulation_id: str, day: date, population: Sequence[Candidate],
    confirmations: Sequence[Reconciliation],
    archived: Sequence[ArchiveRecord] = (),
) -> tuple[tuple[Candidate, ...], tuple[ArchiveRecord, ...], tuple[PredationEvent, ...]]:
    """Archive only confirmed flat/closed plants; propose *free* budget to ledger.

    Events DO NOT change account NAV. A caller can pass the released entitlement
    as a new allocator *input* only after positions and reservations reconcile.
    """
    if not simulation_id or type(day) is not date:
        raise LifecycleError("simulation ID and UTC day required")
    if len({p.strategy_id for p in population}) != len(population):
        raise LifecycleError("duplicate live strategy IDs")
    if len({c.strategy_id for c in confirmations}) != len(confirmations):
        raise LifecycleError("duplicate confirmations")
    if len({a.strategy_id for a in archived}) != len(archived):
        raise LifecycleError("duplicate archived strategy IDs")
    by_id = {p.strategy_id: p for p in population}
    archive_ids = {a.strategy_id for a in archived}
    if set(by_id) & archive_ids:
        raise LifecycleError("already archived strategy still appears in living population")
    if set(c.strategy_id for c in confirmations) - set(by_id):
        raise LifecycleError("unknown strategy in reconciliation")
    confirmed = {c.strategy_id: c for c in confirmations}
    live: list[Candidate] = []
    new_archive = list(archived)
    events: list[PredationEvent] = []
    for p in population:
        confirm = confirmed.get(p.strategy_id)
        if not confirm or not confirm.confirmed:
            live.append(p)
            continue
        if p.status != "closing":
            raise LifecycleError("cannot archive a plant without closing request")
        if p.used_margin != ZERO or p.reserved_margin != ZERO:
            raise LifecycleError("margin still locked: need a fresh flat account snapshot")
        new_archive.append(ArchiveRecord(
            strategy_id=p.strategy_id, account_id=p.account_id,
            tile_id=p.tile_id, archived_on=day,
            genome_version=confirm.genome_version,
            research_ref=confirm.research_ref,
            prior_entitlement=p.entitlement,
            last_vitality=p.vitality,
        ))
        events.append(PredationEvent(
            event_id=f"{simulation_id}:{day.isoformat()}:archive:{p.strategy_id}",
            day=day, strategy_id=p.strategy_id, kind="archived",
            reason="all_fills_and_margin_reconciled",
            account_id=p.account_id, tile_id=p.tile_id,
            previous_entitlement=p.entitlement,
            reusable_entitlement=p.entitlement,
        ))
    return tuple(live), tuple(new_archive), tuple(events)
