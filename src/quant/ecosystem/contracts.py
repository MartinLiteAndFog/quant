"""Versioned, side-effect-free paper ecosystem interchange contracts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"


def _decimal(value: Any) -> Decimal:
    if not isinstance(value, (str, Decimal)):
        raise ValueError("exact decimal values must be decimal strings or Decimal")
    result = Decimal(value)
    if not result.is_finite():
        raise ValueError("decimal must be finite")
    return result


def _utc(value: Any) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must have an explicit timezone")
    return value.astimezone(timezone.utc)


Money = Annotated[Decimal, BeforeValidator(_decimal)]
NonNegative = Annotated[Money, Field(ge=0)]
UTCDateTime = Annotated[datetime, BeforeValidator(_utc)]
Unit = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Positive = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class GenePolicy(Contract):
    gene_id: str
    kind: Literal["immutable_safety_policy", "required_bounded_financial", "optional_strategy_gene"]
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    @model_validator(mode="after")
    def valid_bounds(self):
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("reversed gene bounds")
        return self


class StrategyGenome(Contract):
    genome_id: str
    version: int = Field(ge=1)
    strategy_code_version: str
    genes: dict[str, Union[bool, float]]
    policies: tuple[GenePolicy, ...]
    parent_genome_ids: tuple[str, ...] = ()
    complexity_penalty_weight: Positive = 0

    @model_validator(mode="after")
    def protected_genes(self):
        policies = {p.gene_id: p for p in self.policies}
        if len(policies) != len(self.policies) or set(policies) != set(self.genes):
            raise ValueError("each gene needs exactly one policy")
        required = {"protective_stop", "max_leverage", "max_margin_fraction", "reproduction_budget"}
        if not required <= self.genes.keys() or self.genes["protective_stop"] is not True:
            raise ValueError("mandatory safety/financial genes missing or disabled")
        if policies["protective_stop"].kind != "immutable_safety_policy":
            raise ValueError("protective stop policy must be immutable")
        for name in required - {"protective_stop"}:
            policy = policies[name]
            if policy.kind != "required_bounded_financial" or policy.minimum is None or policy.maximum is None:
                raise ValueError("financial genes need explicit bounded required policies")
        for name, value in self.genes.items():
            p = policies[name]
            if (p.minimum is not None and value < p.minimum) or (p.maximum is not None and value > p.maximum):
                raise ValueError("gene outside authorized bounds")
        if not 0 < self.genes["max_margin_fraction"] <= 1 or not 0 <= self.genes["reproduction_budget"] <= 1 or self.genes["max_leverage"] < 1:
            raise ValueError("invalid financial gene")
        return self


Lifecycle = Literal["proposed", "screened", "backtested", "out_of_sample_verified", "paper_forward", "eligible_for_sowing", "capitalized", "dormant", "closing", "archived"]


class StrategyInstance(Contract):
    strategy_id: str
    genome_id: str
    family_id: str
    account_id: str
    created_at: UTCDateTime
    lifecycle: Lifecycle


class MetricWindow(Contract):
    start: UTCDateTime
    end: UTCDateTime
    score: Optional[float] = None
    net_return: Optional[float] = None
    max_drawdown: Unit = 0
    observations: int = Field(default=0, ge=0)
    out_of_sample: bool = False
    formula_version: str = "1.0"

    @model_validator(mode="after")
    def ordered(self):
        if self.end < self.start:
            raise ValueError("reversed metric window")
        return self


class EvidenceState(Contract):
    effective_trades: Positive = 0
    observed_days: int = Field(default=0, ge=0)
    confidence: Unit = 0
    data_quality: Unit = 0
    opportunities: int = Field(default=0, ge=0)
    out_of_sample: bool = False


class DailyPlantState(Contract):
    strategy_id: str
    as_of: UTCDateTime
    genome_id: str
    lifecycle: Lifecycle
    biomass: Positive
    root_depth: Positive
    root_spread: Positive
    leaf_area: Positive
    bark_strength: Unit
    fitness: MetricWindow
    vitality: Unit
    evidence: EvidenceState

    @model_validator(mode="after")
    def causal(self):
        if self.fitness.end > self.as_of:
            raise ValueError("future fitness")
        return self


class RootNetwork(Contract):
    strategy_id: str
    neighbors: tuple[str, ...] = ()
    depth: Positive
    spread: Positive
    access_weight: Positive


class AccountNAV(Contract):
    account_id: str
    currency: str
    as_of: UTCDateTime
    equity: Money
    free_margin: NonNegative

    @model_validator(mode="after")
    def margin_bound(self):
        if self.free_margin > max(Decimal(0), self.equity):
            raise ValueError("free margin exceeds NAV")
        return self


class CapitalEntitlement(Contract):
    strategy_id: str
    account_id: str
    amount: NonNegative
    used_margin: NonNegative
    reserved_order_margin: NonNegative = Decimal("0")
    realizable_margin: NonNegative
    notional_exposure: NonNegative

    @model_validator(mode="after")
    def bounded_margin(self):
        if not self.realizable_margin <= self.amount:
            raise ValueError("realizable <= entitlement required")
        return self


class GroundwaterState(Contract):
    as_of: UTCDateTime
    currency: str
    accounts: tuple[AccountNAV, ...]
    equity: Money
    reserve: NonNegative
    distributable: NonNegative
    entitlements: tuple[CapitalEntitlement, ...]
    depth: float = Field(gt=0, allow_inf_nan=False)
    risk_breach: bool = False
    allocations_frozen: bool = False

    @model_validator(mode="after")
    def conserved(self):
        accounts = {a.account_id: a for a in self.accounts}
        if len(accounts) != len(self.accounts):
            raise ValueError("duplicate account NAV")
        if any(a.currency != self.currency or a.as_of != self.as_of for a in self.accounts):
            raise ValueError("accounts require reconciled currency and timestamp")
        if sum((a.equity for a in self.accounts), Decimal(0)) != self.equity:
            raise ValueError("NAV is counted exactly once")
        if self.reserve != max(Decimal(0), self.equity) * Decimal("0.10") or self.distributable != max(Decimal(0), self.equity) - self.reserve:
            raise ValueError("90/10 reserve identity violated")
        if len({e.strategy_id for e in self.entitlements}) != len(self.entitlements):
            raise ValueError("duplicate strategy entitlement")
        if any(e.account_id not in accounts for e in self.entitlements):
            raise ValueError("unknown entitlement account")
        if sum((e.amount for e in self.entitlements), Decimal(0)) > self.distributable:
            raise ValueError("budgets exceed distributable NAV")
        if any(a.equity < 0 for a in self.accounts) and not (self.risk_breach and self.allocations_frozen):
            raise ValueError("negative account NAV requires frozen risk breach")
        breached = any(e.used_margin + e.reserved_order_margin > e.amount for e in self.entitlements)
        if breached and not (self.risk_breach and self.allocations_frozen):
            raise ValueError("locked margin above budget requires explicit frozen risk breach")
        for account in self.accounts:
            claims = [e for e in self.entitlements if e.account_id == account.account_id]
            if sum((e.amount for e in claims), Decimal(0)) > max(Decimal(0), account.equity):
                raise ValueError("account overallocated")
            if sum((max(Decimal(0), e.realizable_margin - e.used_margin - e.reserved_order_margin) for e in claims), Decimal(0)) > account.free_margin:
                raise ValueError("promised unused margin exceeds account free margin")
        return self


GroundwaterAllocation = CapitalEntitlement


class ExposureIntent(Contract):
    strategy_id: str
    signal_at: UTCDateTime
    available_at: UTCDateTime
    execute_after: UTCDateTime
    margin: NonNegative
    leverage: Money = Field(ge=1)
    notional: NonNegative

    @model_validator(mode="after")
    def causal(self):
        if not self.signal_at <= self.available_at <= self.execute_after:
            raise ValueError("noncausal intent")
        if self.notional != self.margin * self.leverage:
            raise ValueError("exposure identity violated")
        return self


class MarketSnapshot(Contract):
    symbol: str
    observed_at: UTCDateTime
    available_at: UTCDateTime
    price: Money = Field(gt=0)
    source: str
    synthetic: bool

    @model_validator(mode="after")
    def causal(self):
        if self.available_at < self.observed_at:
            raise ValueError("market data available before observation")
        return self


class TradeFill(Contract):
    fill_id: str
    strategy_id: str
    symbol: str
    signal_at: UTCDateTime
    available_at: UTCDateTime
    filled_at: UTCDateTime
    side: Literal["buy", "sell"]
    quantity: Money = Field(gt=0)
    price: Money = Field(gt=0)
    fee: NonNegative
    funding: Money
    currency: str
    synthetic: bool

    @model_validator(mode="after")
    def causal(self):
        if not self.signal_at <= self.available_at <= self.filled_at:
            raise ValueError("fill precedes known signal")
        return self


class MarketRegime(Contract):
    as_of: UTCDateTime
    label: str
    sun_multiplier: Unit
    source: Literal["synthetic", "observed"]


class BeeAction(Contract):
    action_id: str
    as_of: UTCDateTime
    bee_id: str
    mode: Literal["random", "local", "hive", "hybrid"]
    algorithm: Literal["mixer", "mosaic", "explorer", "collector", "pruner"]
    parent_genome_ids: tuple[str, ...]
    child_genome_id: str
    seed: int
    validated_oos_honey: float = 0


class PredatorEvent(Contract):
    strategy_id: str
    as_of: UTCDateTime
    reason: str
    positions_closed_confirmed: bool
    released_budget: NonNegative

    @model_validator(mode="after")
    def safe_release(self):
        if self.released_budget > 0 and not self.positions_closed_confirmed:
            raise ValueError("cannot release unreconciled capital")
        return self


PredationEvent = PredatorEvent


class ResearchCandidate(Contract):
    candidate_id: str
    genome: StrategyGenome
    stage: Lifecycle
    benchmark_id: str
    verdict: Literal["beats", "does_not_beat", "not_comparable"]
    evidence: EvidenceState


class TickPhase(str, Enum):
    EXECUTION = "execution"
    METRICS = "metrics"
    CAPITAL = "capital"
    ECOLOGY = "ecology"
    BREEDING = "breeding"
    LIFECYCLE = "lifecycle"
    SNAPSHOT = "snapshot"


TICK_PHASES = tuple(TickPhase)


class SimulationEvent(Contract):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    event_id: str
    simulation_id: str
    day_index: int = Field(ge=0)
    sequence: int = Field(ge=0)
    phase: TickPhase
    occurred_at: UTCDateTime
    known_at: UTCDateTime
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def causal(self):
        if self.known_at < self.occurred_at:
            raise ValueError("event knowledge precedes occurrence")
        return self


class SimulationSnapshot(Contract):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    simulation_id: str
    as_of: UTCDateTime
    seed: int
    rng_state: str
    day_index: int = Field(ge=0)
    plants: tuple[DailyPlantState, ...]
    genomes: tuple[StrategyGenome, ...]
    groundwater: GroundwaterState
    events: tuple[SimulationEvent, ...] = ()
    provenance: Literal["synthetic", "historical", "shadow"] = "synthetic"

    @model_validator(mode="after")
    def consistent(self):
        if self.groundwater.as_of != self.as_of:
            raise ValueError("snapshot NAV timestamp differs")
        genome_ids = {g.genome_id for g in self.genomes}
        plant_ids = {p.strategy_id for p in self.plants}
        if len(genome_ids) != len(self.genomes) or len(plant_ids) != len(self.plants):
            raise ValueError("duplicate genome/plant IDs")
        if any(p.genome_id not in genome_ids or p.as_of != self.as_of for p in self.plants):
            raise ValueError("unknown genome or inconsistent plant timestamp")
        if any(e.strategy_id not in plant_ids for e in self.groundwater.entitlements):
            raise ValueError("entitlement has no plant")
        keys = [(e.day_index, TICK_PHASES.index(e.phase), e.sequence) for e in self.events]
        if keys != sorted(keys) or len({e.event_id for e in self.events}) != len(self.events):
            raise ValueError("unordered/duplicate events")
        if len({e.sequence for e in self.events}) != len(self.events):
            raise ValueError("duplicate sequence")
        if any(e.simulation_id != self.simulation_id or e.day_index > self.day_index or e.known_at > self.as_of for e in self.events):
            raise ValueError("foreign or future event")
        return self


def dump_snapshot(snapshot: SimulationSnapshot) -> str:
    """Canonical JSON; no binary floating point money crosses the boundary."""
    return json.dumps(snapshot.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False)


def migrate_snapshot(document: dict[str, Any]) -> dict[str, Any]:
    """Only documented 0.1 (replay_seed -> seed) migration; never guess legacy data."""
    result = dict(document)
    version = result.get("schema_version")
    if version == "0.1":
        if "seed" in result or "replay_seed" not in result:
            raise ValueError("ambiguous legacy replay seed")
        result["seed"] = result.pop("replay_seed")
        result["schema_version"] = SCHEMA_VERSION
    elif version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")
    return result


def load_snapshot(document: str | dict[str, Any]) -> SimulationSnapshot:
    value = json.loads(document) if isinstance(document, str) else document
    return SimulationSnapshot.model_validate(migrate_snapshot(value))
