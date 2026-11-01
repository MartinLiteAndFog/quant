"""ECO-08 pure simulated bee genetics, local pollination and research hive.

All genomes are versioned, gene-policy checked, and only proposed. Central hive
children are research-only. Funding is a distinct paper-ledger transaction and
must preserve existing capital entitlements; no venue or signal execution here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from math import isfinite
import random
from typing import Literal, Mapping, Sequence

Value = float | bool
Policy = Literal["immutable_safety_policy", "required_bounded_financial",
                 "required_axis", "optional_strategy_gene"]
Breed = Literal["mixer", "mosaic", "explorer", "collector", "pruner"]
Mode = Literal["local", "central", "hybrid"]
BREEDS: tuple[Breed, ...] = ("mixer", "mosaic", "explorer", "collector", "pruner")
REQUIRED_FINANCIAL = frozenset((
    "protective_stop", "max_leverage",
    "max_margin_fraction", "reproduction_budget",
))
ZERO = Decimal(0)


class BreedingError(ValueError):
    pass


def unit(x: float, label: str) -> float:
    if type(x) not in (float, int) or not isfinite(x) or not 0 <= x <= 1:
        raise BreedingError(f"{label} must be a finite fraction in [0,1]")
    return float(x)


def dec(value: str | Decimal, label: str) -> Decimal:
    if not isinstance(value, (Decimal, str)):
        raise BreedingError(f"{label} requires exact decimal input")
    try:
        d = Decimal(value)
    except (InvalidOperation, ValueError):
        raise BreedingError(f"{label} invalid") from None
    if not d.is_finite() or d < ZERO:
        raise BreedingError(f"{label} must be finite/nonnegative")
    return d


@dataclass(frozen=True)
class GeneSpec:
    gene_id: str
    policy: Policy
    type: Literal["number", "switch"]
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    default: Value = 0.0

    def __post_init__(self):
        if not self.gene_id or self.policy not in (
            "immutable_safety_policy", "required_bounded_financial",
            "required_axis", "optional_strategy_gene"
        ) or self.type not in ("number", "switch"):
            raise BreedingError("invalid gene schema")
        if self.type == "number":
            if (any(x is None or type(x) not in (float,int) or not isfinite(x)
                    for x in (self.min_value,self.max_value,self.step)) or
                self.min_value >= self.max_value or self.step <= 0):
                raise BreedingError("invalid numeric gene range")
        elif (self.min_value is not None or self.max_value is not None or
              self.step is not None or type(self.default) is not bool):
            raise BreedingError("invalid switch gene schema")
        validate_gene_value(self, self.default)


def validate_gene_value(spec: GeneSpec, val: Value):
    if spec.type == "switch":
        if type(val) is not bool:
            raise BreedingError(f"{spec.gene_id}: expected switch")
    elif (type(val) not in (float, int) or not isfinite(val) or
          val < spec.min_value - 1e-9 or val > spec.max_value + 1e-9):
        raise BreedingError(f"{spec.gene_id}: out of bounds")
    if spec.policy == "immutable_safety_policy" and val != spec.default:
        raise BreedingError(f"{spec.gene_id}: immutable safety gene changed")


def spec_catalog(specs: Sequence[GeneSpec]) -> dict[str, GeneSpec]:
    known = {s.gene_id: s for s in specs}
    if len(known) != len(specs):
        raise BreedingError("duplicate gene spec")
    if not REQUIRED_FINANCIAL <= set(known):
        raise BreedingError("required financial/safety policies missing from catalog")
    if known["protective_stop"].policy != "immutable_safety_policy" or known["protective_stop"].default is not True:
        raise BreedingError("mandatory protective stop cannot be removed")
    if any(known[g].policy != "required_bounded_financial" for g in (
        "max_leverage", "max_margin_fraction", "reproduction_budget",
    )):
        raise BreedingError("mandatory financial genes must be bounded and required")
    if sum(s.policy == "required_axis" for s in specs) != 2:
        raise BreedingError("exactly two parameter-axis genes required per meadow")
    return known


@dataclass(frozen=True)
class Genome:
    genome_id: str
    version: int
    module_family: str
    tile_id: str
    genes: tuple[tuple[str, Value], ...]
    parent_genomes: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.genome_id or not self.module_family or not self.tile_id or (
            type(self.version) is not int or self.version < 1
        ):
            raise BreedingError("incomplete/invalid genome identity")
        if len(set(k for k,_ in self.genes)) != len(self.genes):
            raise BreedingError("duplicate gene IDs in genome")

    def values(self) -> dict[str, Value]:
        return dict(self.genes)


@dataclass(frozen=True)
class Plant:
    strategy_id: str
    genome: Genome
    x: float
    y: float
    tile_id: str
    evidence_score: float

    def __post_init__(self):
        if not self.strategy_id or self.tile_id != self.genome.tile_id:
            raise BreedingError("plant tile/identity inconsistent")
        if not all(type(v) in (int, float) and isfinite(v) and 0 <= v <= 1
                   for v in (self.x, self.y)):
            raise BreedingError("coordinates must be normalized real genetic axes")
        unit(self.evidence_score, "evidence score")


@dataclass(frozen=True)
class Bee:
    bee_id: str
    breed: Breed
    home_tile_id: str
    search_radius: float = 0.4
    mutation_strength: float = 0.1
    optional_gene_drop_rate: float = 0.05
    optional_gene_add_rate: float = 0.15
    hive_share_probability: float = 0.5

    def __post_init__(self):
        if not self.bee_id or not self.home_tile_id or self.breed not in BREEDS:
            raise BreedingError("unknown bee genome")
        for name in ("search_radius", "mutation_strength",
                     "optional_gene_drop_rate", "optional_gene_add_rate",
                     "hive_share_probability"):
            unit(getattr(self,name),name)
        if self.optional_gene_drop_rate > .10:
            raise BreedingError("optional drop rate capped at 10%")


@dataclass(frozen=True)
class SeedProposal:
    seed_id: str
    bee_id: str
    mother_strategy_id: str
    father_strategy_id: str
    child_genome: Genome
    destination_tile_id: str
    research_only: bool
    source: Literal["local", "hive"]
    lineage: tuple[str, str, str]
    normalized_x: float
    normalized_y: float
    funded: bool = False
    allocated_budget: Decimal = ZERO


@dataclass(frozen=True)
class Honey:
    bee_id: str
    verified_offspring: int
    mean_out_of_sample_score: float | None


def validate_genome(genome: Genome, specs: Sequence[GeneSpec]) -> Genome:
    known = spec_catalog(specs)
    values = genome.values()
    if not set(values) <= set(known):
        raise BreedingError("unknown gene cannot be executed")
    for key, spec in known.items():
        if spec.policy != "optional_strategy_gene" and key not in values:
            raise BreedingError(f"missing protected/axis gene: {key}")
    for key, value in values.items():
        validate_gene_value(known[key], value)
    return genome


def _random(seed: int, day: date, mode: Mode, bee_id: str) -> random.Random:
    if type(seed) is not int or type(day) is not date:
        raise BreedingError("reproducible seed and date required")
    data = sha256(f"bees-v1|{seed}|{day.isoformat()}|{mode}|{bee_id}".encode()).digest()
    return random.Random(int.from_bytes(data, "big"))


def _snap_number(value: float, spec: GeneSpec) -> float:
    if spec.type == "switch":
        raise BreedingError("cannot snap a bool gene as numeric")
    v = max(spec.min_value, min(spec.max_value, value))
    steps = int((v - spec.min_value) / spec.step + .5)
    return max(spec.min_value, min(spec.max_value,
               round(spec.min_value + steps * spec.step, 10)))


def breed_genomes(
    mother: Genome, father: Genome, bee: Bee,
    specs: Sequence[GeneSpec], rng: random.Random,
    *, child_id: str, target_tile: str,
) -> Genome:
    known = spec_catalog(specs)
    validate_genome(mother,specs); validate_genome(father,specs)
    if mother.module_family != father.module_family:
        raise BreedingError("incompatible strategy modules: no executable cross")
    if not child_id or child_id in (mother.genome_id, father.genome_id):
        raise BreedingError("child needs unique genome identity")
    left,right=mother.values(),father.values()
    output:dict[str,Value]={}
    for key,spec in sorted(known.items()):
        optional = spec.policy=="optional_strategy_gene"
        a,b=left.get(key),right.get(key)
        if optional:
            if a is None and b is None:
                if rng.random() >= bee.optional_gene_add_rate:
                    continue
                a=spec.default
            if bee.breed=="pruner" and rng.random()<max(.10,bee.optional_gene_drop_rate):
                continue
            if bee.breed!="pruner" and rng.random()<bee.optional_gene_drop_rate:
                continue
        if spec.policy=="immutable_safety_policy":
            output[key]=spec.default
            continue
        values = [v for v in (a,b) if v is not None]
        if not values:
            raise BreedingError("required gene absent in parent")
        if spec.type=="switch":
            output[key]=rng.choice(values)
            continue
        if bee.breed=="mixer" and len(values)==2:
            candidate=(float(values[0])+float(values[1])) / 2
        elif bee.breed=="mosaic":
            candidate=float(rng.choice(values))
        elif bee.breed=="collector" and a is None:
            candidate=float(b)
        elif bee.breed=="collector" and b is None:
            candidate=float(a)
        else:
            candidate=float(rng.choice(values))
        # Mandatory safety/financial alleles may mutate *within* bounds.
        if rng.random()<(.65 if bee.breed=="explorer" else .25):
            multiplier=2.0 if bee.breed=="explorer" else 1.0
            candidate+=rng.uniform(-1,1)*(spec.max_value-spec.min_value)*bee.mutation_strength*multiplier
        output[key]=_snap_number(candidate,spec)
    child=Genome(child_id,1,mother.module_family,target_tile,
                 tuple(sorted(output.items())),
                 (mother.genome_id, father.genome_id))
    return validate_genome(child,specs)


def pollinate(
    day: date, seed: int, bees: Sequence[Bee], plants: Sequence[Plant],
    specs: Sequence[GeneSpec], *, mode: Mode="local",
) -> tuple[SeedProposal,...]:
    """Local seeds and/or central hive research candidates, but NEVER orders."""
    if mode not in ("local","central","hybrid"):
        raise BreedingError("unknown bee mode")
    if len(set(p.strategy_id for p in plants))!=len(plants):
        raise BreedingError("duplicate plant identity")
    if len(set(b.bee_id for b in bees))!=len(bees):
        raise BreedingError("duplicate bee identity")
    catalog = spec_catalog(specs)
    axes = sorted((s for s in catalog.values() if s.policy=="required_axis"),
                  key=lambda spec: spec.gene_id)
    for plant in plants:
        validate_genome(plant.genome,specs)
        values=plant.genome.values()
        true_x=(float(values[axes[0].gene_id])-axes[0].min_value) / (
            axes[0].max_value-axes[0].min_value)
        true_y=(float(values[axes[1].gene_id])-axes[1].min_value) / (
            axes[1].max_value-axes[1].min_value)
        if abs(true_x-plant.x)>1e-7 or abs(true_y-plant.y)>1e-7:
            raise BreedingError("plant coordinates must equal its axis gene values")
    proposals=[]
    seen_genomes={p.genome.genome_id for p in plants}
    for bee in sorted(bees,key=lambda b:b.bee_id):
        rng=_random(seed,day,mode,bee.bee_id)
        available=sorted((p for p in plants if p.tile_id==bee.home_tile_id),
                         key=lambda p:p.strategy_id)
        if not available:
            continue
        mother=rng.choice(available)
        local=[p for p in available if p.strategy_id != mother.strategy_id and
               ((p.x-mother.x)**2+(p.y-mother.y)**2)**.5<=bee.search_radius and
               p.genome.module_family==mother.genome.module_family]
        father=rng.choice(local) if local else mother
        def seed_child(source:str, mate:Plant, target_tile:str, index:int):
            code=f"{day.isoformat()}:{bee.bee_id}:{source}:{index}:{mother.strategy_id}:{mate.strategy_id}"
            genome_id="genome-bee-"+sha256(code.encode()).hexdigest()[:18]
            strategy_id="seed-bee-"+sha256(("strategy:"+code).encode()).hexdigest()[:18]
            if genome_id in seen_genomes:
                raise BreedingError("duplicate born genome")
            seen_genomes.add(genome_id)
            child=breed_genomes(mother.genome,mate.genome,bee,specs,rng,
                                child_id=genome_id,target_tile=target_tile)
            values=child.values()
            x=(float(values[axes[0].gene_id])-axes[0].min_value) / (
               axes[0].max_value-axes[0].min_value)
            y=(float(values[axes[1].gene_id])-axes[1].min_value) / (
               axes[1].max_value-axes[1].min_value)
            proposals.append(SeedProposal(strategy_id,bee.bee_id,mother.strategy_id,
                mate.strategy_id,child,target_tile,source=="hive",source,
                (mother.strategy_id,mate.strategy_id,bee.bee_id),x,y))
        if mode in ("local","hybrid"):
            seed_child("local",father,mother.tile_id,0)
        if mode=="central" or (mode=="hybrid" and
                               rng.random()<=bee.hive_share_probability):
            # Different tiles may provide compatible *research ideas*, never
            # auto-funded orders. All offspring still share real GeneSpec IDs.
            hive=[p for p in sorted(plants,key=lambda p:p.strategy_id)
                  if p.genome.module_family==mother.genome.module_family and
                  p.strategy_id != mother.strategy_id]
            partner=rng.choice(hive) if hive else mother
            seed_child("hive",partner,mother.tile_id,1)
    return tuple(proposals)


def fund_local_seeds(
    proposals: Sequence[SeedProposal],
    current_entitlements: Mapping[str, Decimal],
    locked_margin: Mapping[str, Decimal],
    parent_reproduction_budget: Mapping[str, Decimal],
    *, child_share: str | Decimal = Decimal("0.02"),
) -> tuple[tuple[SeedProposal,...],dict[str,Decimal]]:
    """Exact Decimal entitlement transfer; no NAV minting or central-hive funding.

    Caller must independently reconcile account ownership/FX and avoid mixing
    distinct accounts in a single pool. This function cannot send real orders.
    """
    share=dec(child_share,"seed share")
    if not ZERO < share <= Decimal("1"):
        raise BreedingError("seed share must be in (0,1]")
    if len({p.seed_id for p in proposals})!=len(proposals):
        raise BreedingError("duplicate seed proposal")
    claims={k:dec(v,"entitlement") for k,v in current_entitlements.items()}
    baseline=sum(claims.values(),ZERO)
    original=dict(claims)
    used:dict[str,Decimal]={}
    produced=[]
    for p in proposals:
        if p.research_only:
            produced.append(p)
            continue
        if p.mother_strategy_id not in claims or p.seed_id in claims:
            raise BreedingError("unfunded parent or duplicate child")
        budget_limit=dec(parent_reproduction_budget.get(p.mother_strategy_id, "0"),
                         "parent reproduction ceiling")
        if budget_limit > Decimal("1"):
            raise BreedingError("parent reproduction quota exceeds equity fraction")
        used_amount=used.get(p.mother_strategy_id,ZERO)
        requested=original[p.mother_strategy_id]*share
        if used_amount+requested>original[p.mother_strategy_id]*budget_limit:
            raise BreedingError("reproduction request exceeds parent's gene budget")
        locked=dec(locked_margin.get(p.mother_strategy_id,"0"),"locked margin")
        if claims[p.mother_strategy_id]-requested<locked:
            raise BreedingError("cannot take capital from locked margin")
        claims[p.mother_strategy_id]-=requested
        claims[p.seed_id]=requested
        used[p.mother_strategy_id]=used_amount+requested
        produced.append(replace(p,funded=True,allocated_budget=requested))
    if sum(claims.values(),ZERO)!=baseline:
        raise BreedingError("capital conservation violated")
    return tuple(produced),claims


def bee_honey(bee_id: str, evaluations: Sequence[tuple[str,float,bool]]) -> Honey:
    """Reward only held-out independent OOS child evaluations; no free cash."""
    scores=[]
    seen=set()
    for child_id,score,verified in evaluations:
        if not child_id or child_id in seen:
            raise BreedingError("duplicate/missing evaluation ID")
        seen.add(child_id)
        if type(verified) is not bool or type(score) not in (int,float) or not isfinite(score):
            raise BreedingError("invalid OOS evaluation")
        if verified:
            scores.append(float(score))
    return Honey(bee_id,len(scores),sum(scores)/len(scores) if scores else None)


def mutate_bee(bee:Bee,seed:int,day:date)->Bee:
    """Bee search behavior evolves; protected trading limits remain elsewhere."""
    rng=_random(seed,day,"hybrid",bee.bee_id+":mutation")
    parent=bee
    return replace(parent,
        search_radius=unit(max(0,min(1,parent.search_radius+rng.uniform(-.08,.08))),"radius"),
        mutation_strength=unit(max(0,min(1,parent.mutation_strength+rng.uniform(-.04,.04))),"mutation strength"),
        optional_gene_drop_rate=unit(max(0,min(.10,parent.optional_gene_drop_rate+rng.uniform(-.02,.02))),"drop"),
        hive_share_probability=unit(max(0,min(1,parent.hive_share_probability+rng.uniform(-.08,.08))),"hive share"))
