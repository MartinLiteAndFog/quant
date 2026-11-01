"""ECO-17: deterministic autonomous population in a SYNTHETIC paper world.

Bridges ECO-03 Clock, ECO-05 Groundwater, ECO-06 ecology, ECO-07 metrics,
ECO-08 bees, and ECO-09 safe predation. No live exchange or order interfaces.
An external, causal event producer owns all net PnL and opportunity inputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timezone
from decimal import Decimal
from hashlib import sha256
import json
from typing import Mapping, Sequence

from quant.ecosystem.breeding.bees import (
    Bee, GeneSpec, Genome, Plant as BreedingPlant, fund_local_seeds, pollinate,
    validate_genome,
)
from quant.ecosystem.capital.groundwater import Account, Demand, allocate_groundwater
from quant.ecosystem.ecology.physiology import (
    EcologyConfig, GroundwaterLine, Plant, PlantGenome, PlantInput, Weather,
    ecology_day,
)
from quant.ecosystem.integration.research_world import (
    simple_demo_population, start_world,
)
from quant.ecosystem.lifecycle.predation import (
    Candidate, PredatorConfig, Reconciliation, finalize_archive, select_predators,
)
from quant.ecosystem.metrics.vitality import NetDay, evaluate_strategy
from quant.ecosystem.sim_clock.clock import (
    DayResult, IntradayEvent, SimulationState, advance_day,
)

SCHEMA = "paper-evolution-1"
ZERO = Decimal("0")
BREEDS = ("mixer", "mosaic", "explorer", "collector", "pruner")
GENES = (
    GeneSpec("protective_stop", "immutable_safety_policy", "switch", default=True),
    GeneSpec("max_leverage", "required_bounded_financial", "number", 1, 10, .1, 3),
    GeneSpec("max_margin_fraction", "required_bounded_financial", "number", .01, .9, .01, .5),
    GeneSpec("reproduction_budget", "required_bounded_financial", "number", .08, .15, .01, .10),
    GeneSpec("sensitivity", "required_axis", "number", 12, 25, .1, 17.2),
    GeneSpec("ttp", "required_axis", "number", .25, 1.65, .05, .65),
    GeneSpec("rsi_length", "optional_strategy_gene", "number", 5, 50, 1, 14),
)


class EvolutionError(ValueError):
    """An event, genotype, budget or reconciliation violates the paper contract."""


@dataclass(frozen=True)
class CycleConfig:
    reproduction_every_days: int = 5
    min_parent_trades: int = 2
    min_parent_credibility: float = .01
    max_population: int = 45
    max_hive_records: int = 200
    seed_share: Decimal = Decimal(".02")
    ecology: EcologyConfig = EcologyConfig()
    predators: PredatorConfig = PredatorConfig(attacks_per_day=1)
    hunter: str = "weakest"

    def __post_init__(self):
        if type(self.reproduction_every_days) is not int or self.reproduction_every_days < 1:
            raise EvolutionError("reproduction cadence must be positive")
        if type(self.min_parent_trades) is not int or self.min_parent_trades < 0:
            raise EvolutionError("min_parent_trades must be nonnegative")
        if not 0 <= self.min_parent_credibility <= 1:
            raise EvolutionError("credibility threshold out of range")
        if type(self.max_population) is not int or self.max_population < 24:
            raise EvolutionError("population cap must accommodate initial seedlings")
        if type(self.max_hive_records) is not int or self.max_hive_records < 1:
            raise EvolutionError("max_hive_records must be positive")
        if not isinstance(self.seed_share, Decimal) or not ZERO < self.seed_share <= Decimal(".1"):
            raise EvolutionError("seed_share must be exact Decimal in (0,.1]")
        if self.hunter not in ("weakest", "weighted_random", "age"):
            raise EvolutionError("unknown predator")


def _json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _config_id(cfg: CycleConfig) -> str:
    obj=asdict(cfg)
    obj["seed_share"]=str(cfg.seed_share)
    for field in ("reference_equity", "global_leverage_cap"):
        obj["ecology"][field]=str(obj["ecology"][field])
    return sha256(_json(obj).encode()).hexdigest()


def _persist_genome(g: Genome) -> dict:
    return {
        "genome_id":g.genome_id, "version":g.version,
        "module_family":g.module_family, "tile_id":g.tile_id,
        "genes":dict(g.genes), "parent_genomes":list(g.parent_genomes),
    }


def _read_genome(g: dict) -> Genome:
    found=Genome(g["genome_id"], g["version"], g["module_family"],
                 g["tile_id"], tuple(sorted(g["genes"].items())),
                 tuple(g["parent_genomes"]))
    return validate_genome(found, GENES)


def _functional_dict(g: PlantGenome) -> dict:
    data=asdict(g)
    data["max_leverage"]=str(g.max_leverage)
    return data


def _functional(data: dict) -> PlantGenome:
    return PlantGenome(**{**data, "max_leverage": Decimal(data["max_leverage"])})


def _initial_genome(strategy_id: str) -> Genome:
    if strategy_id.startswith("parent-"):
        index=int(strategy_id.split("-")[-1])
        sensitivity, ttp=[(17.2,.65),(19.2,.85),(15.6,.45)][index]
    else:
        index=int(strategy_id.split("-")[-1])
        root=index%3
        base=[(17.2,.65),(19.2,.85),(15.6,.45)][root]
        sensitivity=round(base[0]+(index%5-2)*.1,10)
        ttp=round(base[1]+((index//3)%3-1)*.05,10)
    genes={
        "protective_stop":True, "max_leverage":3.0,
        "max_margin_fraction":.5,"reproduction_budget":.10,
        "sensitivity":sensitivity,"ttp":ttp,
    }
    g=Genome("genesis:"+strategy_id,1,"imba","sol-imba",
             tuple(sorted(genes.items())))
    return validate_genome(g,GENES)


def start_evolution(
    simulation_id: str, first_day: date, equity: str = "100", *,
    seed: int = 1, config: CycleConfig = CycleConfig(),
) -> SimulationState:
    """Three original plants + twenty funded seedlings from ECO-16."""
    initial=simple_demo_population()
    snapshot=start_world(simulation_id,first_day,equity,initial,seed=seed,config=config.ecology)
    state=snapshot.ecology
    state["schema"]=SCHEMA
    state["config_id"]=_config_id(config)
    state["organisms"]={}
    for p in initial:
        genome=_initial_genome(p.strategy_id)
        state["organisms"][p.strategy_id]={
            "trading_genome":_persist_genome(genome),
            "functional_genome":_functional_dict(p.genome),
            "plant":asdict(p.initial_plant),
            "home_tile_id":"sol-imba",
            "generation":0 if p.parent_id is None else 1,
            "mother":p.parent_id,
            "father":p.parent_id,
            "bee":None,
            "status":"alive",
            "dormancy_days":0,
            "visual_dna_ref":"visual-dna:"+sha256(p.strategy_id.encode()).hexdigest()[:16],
        }
    state["history"]={p.strategy_id:[] for p in initial}
    state["metrics"]={}
    state["archived"]=[]
    state["hive_research"]=[]
    state["evolution_events"]=[]
    state["entry_capacities"]={}
    state["unallocated_budget"]="0"
    state["synthetic"]=True
    return SimulationState.from_json(replace(snapshot,state_json=_json(state)).to_json())


def _water(source: dict) -> GroundwaterLine:
    g=source["groundwater"]
    return GroundwaterLine(date.fromisoformat(g["day"]), Decimal(g["equity"]),
                           float(g["depth"]), float(g["target_depth"]))


def _history(source: dict) -> list[NetDay]:
    return [
        NetDay(date.fromisoformat(item["day"]),float(item["net_return"]),
               int(item["trades"]),int(item["eligible_opportunities"]),
               int(item["expected_signals"]),datetime.fromisoformat(item["known_at"]))
        for item in source
    ]


def _count_inputs(
    ids: set[str], name: str, values: Mapping[str,int] | None,
) -> dict[str,int]:
    source=dict(values or {})
    if set(source)-ids:
        raise EvolutionError(f"{name} contains unknown strategy IDs")
    if any(type(v) is not int or v < 0 for v in source.values()):
        raise EvolutionError(f"{name} must contain nonnegative integer counts")
    return source


def step_evolution(
    previous: SimulationState, day: date, events: Sequence[IntradayEvent],
    weather: Weather, *, trades: Mapping[str,int] | None = None,
    opportunities: Mapping[str,int] | None = None,
    free_margin: str | None = None,
    used_margin: Mapping[str,Decimal] | None = None,
    reserved_margin: Mapping[str,Decimal] | None = None,
    reconciliations: Sequence[Reconciliation] = (),
    config: CycleConfig = CycleConfig(),
) -> DayResult:
    """Advance one synthetic ecology day with birth, hunting and flat-only archive.

    The caller provides *all* PnL, genuine strategy opportunities, trade counts,
    account margin and any verified liquidation; this function invents none.
    """
    original=previous.ecology
    if original.get("schema") != SCHEMA or original.get("config_id") != _config_id(config):
        raise EvolutionError("wrong paper genome/config schema; explicit migration required")
    original_ids=set(original["organisms"])
    if original_ids != set(original["entitlements"]) or original_ids != set(original["history"]):
        raise EvolutionError("inconsistent canonical living population")
    trades=_count_inputs(original_ids,"trades",trades)
    opportunities=_count_inputs(original_ids,"opportunities",opportunities)
    used={k:Decimal(v) for k,v in dict(used_margin or {}).items()}
    reserved={k:Decimal(v) for k,v in dict(reserved_margin or {}).items()}
    if (set(used)|set(reserved)) - original_ids:
        raise EvolutionError("unknown margin owner")
    if any(not x.is_finite() or x < ZERO for x in list(used.values())+list(reserved.values())):
        raise EvolutionError("invalid margin input")
    if len(set(r.strategy_id for r in reconciliations)) != len(reconciliations):
        raise EvolutionError("duplicate external confirmations")
    confirmations={r.strategy_id:r for r in reconciliations}
    if set(confirmations)-original_ids:
        raise EvolutionError("unknown confirmed plant")
    if any(original["organisms"][k]["status"]!="closing" for k in confirmations):
        raise EvolutionError("can only confirm a plant closing before this day")
    by_plant={k:ZERO for k in original_ids}
    for ev in events:
        if ev.kind in ("pnl_delta","fee","funding"):
            if ev.strategy_id not in original_ids:
                raise EvolutionError("unregistered strategy PnL cannot be booked")
            by_plant[ev.strategy_id]+=ev.amount
        if ev.kind=="signal" and ev.strategy_id and ev.strategy_id not in original_ids:
            raise EvolutionError("unregistered signal identity")
    free=Decimal(free_margin) if free_margin is not None else None
    if free is not None and (not free.is_finite() or free < ZERO):
        raise EvolutionError("invalid account-free margin")
    initial_entitlements={k:Decimal(v) for k,v in original["entitlements"].items()}
    for k in original_ids:
        if initial_entitlements[k] <= ZERO and by_plant[k]:
            raise EvolutionError("profit/loss cannot be assigned without prior owned capital")

    def metrics_reducer(payload: dict, _rng) -> dict:
        state=payload["state"]
        cutoff=datetime.combine(day,time.max,timezone.utc)
        for id_ in sorted(original_ids):
            opening=initial_entitlements[id_]
            daily_return=float(by_plant[id_]/opening) if opening>ZERO else 0
            observations=state["history"][id_]
            observations.append({
                "day":day.isoformat(), "net_return":daily_return,
                "trades":trades.get(id_,0),
                "eligible_opportunities":opportunities.get(id_,0),
                "expected_signals":opportunities.get(id_,0),
                "known_at":cutoff.isoformat(),
            })
            m=evaluate_strategy(_history(observations),oos_verified=False)
            state["metrics"][id_]={
                "lifetime_fitness":m.lifetime_fitness,
                "vitality":m.current_vitality,
                "credibility":m.credibility,
                "bark_evidence":m.bark_evidence,
                "effective_trades":m.effective_independent_samples,
                "executed_trades":m.executed_trades,
                "recent_opportunities":sum(x["eligible_opportunities"] for x in observations[-30:]),
                "current_drawdown":m.current_drawdown,
            }
        return state

    def capital_reducer(payload: dict, _rng) -> dict:
        state=payload["state"]
        nav=Decimal(payload["equity"])
        distributable=max(ZERO,nav)*Decimal(".90")
        actual_free=max(ZERO,nav) if free is None else free
        if actual_free>max(ZERO,nav):
            raise EvolutionError("account free margin exceeds equity snapshot")
        # Ring-fence closing budgets. Until external flat/order confirmation,
        # none of their prior entitlement is made available to other plants.
        closing=[id_ for id_ in sorted(state["organisms"])
                 if state["organisms"][id_]["status"]=="closing"]
        active=[id_ for id_ in sorted(state["organisms"])
                if state["organisms"][id_]["status"]!="closing"]
        old_claims={id_:Decimal(value) for id_,value in state["entitlements"].items()}
        escrow=sum((old_claims[id_] for id_ in closing),ZERO)
        risk_freeze=nav<ZERO or escrow>distributable
        escrow_scale=distributable/escrow if escrow>distributable and escrow else Decimal("1")
        claims={id_:old_claims[id_]*escrow_scale for id_ in closing}
        remaining=max(ZERO,distributable-sum(claims.values(),ZERO))
        if active and remaining>ZERO:
            demands=[
                Demand(id_,"paper",max(Decimal(".000001"),old_claims[id_]) *
                       (Decimal(".80")+Decimal(str(state["metrics"][id_]["vitality"]))*Decimal(".4")),
                       used_margin=used.get(id_,ZERO),
                       reserved_order_margin=reserved.get(id_,ZERO))
                for id_ in active
            ]
            plan=allocate_groundwater([Account("paper","USDT",nav,actual_free)],demands)
            ratio=remaining/distributable if distributable else ZERO
            claims.update({e.strategy_id:e.amount*ratio for e in plan.entitlements})
            risk_freeze=risk_freeze or plan.risk_breach
        else:
            claims.update({id_:ZERO for id_ in active})
        if sum(claims.values(),ZERO)>distributable:
            raise EvolutionError("reserved plus funded claims exceed global 90% NAV")
        headroom={
            id_:max(ZERO,claims[id_]-used.get(id_,ZERO)-reserved.get(id_,ZERO))
            for id_ in active
        }
        if any(used.get(id_,ZERO)+reserved.get(id_,ZERO)>claims[id_] for id_ in state["organisms"]):
            risk_freeze=True
        available_free=ZERO if risk_freeze else min(actual_free,sum(headroom.values(),ZERO))
        denominator=sum(headroom.values(),ZERO)
        capacity={
            id_:str(available_free*headroom[id_]/denominator if denominator else ZERO)
            for id_ in active
        }
        capacity.update({id_:"0" for id_ in closing})
        state["entitlements"]={id_:str(value) for id_,value in sorted(claims.items())}
        state["entry_capacities"]=capacity
        state["unallocated_budget"]=str(distributable-sum(claims.values(),ZERO))
        state["risk_frozen"]=risk_freeze
        return state

    def ecology_reducer(payload: dict, _rng) -> dict:
        state=payload["state"]
        inputs=[]
        for id_ in sorted(state["organisms"]):
            organ=state["organisms"][id_]
            m=state["metrics"][id_]
            opening=initial_entitlements[id_]
            inputs.append(PlantInput(
                Plant(**organ["plant"]),_functional(organ["functional_genome"]),
                float(by_plant[id_]/opening) if opening>ZERO else 0,
                Decimal(state["entitlements"][id_]),
                m["credibility"]>=.10 and m["lifetime_fitness"]>=25,
            ))
        result=ecology_day(_water(state),day,payload["equity"],inputs,weather,config.ecology)
        intents={}
        for output in result.plants:
            id_=output.plant.strategy_id
            organ=state["organisms"][id_]
            organ["plant"]=asdict(output.plant)
            limit=Decimal(state["entry_capacities"][id_])
            capacity=min(limit,output.intended_margin)
            if organ["status"]=="closing" or state["risk_frozen"]:
                capacity=ZERO
            intents[id_]={
                "entry_margin":str(capacity),
                "notional":str(capacity*output.effective_leverage),
                "leverage":str(output.effective_leverage),
                "water_eligible":output.water_eligible,
                "light_area":output.effective_leaf_area,
                "root_depth":output.plant.root_depth,
            }
        state["intents"]=intents
        state["groundwater"]={
            "day":result.groundwater.day.isoformat(),
            "equity":str(result.groundwater.equity),
            "depth":result.groundwater.depth,
            "target_depth":result.groundwater.target_depth,
        }
        return state

    def breeding_reducer(payload: dict, _rng) -> dict:
        state=payload["state"]
        if (state["risk_frozen"] or (payload["day_index"]+1)%config.reproduction_every_days
            or len(state["organisms"])>=config.max_population):
            return state
        viable=[
            (id_,organ) for id_,organ in state["organisms"].items()
            if organ["status"]=="alive"
            and Decimal(state["entitlements"][id_])>ZERO
            and organ["plant"]["age_days"]>=3
            and state["metrics"].get(id_,{}).get("executed_trades",0)>=config.min_parent_trades
            and state["metrics"].get(id_,{}).get("credibility",0)>=config.min_parent_credibility
        ]
        if not viable:
            return state
        # Favor matured later generations, then credibility. This is a *paper
        # search policy*, not a claim that visual descendants are more profitable.
        viable.sort(key=lambda item:(
            -item[1]["generation"],
            -state["metrics"][item[0]]["credibility"],item[0]))
        highest=viable[0][1]["generation"]
        eligible=[(id_,v) for id_,v in viable if v["generation"]==highest][:3]
        plants=[]
        for id_,organ in eligible:
            g=_read_genome(organ["trading_genome"])
            values=g.values()
            plants.append(BreedingPlant(
                id_,g,(float(values["sensitivity"])-12)/13,
                (float(values["ttp"])-.25)/1.4,organ["home_tile_id"],
                min(1.0,state["metrics"][id_]["credibility"]),
            ))
        bee_kind=BREEDS[(payload["day_index"]//config.reproduction_every_days)%len(BREEDS)]
        bee=Bee("hive:"+bee_kind,bee_kind,"sol-imba",
                search_radius=1.0,mutation_strength=.30,
                optional_gene_drop_rate=.05,
                optional_gene_add_rate=.10,
                hive_share_probability=1.0)
        proposals=pollinate(day,previous.seed,[bee],plants,GENES,mode="hybrid")
        local=[p for p in proposals if not p.research_only][:1]
        research=[p for p in proposals if p.research_only]
        for p in research:
            if len(state["hive_research"])<config.max_hive_records:
                state["hive_research"].append({
                    "seed_id":p.seed_id,"day":day.isoformat(),
                    "parents":list(p.lineage[:2]),"bee_id":p.bee_id,
                    "genome":_persist_genome(p.child_genome),
                    "research_only":True,"allocated_budget":"0",
                })
        if not local:
            return state
        previous_claims={id_:Decimal(value) for id_,value in state["entitlements"].items()}
        from quant.ecosystem.breeding.bees import BreedingError
        try:
            financed,claims=fund_local_seeds(
                local,previous_claims,
                {id_:used.get(id_,ZERO)+reserved.get(id_,ZERO) for id_ in original_ids},
                {id_:Decimal(state["organisms"][id_]["trading_genome"]["genes"]["reproduction_budget"])
                 for id_ in state["organisms"]},
                child_share=config.seed_share,
            )
        except BreedingError:
            # Rejected proposal is recorded as research, never a phantom birth.
            return state
        for child in financed:
            if not child.funded or child.seed_id in state["organisms"]:
                raise EvolutionError("unfunded or duplicate offspring")
            mother=state["organisms"][child.mother_strategy_id]
            father=state["organisms"][child.father_strategy_id]
            baby=Plant(child.seed_id,0,.18,.01,.02,.05,.50,False)
            state["organisms"][child.seed_id]={
                "trading_genome":_persist_genome(child.child_genome),
                "functional_genome":dict(mother["functional_genome"]),
                "plant":asdict(baby),
                "home_tile_id":child.destination_tile_id,
                "generation":max(mother["generation"],father["generation"])+1,
                "mother":child.mother_strategy_id,
                "father":child.father_strategy_id,
                "bee":child.bee_id,"status":"alive","dormancy_days":0,
                "visual_dna_ref":"visual-dna:"+sha256(child.seed_id.encode()).hexdigest()[:16],
            }
            state["history"][child.seed_id]=[]
            state["entitlements"][child.seed_id]=str(child.allocated_budget)
            state["entry_capacities"][child.seed_id]="0"
            state["intents"][child.seed_id]={
                "entry_margin":"0","notional":"0","leverage":"1",
                "water_eligible":False,"light_area":0,"root_depth":.01,
            }
            state["evolution_events"].append({
                "event_id":"birth:"+child.seed_id,"day":day.isoformat(),
                "kind":"birth","strategy_id":child.seed_id,
                "mother":child.mother_strategy_id,"father":child.father_strategy_id,
                "bee":child.bee_id,
                "generation":state["organisms"][child.seed_id]["generation"],
                "new_capital":str(child.allocated_budget),
                "new_x":child.normalized_x,"new_y":child.normalized_y,
            })
        for id_,amount in claims.items():
            state["entitlements"][id_]=str(amount)
        if sum((Decimal(x) for x in state["entitlements"].values()),ZERO)>Decimal(payload["distributable"]):
            raise EvolutionError("birth created unbacked equity")
        return state

    def lifecycle_reducer(payload: dict, _rng) -> dict:
        state=payload["state"]
        candidates=[]
        for id_,organ in sorted(state["organisms"].items()):
            m=state["metrics"].get(id_,{})
            if organ["status"]=="alive" and _functional(organ["functional_genome"]).dormancy>=.75:
                if (m.get("recent_opportunities",0)==0 and by_plant.get(id_,ZERO)==ZERO):
                    organ["dormancy_days"]+=1
                    if organ["dormancy_days"]>=30:
                        organ["status"]="dormant"
                else:
                    organ["dormancy_days"]=0
            elif organ["status"]=="dormant" and (
                m.get("recent_opportunities",0)>0 or by_plant.get(id_,ZERO)!=ZERO
            ):
                organ["status"]="alive"
                organ["dormancy_days"]=0
            candidates.append(Candidate(
                strategy_id=id_,account_id="paper",tile_id=organ["home_tile_id"],
                age_days=organ["plant"]["age_days"],
                vitality=m.get("vitality",.5),credibility=m.get("credibility",0),
                bark_strength=_functional(organ["functional_genome"]).bark_strength,
                bark_evidence=m.get("bark_evidence",0),
                effective_trades=m.get("effective_trades",0),
                recent_eligible_opportunities=m.get("recent_opportunities",0),
                dormancy_days=organ["dormancy_days"],
                recent_mark_to_market_loss=by_plant.get(id_,ZERO)<ZERO,
                entitlement=Decimal(state["entitlements"][id_]),
                used_margin=used.get(id_,ZERO),reserved_margin=reserved.get(id_,ZERO),
                status=organ["status"],
            ))
        updated,attacks=select_predators(
            previous.simulation_id,previous.seed,day,candidates,
            hunter=config.hunter,config=config.predators,
        )
        for candidate in updated:
            state["organisms"][candidate.strategy_id]["status"]=candidate.status
        for e in attacks:
            state["evolution_events"].append({
                "event_id":e.event_id,"day":day.isoformat(),"kind":e.kind,
                "strategy_id":e.strategy_id,"reusable_entitlement":"0",
            })
        # Explicit acknowledgments may only refer to already closing plants
        # from the PRIOR day, never to a freshly attacked same-day candidate.
        confirmed=[r for r in reconciliations if r.confirmed]
        prior_archive_ids={x["strategy_id"] for x in state["archived"]}
        if len(prior_archive_ids)!=len(state["archived"]):
            raise EvolutionError("duplicate archive")
        _,archives,archived_events=finalize_archive(
            previous.simulation_id,day,updated,confirmed,
        )
        for record in archives:
            if record.strategy_id not in state["organisms"]:
                raise EvolutionError("archive unknown strategy")
            new_record={
                "strategy_id":record.strategy_id,"account_id":record.account_id,
                "tile_id":record.tile_id,"archived_on":record.archived_on.isoformat(),
                "genome_version":record.genome_version,
                "research_ref":record.research_ref,
                "prior_entitlement":str(record.prior_entitlement),
                "last_vitality":record.last_vitality,
                "trading_genome":state["organisms"][record.strategy_id]["trading_genome"],
                "history":state["history"][record.strategy_id],
            }
            state["archived"].append(new_record)
            for field in ("organisms","entitlements","history","metrics",
                          "entry_capacities","intents"):
                state[field].pop(record.strategy_id,None)
        for e in archived_events:
            state["evolution_events"].append({
                "event_id":e.event_id,"day":day.isoformat(),"kind":e.kind,
                "strategy_id":e.strategy_id,
                "reusable_entitlement":str(e.reusable_entitlement),
            })
        state["unallocated_budget"]=str(
            max(ZERO,Decimal(payload["distributable"]) -
                sum((Decimal(x) for x in state["entitlements"].values()),ZERO))
        )
        return state

    return advance_day(previous,day,events,{
        "metrics":metrics_reducer,"capital":capital_reducer,
        "ecology":ecology_reducer,"breeding":breeding_reducer,
        "lifecycle":lifecycle_reducer,
    })


def evolution_report(state: SimulationState) -> dict:
    """Stable read-only summary; absolutely no trading side effects."""
    x=state.ecology
    if x.get("schema")!=SCHEMA:
        raise EvolutionError("not an ECO-17 checkpoint")
    alive=x["organisms"]
    return {
        "mode":"SYNTHETIC PAPER EVOLUTION; NOT VERIFIED LIVE/BACKTEST",
        "day":state.as_of.isoformat(),
        "equity":str(state.equity),"reserve":str(state.reserve),
        "distributable":str(state.distributable),
        "entitlement_total":str(sum((Decimal(s) for s in x["entitlements"].values()),ZERO)),
        "unallocated_budget":x["unallocated_budget"],
        "population":len(alive),"max_generation":max(
            (v["generation"] for v in alive.values()),default=0),
        "pending_close":sum(v["status"]=="closing" for v in alive.values()),
        "archived":len(x["archived"]),"hive_research":len(x["hive_research"]),
        "births":sum(y["kind"]=="birth" for y in x["evolution_events"]),
        "water_depth":x["groundwater"]["depth"],
        "synthetic":True,"live_orders_sent":0,
    }
