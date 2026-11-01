"""ECO-17: audited, cash-conserving, synthetic multi-generation research world."""
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal as D
import unittest
import json

from quant.ecosystem.integration.evolution_cycle import (
    CycleConfig, EvolutionError, start_evolution, step_evolution, evolution_report,
)
from quant.ecosystem.ecology.physiology import Weather
from quant.ecosystem.lifecycle.predation import PredatorConfig, Reconciliation
from quant.ecosystem.sim_clock.clock import ClockError, IntradayEvent, SimulationState


START=date(2026,1,1)
UTC=timezone.utc


def weather(day, sun=.8, next_day=False):
    at=datetime(day.year,day.month,day.day,12,tzinfo=UTC)
    if next_day:
        at+=timedelta(days=1)
    return Weather(at,at,sun,True)


def pnl(day, n, strategy="parent-00", delta=".02"):
    known=datetime(day.year,day.month,day.day,10,tzinfo=UTC)
    return [IntradayEvent("synthetic:"+str(n)+":"+strategy,known,known,
                          "pnl_delta",D(delta),strategy)]


def synthetic_day(state, n, *, config, finance="20", trade_everyone=True):
    day=START+timedelta(days=n)
    living=state.ecology["organisms"]
    counts={id_:1 for id_,organ in living.items() if organ["status"]!="closing"} if trade_everyone else {"parent-00":1}
    return step_evolution(
        state,day,pnl(day,n) if "parent-00" in living else [],
        weather(day),trades=counts,opportunities=counts,
        free_margin=finance,config=config,
    ).state


def journey(days, split=None, *, config):
    state=start_evolution("paper-evo",START,"100",seed=2026,config=config)
    for i in range(days):
        if split is not None and i==split:
            state=SimulationState.from_json(state.to_json())
        state=synthetic_day(state,i,config=config)
    return state


class EvolutionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.config=CycleConfig(
            reproduction_every_days=5,min_parent_trades=2,
            min_parent_credibility=.01,max_population=43,
            predators=PredatorConfig(attacks_per_day=0),
        )

    def test_initial_3_parents_20_seedlings_and_no_extra_equity(self):
        state=start_evolution("seed",START,"100",seed=8,config=self.config)
        d=state.ecology
        self.assertEqual(len(d["organisms"]),23)
        self.assertEqual(sum(o["generation"]==0 for o in d["organisms"].values()),3)
        self.assertEqual(sum(o["generation"]==1 for o in d["organisms"].values()),20)
        self.assertEqual(sum((D(v) for v in d["entitlements"].values()),D(0)),D("90"))
        self.assertEqual(state.reserve,D("10"))
        self.assertEqual(state.distributable,D("90"))
        self.assertTrue(d["synthetic"])
        self.assertTrue(all(
            o["trading_genome"]["genes"]["protective_stop"] is True
            for o in d["organisms"].values()
        ))

    def test_thirty_days_generations_gene_positions_and_resume_bytes(self):
        whole=journey(30,config=self.config)
        restarted=journey(30,split=12,config=self.config)
        again=journey(30,config=self.config)
        self.assertEqual(whole.to_json(),restarted.to_json())
        self.assertEqual(whole.to_json(),again.to_json())
        view=evolution_report(whole)
        self.assertEqual(whole.day_index,29)
        self.assertGreaterEqual(view["births"],4)
        self.assertGreaterEqual(view["max_generation"],2)
        self.assertGreater(view["hive_research"],0)
        self.assertEqual(view["archived"],0)
        self.assertEqual(view["live_orders_sent"],0)
        self.assertLessEqual(D(view["entitlement_total"]),whole.distributable)
        self.assertEqual(whole.reserve,whole.equity*D(".10"))
        self.assertLessEqual(view["population"],self.config.max_population)
        born=[e for e in whole.ecology["evolution_events"] if e["kind"]=="birth"]
        self.assertEqual(len(born),len(set(e["strategy_id"] for e in born)))
        initial=[(17.2,.65),(19.2,.85),(15.6,.45)]
        self.assertTrue(any(
            any(abs(e["new_x"]-(x-12)/13)>.001 or
                abs(e["new_y"]-(y-.25)/1.4)>.001 for x,y in initial)
            for e in born
        ))
        for p in whole.ecology["organisms"].values():
            genes=p["trading_genome"]["genes"]
            self.assertTrue(genes["protective_stop"])
            self.assertTrue(1<=genes["max_leverage"]<=10)
            self.assertTrue(.01<=genes["max_margin_fraction"]<=.9)
            self.assertTrue(.08<=genes["reproduction_budget"]<=.15)
        self.assertEqual(
            len({e["event_id"] for e in whole.ecology["evolution_events"]}),
            len(whole.ecology["evolution_events"]),
        )

    def test_all_intents_obey_capital_and_gene_limits(self):
        state=journey(12,config=self.config)
        view=state.ecology
        total=sum((D(x["entry_margin"]) for x in view["intents"].values()),D(0))
        self.assertLessEqual(total,D("20"))
        self.assertLessEqual(
            sum((D(v) for v in view["entitlements"].values()),D(0)),
            state.distributable,
        )
        for id_,intent in view["intents"].items():
            genome=view["organisms"][id_]["trading_genome"]["genes"]
            own_limit=D(view["entitlements"][id_])*D(str(genome["max_margin_fraction"]))
            self.assertLessEqual(D(intent["entry_margin"]),own_limit)
            self.assertLessEqual(D(intent["leverage"]),D(str(genome["max_leverage"])))
            self.assertEqual(
                D(intent["notional"]),
                D(intent["entry_margin"])*D(intent["leverage"]),
            )
        self.assertGreater(
            view["organisms"]["parent-02"]["plant"]["root_depth"],
            view["organisms"]["parent-00"]["plant"]["root_depth"],
        )

    def test_predation_no_released_cash_before_confirmed_flat(self):
        config=CycleConfig(
            reproduction_every_days=100,
            predators=PredatorConfig(
                attacks_per_day=23,base_attack_chance=1,
                seedling_grace_days=0,evidence_grace_days=0,
                min_effective_trades=0,
            ),
        )
        state=start_evolution("hunter",START,"100",seed=7,config=config)
        first=step_evolution(state,START,[],weather(START),
                             free_margin="90",config=config).state
        closing=[k for k,o in first.ecology["organisms"].items()
                 if o["status"]=="closing"]
        self.assertGreater(len(closing),0)
        self.assertTrue(all(
            event.get("reusable_entitlement")=="0"
            for event in first.ecology["evolution_events"] if event["kind"]=="closing_requested"
        ))
        lost_before=sum((D(first.ecology["entitlements"][k]) for k in closing),D(0))
        self.assertGreater(lost_before,0)
        second=step_evolution(
            first,START+timedelta(days=1),[],
            weather(START+timedelta(days=1)),
            free_margin="90",config=config,
        ).state
        self.assertTrue(all(k in second.ecology["entitlements"] for k in closing))
        self.assertTrue(all(second.ecology["intents"][k]["entry_margin"]=="0"
                            for k in closing))
        target=closing[0]
        with self.assertRaisesRegex(EvolutionError,"before this day"):
            step_evolution(state,START,[],weather(START),config=config,
                           reconciliations=[Reconciliation(
                               target,True,True,True,True,"gen","research://x"
                           )])
        acknowledged=Reconciliation(
            target,True,True,True,True,"v1","research://"+target,
        )
        nextday=START+timedelta(days=2)
        archived=step_evolution(
            second,nextday,[],weather(nextday),config=config,
            reconciliations=[acknowledged],free_margin="90",
        ).state
        self.assertNotIn(target,archived.ecology["organisms"])
        self.assertNotIn(target,archived.ecology["entitlements"])
        self.assertEqual(len([a for a in archived.ecology["archived"]
                              if a["strategy_id"]==target]),1)
        self.assertEqual(archived.equity,D("100"))
        self.assertEqual(archived.reserve,D("10"))
        self.assertEqual(
            [e["reusable_entitlement"] for e in archived.ecology["evolution_events"]
             if e["kind"]=="archived" and e["strategy_id"]==target],
            [second.ecology["entitlements"][target]],
        )
        with self.assertRaisesRegex(EvolutionError,"unknown confirmed plant"):
            step_evolution(
                archived,nextday+timedelta(days=1),[],
                weather(nextday+timedelta(days=1)),config=config,
                reconciliations=[acknowledged],
            )

    def test_locked_margin_blocks_archive_even_if_flagged_confirmed(self):
        config=CycleConfig(
            predators=PredatorConfig(attacks_per_day=23,base_attack_chance=1,
                                     seedling_grace_days=0,evidence_grace_days=0,
                                     min_effective_trades=0),
            reproduction_every_days=100,
        )
        state=start_evolution("lock",START,"100",seed=7,config=config)
        first=step_evolution(state,START,[],weather(START),config=config).state
        closing=next(id_ for id_,o in first.ecology["organisms"].items()
                     if o["status"]=="closing")
        day=START+timedelta(days=1)
        ack=Reconciliation(closing,True,True,True,True,"v1","research://locked")
        with self.assertRaisesRegex(Exception,"margin still locked"):
            step_evolution(
                first,day,[],weather(day),config=config,
                used_margin={closing:D("2")},reconciliations=[ack],
            )

    def test_dormant_cactus_stays_in_world_when_no_opportunity(self):
        config=CycleConfig(
            reproduction_every_days=100,
            predators=PredatorConfig(attacks_per_day=5,seedling_grace_days=400),
        )
        world=start_evolution("cactus",START,"100",seed=9,config=config)
        for i in range(35):
            day=START+timedelta(days=i)
            world=step_evolution(world,day,[],weather(day),
                                 config=config).state
        cactus=world.ecology["organisms"]["parent-02"]
        self.assertEqual(cactus["status"],"dormant")
        self.assertGreaterEqual(cactus["dormancy_days"],30)
        self.assertEqual(cactus["functional_genome"]["leaf_count"],1)
        self.assertEqual(evolution_report(world)["archived"],0)

    def test_external_cashflows_never_mint_returns(self):
        world=start_evolution("deposits",START,"100",seed=3,config=self.config)
        at=datetime(START.year,START.month,START.day,12,tzinfo=UTC)
        cash=IntradayEvent("deposit",at,at,"external_cashflow",D("50"))
        next_world=step_evolution(
            world,START,[cash],weather(START),config=self.config,
        ).state
        self.assertEqual(next_world.equity,D("150"))
        self.assertEqual(next_world.reserve,D("15"))
        self.assertTrue(all(
            row[0]["net_return"] == 0
            for row in next_world.ecology["history"].values()
        ))

    def test_negative_equity_frozen_without_pretending_to_flatten(self):
        world=start_evolution("withdraw",START,"100",seed=4,config=self.config)
        at=datetime(START.year,START.month,START.day,13,tzinfo=UTC)
        cash=IntradayEvent("cash-out",at,at,"external_cashflow",D("-110"))
        next_world=step_evolution(world,START,[cash],weather(START),config=self.config).state
        self.assertEqual(next_world.equity,D("-10"))
        self.assertTrue(next_world.ecology["risk_frozen"])
        self.assertEqual(next_world.distributable,D("0"))
        self.assertTrue(all(D(i["entry_margin"])==0 for i in
                            next_world.ecology["intents"].values()))
        self.assertFalse(any(e["kind"]=="birth" for e in
                             next_world.ecology["evolution_events"]))

    def test_optical_id_changes_never_change_financial_output(self):
        world=start_evolution("visual",START,"100",seed=10,config=self.config)
        before=world.ecology
        before["organisms"]["parent-00"]["visual_dna_ref"]="visual-dna:new-color"
        patched=SimulationState.from_json(replace(world,state_json=json.dumps(
            before,sort_keys=True,separators=(",",":"))).to_json())
        a=step_evolution(world,START,pnl(START,0),weather(START),config=self.config).state
        b=step_evolution(patched,START,pnl(START,0),weather(START),config=self.config).state
        self.assertEqual(a.equity,b.equity)
        self.assertEqual(a.ecology["entitlements"],b.ecology["entitlements"])
        self.assertEqual(a.ecology["metrics"],b.ecology["metrics"])
        self.assertEqual(a.ecology["intents"],b.ecology["intents"])

    def test_unregistered_events_future_weather_and_duplicate_replay_rejected(self):
        world=start_evolution("reject",START,"100",seed=10,config=self.config)
        with self.assertRaisesRegex(EvolutionError,"unregistered strategy"):
            step_evolution(
                world,START,pnl(START,0,strategy="ghost"),weather(START),
                config=self.config,
            )
        with self.assertRaisesRegex(Exception,"no lookahead"):
            step_evolution(
                world,START,[],weather(START,next_day=True),config=self.config,
            )
        next_world=step_evolution(world,START,[],weather(START),config=self.config).state
        with self.assertRaisesRegex(ClockError,"next consecutive"):
            step_evolution(next_world,START,[],weather(START),config=self.config)


if __name__ == "__main__":
    unittest.main()
