"""ECO-16: cross-module 30+ day repeatable PAPER life-cycle acceptance checks."""
import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal as D
from hashlib import sha256

from quant.ecosystem.integration.research_world import (
    IntegrationError, ResearchPlant, simple_demo_population, start_world, step_world,
)
from quant.ecosystem.ecology.physiology import EcologyConfig, Weather
from quant.ecosystem.sim_clock.clock import ClockError, IntradayEvent, SimulationState

START = date(2026, 1, 1)
UTC = timezone.utc


def day_events(day, number):
    at = datetime(day.year, day.month, day.day, 9, tzinfo=UTC)
    fees = datetime(day.year, day.month, day.day, 11, tzinfo=UTC)
    # All PnL comes from these supplied, synthetically labelled incremental
    # events; none is generated from the plants' apparent fitness.
    return [
        IntradayEvent(f"pnl-{number}", at, at, "pnl_delta",
                      D("0.08") if number % 3 else D("-0.06"), "parent-00"),
        IntradayEvent(f"fee-{number}", fees, fees, "fee", D("-0.01"), "parent-00"),
    ]


def weather(day, *, sunlight=.8, known_hour=12):
    known = datetime(day.year, day.month, day.day, known_hour, tzinfo=UTC)
    return Weather(known, known, sunlight, synthetic=True)


def run_dates(state, pop, start, stop, *, margin=None):
    for i in range(start, stop):
        d = START + timedelta(days=i)
        state = step_world(state, d, day_events(d, i), pop, weather(d),
                           free_margin=margin).state
    return state


class PaperWorldIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.pop = simple_demo_population()
        self.world = start_world("synthetic-ecosystem", START, "100", self.pop, seed=2026)

    def test_bootstrap_3_parents_20_seeds_funded_without_minting_money(self):
        snap = self.world.ecology
        self.assertEqual(len(self.pop), 23)
        self.assertEqual(sum(p.parent_id is None for p in self.pop), 3)
        self.assertEqual(sum(p.parent_id is not None for p in self.pop), 20)
        self.assertEqual(sum((D(v) for v in snap["entitlements"].values()), D(0)), D("90"))
        self.assertEqual(self.world.reserve, D("10"))
        self.assertEqual(self.world.distributable, D("90"))
        self.assertTrue(all(D(v) > 0 for v in snap["entitlements"].values()))
        self.assertTrue(snap["synthetic"])

    def test_30_days_no_double_count_and_stable_replay(self):
        full = run_dates(self.world, self.pop, 0, 30)
        repeated = run_dates(self.world, self.pop, 0, 30)
        half = run_dates(self.world, self.pop, 0, 12)
        recovered = SimulationState.from_json(half.to_json())
        resumed = run_dates(recovered, self.pop, 12, 30)
        self.assertEqual(full.to_json(), repeated.to_json())
        self.assertEqual(full.to_json(), resumed.to_json())
        self.assertEqual(full.day_index, 29)
        self.assertEqual(full.reserve, full.equity * D("0.10"))
        self.assertEqual(full.distributable, full.equity * D("0.90"))
        self.assertEqual(
            sum((D(v) for v in full.ecology["entitlements"].values()), D(0)),
            full.distributable,
        )
        # Ensure 30 real daily biology steps occurred for all registered plants.
        self.assertEqual(full.ecology["plants"]["parent-00"]["age_days"], 50)
        self.assertEqual(full.ecology["plants"]["seed-00"]["age_days"], 30)
        self.assertEqual(full.ecology["groundwater"]["day"], (START + timedelta(days=29)).isoformat())
        self.assertEqual(
            len({e.event_id for e in full.audit}), len(full.audit),
        )
        self.assertEqual(
            sha256(full.to_json().encode()).hexdigest(),
            sha256(repeated.to_json().encode()).hexdigest(),
        )

    def test_margin_intents_respect_observed_free_margin(self):
        step = step_world(
            self.world, START, day_events(START, 0), self.pop, weather(START),
            free_margin="5",
        )
        intents = step.state.ecology["intents"]
        self.assertEqual(len(intents), 23)
        total_margin = sum((D(v["entry_margin"]) for v in intents.values()), D(0))
        self.assertLessEqual(total_margin, D("5"))
        self.assertTrue(all(D(v["leverage"]) <= D("10") for v in intents.values()))
        self.assertLessEqual(
            sum((D(v["notional"]) for v in intents.values()), D(0)),
            D("50"),
        )

    def test_single_leaf_cactus_genome_grows_roots_but_uses_less_margin(self):
        step = step_world(
            self.world, START, day_events(START, 0), self.pop, weather(START),
        )
        p = step.state.ecology["plants"]
        intents = step.state.ecology["intents"]
        cactus = next(x for x in self.pop if x.strategy_id == "parent-02")
        self.assertEqual(cactus.genome.leaf_count, 1)
        self.assertTrue(cactus.genome.photosynthetic_rind)
        self.assertGreater(intents["parent-02"]["light_area"], 0)
        self.assertGreater(p["parent-02"]["root_depth"], p["parent-00"]["root_depth"])
        self.assertLess(D(intents["parent-02"]["entry_margin"]),
                        D(intents["parent-00"]["entry_margin"]))

    def test_no_alpha_from_deposits_and_free_margin_not_invented(self):
        midday = datetime(2026, 1, 1, 12, tzinfo=UTC)
        external = IntradayEvent("cash-deposit", midday, midday,
                                 "external_cashflow", D("50"))
        step = step_world(
            self.world, START, [external], self.pop, weather(START),
            free_margin="0",
        )
        self.assertEqual(step.external_cashflow, D("50"))
        self.assertEqual(step.daily_net_pnl, D("0"))
        self.assertEqual(step.state.equity, D("150"))
        self.assertEqual(step.state.reserve, D("15"))
        self.assertEqual(sum((D(v["entry_margin"])
                              for v in step.state.ecology["intents"].values()), D(0)), D("0"))
        self.assertEqual(step.state.ecology["plants"]["parent-00"]["biomass"], 1.0)

    def test_negative_equity_freezes_new_intents_without_flatten(self):
        midday = datetime(2026, 1, 1, 12, tzinfo=UTC)
        withdraw = IntradayEvent("withdrawal", midday, midday,
                                 "external_cashflow", D("-110"))
        s = step_world(self.world, START, [withdraw], self.pop, weather(START)).state
        self.assertEqual(s.equity, D("-10"))
        self.assertEqual(s.distributable, D("0"))
        self.assertTrue(s.ecology["risk_frozen"])
        self.assertTrue(all(D(v["entry_margin"]) == 0 for v in s.ecology["intents"].values()))
        self.assertTrue(all("close" not in x.kind for x in s.audit))

    def test_population_manifest_change_rejected_after_restart(self):
        with self.assertRaisesRegex(IntegrationError, "identity mismatch"):
            ResearchPlant("different", self.pop[0].genome, self.pop[0].initial_plant)
        spec = self.pop[0]
        altered = (
            ResearchPlant(spec.strategy_id, spec.genome, spec.initial_plant,
                          root_weight=D("5")),
            *self.pop[1:],
        )
        with self.assertRaisesRegex(IntegrationError, "changed since checkpoint"):
            step_world(self.world, START, [], altered, weather(START))

    def test_trade_pnl_without_registered_owner_rejected(self):
        at = datetime(2026, 1, 1, 12, tzinfo=UTC)
        other = IntradayEvent("anonymous", at, at, "pnl_delta", D("1"), "ghost")
        with self.assertRaisesRegex(IntegrationError, "registered plant"):
            step_world(self.world, START, [other], self.pop, weather(START))

    def test_unknown_future_weather_and_repeat_days(self):
        tomorrow = START + timedelta(days=1)
        with self.assertRaisesRegex(Exception, "no lookahead"):
            step_world(self.world, START, [], self.pop, weather(tomorrow))
        step = step_world(self.world, START, [], self.pop, weather(START))
        with self.assertRaisesRegex(ClockError, "next consecutive"):
            step_world(step.state, START, [], self.pop, weather(START))


if __name__ == "__main__":
    unittest.main()
