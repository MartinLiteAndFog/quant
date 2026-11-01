"""ECO-06 deterministic paper plant physiology and global reserve invariants."""
import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal as D

from quant.ecosystem.ecology.physiology import (
    EcologyConfig, EcologyError, GroundwaterLine, Plant, PlantGenome,
    PlantInput, Weather, ecology_day, initial_groundwater, step_groundwater,
    water_target,
)

UTC = timezone.utc
TODAY = date(2026, 11, 1)
OBSERVED = datetime(2026, 11, 1, 17, tzinfo=UTC)
WEATHER = Weather(OBSERVED, OBSERVED, 0.8, synthetic=True)
CFG = EcologyConfig()


def gene(**kwargs):
    base = dict(
        root_growth=0.45, root_depth=0.6, root_spread=0.6,
        branching=0.5, leaf_count=20, leaf_size=0.6, bark_strength=0.4,
        dormancy=0.05, photosynthetic_rind=False, rind_efficiency=0.5,
        reproduction_budget=0.10, protective_stop=True,
        max_leverage=D("9"), max_margin_fraction=0.9,
    )
    base.update(kwargs)
    return PlantGenome(**base)


def plant(name, root=0.2, leaf=0.5, access=False):
    return Plant(name, 12, 1.0, root, root, leaf, 0.8, access)


class EcologicalPhysiologyTests(unittest.TestCase):
    def setUp(self):
        self.initial = initial_groundwater(TODAY, "100", CFG)

    def test_daily_growth_and_90_10_budget_without_new_money(self):
        items = [
            PlantInput(plant("grass"), gene(), 0.02, D("30"), True),
            PlantInput(plant("tree"), gene(root_growth=0.9, root_depth=1,
                                           root_spread=1), 0.02, D("30"), True),
            PlantInput(plant("bush"), gene(leaf_count=5, branching=0.1),
                       -0.02, D("30"), False),
        ]
        d = ecology_day(self.initial, TODAY, "100", items, WEATHER)
        self.assertEqual(d.reserve, D("10"))
        self.assertEqual(d.distributable, D("90"))
        self.assertEqual(d.unassigned_budget, D("0"))
        self.assertEqual(d.groundwater.day, TODAY)
        self.assertTrue(all(o.plant.age_days == 13 for o in d.plants))
        self.assertGreater(d.plants[1].plant.root_depth, d.plants[0].plant.root_depth)
        self.assertGreater(d.plants[1].plant.root_spread, d.plants[0].plant.root_spread)
        self.assertLess(d.plants[1].own_utilization, d.plants[0].own_utilization)
        self.assertLess(d.plants[2].plant.biomass, 1)
        self.assertLessEqual(sum((o.intended_margin for o in d.plants), D(0)), D("90"))
        self.assertLessEqual(sum((o.intended_notional for o in d.plants), D(0)), D("900"))

    def test_cactus_one_leaf_green_rind_separate_from_armour(self):
        cactus = gene(leaf_count=1, leaf_size=0, branching=0,
                      photosynthetic_rind=True, rind_efficiency=1, bark_strength=0.95,
                      root_growth=1, root_depth=1, root_spread=0.1)
        leaf_only = gene(leaf_count=1, leaf_size=0, branching=0,
                         photosynthetic_rind=False, rind_efficiency=1, bark_strength=0.95)
        inputs = [PlantInput(plant("cactus", leaf=0), cactus, 0.01, D("40"), True),
                  PlantInput(plant("leaf-only", leaf=0), leaf_only, 0.01, D("40"), True)]
        d = ecology_day(self.initial, TODAY, "100", inputs, WEATHER)
        self.assertGreater(d.plants[0].effective_leaf_area, 0)
        self.assertEqual(d.plants[1].effective_leaf_area, 0)
        self.assertGreater(d.plants[0].intended_margin, d.plants[1].intended_margin)
        # Same bark protects; alone it does NOT create sunlight absorption.
        self.assertEqual(cactus.bark_strength, leaf_only.bark_strength)
        self.assertEqual(d.plants[0].plant.age_days, 13)
        with self.assertRaisesRegex(EcologyError, "leaf_count"):
            gene(leaf_count=0)

    def test_high_branches_leaves_give_more_sun_and_margin_within_caps(self):
        low = gene(branching=0, leaf_count=2, leaf_size=0.5, root_growth=0)
        high = gene(branching=1, leaf_count=64, leaf_size=1, root_growth=0)
        d = ecology_day(self.initial, TODAY, "100", [
            PlantInput(plant("low"), low, 0, D("40"), False),
            PlantInput(plant("high"), high, 0, D("40"), False),
        ], WEATHER)
        self.assertGreater(d.plants[1].effective_leaf_area, d.plants[0].effective_leaf_area)
        self.assertGreater(d.plants[1].own_utilization, d.plants[0].own_utilization)
        self.assertGreater(d.plants[1].effective_leverage, d.plants[0].effective_leverage)
        self.assertLessEqual(d.plants[1].effective_leverage, D("9"))

    def test_dormancy_keeps_capital_as_entitlement_without_trading(self):
        sleeping = gene(dormancy=1, root_growth=0)
        awake = gene(dormancy=0, root_growth=0)
        d = ecology_day(self.initial, TODAY, "100", [
            PlantInput(plant("sleeper"), sleeping, 0, D("40"), True),
            PlantInput(plant("awake"), awake, 0, D("40"), True),
        ], WEATHER)
        self.assertEqual(d.plants[0].intended_margin, D("0"))
        self.assertGreater(d.plants[1].intended_margin, D("0"))
        self.assertEqual(d.unassigned_budget, D("10"))
        self.assertEqual(d.distributable, D("90"))

    def test_global_limits_override_sun_and_genome(self):
        cfg = EcologyConfig(global_leverage_cap=D("3"), global_margin_fraction_cap=0.2)
        out = ecology_day(initial_groundwater(TODAY, "100", cfg), TODAY, "100",
                          [PlantInput(plant("giant"), gene(branching=1, leaf_count=64,
                                                         leaf_size=1, max_leverage=D("20")), 0,
                                      D("50"), True)], Weather(OBSERVED, OBSERVED, 1, True), cfg)
        self.assertLessEqual(out.plants[0].effective_leverage, D("3"))
        self.assertLessEqual(out.plants[0].intended_margin, D("10"))

    def test_groundwater_target_asymptotic_and_rate_limited(self):
        self.assertGreater(water_target("0", CFG), water_target("100", CFG))
        self.assertGreater(water_target("100", CFG), water_target("1000", CFG))
        self.assertGreater(water_target("1e100", CFG), 0)
        up = step_groundwater(self.initial, TODAY, "1000", CFG)
        self.assertLess(up.depth, self.initial.depth)
        self.assertGreater(up.depth, up.target_depth)
        down = step_groundwater(up, TODAY + timedelta(days=1), "10", CFG)
        self.assertGreater(down.depth, up.depth)
        self.assertLess(down.depth, down.target_depth)
        self.assertGreater(down.depth, CFG.min_water_depth)

    def test_access_requires_roots_and_evidence_with_hysteresis(self):
        cfg = EcologyConfig(reference_equity=D("100"), hysteresis_depth=0.10)
        prev = initial_groundwater(TODAY, "100", cfg)
        depth = prev.depth
        q = [
            PlantInput(plant("young", root=0), gene(root_growth=0), 0, D("10"), True),
            PlantInput(plant("deep", root=depth + 1), gene(root_growth=0), 0, D("10"), True),
            PlantInput(plant("no-evidence", root=depth + 1), gene(root_growth=0), 0, D("10"), False),
        ]
        d = ecology_day(prev, TODAY, "100", q, WEATHER, cfg)
        self.assertEqual([p.water_eligible for p in d.plants], [False, True, False])
        self.assertFalse(d.plants[0].plant.water_access)
        self.assertTrue(d.plants[1].plant.water_access)
        self.assertFalse(d.plants[2].plant.water_access)

    def test_access_cap_never_forces_extra_water_eligibility(self):
        cfg = EcologyConfig(max_water_access_fraction=0.34)
        base = initial_groundwater(TODAY, "100", cfg)
        deep = base.depth + 2
        q = [PlantInput(plant(str(i), root=deep), gene(root_growth=0), 0, D("10"), True)
             for i in range(3)]
        d = ecology_day(base, TODAY, "100", q, WEATHER, cfg)
        self.assertEqual(sum(p.water_eligible for p in d.plants), 1)

    def test_no_forward_looking_weather_and_no_double_counted_budgets(self):
        q = [PlantInput(plant("x"), gene(), 0, D("91"), True)]
        with self.assertRaisesRegex(EcologyError, "90%"):
            ecology_day(self.initial, TODAY, "100", q, WEATHER)
        later = Weather(OBSERVED, datetime(2026, 11, 2, tzinfo=UTC), 0.8, True)
        with self.assertRaisesRegex(EcologyError, "no lookahead"):
            ecology_day(self.initial, TODAY, "100", [], later)
        with self.assertRaisesRegex(EcologyError, "duplicate strategy"):
            ecology_day(self.initial, TODAY, "100", [
                PlantInput(plant("x"), gene(), 0, D("10"), True),
                PlantInput(plant("x"), gene(), 0, D("10"), True),
            ], WEATHER)

    def test_no_photo_bark_hardcap_removal_or_nonfinite_inputs(self):
        with self.assertRaisesRegex(EcologyError, "protective_stop"):
            gene(protective_stop=False)
        with self.assertRaisesRegex(EcologyError, "root_spread"):
            gene(root_spread=3)
        with self.assertRaisesRegex(EcologyError, "invalid daily"):
            PlantInput(plant("x"), gene(), float("nan"), D("10"), True)
        self.assertEqual(ecology_day(self.initial, TODAY, "-10", [], WEATHER).distributable, D("0"))


if __name__ == "__main__":
    unittest.main()
