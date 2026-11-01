"""ECO-09 paper-only lifecycle invariants."""
from dataclasses import replace
from datetime import date
from decimal import Decimal as D
import unittest

from quant.ecosystem.lifecycle.predation import (
    Candidate, PredatorConfig, Reconciliation, LifecycleError,
    attack_probability, huntable, select_predators, finalize_archive,
)

DAY = date(2026, 11, 1)
CFG = PredatorConfig(base_attack_chance=1, attacks_per_day=3)


def plant(name="x", **patch):
    v = dict(strategy_id=name, account_id="paper", tile_id="sol-imba", age_days=500,
             vitality=.3, credibility=.9, bark_strength=.2, bark_evidence=.5,
             effective_trades=40, recent_eligible_opportunities=4, dormancy_days=0,
             recent_mark_to_market_loss=False, entitlement=D("20"),
             used_margin=D("0"), reserved_margin=D("0"), status="alive")
    v.update(patch)
    return Candidate(**v)


def confirmation(name="x", **patch):
    v = dict(strategy_id=name, positions_flat=True, orders_cancelled=True,
             fills_reconciled=True, free_margin_confirmed=True,
             genome_version="genome-v3", research_ref="archive://"+name)
    v.update(patch)
    return Reconciliation(**v)


class PredatorTests(unittest.TestCase):
    def test_protected_band_and_seedlings_never_hunted(self):
        all_protected = [
            plant("healthy", vitality=.9), plant("band", vitality=.75),
            plant("exact", vitality=.7), plant("baby", vitality=.1, age_days=3),
        ]
        for hunter in ("weakest", "weighted_random", "age"):
            result, events = select_predators("sim", 1, DAY, all_protected,
                                             hunter=hunter, config=CFG)
            self.assertEqual(events, ())
            self.assertTrue(all(p.status == "alive" for p in result))

    def test_evidence_grace_is_limited_and_bark_needs_evidence(self):
        young = plant("young", age_days=50, vitality=.1, effective_trades=2)
        old = replace(young, age_days=180)
        self.assertFalse(huntable(young, CFG))
        self.assertTrue(huntable(old, CFG))
        shielded = replace(old, bark_strength=1, bark_evidence=1, credibility=1)
        unproven = replace(old, bark_strength=1, bark_evidence=0)
        self.assertLess(attack_probability(shielded, CFG),
                        attack_probability(unproven, CFG))
        self.assertGreater(attack_probability(shielded, CFG), 0)

    def test_cactus_no_eligible_opportunities_in_sleep(self):
        cactus = plant("cactus", status="dormant", vitality=.25,
                       bark_strength=1, bark_evidence=.8, credibility=.9,
                       dormancy_days=300, recent_eligible_opportunities=0)
        self.assertFalse(huntable(cactus, CFG))
        self.assertTrue(huntable(replace(cactus, recent_mark_to_market_loss=True), CFG))
        self.assertTrue(huntable(replace(cactus, dormancy_days=400), CFG))

    def test_seeded_hunters_deterministic_with_no_order_or_capital_release(self):
        original = [plant(str(i), vitality=.06 + i*.05, bark_strength=0,
                          bark_evidence=0) for i in range(8)]
        for hunter in ("weakest", "weighted_random", "age"):
            a, ae = select_predators("sim", 771, DAY, original,
                                    hunter=hunter, config=CFG)
            b, be = select_predators("sim", 771, DAY, list(reversed(original)),
                                    hunter=hunter, config=CFG)
            self.assertEqual({e.strategy_id for e in ae}, {e.strategy_id for e in be})
            self.assertGreater(len(ae), 0)
            self.assertTrue(all(e.reusable_entitlement == 0 for e in ae))
            self.assertEqual(sum(p.entitlement for p in a), D("160"))
            self.assertTrue(all(p.status == "closing" for p in a if
                                p.strategy_id in {e.strategy_id for e in ae}))

    def test_locked_margin_never_released_without_reconciled_snapshot(self):
        p = replace(plant("x", vitality=0, used_margin=D("4"), reserved_margin=D("2")),
                    status="closing")
        with self.assertRaisesRegex(LifecycleError, "margin still locked"):
            finalize_archive("sim", DAY, [p], [confirmation()])
        live, archived, events = finalize_archive(
            "sim", DAY, [p], [confirmation(fills_reconciled=False)])
        self.assertEqual(live, (p,))
        self.assertEqual(archived, ())
        self.assertEqual(events, ())

    def test_confirmed_flat_archive_releases_entitlement_not_equity(self):
        p = replace(plant("x", vitality=0), status="closing")
        live, archive, events = finalize_archive("sim", DAY, [p], [confirmation()])
        self.assertEqual(live, ())
        self.assertEqual(len(archive), 1)
        self.assertEqual(archive[0].research_ref, "archive://x")
        self.assertEqual(events[0].reusable_entitlement, D("20"))
        self.assertEqual(events[0].kind, "archived")
        again = finalize_archive("sim", DAY, live, (), archive)
        self.assertEqual(again[1], archive)
        self.assertEqual(again[2], ())
        with self.assertRaisesRegex(LifecycleError, "already archived"):
            finalize_archive("sim", DAY, [p], (), archive)

    def test_cannot_archive_still_living(self):
        with self.assertRaisesRegex(LifecycleError, "closing request"):
            finalize_archive("sim", DAY, [plant()], [confirmation()])

    def test_invalid_values_or_duplicates_rejected(self):
        with self.assertRaisesRegex(LifecycleError, "vitality"):
            plant(vitality=float("nan"))
        with self.assertRaisesRegex(LifecycleError, "entitlement"):
            plant(entitlement="-3")
        with self.assertRaisesRegex(LifecycleError, "duplicate"):
            select_predators("sim", 2, DAY, [plant(), plant()])
        with self.assertRaisesRegex(LifecycleError, "unknown strategy"):
            finalize_archive("sim", DAY, [plant()], [confirmation("other")])
        with self.assertRaisesRegex(LifecycleError, "attacks_per_day"):
            PredatorConfig(attacks_per_day=-1)
        with self.assertRaisesRegex(LifecycleError, "hunt/protected thresholds"):
            PredatorConfig(hunt_threshold=.9, protected_threshold=.8)


if __name__ == "__main__":
    unittest.main()
