#!/usr/bin/env python3
"""Per-tile evolution may multiply research experiments; money does not multiply."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import unittest

from plan_tile_experiments import default_plan, plan, validate_activation
from validate_gene_physics import load


class TileResearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = load("fixtures/scene_demo_v0.json")
        self.tiles = load("contracts/parameter_tiles_v1.json")
        self.genomes = load("fixtures/tile_genomes_v1.json")
        self.activation = load("contracts/tile_research_activation_v1.json")
        self.profiles = load("fixtures/tile_seed_profiles_v1.json")
        self.registry = load("contracts/parameter_registry_v1.json")
        self.reference = load("fixtures/parameter_snapshot_v1.json")

    def run_plan(self, *, active=None, profiles=None, genomes=None):
        return plan(self.scene, self.tiles, genomes or self.genomes,
                    active or self.activation, profiles or self.profiles,
                    self.registry, self.reference)

    def test_six_independent_tile_releases_one_canonical_plant(self):
        result = self.run_plan()
        self.assertEqual(result["canonicalPlantCount"], 4)
        self.assertEqual(result["renderInstanceCount"], 6)
        self.assertEqual(result["totalIndependentReleases"], 6)
        self.assertEqual(result["totalSimulatedSeedOutcomes"], 16)
        self.assertEqual(len(set(r["eventId"] for r in result["releases"])), 6)
        self.assertEqual(
            sum(1 for r in result["releases"] if r["plantId"] == "plant-a"), 4
        )
        self.assertEqual(
            sum(1 for r in result["releases"] if r["plantId"] == "plant-c"), 2
        )
        self.assertEqual(result["requestedPaperBudgetTotalUsdt"], "60.00")
        self.assertFalse(result["virtualBudgetRequestsAreFunded"])
        self.assertFalse(result["liveAccountOrOrderWrites"])
        self.assertEqual(result, default_plan())

    def test_same_parent_each_tile_proposes_different_axis_pairs(self):
        result = self.run_plan()
        parent = [r for r in result["releases"] if r["plantId"] == "plant-a"]
        self.assertEqual(len(parent), 4)
        different_pairs = set()
        ids = []
        for r in parent:
            self.assertEqual(len(r["outcomes"]), 3)
            first = r["outcomes"][0]
            self.assertEqual(first["status"], "unapproved_offspring_gene_patch")
            pair = tuple(sorted(first["proposedAxes"]))
            different_pairs.add(pair)
            self.assertEqual(first["notBeforeDay"], 8)
            self.assertFalse(first["moneyBooked"])
            self.assertFalse(first["newCanonicalPlantCreated"])
            self.assertEqual(first["baseGenomeRevision"], "synthetic-plant-a-v1")
            ids.extend(o["seedId"] for o in r["outcomes"])
        self.assertEqual(len(different_pairs), 4)
        self.assertEqual(len(set(ids)), 12)

    def test_capital_none_still_can_evolve(self):
        result = self.run_plan()
        self.assertTrue(any(r["tileId"] == "sol-sensitivity-ttp" for r in result["releases"]))
        zero = next(x for x in result["tileStates"] if x["tileId"] == "sol-sensitivity-ttp")
        self.assertEqual(zero["capitalMode"], "off")
        self.assertEqual(zero["paperBudgetRequestedUsdt"], "0.00")
        self.assertTrue(zero["evolutionEnabled"])

    def test_evolution_pause_does_not_change_virtual_paper_budget(self):
        cfg = deepcopy(self.activation)
        cfg["tiles"][0]["evolutionEnabled"] = False
        result = self.run_plan(active=cfg)
        self.assertEqual(result["totalIndependentReleases"], 4)
        self.assertEqual(result["totalSimulatedSeedOutcomes"], 11)
        self.assertEqual(result["requestedPaperBudgetTotalUsdt"], "60.00")
        self.assertEqual([s["plantId"] for s in result["skipped"]
                          if s["tileId"] == "sol-imbase"], ["plant-a", "plant-c"])

    def test_paper_off_does_not_turn_off_seed_emission(self):
        cfg = deepcopy(self.activation)
        cfg["tiles"][0].update(capitalMode="off", paperBudgetRequestedUsdt="0.00",
                               paperFeedMode="none")
        result = self.run_plan(active=cfg)
        self.assertEqual(result["totalIndependentReleases"], 6)
        self.assertEqual(result["requestedPaperBudgetTotalUsdt"], "35.00")
        self.assertFalse(result["liveAccountOrOrderWrites"])

    def test_switch_to_realtime_feed_is_still_paper(self):
        cfg = deepcopy(self.activation)
        cfg["tiles"][1]["paperFeedMode"] = "realtime"
        result = self.run_plan(active=cfg)
        row = next(t for t in result["tileStates"] if t["tileId"] == "sol-lookback-sensitivity")
        self.assertEqual(row["capitalMode"], "paper")
        self.assertEqual(row["paperFeedMode"], "realtime")
        self.assertFalse(result["liveAccountOrOrderWrites"])

    def test_fail_closed_on_live_without_backend_gate(self):
        cfg = deepcopy(self.activation)
        cfg["tiles"][0].update(capitalMode="live",paperBudgetRequestedUsdt="0.00",
                               paperFeedMode="none")
        with self.assertRaisesRegex(ValueError, "Live trading must fail closed"):
            self.run_plan(active=cfg)

    def test_no_oversubscribed_or_faked_paper_budgets(self):
        cfg = deepcopy(self.activation)
        cfg["tiles"][0]["paperBudgetRequestedUsdt"] = "99.00"
        with self.assertRaisesRegex(ValueError, "exceed"):
            self.run_plan(active=cfg)
        cfg2 = deepcopy(self.activation)
        cfg2["tiles"][1]["paperBudgetRequestedUsdt"] = "25"
        with self.assertRaisesRegex(ValueError, "two-decimal"):
            self.run_plan(active=cfg2)
        cfg3 = deepcopy(self.activation)
        cfg3["tiles"][3]["capitalMode"] = "off"
        with self.assertRaisesRegex(ValueError, "Off mode"):
            self.run_plan(active=cfg3)

    def test_missing_explicit_release_profile_blocks_seed_experiment_only(self):
        profiles = deepcopy(self.profiles)
        profiles["profiles"] = [p for p in profiles["profiles"] if p["plantId"] == "plant-a"]
        result = self.run_plan(profiles=profiles)
        self.assertEqual(result["renderInstanceCount"], 6)
        self.assertEqual(result["totalIndependentReleases"], 4)
        self.assertEqual(result["totalSimulatedSeedOutcomes"], 12)
        self.assertEqual({s["plantId"] for s in result["skipped"]}, {"plant-c"})

    def test_bad_or_stale_profile_fails_closed(self):
        cfg = deepcopy(self.profiles)
        cfg["profiles"][0]["heightM"] = 99
        with self.assertRaisesRegex(ValueError, "outside gene range"):
            self.run_plan(profiles=cfg)
        cfg2 = deepcopy(self.profiles)
        cfg2["profiles"][0]["heightM"] = 1.9
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.run_plan(profiles=cfg2)
        cfg3 = deepcopy(self.profiles)
        cfg3["sceneRef"]["dayIndex"] = 9
        with self.assertRaisesRegex(ValueError, "stale"):
            self.run_plan(profiles=cfg3)

    def test_genome_membership_controls_releases_even_when_evolution_on(self):
        genomes = deepcopy(self.genomes)
        g = next(p for p in genomes["plants"] if p["plantId"] == "plant-a")
        del g["geneValues"]["research.ttp_pct"]
        result = self.run_plan(genomes=genomes)
        self.assertEqual(result["renderInstanceCount"], 4)
        self.assertEqual(result["totalIndependentReleases"], 4)
        self.assertEqual(result["canonicalPlantCount"], 4)

    def test_determinism_and_unique_seed_event_ids_across_tiles(self):
        result = self.run_plan()
        pairs = set()
        for release in result["releases"]:
            for item in release["outcomes"]:
                pairs.add(item["seedId"])
                self.assertEqual(item["sourcePlantId"], release["plantId"])
                self.assertEqual(item["releaseEventId"], release["eventId"])
                self.assertIn(item["status"], ("unapproved_offspring_gene_patch",
                                               "outside_tile_unassigned"))
        self.assertEqual(len(pairs), result["totalSimulatedSeedOutcomes"])
        self.assertEqual(result, self.run_plan())


if __name__ == "__main__":
    unittest.main(verbosity=2)
