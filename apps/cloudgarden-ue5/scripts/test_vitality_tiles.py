#!/usr/bin/env python3
"""Synthetic UE5 preflight: one vitality owner, many copies, zero extra ticks."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import unittest
from validate_gene_physics import load
from project_vitality_tiles import default_projection, project_vitality


class VitalityTileTests(unittest.TestCase):
    def setUp(self):
        self.scene = load("fixtures/scene_demo_v0.json")
        self.catalog = load("contracts/parameter_tiles_v1.json")
        self.genes = load("fixtures/tile_genomes_v1.json")
        self.snap = load("fixtures/vitality_snapshot_v1.json")

    def run_projection(self, snapshot=None, genes=None, scene=None):
        return project_vitality(scene or self.scene, self.catalog,
                                genes or self.genes, snapshot or self.snap)

    def test_repeated_instances_share_canonical_unchanged_values(self):
        original = deepcopy(self.snap)
        result = self.run_projection()
        self.assertEqual((result["canonicalPlantCount"], result["renderInstanceCount"]), (4, 6))
        self.assertEqual(Counter(x["plantId"] for x in result["instances"]),
                         {"plant-a": 4, "plant-c": 2})
        copies = [x for x in result["instances"] if x["plantId"] == "plant-a"]
        self.assertEqual(len({(x["vigor"], x["patience"], x["inactiveDays"])
                              for x in copies}), 1)
        self.assertEqual([x["vigor"] for x in copies], [81.991] * 4)
        self.assertEqual(result["simulationTicksApplied"], 0)
        self.assertFalse(result["capitalOrGenomeWrites"])
        self.assertEqual(self.snap, original)
        self.assertEqual(result, default_projection())

    def test_grace_period_is_metadata_only_and_not_a_tick(self):
        self.assertFalse(any(x["inactivityDecayEligible"]
                             for x in self.run_projection()["instances"]))
        snap = deepcopy(self.snap)
        scene = deepcopy(self.scene)
        scene["scene"]["dayIndex"] = 21
        snap["sceneRef"]["dayIndex"] = 21
        catalog = deepcopy(self.catalog)
        catalog["sceneRef"]["dayIndex"] = 21
        genes = deepcopy(self.genes)
        genes["sceneRef"]["dayIndex"] = 21
        snap["plants"][0]["inactiveDays"] = 15
        output = project_vitality(scene, catalog, genes, snap)
        a = [x for x in output["instances"] if x["plantId"] == "plant-a"]
        self.assertTrue(all(x["inactivityDecayEligible"] for x in a))
        self.assertTrue(all(x["vigor"] == 81.991 for x in a))
        self.assertEqual(output["simulationTicksApplied"], 0)

    def test_missing_duplicate_or_stale_record_rejected(self):
        missing = deepcopy(self.snap)
        missing["plants"].pop()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.run_projection(snapshot=missing)
        duplicate = deepcopy(self.snap)
        duplicate["plants"].append(deepcopy(duplicate["plants"][0]))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.run_projection(snapshot=duplicate)
        stale = deepcopy(self.snap)
        stale["sceneRef"]["dayIndex"] = 2
        with self.assertRaisesRegex(ValueError, "stale"):
            self.run_projection(snapshot=stale)

    def test_invalid_vigor_patience_idle_and_identity_fail_closed(self):
        for key, bad in (("vigor", 101), ("vigor", float("nan")), ("patience", 1.01),
                         ("patience", True), ("inactiveDays", -1),
                         ("inactiveDays", 8), ("inactiveDays", 1.5),
                         ("status", "archived"), ("revision", "out-of-date")):
            snap = deepcopy(self.snap)
            snap["plants"][0][key] = bad
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                self.run_projection(snapshot=snap)

    def test_unsafe_mode_or_attempted_writes_rejected(self):
        for key, bad in (("oneDailyUpdatePerCanonicalPlant", False),
                         ("individualTileVitalityState", True),
                         ("readOnly", False), ("liveTrading", True)):
            snap = deepcopy(self.snap)
            snap["policy"][key] = bad
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "writes"):
                self.run_projection(snapshot=snap)
        snap = deepcopy(self.snap)
        snap["authority"] = "external_unverified"
        with self.assertRaisesRegex(ValueError, "untrusted"):
            self.run_projection(snapshot=snap)

    def test_canonical_g2_readonly_export_contract_is_type_compatible(self):
        # Transport authentication is a separate future task; this verifies
        # only typed, identity-matched read-only consumption.
        snap = deepcopy(self.snap)
        snap["authority"] = "canonical_readonly_export"
        output = self.run_projection(snapshot=snap)
        self.assertEqual(output["declaredSourceAuthority"], "canonical_readonly_export")
        self.assertEqual(output["authority"], "read_only_preview_no_transport_trust")
        self.assertEqual(output["simulationTicksApplied"], 0)
        self.assertEqual(output["renderInstanceCount"], 6)

    def test_archived_or_missing_gene_is_not_rendered(self):
        result = self.run_projection()
        self.assertFalse(any(x["plantId"] == "plant-b" for x in result["instances"]))
        self.assertFalse(any(x["plantId"] == "plant-d" for x in result["instances"]))
        genes = deepcopy(self.genes)
        genes["plants"][0]["geneValues"].pop("research.ttp_pct")
        result2 = self.run_projection(genes=genes)
        self.assertEqual(sum(x["plantId"] == "plant-a" for x in result2["instances"]), 2)
        self.assertEqual(result2["canonicalPlantCount"], 4)

if __name__ == "__main__":
    unittest.main(verbosity=2)
