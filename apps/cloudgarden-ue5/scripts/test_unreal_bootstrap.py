#!/usr/bin/env python3
"""Verify the one-file, synthetic UE5 scene handoff can be safely consumed."""
from __future__ import annotations

from copy import deepcopy
import json
import unittest

from export_unreal_bootstrap import build_bootstrap, verify_bootstrap
from tile_research_journal import digest
from validate_gene_physics import load


class UnrealBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = build_bootstrap()

    def test_exports_one_complete_deterministic_non_live_bundle(self):
        b = self.bundle
        self.assertEqual(b, build_bootstrap())
        self.assertEqual(b["sceneRef"], {"seed": "ue5-visual-preview-v0", "dayIndex": 7})
        self.assertEqual(b["totals"], {
            "canonicalPlants": 4, "visualInstances": 6,
            "independentResearchReleases": 6, "referenceSeedOutcomes": 16,
            "journalEvents": 42, "paperBudgetRequestedUsdt": "60.00",
            "paperFundsActuallyBookedUsdt": "0.00", "liveOrders": 0,
        })
        self.assertEqual(len(b["geneAxisCatalog"]), 4)
        self.assertEqual(len(b["parameterTiles"]), 4)
        self.assertEqual(len(b["tileDecisions"]), 16)
        self.assertEqual(len(b["unrealPropertyPreview"]["assignments"]), 13)
        self.assertEqual(verify_bootstrap(b), b["totals"])

    def test_json_roundtrip_and_file_fingerprints(self):
        b = json.loads(json.dumps(self.bundle))
        self.assertEqual(verify_bootstrap(b), b["totals"])
        self.assertEqual(len(b["sourceFileSha256"]), 8)
        self.assertTrue(all(len(digest) == 64 for digest in b["sourceFileSha256"].values()))
        self.assertEqual(
            b["bundleSha256"],
            digest({k: v for k, v in b.items() if k != "bundleSha256"}),
        )

    def test_same_plant_repeated_but_no_duplicated_canonical_entities(self):
        b = self.bundle
        by_id = {p["id"] for p in b["sourcePlants"]}
        self.assertEqual(len(by_id), 4)
        self.assertEqual({p["plantId"] for p in b["tileVisualInstances"]}, {"plant-a", "plant-c"})
        self.assertEqual(len(b["plantToVisualInstances"]["plant-a"]), 4)
        self.assertEqual(len(b["plantToVisualInstances"]["plant-c"]), 2)
        self.assertEqual(b["plantToVisualInstances"]["plant-b"], [])
        self.assertEqual(len(b["seedResearchPlan"]), 6)
        self.assertEqual(len({p["eventId"] for p in b["seedResearchPlan"]}), 6)

    def test_no_source_write_permissions_or_funding(self):
        perm = self.bundle["permissions"]
        for key in ("unrealRuntimeMayModifySourceGenome",
                    "unrealRuntimeMayBookCapital",
                    "unrealRuntimeMaySendExchangeOrders",
                    "unrealRuntimeMayAdoptResearchCandidates"):
            self.assertIs(perm[key], False)
        self.assertIs(perm["modeSwitchesRequireBackendAcceptance"], True)
        self.assertEqual(self.bundle["totals"]["paperFundsActuallyBookedUsdt"], "0.00")
        self.assertEqual(self.bundle["totals"]["liveOrders"], 0)

    def test_modification_without_new_digest_rejected(self):
        damaged = deepcopy(self.bundle)
        damaged["parameterTiles"][0]["sizeM"] = [100, 100]
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            verify_bootstrap(damaged)

    def test_self_rehashed_tamper_still_rejected_against_source(self):
        damaged = deepcopy(self.bundle)
        damaged["totals"]["liveOrders"] = 5
        damaged["bundleSha256"] = digest({k: v for k, v in damaged.items()
                                           if k != "bundleSha256"})
        with self.assertRaisesRegex(ValueError, "does not match source"):
            verify_bootstrap(damaged)

    def test_camera_does_not_have_a_separate_population(self):
        b = self.bundle
        self.assertEqual(b["cameraJourney"]["worldRef"]["sceneSeed"], b["sceneRef"]["seed"])
        self.assertEqual(b["cameraJourney"]["worldRef"]["dayIndex"], b["sceneRef"]["dayIndex"])
        self.assertEqual(
            {s["mode"] for s in b["cameraJourney"]["states"]}, {"atlas", "explore"}
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
