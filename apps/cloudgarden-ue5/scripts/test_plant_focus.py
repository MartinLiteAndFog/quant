#!/usr/bin/env python3
"""Independent visual plant focusing must not duplicate strategy/finance/biology."""
from __future__ import annotations

from copy import deepcopy
import unittest

from build_hud_snapshot import build_hud_snapshot
from build_plant_inspector import build_plant_inspectors, make_focus_interaction
from export_unreal_bootstrap import build_bootstrap
from validate_gene_physics import load


class PlantFocusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bootstrap = build_bootstrap()
        cls.mock = load("fixtures/hud_mock_finance_v1.json")

    def test_all_canonical_plants_have_only_one_inspector(self):
        result = build_hud_snapshot(self.bootstrap, self.mock)
        entries = result["plantInspectors"]
        self.assertEqual(len(entries), 4)
        self.assertEqual(set(entries), {"plant-a", "plant-b", "plant-c", "plant-d"})
        self.assertEqual(len(entries["plant-a"]["tileInstanceIds"]), 4)
        self.assertEqual(len(entries["plant-c"]["tileInstanceIds"]), 2)
        self.assertEqual(entries["plant-b"]["tileInstanceIds"], [])
        self.assertEqual(entries["plant-a"]["visualPreview"]["sourcePlantId"], "plant-a")
        self.assertTrue(all(not p["visualPreview"]["spawnNewCanonicalEntity"]
                            and not p["visualPreview"]["enablePhysics"]
                            and not p["visualPreview"]["enableSeedEmitters"]
                            and not p["visualPreview"]["enableTrading"]
                            for p in entries.values()))

    def test_one_fictional_plant_equity_series_and_high_water_mark(self):
        entries = build_plant_inspectors(self.bootstrap, self.mock)
        a = entries["plant-a"]
        self.assertEqual(a["metrics"]["individualEquityUsdt"], "31.50")
        self.assertEqual(a["metrics"]["netPnlUsdt"], "11.50")
        self.assertEqual(a["metrics"]["returnPct"], "57.50")
        self.assertEqual(a["metrics"]["highWaterMarkUsdt"], "31.50")
        self.assertEqual(a["metrics"]["maxDrawdownPct"], "4.35")
        self.assertEqual(a["metrics"]["peakToCurrentDrawdownPct"], "0.00")
        self.assertEqual(a["metrics"]["closedTrades"], 13)
        self.assertEqual(a["metrics"]["winRatePct"], "61.54")
        self.assertEqual(a["equitySeries"][0]["dayIndex"], 0)
        self.assertEqual(a["equitySeries"][-1]["dayIndex"], 7)
        self.assertEqual(a["metrics"]["biologicalBiomass"], None)

    def test_missing_source_does_not_invent_selected_plant_biomass(self):
        entries = build_plant_inspectors(self.bootstrap, None)
        for entry in entries.values():
            self.assertIsNone(entry["metrics"]["individualEquityUsdt"])
            self.assertIsNone(entry["metrics"]["highWaterMarkUsdt"])
            self.assertIsNone(entry["metrics"]["maxDrawdownPct"])
            self.assertEqual(entry["equitySeries"], [])
            self.assertEqual(entry["financialSource"], "unavailable_no_plant_ledger")

    def test_archived_plant_history_ends_on_archive_day(self):
        entry = build_plant_inspectors(self.bootstrap, self.mock)["plant-b"]
        self.assertEqual(entry["status"], "archived")
        self.assertEqual(entry["equitySeries"][-1]["dayIndex"], 5)
        self.assertEqual(entry["metrics"]["individualEquityUsdt"], "17.50")
        self.assertEqual(entry["metrics"]["netPnlUsdt"], "-2.50")

    def test_offspring_history_starts_at_birth_not_before(self):
        entry = build_plant_inspectors(self.bootstrap, self.mock)["plant-c"]
        self.assertEqual(entry["bornDay"], 6)
        self.assertEqual(entry["generation"], 1)
        self.assertEqual(entry["ageDays"], 1)
        self.assertEqual(entry["equitySeries"][0]["dayIndex"], 6)
        self.assertEqual(entry["metrics"]["individualEquityUsdt"], "19.90")

    def test_equity_history_forgery_or_time_jump_rejected(self):
        bad = deepcopy(self.mock)
        bad["plantResults"][0]["plantNavHistory"][-1]["equityUsdt"] = "999.00"
        with self.assertRaisesRegex(ValueError, "does not reconcile"):
            build_plant_inspectors(self.bootstrap, bad)
        bad2 = deepcopy(self.mock)
        bad2["plantResults"][2]["plantNavHistory"][0]["dayIndex"] = 0
        with self.assertRaisesRegex(ValueError, "temporally invalid"):
            build_plant_inspectors(self.bootstrap, bad2)
        bad3 = deepcopy(self.mock)
        bad3["plantResults"][1]["plantNavHistory"][-1]["dayIndex"] = 7
        with self.assertRaisesRegex(ValueError, "temporally invalid"):
            build_plant_inspectors(self.bootstrap, bad3)

    def test_focus_interaction_miniature_and_main_camera_are_visual_only(self):
        focus = make_focus_interaction(self.bootstrap)
        self.assertEqual(focus["selectionKey"], "canonical_plant_id")
        self.assertEqual(focus["world"], "single_continuous_garden")
        self.assertEqual(focus["hudPlantPreview"]["anchor"], "lower_left")
        self.assertTrue(focus["hudPlantPreview"]["no_CanonicalPlantActor_spawn"])
        self.assertTrue(focus["mainCamera"]["preserveResearchDay"])
        self.assertTrue(focus["mainCamera"]["noHardCameraCut"])
        self.assertEqual(focus["mainCamera"]["blendSeconds"], 0.65)
        self.assertEqual(focus["sideEffects"], {
            "canonicalPlantBirths": 0, "extraStrategies": 0,
            "marketOrders": 0, "capitalBookings": 0
        })

    def test_camera_focus_does_not_change_scene_or_ledger(self):
        from build_hud_snapshot import build_hud_snapshot
        x = build_hud_snapshot(self.bootstrap, self.mock)
        self.assertEqual(x["focus"]["selectedPlantId"], "plant-a")
        self.assertEqual(x["financialMetrics"]["totalBiomassPortfolioUsdt"]["value"], "108.90")
        self.assertEqual(x["ecologyMetrics"]["canonicalPlants"], 4)
        self.assertEqual(x["plantInspectors"]["plant-a"]["tileInstanceIds"],
                         self.bootstrap["plantToVisualInstances"]["plant-a"])
        self.assertFalse(x["permissions"]["unrealRuntimeMaySendExchangeOrders"]
                         if "permissions" in x else self.bootstrap["permissions"]["unrealRuntimeMaySendExchangeOrders"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
