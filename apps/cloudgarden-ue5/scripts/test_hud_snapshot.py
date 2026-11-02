#!/usr/bin/env python3
"""HUD contract: unique-plant finance rankings and optional explicit mock series."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import unittest

from build_hud_snapshot import build_hud_snapshot, dec, default_hud, rank_rows
from export_unreal_bootstrap import build_bootstrap
from validate_gene_physics import load


class HUDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bootstrap = build_bootstrap()
        cls.mock = load("fixtures/hud_mock_finance_v1.json")

    def test_default_never_fabricates_portfolio_nav_or_winners(self):
        hud = build_hud_snapshot(self.bootstrap)
        self.assertEqual(hud["financialSource"], "unavailable_no_ledger")
        self.assertEqual(hud["sourceBanner"], "DEMO · SYNTHETISCHE DATEN · KEIN LIVE-HANDEL")
        self.assertTrue(all(m["value"] is None and m["availability"] == "unavailable"
                            for m in hud["financialMetrics"].values()))
        self.assertEqual(hud["leaderboard"]["winnerRows"], [])
        self.assertEqual(hud["leaderboard"]["loserRows"], [])
        self.assertEqual(hud["portfolioNavSeries"], [])
        self.assertIn("NAV_UNAVAILABLE", [a["code"] for a in hud["alerts"]])
        self.assertEqual(hud["paperBudget"]["fundedTotalUsdt"], "0.00")

    def test_mock_shows_biomass_as_nav_one_source_not_summed_plant_biomass(self):
        hud = build_hud_snapshot(self.bootstrap, self.mock)
        self.assertEqual(hud["financialMetrics"]["totalBiomassPortfolioUsdt"]["value"], "108.90")
        self.assertEqual(hud["financialMetrics"]["portfolioReturnPct"]["value"], "8.90")
        self.assertEqual(hud["financialMetrics"]["netPnlUsdt"]["value"], "8.90")
        self.assertEqual(hud["financialMetrics"]["maxDrawdownPct"]["value"], "3.67")
        self.assertEqual(hud["financialMetrics"]["winRatePct"]["value"], "50.00")
        self.assertTrue(all(x["availability"] == "mock_only"
                            for x in hud["financialMetrics"].values()))
        self.assertEqual(hud["financialSource"], "synthetic_only")
        self.assertEqual(hud["ecologyMetrics"]["canonicalPlants"], 4)
        self.assertEqual(hud["ecologyMetrics"]["activePlants"], 3)
        self.assertEqual(hud["ecologyMetrics"]["visibleTileRepresentations"], 6)

    def test_winners_losers_are_unique_plant_id_and_top_five(self):
        hud = build_hud_snapshot(self.bootstrap, self.mock)
        lead = hud["leaderboard"]
        self.assertEqual(lead["defaultToggle"], "winners")
        self.assertEqual(lead["toggleOptions"], ["winners", "losers"])
        self.assertEqual(lead["rankBy"], "netPnlUsdt")
        self.assertEqual(lead["limit"], 5)
        self.assertEqual([x["plantId"] for x in lead["winnerRows"]], ["plant-a"])
        self.assertEqual([x["plantId"] for x in lead["loserRows"]], ["plant-b", "plant-c"])
        self.assertEqual(lead["winnerRows"][0]["netPnlUsdt"], "11.50")
        self.assertEqual(lead["loserRows"][0]["netPnlUsdt"], "-2.50")
        self.assertEqual(lead["loserRows"][1]["netPnlUsdt"], "-0.10")
        self.assertEqual(lead["scope"], "canonical_plant_unique_only")
        self.assertEqual(lead["loserRows"][0]["status"], "archived")
        self.assertEqual(lead["winnerRows"][0]["tileId"], None)

    def test_exact_portfolio_values_and_mock_evidence(self):
        hud = build_hud_snapshot(self.bootstrap, self.mock)
        self.assertEqual(hud["portfolioNavSeries"][0], {"dayIndex": 0, "navUsdt": "100.00"})
        self.assertEqual(hud["portfolioNavSeries"][-1], {"dayIndex": 7, "navUsdt": "108.90"})
        self.assertTrue(all(row["source"] == "synthetic_only"
                            for row in hud["leaderboard"]["winnerRows"] + hud["leaderboard"]["loserRows"]))
        self.assertEqual(hud["ecologyMetrics"]["source"], "synthetic_only")
        self.assertFalse(hud["tileCards"][0]["paperBudgetIsFunded"])
        self.assertEqual(hud["paperBudget"]["requestedTotalUsdt"], "60.00")

    def test_no_duplicate_ranking_if_one_plant_is_on_four_tiles(self):
        hud = build_hud_snapshot(self.bootstrap, self.mock)
        self.assertEqual(len(hud["focus"]["plantInstanceIds"]), 4)
        self.assertEqual(len(hud["leaderboard"]["winnerRows"]), 1)
        self.assertEqual(hud["leaderboard"]["winnerRows"][0]["netPnlUsdt"], "11.50")
        self.assertEqual(hud["ecologyMetrics"]["independentSeedReleases"], 6)
        self.assertEqual(hud["ecologyMetrics"]["referenceSeedLandings"], 16)
        self.assertEqual(hud["ecologyMetrics"]["newCanonicalBirthsApproved"], 0)
        self.assertEqual(hud["ecologyMetrics"]["liveTilesApproved"], 0)

    def test_mock_has_to_reconcile_or_is_rejected(self):
        bad = deepcopy(self.mock)
        bad["navUsdt"] = "109.90"
        with self.assertRaisesRegex(ValueError, "reconcile"):
            build_hud_snapshot(self.bootstrap, bad)
        bad2 = deepcopy(self.mock)
        bad2["plantResults"][0]["netPnlUsdt"] = "25.00"
        with self.assertRaisesRegex(ValueError, "reconcile"):
            build_hud_snapshot(self.bootstrap, bad2)

    def test_reject_unknown_plant_duplicate_and_wrong_scene(self):
        bad = deepcopy(self.mock)
        bad["plantResults"][0]["plantId"] = "nonexistent"
        with self.assertRaisesRegex(ValueError, "unknown"):
            build_hud_snapshot(self.bootstrap, bad)
        bad2 = deepcopy(self.mock)
        bad2["plantResults"][0]["plantId"] = "plant-b"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_hud_snapshot(self.bootstrap, bad2)
        bad3 = deepcopy(self.mock)
        bad3["sceneRef"]["dayIndex"] += 1
        with self.assertRaisesRegex(ValueError, "not a validated"):
            build_hud_snapshot(self.bootstrap, bad3)

    def test_reject_non_demo_finance_claims_and_broker_permission(self):
        bad = deepcopy(self.mock)
        bad["authority"] = "real_live_venue"
        with self.assertRaisesRegex(ValueError, "not a validated"):
            build_hud_snapshot(self.bootstrap, bad)
        bad2 = deepcopy(self.mock)
        bad2["allowBrokerUse"] = True
        with self.assertRaisesRegex(ValueError, "not a validated"):
            build_hud_snapshot(self.bootstrap, bad2)
        bootstrap = deepcopy(self.bootstrap)
        bootstrap["permissions"]["unrealRuntimeMayBookCapital"] = True
        with self.assertRaisesRegex(ValueError, "cannot mutate"):
            build_hud_snapshot(bootstrap, self.mock)

    def test_invalid_evidence_or_nav_series_rejected(self):
        bad = deepcopy(self.mock)
        bad["plantResults"][1]["closedTrades"] = -3
        with self.assertRaisesRegex(ValueError, "evidence"):
            build_hud_snapshot(self.bootstrap, bad)
        bad2 = deepcopy(self.mock)
        bad2["navHistory"][-1]["navUsdt"] = "105.00"
        with self.assertRaisesRegex(ValueError, "start and end"):
            build_hud_snapshot(self.bootstrap, bad2)
        bad3 = deepcopy(self.mock)
        bad3["navHistory"][-1]["dayIndex"] = 8
        with self.assertRaisesRegex(ValueError, "start and end"):
            build_hud_snapshot(self.bootstrap, bad3)

    def test_financial_decimal_not_float_nan_or_scientific(self):
        for bad in (1.3, "NaN", "Infinity", "1e3", " 20.00", None, True):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    dec(bad)

    def test_three_modes_and_separate_context_invariant(self):
        hud = build_hud_snapshot(self.bootstrap)
        self.assertEqual(hud["hudControls"]["atlasVsExplore"], ["atlas", "explore"])
        self.assertTrue(hud["hudControls"]["exploreStartsCompact"])
        self.assertEqual(hud["hudControls"]["capitalModeAction"],
                         "preview_only_requires_backend_acceptance")
        self.assertEqual(hud["hudControls"]["liveModeClickAction"], "blocked")
        self.assertEqual(hud["focus"]["selectedPlantId"], "plant-a")
        self.assertEqual(len(hud["tileCards"]), 4)
        self.assertEqual(hud["ecologyMetrics"]["tilesEvolving"], 4)
        self.assertEqual(hud["ecologyMetrics"]["paperTilesRequested"], 3)

    def test_repeatability_and_mock_switch_keeps_ecology_counts_same(self):
        empty = build_hud_snapshot(self.bootstrap)
        full = build_hud_snapshot(self.bootstrap, self.mock)
        self.assertEqual(empty, build_hud_snapshot(self.bootstrap))
        self.assertEqual(full, build_hud_snapshot(self.bootstrap, self.mock))
        self.assertEqual(empty["ecologyMetrics"], full["ecologyMetrics"])
        self.assertEqual(empty["tileCards"], full["tileCards"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
