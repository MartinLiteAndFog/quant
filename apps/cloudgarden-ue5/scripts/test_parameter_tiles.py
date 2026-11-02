#!/usr/bin/env python3
"""Regression tests: zero or many tile representations, one immutable plant ID."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import unittest

from validate_gene_physics import load
from project_parameter_tiles import project_tiles


class MultiTileProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = load("fixtures/scene_demo_v0.json")
        self.tiles = load("contracts/parameter_tiles_v1.json")
        self.genomes = load("fixtures/tile_genomes_v1.json")

    def build(self, scene=None, tiles=None, genomes=None):
        return project_tiles(scene if scene is not None else self.scene,
                             tiles if tiles is not None else self.tiles,
                             genomes if genomes is not None else self.genomes)

    def decision(self, result, plant, tile):
        return next(d for d in result["decisions"]
                    if d["plantId"] == plant and d["tileId"] == tile)

    def test_multiple_tiles_one_canonical_identity(self):
        output = self.build()
        self.assertEqual(output["canonicalPlantCount"], 4)
        self.assertEqual(output["renderInstanceCount"], 6)
        self.assertEqual(
            {tile["tileId"]: tile["visible"] for tile in output["tileSummaries"]},
            {"sol-imbase": 2, "sol-lookback-sensitivity": 2,
             "sol-sensitivity-ttp": 1, "sol-lookback-ttp": 1},
        )
        self.assertEqual(len(output["decisions"]), 16)
        counts = Counter(i["plantId"] for i in output["instances"])
        self.assertEqual(counts, {"plant-a": 4, "plant-c": 2})
        self.assertEqual(len(set(i["instanceId"] for i in output["instances"])), 6)
        self.assertEqual(len(set(i["canonicalPlantRevision"] for i in output["instances"]
                                 if i["plantId"] == "plant-a")), 1)
        self.assertEqual(output["plantToInstances"]["plant-b"], [])
        self.assertEqual(output["plantToInstances"]["plant-d"], [])
        self.assertNotIn("navUsdt", str(output))
        self.assertNotIn("paperFeesUsdt", str(output))
        self.assertTrue(all(i["capitalEntityCount"] == 0 for i in output["instances"]))

    def test_distinct_locations_match_two_axis_normalization(self):
        output = self.build()
        by_key = {i["instanceId"]: i for i in output["instances"]}
        positions = {
            "sol-imbase::plant-a": [0.625, 4.444444, 0],
            "sol-lookback-sensitivity::plant-a": [14.625, 4, 0],
            "sol-sensitivity-ttp::plant-a": [4, 16.857143, 0],
            "sol-lookback-ttp::plant-a": [14.625, 16.857143, 0],
        }
        for key, expected in positions.items():
            self.assertEqual(by_key[key]["worldPositionM"], expected, key)
            self.assertEqual(by_key[key]["plantId"], "plant-a")
        self.assertEqual(by_key["sol-lookback-sensitivity::plant-a"]["geneValues"],
                         {"trade.channel_lookback": 50, "research.sensitivity": 17.2})

    def test_missing_gene_is_no_not_a_default(self):
        result = self.build()
        for tile in ("sol-sensitivity-ttp", "sol-lookback-ttp"):
            excluded = self.decision(result, "plant-c", tile)
            self.assertFalse(excluded["eligible"])
            self.assertEqual(excluded["reason"], "missing_axes")
            self.assertEqual(excluded["missingAxes"], ["research.ttp_pct"])
        self.assertFalse(self.decision(result, "plant-d", "sol-sensitivity-ttp")["eligible"])
        self.assertEqual(self.decision(result, "plant-d", "sol-imbase")["missingAxes"],
                         ["trade.channel_lookback", "trade.stop_loss_bps"])

    def test_archived_always_excluded_but_canonical_id_retained(self):
        result = self.build()
        self.assertEqual(len(result["plantToInstances"]["plant-b"]), 0)
        self.assertTrue(all(d["reason"] == "archived" for d in result["decisions"]
                            if d["plantId"] == "plant-b"))
        self.assertTrue(any(p["id"] == "plant-b" for p in self.scene["plants"]))

    def test_gene_added_only_enables_matching_tiles_without_new_plant(self):
        population = deepcopy(self.genomes)
        next(p for p in population["plants"] if p["plantId"] == "plant-c")[
            "geneValues"]["research.ttp_pct"] = 0.85
        result = self.build(genomes=population)
        self.assertEqual(result["renderInstanceCount"], 8)
        self.assertEqual(result["canonicalPlantCount"], 4)
        self.assertEqual(len(result["plantToInstances"]["plant-c"]), 4)
        self.assertEqual(len(result["plantToInstances"]["plant-a"]), 4)
        self.assertEqual(self.genomes["plants"][2]["geneValues"].get("research.ttp_pct"), None)

    def test_bad_or_out_of_bounds_gene_excludes_only_relevant_tiles(self):
        bad = deepcopy(self.genomes)
        first = next(p for p in bad["plants"] if p["plantId"] == "plant-a")
        first["geneValues"]["research.sensitivity"] = 30
        result = self.build(genomes=bad)
        self.assertEqual(self.decision(result, "plant-a", "sol-lookback-sensitivity")["reason"],
                         "axis_out_of_range")
        self.assertEqual(self.decision(result, "plant-a", "sol-sensitivity-ttp")["reason"],
                         "axis_out_of_range")
        self.assertTrue(self.decision(result, "plant-a", "sol-imbase")["eligible"])
        self.assertTrue(self.decision(result, "plant-a", "sol-lookback-ttp")["eligible"])
        bad2 = deepcopy(self.genomes)
        bad2["plants"][0]["geneValues"]["research.sensitivity"] = True
        result2 = self.build(genomes=bad2)
        self.assertEqual(self.decision(result2, "plant-a", "sol-lookback-sensitivity")["reason"],
                         "invalid_axis_type")

    def test_one_physics_home_tile_not_four_seed_emitters(self):
        instances = self.build()["instances"]
        self.assertEqual([i["instanceId"] for i in instances
                          if i["physicsCandidateOnlyOnAuthorizedHomeTile"]],
                         ["sol-imbase::plant-a", "sol-imbase::plant-c"])
        self.assertTrue(all(i["role"] == "read_only_visual_projection" for i in instances))

    def test_reject_untrusted_or_inconsistent_source(self):
        bad = deepcopy(self.genomes)
        bad["plants"][0]["geneValues"]["trade.channel_lookback"] = 80
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.build(genomes=bad)
        bad2 = deepcopy(self.genomes)
        bad2["plants"][0]["geneValues"]["trade.fake_gene"] = 1
        with self.assertRaisesRegex(ValueError, "Unregistered"):
            self.build(genomes=bad2)
        stale = deepcopy(self.genomes)
        stale["sceneRef"]["dayIndex"] += 1
        with self.assertRaisesRegex(ValueError, "stale"):
            self.build(genomes=stale)

    def test_reject_duplicated_and_overlapping_tiles(self):
        duplicate = deepcopy(self.tiles)
        duplicate["tiles"][1]["id"] = duplicate["tiles"][0]["id"]
        with self.assertRaisesRegex(ValueError, "duplicate tile"):
            self.build(tiles=duplicate)
        reversed_pair = deepcopy(self.tiles)
        reversed_pair["tiles"][1]["axes"] = {
            "x": "trade.stop_loss_bps", "y": "trade.channel_lookback"}
        with self.assertRaisesRegex(ValueError, "Duplicate unordered"):
            self.build(tiles=reversed_pair)
        overlapped = deepcopy(self.tiles)
        overlapped["tiles"][1]["originM"] = [0, 0, 0]
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.build(tiles=overlapped)
        one_axis = deepcopy(self.tiles)
        one_axis["tiles"][0]["axes"]["y"] = one_axis["tiles"][0]["axes"]["x"]
        with self.assertRaisesRegex(ValueError, "different"):
            self.build(tiles=one_axis)

    def test_repeatable_and_unaffected_by_camera_mode(self):
        first = self.build()
        self.assertEqual(first, self.build())
        scene = deepcopy(self.scene)
        scene["scene"]["camera"] = "immersive_explore"
        self.assertEqual(first, self.build(scene=scene))
        self.assertEqual(first["plantToInstances"]["plant-a"],
                         [t["id"] + "::plant-a" for t in self.tiles["tiles"]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
