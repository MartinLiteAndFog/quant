#!/usr/bin/env python3
"""Causal replay, idempotent research receipts and fail-closed capital UI tests."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import unittest

from plan_tile_experiments import default_plan
from preview_tile_mode import preview_transition
from tile_research_journal import ZERO_HASH, build_journal, digest, replay_journal
from validate_gene_physics import load


class TileJournalTests(unittest.TestCase):
    def setUp(self):
        self.plan = default_plan()
        self.journal = build_journal(self.plan)
        self.catalog = load("contracts/parameter_tiles_v1.json")
        self.config = load("contracts/tile_research_activation_v1.json")

    def preview(self, *, tile="sol-imbase", mode="off", budget="0.00",
                feed="none", evolution=None, digest_override=None, config=None):
        src = config if config is not None else self.config
        return preview_transition(
            src, self.catalog, tile_id=tile, next_mode=mode,
            paper_budget=budget, paper_feed=feed, evolution_enabled=evolution,
            expected_config_digest=digest(src) if digest_override is None else digest_override,
        )

    def test_reference_journal_is_deterministic_and_causal(self):
        first, again = self.journal, build_journal(self.plan)
        self.assertEqual(first, again)
        self.assertEqual(first["planDigest"], digest(self.plan))
        result = replay_journal(first, self.plan)
        self.assertEqual(result["tileCount"], 4)
        self.assertEqual(result["releaseCount"], 6)
        self.assertEqual(result["landingCount"], 16)
        self.assertEqual(result["candidateCount"] + result["outsideTileCount"], 16)
        self.assertEqual(first["eventCount"], 42)
        self.assertEqual(first["genesisHash"], ZERO_HASH)
        self.assertEqual(result["headHash"], first["events"][-1]["eventHash"])
        self.assertEqual(result["capitalLedgerWrites"], 0)
        self.assertEqual(result["marketOrders"], 0)
        self.assertTrue(result["noCanonicalBirths"])

    def test_unique_events_and_only_future_genetic_patches(self):
        events = self.journal["events"]
        self.assertEqual(len({e["eventId"] for e in events}), len(events))
        counts = Counter(e["type"] for e in events)
        self.assertEqual(counts["tile.mode_declared"], 4)
        self.assertEqual(counts["seed.release_reference"], 6)
        self.assertEqual(counts["seed.terminal_landing_reference"], 16)
        self.assertEqual(counts["offspring.axis_patch_proposed"] +
                         counts["offspring.outside_tile_unassigned"], 16)
        for ev in events:
            if ev["type"].startswith("offspring."):
                self.assertEqual(ev["dayIndex"], self.plan["sourceScene"]["dayIndex"] + 1)
                self.assertFalse(ev["payload"]["canonicalBirthAccepted"])
                self.assertEqual(ev["payload"]["notBeforeDay"], 8)
                if "proposedAxes" in ev["payload"]:
                    self.assertEqual(len(ev["payload"]["proposedAxes"]), 2)

    def test_tampered_reference_fails_even_with_recomputed_hash_chain(self):
        tamper = deepcopy(self.journal)
        next(e for e in tamper["events"] if e["type"] == "seed.terminal_landing_reference")[
            "payload"]["impactWorldM"][0] += 4
        prev = ZERO_HASH
        for ev in tamper["events"]:
            ev["previousHash"] = prev
            ev["eventHash"] = digest({k: v for k, v in ev.items() if k != "eventHash"})
            prev = ev["eventHash"]
        tamper["headHash"] = prev
        with self.assertRaisesRegex(ValueError, "mismatch"):
            replay_journal(tamper, self.plan)

    def test_duplicate_and_out_of_order_receipts_rejected(self):
        dup = deepcopy(self.journal)
        dup["events"][2] = deepcopy(dup["events"][1])
        with self.assertRaises(ValueError):
            replay_journal(dup, self.plan)
        swapped = deepcopy(self.journal)
        swapped["events"][5], swapped["events"][6] = swapped["events"][6], swapped["events"][5]
        with self.assertRaises(ValueError):
            replay_journal(swapped, self.plan)

    def test_canonical_birth_or_money_not_permitted(self):
        illegal = deepcopy(self.plan)
        illegal["releases"][0]["outcomes"][0]["newCanonicalPlantCreated"] = True
        with self.assertRaisesRegex(ValueError, "cannot affect"):
            build_journal(illegal)
        illegal2 = deepcopy(self.plan)
        illegal2["releases"][0]["outcomes"][0]["moneyBooked"] = True
        with self.assertRaisesRegex(ValueError, "cannot affect"):
            build_journal(illegal2)

    def test_cross_tile_seed_identity_rewrite_rejected(self):
        illegal = deepcopy(self.plan)
        illegal["releases"][0]["outcomes"][0]["tileId"] = "sol-sensitivity-ttp"
        with self.assertRaisesRegex(ValueError, "wrong source identity"):
            build_journal(illegal)
        illegal2 = deepcopy(self.plan)
        illegal2["releases"][1]["outcomes"][0]["seedId"] = illegal2["releases"][0]["outcomes"][0]["seedId"]
        with self.assertRaisesRegex(ValueError, "Duplicate seed"):
            build_journal(illegal2)

    def test_plan_change_invalidates_saved_reference_journal(self):
        changed = deepcopy(self.plan)
        changed["tileStates"][0]["capitalMode"] = "off"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            replay_journal(self.journal, changed)

    def test_mode_preview_off_preserves_evolution_and_does_not_apply(self):
        before = deepcopy(self.config)
        result = self.preview()
        self.assertEqual(self.config, before)
        self.assertEqual(result["change"]["before"]["capitalMode"], "paper")
        self.assertEqual(result["change"]["proposed"]["capitalMode"], "off")
        self.assertTrue(result["change"]["proposed"]["evolutionEnabled"])
        self.assertEqual(result["requestedPaperPoolAfterUsdt"], "35.00")
        self.assertEqual(result["change"]["effectiveNotBeforeDay"], 8)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["cashBooked"])
        self.assertFalse(result["paperFunded"])
        self.assertEqual(result["venueOrders"], 0)
        self.assertEqual(result, self.preview())

    def test_evolution_pause_is_independent_of_paper_mode(self):
        request = self.preview(tile="sol-lookback-sensitivity", mode="paper",
                               budget="25.00", feed="historical", evolution=False)
        self.assertFalse(request["change"]["proposed"]["evolutionEnabled"])
        self.assertEqual(request["change"]["proposed"]["capitalMode"], "paper")
        self.assertEqual(request["requestedPaperPoolAfterUsdt"], "60.00")
        self.assertTrue(self.config["tiles"][1]["evolutionEnabled"])

    def test_switch_an_off_tile_to_paper_is_virtual_request_only(self):
        request = self.preview(tile="sol-sensitivity-ttp", mode="paper",
                               budget="15.00", feed="realtime")
        self.assertEqual(request["requestedPaperPoolAfterUsdt"], "75.00")
        self.assertEqual(request["change"]["proposed"]["paperFeedMode"], "realtime")
        self.assertTrue(request["change"]["proposed"]["evolutionEnabled"])
        self.assertFalse(request["accepted"])
        self.assertFalse(request["approvedByBackend"])
        self.assertIn("requires_backend_acceptance", request["state"])

    def test_stale_mode_snapshot_rejected_by_compare_and_swap(self):
        with self.assertRaisesRegex(ValueError, "Stale"):
            self.preview(digest_override=ZERO_HASH)

    def test_live_order_attempt_hard_rejected(self):
        with self.assertRaisesRegex(ValueError, "Live mode is blocked"):
            self.preview(mode="live", budget="0.00", feed="none")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.preview(mode="some-exchange", budget="0.00", feed="none")

    def test_mode_and_budget_validation_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Off mode"):
            self.preview(mode="off", budget="25.00", feed="none")
        with self.assertRaisesRegex(ValueError, "positive virtual"):
            self.preview(mode="paper", budget="0.00", feed="historical")
        with self.assertRaisesRegex(ValueError, "two-decimal"):
            self.preview(mode="paper", budget="5", feed="historical")
        with self.assertRaisesRegex(ValueError, "exceed"):
            self.preview(tile="sol-sensitivity-ttp", mode="paper",
                         budget="70.00", feed="historical")
        with self.assertRaisesRegex(ValueError, "Evolution flag"):
            self.preview(evolution=1)

    def test_unknown_tile_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown tile"):
            self.preview(tile="bogus")


if __name__ == "__main__":
    unittest.main(verbosity=2)
