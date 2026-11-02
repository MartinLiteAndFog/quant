#!/usr/bin/env python3
"""Build/verify a reproducible, synthetic per-tile research event journal.

The hash chain detects changes against a separately trusted source plan. It is
NOT a signature or trusted Chaos/capital event receipt. No persistence/markets.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json

from plan_tile_experiments import default_plan
from validate_gene_physics import load

ZERO_HASH = "0" * 64


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def digest(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def build_journal(plan: dict) -> dict:
    if (plan.get("schemaVersion") != 1
            or plan.get("kind") != "cloudgarden_per_tile_seed_and_capital_plan"
            or plan.get("authority") != "synthetic_research_fixture_only"
            or plan.get("liveAccountOrOrderWrites") is not False
            or plan.get("virtualBudgetRequestsAreFunded") is not False
            or plan.get("resultPolicy") != "proposed_genes_only_no_automatic_child_birth_no_trade_execution"):
        raise ValueError("Journal requires trusted synthetic-only, no-money plan")
    scene = plan["sourceScene"]
    if not isinstance(scene["dayIndex"], int) or scene["dayIndex"] < 0:
        raise ValueError("Invalid source day")
    events: list[dict] = []
    seen: set[str] = set()
    prior = ZERO_HASH

    def append(event_id: str, event_type: str, tile_id: str,
               plant_id: str | None, event_day: int, payload: dict) -> None:
        nonlocal prior
        if not event_id or event_id in seen:
            raise ValueError("Duplicated/empty research event ID")
        if event_day not in (scene["dayIndex"], scene["dayIndex"] + 1):
            raise ValueError("Research event outside source/future day")
        if "moneyBooked" in payload and payload["moneyBooked"] is not False:
            raise ValueError("Cannot book capital")
        seen.add(event_id)
        base = {
            "seq": len(events) + 1,
            "eventId": event_id,
            "type": event_type,
            "tileId": tile_id,
            "plantId": plant_id,
            "dayIndex": event_day,
            "payload": payload,
            "previousHash": prior,
        }
        current = digest(base)
        events.append({**base, "eventHash": current})
        prior = current

    tiles = plan["tileStates"]
    if len({t["tileId"] for t in tiles}) != len(tiles):
        raise ValueError("Duplicate tile mode")
    for tile in tiles:
        append(f"tile-mode:{scene['seed']}:{scene['dayIndex']}:{tile['tileId']}",
               "tile.mode_declared", tile["tileId"], None, scene["dayIndex"],
               {**tile, "authority": "synthetic_unfunded_mode_request"})

    release_ids: set[str] = set()
    seed_ids: set[str] = set()
    for release in plan["releases"]:
        release_id = release["eventId"]
        if release_id in release_ids:
            raise ValueError("Duplicate release")
        release_ids.add(release_id)
        if release["tileId"] not in {t["tileId"] for t in tiles}:
            raise ValueError("Release refers to unknown tile")
        append(release_id, "seed.release_reference",
               release["tileId"], release["plantId"], scene["dayIndex"],
               {"experimentId": release["experimentId"],
                "releaseProfileRevision": release["releaseProfileRevision"],
                "rng": release["rng"],
                "seedIds": [x["seedId"] for x in release["outcomes"]],
                "source": "analytic_reference_not_chaos",
                "candidateOnly": True})
        for outcome in release["outcomes"]:
            if (outcome["releaseEventId"] != release_id
                    or outcome["tileId"] != release["tileId"]
                    or outcome["sourcePlantId"] != release["plantId"]):
                raise ValueError("Outcome has wrong source identity")
            seed_id = outcome["seedId"]
            if seed_id in seed_ids:
                raise ValueError("Duplicate seed ID")
            seed_ids.add(seed_id)
            if (outcome["notBeforeDay"] != scene["dayIndex"] + 1
                    or outcome["moneyBooked"] is not False
                    or outcome["newCanonicalPlantCreated"] is not False):
                raise ValueError("Research outcomes cannot affect present/capital")
            append(seed_id + ":landing", "seed.terminal_landing_reference",
                   release["tileId"], release["plantId"], scene["dayIndex"],
                   {"seedId": seed_id, "releaseEventId": release_id,
                    "impactWorldM": outcome["impactWorldM"],
                    "flightSeconds": outcome["flightSeconds"],
                    "impactMode": outcome["impactMode"],
                    "status": outcome["status"], "moneyBooked": False})
            if outcome["status"] == "unapproved_offspring_gene_patch":
                if (not isinstance(outcome.get("proposedAxes"), dict)
                        or len(outcome["proposedAxes"]) != 2):
                    raise ValueError("Missing two proposed gene axes")
                payload = {"seedId": seed_id, "releaseEventId": release_id,
                           "proposedAxes": outcome["proposedAxes"],
                           "baseGenomeRevision": outcome["baseGenomeRevision"],
                           "notBeforeDay": outcome["notBeforeDay"],
                           "canonicalBirthAccepted": False}
                kind = "offspring.axis_patch_proposed"
            elif outcome["status"] == "outside_tile_unassigned":
                if "proposedAxes" in outcome:
                    raise ValueError("Out-of-range seeds cannot propose a valid genome")
                payload = {"seedId": seed_id, "releaseEventId": release_id,
                           "notBeforeDay": outcome["notBeforeDay"],
                           "canonicalBirthAccepted": False}
                kind = "offspring.outside_tile_unassigned"
            else:
                raise ValueError("Unknown seed outcome state")
            append(seed_id + ":candidate", kind, release["tileId"], release["plantId"],
                   scene["dayIndex"] + 1, payload)

    if (len(plan["releases"]) != plan["totalIndependentReleases"]
            or len(seed_ids) != plan["totalSimulatedSeedOutcomes"]):
        raise ValueError("Plan totals differ from source releases")
    return {
        "schemaVersion": 1,
        "kind": "cloudgarden_synthetic_tile_research_journal",
        "authority": "synthetic_untrusted_reference_not_canonical",
        "sceneRef": scene,
        "planDigest": digest(plan),
        "hashAlgorithm": "sha256_canonical_json_v1",
        "genesisHash": ZERO_HASH,
        "headHash": prior,
        "eventCount": len(events),
        "events": events,
        "disclaimer": "Hash chain proves source-plan consistency, not authenticity. No accepted Chaos receipts, venue orders, fills, NAV or canonical child births.",
    }


def replay_journal(journal: dict, source_plan: dict) -> dict:
    # Regenerate independently from a trusted source plan. A client that merely
    # rewrites the hash chain must still fail against expected event contents.
    expected = build_journal(source_plan)
    if journal != expected:
        raise ValueError("Journal mismatch vs independently recomputed source plan")
    previous = journal["genesisHash"]
    ids = set()
    releases = set()
    seeds = set()
    proposed = []
    excluded = []
    modes = {}
    for index, entry in enumerate(journal["events"], 1):
        if (entry["seq"] != index or entry["previousHash"] != previous
                or entry["eventHash"] != digest({k: v for k, v in entry.items()
                                                  if k != "eventHash"})
                or entry["eventId"] in ids):
            raise ValueError("Corrupted, out-of-order or duplicate journal entry")
        ids.add(entry["eventId"])
        previous = entry["eventHash"]
        kind, payload = entry["type"], entry["payload"]
        if kind == "tile.mode_declared":
            modes[entry["tileId"]] = payload
        elif kind == "seed.release_reference":
            releases.add(entry["eventId"])
            if not payload["candidateOnly"]:
                raise ValueError("Research-only seed release changed")
        elif kind == "seed.terminal_landing_reference":
            if payload["releaseEventId"] not in releases:
                raise ValueError("Landing precedes its release")
            seeds.add(payload["seedId"])
        elif kind in ("offspring.axis_patch_proposed", "offspring.outside_tile_unassigned"):
            if (payload["seedId"] not in seeds
                    or entry["dayIndex"] < journal["sceneRef"]["dayIndex"] + 1
                    or payload["canonicalBirthAccepted"] is not False):
                raise ValueError("Candidate is not forward-only/causal")
            if kind == "offspring.axis_patch_proposed":
                proposed.append(entry["eventId"])
            else:
                excluded.append(entry["eventId"])
        else:
            raise ValueError("Unsupported journal event type")
    if previous != journal["headHash"] or len(journal["events"]) != journal["eventCount"]:
        raise ValueError("Final journal hash/count incorrect")
    return {
        "sceneRef": journal["sceneRef"],
        "headHash": journal["headHash"],
        "tileCount": len(modes),
        "releaseCount": len(releases),
        "landingCount": len(seeds),
        "candidateCount": len(proposed),
        "outsideTileCount": len(excluded),
        "candidateIds": proposed,
        "noCanonicalBirths": True,
        "capitalLedgerWrites": 0,
        "marketOrders": 0,
        "authority": "synthetic_replay_not_canonical",
    }


if __name__ == "__main__":
    plan = default_plan()
    journal = build_journal(plan)
    replay_journal(journal, plan)
    print(json.dumps(journal, indent=2, ensure_ascii=False, sort_keys=True))
