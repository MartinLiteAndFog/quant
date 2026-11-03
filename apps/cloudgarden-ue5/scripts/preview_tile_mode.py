#!/usr/bin/env python3
"""Preview one tile's evolution/capital-mode transition, with CAS and no writes.

No mutation of a source configuration, no funded allocations, and NO live orders.
Acceptance, persistence, day transition and funding belong to a later owner.
"""
from __future__ import annotations

from copy import deepcopy
import json

from plan_tile_experiments import validate_activation, validate_money
from tile_research_journal import digest
from validate_gene_physics import load


def preview_transition(config: dict, catalog: dict, *,
                       tile_id: str, next_mode: str, paper_budget: str,
                       paper_feed: str, evolution_enabled: bool | None,
                       expected_config_digest: str) -> dict:
    # Check-and-set for stale clients. Even preview cannot assume that a user's
    # last-seen activation snapshot is still the source configuration.
    before_digest = digest(config)
    if expected_config_digest != before_digest:
        raise ValueError("Stale tile activation configuration")
    activated = validate_activation(config, catalog)
    if tile_id not in activated:
        raise ValueError("Unknown tile")
    if evolution_enabled is not None and type(evolution_enabled) is not bool:
        raise ValueError("Evolution flag must be boolean")
    if next_mode == "live":
        raise ValueError("Live mode is blocked until a trusted backend risk/venue gate exists")
    if next_mode not in ("off", "paper"):
        raise ValueError("Unsupported tile mode")
    # Deliberately fail rather than silently rewriting requested budgets.
    validate_money(paper_budget)
    proposal = deepcopy(config)
    before = deepcopy(activated[tile_id])
    after = next(row for row in proposal["tiles"] if row["tileId"] == tile_id)
    after["capitalMode"] = next_mode
    after["paperBudgetRequestedUsdt"] = paper_budget
    after["paperFeedMode"] = paper_feed
    if evolution_enabled is not None:
        after["evolutionEnabled"] = evolution_enabled
    validate_activation(proposal, catalog)
    next_digest = digest(proposal)
    change = {
        "tileId": tile_id,
        "before": before,
        "proposed": deepcopy(after),
        "effectiveNotBeforeDay": catalog["sceneRef"]["dayIndex"] + 1,
        "sourceConfigDigest": before_digest,
        "proposedConfigDigest": next_digest,
    }
    return {
        "schemaVersion": 1,
        "kind": "cloudgarden_tile_mode_change_preview",
        "authority": "synthetic_preview_unapplied",
        "state": "request_only_requires_backend_acceptance",
        "requestId": "tile-mode-preview:" + digest(change)[:24],
        "sceneRef": catalog["sceneRef"],
        "change": change,
        "requestedPaperPoolAfterUsdt": (
            f"{sum((validate_money(x['paperBudgetRequestedUsdt']) for x in proposal['tiles'])):.2f}"
        ),
        "cashBooked": False,
        "paperFunded": False,
        "venueOrders": 0,
        "accepted": False,
        "approvedByBackend": False,
        "proposedActivation": proposal,
    }


if __name__ == "__main__":
    conf = load("contracts/tile_research_activation_v1.json")
    tiles = load("contracts/parameter_tiles_v1.json")
    preview = preview_transition(
        conf, tiles, tile_id="sol-imbase", next_mode="off",
        paper_budget="0.00", paper_feed="none",
        evolution_enabled=None, expected_config_digest=digest(conf),
    )
    print(json.dumps(preview, indent=2, sort_keys=True))
