#!/usr/bin/env python3
"""Project seed impacts into NEW offspring-genome proposals, never live trading.

All points are synthetic relative to the parent plant on a fixed 10m tile.
Authoritative settlement, selection, and performance are out of scope.
"""
from __future__ import annotations
import json
import math
from validate_gene_physics import load, check_params, check_seedfall


def project_genome_proposals(scene: dict, registry: dict, snapshot: dict,
                             release: dict, receipt: dict, tile: dict) -> dict:
    check_params(registry, snapshot, scene)
    check_seedfall(release, receipt, snapshot, scene)
    assert tile["schemaVersion"] == 1
    assert tile["kind"] == "cloudgarden_gene_meadow_inverse_mapping"
    assert tile["mutationAuthority"] == "owner_review_only"
    assert tile["outOfBounds"] == "unassigned_no_clamping"
    assert tile["positionFrame"] == "parent_local_offsets_m"
    assert tile["sceneRef"] == snapshot["sceneRef"]
    assert any(t["id"] == tile["tileId"] for t in scene["tiles"])
    source = next(t for t in scene["tiles"] if t["id"] == tile["tileId"])
    axes = [tile["axisX"], tile["axisY"]]
    limits = tile["tileMeters"]
    for axis, scene_axis in zip(axes, (source["axisX"], source["axisY"])):
        assert axis["source"] in ("trade.channel_lookback", "trade.stop_loss_bps")
        assert math.isfinite(axis["quantizationStep"]) and axis["quantizationStep"] > 0
        assert (axis["min"], axis["max"]) == (scene_axis["min"], scene_axis["max"])
        assert axis["source"].split("trade.")[1].replace("_", "").lower() == scene_axis["gene"].lower()
    for low, high in ((limits["minX"], limits["maxX"]), (limits["minY"], limits["maxY"])):
        assert math.isfinite(low) and math.isfinite(high) and low < high
    origin = []
    for axis, lo, hi in zip(axes, (limits["minX"], limits["minY"]), (limits["maxX"], limits["maxY"])):
        v = snapshot["values"][axis["source"]]
        assert axis["min"] <= v <= axis["max"]
        origin.append(lo + (v - axis["min"]) / (axis["max"] - axis["min"]) * (hi - lo))
    proposals = []
    for landing in receipt["outcomes"]:
        x = origin[0] + landing["impactM"][0]
        y = origin[1] + landing["impactM"][1]
        inside = limits["minX"] <= x <= limits["maxX"] and limits["minY"] <= y <= limits["maxY"]
        row = {"seedId": landing["seedId"], "tileId": tile["tileId"] if inside else None,
               "landedXYMeters": [round(x, 6), round(y, 6)],
               "state": "research_candidate_only" if inside else "unassigned_outside_tile"}
        if inside:
            norm = [(x - limits["minX"]) / (limits["maxX"] - limits["minX"]),
                    (y - limits["minY"]) / (limits["maxY"] - limits["minY"])]
            values = {}
            for axis, n in zip(axes, norm):
                exact = axis["min"] + n * (axis["max"] - axis["min"])
                step = axis["quantizationStep"]
                value = axis["min"] + math.floor((exact - axis["min"]) / step + 0.5) * step
                value = max(axis["min"], min(axis["max"], value))
                values[axis["source"]] = int(value) if axis["outputType"] == "integer" else float(value)
            row["proposedGenomeAxes"] = values
            row["normalizedXY"] = [round(v, 8) for v in norm]
        proposals.append(row)
    return {
        "schemaVersion": 1,
        "kind": "cloudgarden_seedfall_offspring_proposals",
        "authority": "synthetic_fixture_proposal_only",
        "sourceReleaseEventId": release["eventId"],
        "sourceReceiptId": receipt["receiptId"],
        "sourcePlantId": snapshot["plantId"],
        "earliestEffectiveDay": receipt["applyAtDay"],
        "source": "reference_ballistic_not_unreal_chaos",
        "tileId": tile["tileId"],
        "parentXYMeters": [round(n, 6) for n in origin],
        "proposals": proposals,
        "policy": "never_write_back_to_trading_or_ledger_without_owner_approval",
    }


if __name__ == "__main__":
    scene = load("fixtures/scene_demo_v0.json")
    result = project_genome_proposals(
        scene,
        load("contracts/parameter_registry_v1.json"),
        load("fixtures/parameter_snapshot_v1.json"),
        load("fixtures/seed_release_v1.json"),
        load("fixtures/seed_landing_v1.json"),
        load("contracts/gene_meadow_inverse_mapping_v1.json"),
    )
    assert result == load("fixtures/offspring_proposals_v1.json"), "Golden offspring proposals differ"
    print(f"PASS: {len(result['proposals'])} seedfall -> two-gene research candidates, no trading writes.")
    print(json.dumps(result, indent=2, sort_keys=True))
