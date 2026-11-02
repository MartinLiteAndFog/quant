#!/usr/bin/env python3
"""Project Cloudgarden G2 vitality onto every visible UE tile, WITHOUT updating it.

The upstream simulation owns the single biological tick per canonical plant.
This offline adapter consumes only synthetic preview fixtures for now; it NEVER
estimates inactivity from Unreal frames or advances a plant on a tile.
"""
from __future__ import annotations
import json
import math
from validate_gene_physics import load
from project_parameter_tiles import project_tiles


def valid_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def project_vitality(scene: dict, catalog: dict, gene_source: dict, snapshot: dict) -> dict:
    tiles = project_tiles(scene, catalog, gene_source)
    scene_ref = tiles["sceneRef"]
    if (snapshot.get("schemaVersion") != 1
            or snapshot.get("kind") != "cloudgarden_vitality_snapshot"
            or snapshot.get("authority") not in ("synthetic_fixture_only", "canonical_readonly_export")
            or snapshot.get("sceneRef") != scene_ref):
        raise ValueError("Invalid, untrusted or stale vitality snapshot")
    policy = snapshot.get("policy", {})
    if (policy.get("oneDailyUpdatePerCanonicalPlant") is not True
            or policy.get("individualTileVitalityState") is not False
            or policy.get("readOnly") is not True
            or policy.get("liveTrading") is not False):
        raise ValueError("Vitality snapshot would permit duplicate updates or writes")
    records = snapshot.get("plants")
    if not isinstance(records, list):
        raise ValueError("Missing canonical vitality records")
    source_by_id = {r["plantId"]: r for r in gene_source["plants"]}
    scene_by_id = {p["id"]: p for p in scene["plants"]}
    ids = [r.get("plantId") for r in records if isinstance(r, dict)]
    if (len(ids) != len(records) or len(ids) != len(set(ids))
            or set(ids) != set(scene_by_id) or set(ids) != set(source_by_id)):
        raise ValueError("Every canonical plant must have exactly one vitality record")
    by_plant = {}
    for record in records:
        pid = record["plantId"]
        original = scene_by_id[pid]
        if (record.get("revision") != source_by_id[pid]["revision"]
                or record.get("status") != original["status"]):
            raise ValueError("Vitality revision/status disagrees with canonical scene")
        vigor = record.get("vigor")
        patience = record.get("patience")
        idle = record.get("inactiveDays")
        if (not valid_number(vigor) or not 0 <= vigor <= 100
                or not valid_number(patience) or not 0.05 <= patience <= 0.95
                or type(idle) is not int or idle < 0
                or idle > scene_ref["dayIndex"] - original["bornDay"]):
            raise ValueError("Invalid biological gene, vitality or inactivity streak")
        by_plant[pid] = record
    instances = []
    for tile in tiles["instances"]:
        p = by_plant[tile["plantId"]]
        inactive_days = p["inactiveDays"]
        instances.append({
            "instanceId": tile["instanceId"],
            "plantId": tile["plantId"],
            "tileId": tile["tileId"],
            "canonicalPlantRevision": tile["canonicalPlantRevision"],
            "vigor": p["vigor"],
            "patience": p["patience"],
            "inactiveDays": inactive_days,
            "inactivityDecayEligible": inactive_days > 14,
            "unrealReadOnlyBindings": {
                "WBP_CloudPlant.Vigor": p["vigor"],
                "WBP_CloudPlant.Patience": p["patience"],
                "WBP_CloudPlant.InactiveDays": inactive_days,
                "WBP_CloudPlant.InactivityDecayEligible": inactive_days > 14,
            },
        })
    # A given plant can have four world instances but only one authoritative
    # physiological snapshot. No simulated day, NAV, PnL, genome or account changes.
    return {
        "schemaVersion": 1,
        "kind": "cloudgarden_readonly_tile_vitality_projection",
        "authority": "read_only_preview_no_transport_trust",
        "declaredSourceAuthority": snapshot["authority"],
        "sceneRef": scene_ref,
        "ownership": "single_canonical_plant_per_plantId",
        "simulationTicksApplied": 0,
        "capitalOrGenomeWrites": False,
        "canonicalPlantCount": len(by_plant),
        "renderInstanceCount": len(instances),
        "instances": instances,
    }


def default_projection() -> dict:
    return project_vitality(
        load("fixtures/scene_demo_v0.json"),
        load("contracts/parameter_tiles_v1.json"),
        load("fixtures/tile_genomes_v1.json"),
        load("fixtures/vitality_snapshot_v1.json"),
    )


if __name__ == "__main__":
    print(json.dumps(default_projection(), sort_keys=True, indent=2))
