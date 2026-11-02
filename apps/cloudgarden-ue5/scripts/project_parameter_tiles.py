#!/usr/bin/env python3
"""Project ONE canonical plant into ZERO OR MORE independent tile research views.

Synthetic view and tile-scoped seed experiment reference. No Unreal Editor, order or account API.
No genes are invented; incompatible/missing/archived plants are excluded PER tile.
"""
from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
import json
import math
import sys
from validate_gene_physics import load


def finite_number(value: object) -> bool:
    return type(value) in (float, int) and math.isfinite(value)


def axis_value_status(raw: object, axis: dict) -> str:
    if axis["kind"] == "integer":
        if type(raw) is not int:
            return "invalid_axis_type"
    elif axis["kind"] == "number":
        if not finite_number(raw):
            return "invalid_axis_type"
    else:
        raise ValueError("Unsupported axis type")
    if raw < axis["min"] or raw > axis["max"]:
        return "axis_out_of_range"
    return "ok"


def validate_catalog(catalog: dict, scene: dict) -> dict[str, dict]:
    if catalog["schemaVersion"] != 1 or catalog["kind"] != "cloudgarden_parameter_tile_catalog":
        raise ValueError("Unsupported parameter tile catalog")
    if catalog["authority"] != "synthetic_research_fixture_only":
        raise ValueError("Expected a non-empirical tile catalog")
    if catalog["sceneRef"] != {"seed": scene["scene"]["seed"], "dayIndex": scene["scene"]["dayIndex"]}:
        raise ValueError("Tile catalog is from a different world/day")
    settings = catalog["rendering"]
    if (settings["selection"] != "plantId" or settings["identity"] != "tileId::plantId"
            or settings["excludeWithoutBothAxes"] is not True
            or settings["excludeOutOfRange"] is not True
            or settings["excludeArchived"] is not True
            or settings["multipleInstancesOfOneCanonicalPlant"] is not True
            or settings["physicalSideEffects"] != "tile_scoped_candidate_only"
            or settings["missingGenePolicy"] != "no_default_or_imputation"
            or settings["tileExperimentIdentity"] != "tileId::plantId::revision"
            or settings["reproductionPolicy"] != "distinct_tile_candidates_never_automatic_canonical_birth"
            or settings["units"] != "m"):
        raise ValueError("Unsafe tile display or physics policy")
    axes = catalog["axisDefinitions"]
    tiles = catalog["tiles"]
    if not 1 <= len(axes) <= 2000 or not 1 <= len(tiles) <= 2000:
        raise ValueError("Invalid axis/tile count")
    axis_ids = [a["id"] for a in axes]
    if len(set(axis_ids)) != len(axes):
        raise ValueError("Duplicate axis IDs")
    for axis in axes:
        if (axis["kind"] not in ("number", "integer")
                or not finite_number(axis["min"]) or not finite_number(axis["max"])
                or axis["min"] >= axis["max"]
                or not finite_number(axis.get("step")) or axis["step"] <= 0
                or not isinstance(axis["unit"], str) or not axis["unit"]
                or not isinstance(axis["sourceStatus"], str)):
            raise ValueError("Invalid axis definition")
        if axis["kind"] == "integer" and (type(axis["min"]) is not int or type(axis["max"]) is not int):
            raise ValueError("Integer axis bounds must be integer")
    by_axis = {a["id"]: a for a in axes}
    ids: set[str] = set()
    pairs: set[tuple[str, str, str]] = set()
    rects: list[tuple[str, str, float, float, float, float]] = []
    for tile in tiles:
        tile_id = tile["id"]
        if (not isinstance(tile_id, str) or not tile_id or "::" in tile_id
                or tile_id in ids or not isinstance(tile["label"], str) or not tile["label"]):
            raise ValueError("Invalid or duplicate tile ID")
        ids.add(tile_id)
        x, y = tile["axes"]["x"], tile["axes"]["y"]
        if x not in by_axis or y not in by_axis or x == y:
            raise ValueError("Each tile requires exactly two different supported genes")
        pair = (tile["assetRegion"], *sorted((x, y)))
        if pair in pairs:
            raise ValueError("Duplicate unordered pair in same asset region")
        pairs.add(pair)
        origin = tile["originM"]
        size = tile["sizeM"]
        if (not isinstance(origin, list) or len(origin) != 3
                or not isinstance(size, list) or len(size) != 2
                or not all(finite_number(v) for v in origin + size)
                or any(v <= 0 for v in size)
                or tile["yawDeg"] != 0
                or tile["physicsPolicy"] != "tile_scoped_research_candidate_only"):
            raise ValueError("Invalid physical tile transform or policy")
        rects.append((tile_id, tile["assetRegion"], origin[0], origin[1],
                      origin[0] + size[0], origin[1] + size[1]))
    for i, one in enumerate(rects):
        for two in rects[i + 1:]:
            if (one[1] == two[1] and max(one[2], two[2]) < min(one[4], two[4])
                    and max(one[3], two[3]) < min(one[5], two[5])):
                raise ValueError("Parameter tiles overlap in world coordinates")
    return by_axis


def validate_population(source: dict, scene: dict, catalog: dict, by_axis: dict) -> dict[str, dict]:
    if (source["schemaVersion"] != 1 or source["kind"] != "cloudgarden_tile_genome_projection_source"
            or source["authority"] != "synthetic_research_fixture_only"
            or source["sceneRef"] != catalog["sceneRef"]):
        raise ValueError("Unsupported or stale tile gene sidecar")
    scene_by_id = {p["id"]: p for p in scene["plants"]}
    if len(scene_by_id) != len(scene["plants"]):
        raise ValueError("Canonical plant IDs are duplicated")
    records = source["plants"]
    ids = [p["plantId"] for p in records]
    if len(ids) != len(set(ids)) or set(ids) != set(scene_by_id):
        raise ValueError("Sidecar must reference each source plant exactly once")
    for p in records:
        owner = scene_by_id[p["plantId"]]
        if not isinstance(p["assetRegion"], str) or not p["assetRegion"]:
            raise ValueError("Missing asset region")
        if not isinstance(p["revision"], str) or not p["revision"]:
            raise ValueError("Missing plant version")
        values = p["geneValues"]
        if not isinstance(values, dict) or any(g not in by_axis for g in values):
            raise ValueError("Unregistered genome axis")
        # These two current source values must match the source object, not a made-up
        # sidecar override. Other research genes are clearly synthetic fixture only.
        for gene, raw_key in (("trade.channel_lookback", "channelLookback"),
                              ("trade.stop_loss_bps", "stopLossBps")):
            present = raw_key in owner["genome"]
            if (gene in values) != present:
                raise ValueError("Sidecar omits or invents an existing trading gene")
            if present and values[gene] != owner["genome"][raw_key]:
                raise ValueError("Sidecar trading gene conflicts with canonical source")
    return {p["plantId"]: p for p in records}


def decide(plant: dict, values: dict, tile: dict, by_axis: dict) -> dict:
    axes = [tile["axes"]["x"], tile["axes"]["y"]]
    if plant["status"] != "active":
        return {"eligible": False, "reason": "archived"}
    if plant["bornDay"] > values["dayIndex"]:
        return {"eligible": False, "reason": "not_born"}
    if values["assetRegion"] != tile["assetRegion"]:
        return {"eligible": False, "reason": "asset_mismatch"}
    absent = [key for key in axes if key not in values["geneValues"]]
    if absent:
        return {"eligible": False, "reason": "missing_axes", "missingAxes": absent}
    for key in axes:
        status = axis_value_status(values["geneValues"][key], by_axis[key])
        if status != "ok":
            return {"eligible": False, "reason": status, "failedAxis": key}
    return {"eligible": True, "reason": "has_both_valid_genes"}


def project_tiles(scene: dict, catalog: dict, source: dict) -> dict:
    by_axis = validate_catalog(catalog, scene)
    by_gene_owner = validate_population(source, scene, catalog, by_axis)
    decisions: list[dict] = []
    instances: list[dict] = []
    by_plant: dict[str, list[str]] = {p["id"]: [] for p in scene["plants"]}
    count: Counter[str] = Counter()
    for tile in catalog["tiles"]:
        for plant in scene["plants"]:
            p = by_gene_owner[plant["id"]]
            joined = {**p, "dayIndex": scene["scene"]["dayIndex"]}
            eligibility = decide(plant, joined, tile, by_axis)
            decisions.append({"tileId": tile["id"], "plantId": plant["id"], **eligibility})
            if not eligibility["eligible"]:
                continue
            x_id, y_id = tile["axes"]["x"], tile["axes"]["y"]
            x, y = p["geneValues"][x_id], p["geneValues"][y_id]
            ax, ay = by_axis[x_id], by_axis[y_id]
            u = (x - ax["min"]) / (ax["max"] - ax["min"])
            v = (y - ay["min"]) / (ay["max"] - ay["min"])
            origin, size = tile["originM"], tile["sizeM"]
            position = [origin[0] + u * size[0], origin[1] + v * size[1], origin[2]]
            instance_id = tile["id"] + "::" + plant["id"]
            # Every eligible tile is a separate RESEARCH SEED EXPERIMENT.
            # Source identity (plantId) remains singular. A tile experiment must
            # never create a canonical birth, funded account or trade automatically.
            experiment_id = "experiment::" + tile["id"] + "::" + plant["id"] + "::" + p["revision"]
            instance = {
                "instanceId": instance_id, "plantId": plant["id"], "tileId": tile["id"],
                "canonicalPlantRevision": p["revision"],
                "normalizedXY": [round(u, 8), round(v, 8)],
                "worldPositionM": [round(value, 6) for value in position],
                "geneValues": {x_id: x, y_id: y},
                "role": "shared_canonical_plant_tile_experiment_projection",
                "experimentId": experiment_id,
                "seedResearchAllowedSubjectToTilePolicy": True,
                "physicsCandidateAuthority": "untrusted_tile_scoped_research_only",
                "capitalEntityCount": 0,
            }
            instances.append(instance)
            by_plant[plant["id"]].append(instance_id)
            count[tile["id"]] += 1
    by_instance = [i["instanceId"] for i in instances]
    if len(set(by_instance)) != len(by_instance):
        raise ValueError("Duplicate tile/plant instance key")
    return {
        "schemaVersion": 1,
        "kind": "cloudgarden_multi_tile_plant_projections",
        "authority": "synthetic_visual_fixture_only",
        "sceneRef": catalog["sceneRef"],
        "accountingPolicy": "zero_view_side_effects_one_canonical_plant_per_id",
        "axisSourceStatus": "research.sensitivity and research.ttp_pct are synthetic demo only",
        "tileSummaries": [{"tileId": t["id"], "visible": count[t["id"]],
                           "testedPlants": len(scene["plants"])} for t in catalog["tiles"]],
        "decisions": decisions,
        "instances": instances,
        "plantToInstances": by_plant,
        "canonicalPlantCount": len(scene["plants"]),
        "renderInstanceCount": len(instances),
    }


if __name__ == "__main__":
    data = project_tiles(
        load("fixtures/scene_demo_v0.json"),
        load("contracts/parameter_tiles_v1.json"),
        load("fixtures/tile_genomes_v1.json"),
    )
    print(json.dumps(data, sort_keys=True, indent=2))
