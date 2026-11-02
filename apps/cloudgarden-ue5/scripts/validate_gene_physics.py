#!/usr/bin/env python3
"""Cloudgarden UE5: stdlib-only NON-EMPIRICAL gene -> physics -> telemetry preflight.

This is intentionally not the capital/strategy simulation or a network endpoint.
The reference flight model is analytically ballistic, NOT Unreal Chaos.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MASK = (1 << 32) - 1
TOL = 0.000002


def load(name: str) -> dict:
    def unique_pairs(pairs: list[tuple[str, object]]) -> dict:
        obj: dict = {}
        for key, val in pairs:
            if key in obj:
                raise ValueError("Duplicated JSON key: " + key)
            obj[key] = val
        return obj
    return json.loads((ROOT / name).read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)


def number(n: object) -> bool:
    return type(n) in (int, float) and math.isfinite(n)


def same_float(a: object, b: object, tolerance: float = TOL) -> bool:
    return number(a) and number(b) and abs(a - b) <= tolerance


def xorshift32(x: int) -> int:
    x ^= (x << 13) & MASK
    x ^= x >> 17
    x ^= (x << 5) & MASK
    return x & MASK


def check_params(registry: dict, snapshot: dict, scene: dict) -> None:
    assert registry["schemaVersion"] == 1
    assert registry["kind"] == "cloudgarden_ue5_parameter_registry"
    assert registry["status"] == "proposed_not_implemented"
    assert snapshot["schemaVersion"] == snapshot["registryVersion"] == 1
    assert snapshot["authority"] == "synthetic_fixture_only"
    assert snapshot["kind"] == "cloudgarden_ue5_parameter_snapshot"
    assert snapshot["sceneRef"] == {"seed": scene["scene"]["seed"], "dayIndex": scene["scene"]["dayIndex"]}
    plant = next(p for p in scene["plants"] if p["id"] == snapshot["plantId"])
    assert plant["status"] == "active"
    definitions = registry["parameters"]
    assert len(definitions) <= 2000
    ids = [item["id"] for item in definitions]
    targets = [item["binding"]["target"] for item in definitions]
    assert len(ids) == len(set(ids)) and len(targets) == len(set(targets))
    assert set(ids) == set(snapshot["values"])
    assert set(targets) == set(snapshot["expectedProjections"])
    mapping_to_scene = {
        "visual.hue_deg": "hue",
        "visual.saturation_ratio": "saturation",
        "morph.branching_ratio": "branching",
        "morph.leaf_count": "leafCount",
        "morph.photosynthetic_rind": "photosyntheticRind",
    }
    for param in definitions:
        name = param["id"]
        role = param["role"]
        assert name.startswith(("visual.", "morph.", "eco.", "trade."))
        assert role in ("visual_only", "visual_ecology", "ecology_input", "trading_readonly")
        assert param["sourceStatus"] in ("existing_model_proposed_in_pr", "proposed_new_gene")
        if name.startswith("eco."):
            assert param["sourceStatus"] == "proposed_new_gene"
            assert role == "ecology_input"
        if name.startswith("trade."):
            assert role == "trading_readonly"
        if param["sourceStatus"] == "existing_model_proposed_in_pr":
            assert name not in {"eco.seed_release_height_m", "eco.seed_azimuth_deg"}
        binding = param["binding"]
        assert binding["transform"] == "linear_or_identity"
        assert binding["writeFromUnreal"] is False
        assert number(binding["scale"]) and binding["scale"] > 0
        assert number(binding["offset"])
        assert binding["unit"] != "USDT"
        assert binding["target"].startswith(("BP_CloudPlant.", "PCG_CloudPlant.", "BP_SeedEmitter.", "WBP_PlantInspector."))
        v = snapshot["values"][name]
        kind = param["kind"]
        if kind == "boolean":
            assert type(v) is bool and v in param["allowed"]
            assert type(param["defaultValue"]) is bool
            expected = v
        else:
            assert kind in ("integer", "float")
            assert number(v) and number(param["defaultValue"])
            assert number(param["min"]) and number(param["max"])
            assert param["min"] <= v <= param["max"]
            assert param["min"] <= param["defaultValue"] <= param["max"]
            if kind == "integer":
                assert type(v) is int and type(param["defaultValue"]) is int
            expected = v * binding["scale"] + binding["offset"]
        assert expected == snapshot["expectedProjections"][binding["target"]] if kind == "boolean" else same_float(
            expected, snapshot["expectedProjections"][binding["target"]], 0.0000001
        ), name
        if name in mapping_to_scene:
            actual = plant["appearance"][mapping_to_scene[name]]
            assert actual == v, "Existing visual/morph data cannot be silently redefined"
    assert "nav" not in snapshot and "orders" not in snapshot and "balance" not in snapshot
    print(f"PASS: {len(ids)} typed gene/inspector bindings; UE units explicit; existing model mappings and read-only isolation.")


def check_seedfall(release: dict, receipt: dict, snapshot: dict, scene: dict) -> None:
    assert release["schemaVersion"] == receipt["schemaVersion"] == 1
    assert release["kind"] == "cloudgarden_seed_release_request"
    assert receipt["kind"] == "cloudgarden_seed_landing_receipt"
    assert release["authority"] == receipt["authority"] == "synthetic_fixture_only"
    assert release["sceneRef"]["seed"] == receipt["sceneRef"]["seed"] == scene["scene"]["seed"]
    assert release["sceneRef"]["dayIndex"] == receipt["sceneRef"]["dayIndex"] == scene["scene"]["dayIndex"]
    assert release["createdDay"] == receipt["sceneRef"]["dayIndex"]
    assert release["earliestApplyDay"] == receipt["applyAtDay"] == release["createdDay"] + 1
    assert release["plantId"] == receipt["sourcePlantId"] == snapshot["plantId"]
    assert any(p["id"] == release["plantId"] and p["status"] == "active" for p in scene["plants"])
    assert receipt["releaseEventId"] == release["eventId"]
    assert receipt["source"] == "reference_ballistic_not_unreal_chaos"
    assert receipt["commitPolicy"] == "proposal_only_until_canonical_ecology_accepts"
    assert receipt["physicsSolver"] == release["physics"]["solver"] == "ballistic_ground_plane_v1"
    assert release["sceneRef"]["terrainRevision"] == release["physics"]["terrainRevision"]
    assert receipt["sourceDataProvenance"]["collisionGeometryRevision"] == release["sceneRef"]["terrainRevision"]
    assert receipt["sourceDataProvenance"]["sourceHash"] == "synthetic-fixture-only"
    assert "nav" not in receipt and "orders" not in receipt and "balance" not in receipt
    p = release["physics"]
    assert p["terrain"] == "uniform_loam"
    assert p["drag"] == "none"
    assert p["windMps"] == [0, 0, 0]
    assert same_float(p["gravityMps2"], 9.81)
    assert same_float(p["groundZ_m"], 0)
    launch = release["launch"]
    v = snapshot["values"]
    assert len(launch["originM"]) == 3
    assert launch["originM"] == [0, 0, v["eco.seed_release_height_m"]]
    for a, b in (
        ("azimuthDeg", "eco.seed_azimuth_deg"),
        ("elevationDeg", "eco.seed_elevation_deg"),
        ("speedMps", "eco.seed_launch_speed_mps"),
        ("yawSpreadDeg", "eco.seed_spread_deg"),
        ("seedCount", "eco.seed_count"),
    ):
        assert launch[a] == v[b], "Release params must come from the authorized source snapshot"
    assert type(launch["seedCount"]) is int and 1 <= launch["seedCount"] <= 30
    assert len(receipt["outcomes"]) == len(receipt["derivedSignals"]) == launch["seedCount"]
    rng = release["rng"]
    assert rng["algorithm"] == "xorshift32_v1"
    assert type(rng["seedU32"]) is int and 0 < rng["seedU32"] <= MASK
    s = rng["seedU32"]
    by_seed = {o["seedId"]: o for o in receipt["outcomes"]}
    assert len(by_seed) == launch["seedCount"]
    features = {o["seedId"]: o for o in receipt["derivedSignals"]}
    assert len(features) == launch["seedCount"] and by_seed.keys() == features.keys()
    for index in range(launch["seedCount"]):
        seed_id = f"seed-{index + 1}"
        item = by_seed[seed_id]
        s = xorshift32(s)
        assert item["randomU32"] == s
        yaw = launch["azimuthDeg"] + (2 * (s / (1 << 32)) - 1) * launch["yawSpreadDeg"]
        elev = math.radians(launch["elevationDeg"])
        vel_horizontal = launch["speedMps"] * math.cos(elev)
        vel_z = launch["speedMps"] * math.sin(elev)
        t = (vel_z + math.sqrt(vel_z ** 2 + 2 * p["gravityMps2"] * launch["originM"][2])) / p["gravityMps2"]
        x = launch["originM"][0] + t * vel_horizontal * math.cos(math.radians(yaw))
        y = launch["originM"][1] + t * vel_horizontal * math.sin(math.radians(yaw))
        distance = math.hypot(x - launch["originM"][0], y - launch["originM"][1])
        assert same_float(item["yawDeg"], yaw)
        assert same_float(item["timeSeconds"], t)
        assert len(item["impactM"]) == 3
        assert all(same_float(a, b) for a, b in zip(item["impactM"], [x, y, 0]))
        assert item["groundMaterial"] == p["terrain"]
        assert same_float(item["distanceM"], distance)
        assert item["cellXY"] == [math.floor(x), math.floor(y)]
        derived = features[seed_id]
        assert derived["ecologyOnly"] is True
        assert derived["landingCellXY"] == item["cellXY"]
        assert same_float(derived["dispersalDistanceM"], item["distanceM"])
    print(f"PASS: {len(by_seed)} deterministic reference seed falls, collision coordinates, distances and ecology-only forward signals.")


def negative_test(scene: dict, registry: dict, snapshot: dict, release: dict, receipt: dict) -> None:
    def rejects(kind: str, mutate) -> None:
        r = deepcopy(registry)
        s = deepcopy(snapshot)
        q = deepcopy(release)
        o = deepcopy(receipt)
        mutate(r, s, q, o)
        try:
            if kind == "params":
                check_params(r, s, scene)
            else:
                check_seedfall(q, o, s, scene)
        except (AssertionError, KeyError, TypeError, ValueError, StopIteration, OverflowError):
            return
        raise RuntimeError(f"Tampered {kind} fixture was accepted")
    rejects("params", lambda r, s, q, o: s["values"].update({"eco.seed_release_height_m": 1500}))
    rejects("params", lambda r, s, q, o: s["values"].update({"morph.photosynthetic_rind": 1}))
    rejects("params", lambda r, s, q, o: r["parameters"][0]["binding"].update({"writeFromUnreal": True}))
    rejects("physics", lambda r, s, q, o: o["outcomes"][0].update({"distanceM": 999}))
    rejects("physics", lambda r, s, q, o: o.update({"applyAtDay": q["createdDay"]}))
    rejects("physics", lambda r, s, q, o: o["outcomes"][1].update({"seedId": "seed-1"}))
    rejects("physics", lambda r, s, q, o: o.update({"orders": ["BUY SOL-USDT"]}))
    print("PASS: 7 tamper tests rejected (bounds, types, forbidden writeback, physics forgery, time, duplicate ID, order injection).")


if __name__ == "__main__":
    scene = load("fixtures/scene_demo_v0.json")
    registry = load("contracts/parameter_registry_v1.json")
    snapshot = load("fixtures/parameter_snapshot_v1.json")
    release = load("fixtures/seed_release_v1.json")
    receipt = load("fixtures/seed_landing_v1.json")
    check_params(registry, snapshot, scene)
    check_seedfall(release, receipt, snapshot, scene)
    negative_test(scene, registry, snapshot, release, receipt)
