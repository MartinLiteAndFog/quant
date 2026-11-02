#!/usr/bin/env python3
"""Validate NON-EMPIRICAL UE5 view-only demo fixture; stdlib only."""
from __future__ import annotations
import json
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "fixtures" / "scene_demo_v0.json"

def validate(data: dict) -> None:
    assert data["schemaVersion"] == 0
    assert data["fixture"] is True
    assert data["kind"] == "synthetic_visual_fixture_only"
    assert data["scene"]["sourceStatus"] == "synthetic_visual_only"
    assert "nav" not in data and "balance" not in data and "trades" not in data
    tiles = {t["id"]: t for t in data["tiles"]}
    assert tiles and len(tiles) == len(data["tiles"])
    ids = [p["id"] for p in data["plants"]]
    assert len(ids) == len(set(ids)), "Plant IDs must be unique"
    idset = set(ids)
    assert data["events"]
    seen_events = set()
    for event in data["events"]:
        assert event["id"] not in seen_events
        seen_events.add(event["id"])
        assert event["plantId"] in idset
        assert 0 <= event["dayIndex"] <= data["scene"]["dayIndex"]
    unassigned = 0
    for plant in data["plants"]:
        assert plant["role"] in ("parent", "seedling")
        assert plant["status"] in ("active", "archived")
        assert 0 <= plant["bornDay"] <= data["scene"]["dayIndex"]
        if plant["status"] == "archived":
            assert plant["archivedDay"] <= data["scene"]["dayIndex"]
            assert plant["archivedDay"] >= plant["bornDay"]
        for parent in plant["parentIds"]:
            assert parent in idset and parent != plant["id"]
            assert next(p for p in data["plants"] if p["id"] == parent)["bornDay"] < plant["bornDay"]
        look = plant["appearance"]
        assert 0 <= look["hue"] < 360
        assert 0 <= look["saturation"] <= 1
        assert 0 <= look["rootDepth"] <= 1
        if plant["phenotype"] == "cactus":
            assert look["photosyntheticRind"] is True
            assert look["leafCount"] >= 1
        if plant["tileId"] is None:
            unassigned += 1
            continue
        tile = tiles[plant["tileId"]]
        assert tile["axisX"]["gene"] != tile["axisY"]["gene"]
        for axis in (tile["axisX"], tile["axisY"]):
            assert axis["gene"] in plant["genome"], "No invented axis coordinates"
            value = plant["genome"][axis["gene"]]
            assert isinstance(value, (int, float)) and axis["min"] <= value <= axis["max"]
    assert unassigned >= 1, "Missing-axis plants need a visible unassigned region"
    assert any(p["status"] == "archived" for p in data["plants"])
    assert any(p["parentIds"] for p in data["plants"])
    print(f"PASS: synthetic UE5 scene fixture, {len(data['plants'])} plants, {len(tiles)} two-gene tile(s), archived genealogy, unassigned display; no ledger.")

def validate_camera_journey(journey: dict, scene: dict) -> None:
    from math import isfinite
    assert journey["schemaVersion"] == 0
    assert journey["fixture"] is True
    assert journey["kind"] == "synthetic_camera_navigation_fixture_only"
    assert "nav" not in journey and "balance" not in journey and "trades" not in journey
    world = journey["worldRef"]
    assert world["sceneSeed"] == scene["scene"]["seed"]
    assert world["dayIndex"] == scene["scene"]["dayIndex"]
    assert world["sourceStatus"] == "synthetic_visual_only"
    policy = journey["policy"]
    assert policy["cameraProjection"] == "perspective_in_both_modes"
    assert policy["readOnly"] is True and policy["noLevelReload"] is True
    assert policy["allowCancel"] is True and policy["allowReverse"] is True
    states = journey["states"]
    ids = {p["id"] for p in scene["plants"]}
    assert len(states) == 4
    assert len({s["id"] for s in states}) == len(states)
    assert {s["mode"] for s in states} == {"atlas", "explore"}
    assert {s["layer"] for s in states} == {"surface", "underworld"}
    for s in states:
        assert s["selectedPlantId"] in ids and s["focusId"] in ids
        assert s["dayIndex"] == world["dayIndex"] and s["worldSeed"] == world["sceneSeed"]
        for key in ("position", "lookAt"):
            vec = s["camera"][key]
            assert len(vec) == 3
            assert all(isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v) for v in vec)
        assert 10 <= s["camera"]["fovDegrees"] <= 120
        assert s["camera"]["position"] != s["camera"]["lookAt"]
    assert [t["from"] for t in journey["transitions"]] == [s["id"] for s in states[:-1]]
    assert [t["to"] for t in journey["transitions"]] == [s["id"] for s in states[1:]]
    by_id = {s["id"]: s for s in states}
    for t in journey["transitions"]:
        assert t["sourceTicks"] == 0 and t["bookings"] == 0
        assert t["sceneReload"] is False and t["preserveSelection"] is True
        a, b = by_id[t["from"]], by_id[t["to"]]
        assert a["selectedPlantId"] == b["selectedPlantId"]
        assert a["dayIndex"] == b["dayIndex"] and a["worldSeed"] == b["worldSeed"]
    assert states[0]["mode"] == states[-1]["mode"] == "atlas"
    assert states[0]["selectedPlantId"] == states[-1]["selectedPlantId"]
    print("PASS: synthetic equal Atlas/Explore path, same plant/day/world, no bookings or scene reload.")

if __name__ == "__main__":
    scene = json.loads(PATH.read_text(encoding="utf-8"))
    validate(scene)
    journey = json.loads((PATH.parent / "camera_journey_v0.json").read_text(encoding="utf-8"))
    validate_camera_journey(journey, scene)
