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

if __name__ == "__main__":
    validate(json.loads(PATH.read_text(encoding="utf-8")))
