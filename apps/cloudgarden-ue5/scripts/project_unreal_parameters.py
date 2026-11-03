#!/usr/bin/env python3
"""Generate a DRY-RUN, machine-readable Unreal property binding projection.

No Unreal Editor API, account API, or arbitrary code execution is used.
Only target names from the registry allowlist are projected.
"""
from __future__ import annotations
from contextlib import redirect_stdout
from hashlib import sha256
import json
from pathlib import Path
import sys
from validate_gene_physics import load, check_params, ROOT


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def build_projection(scene: dict, registry: dict, snapshot: dict) -> dict:
    with redirect_stdout(sys.stderr):
        check_params(registry, snapshot, scene)
    assignments = []
    for spec in registry["parameters"]:
        source = spec["id"]
        target = spec["binding"]["target"]
        value = snapshot["expectedProjections"][target]
        assignments.append({
            "sourceId": source,
            "target": target,
            "targetValue": value,
            "targetUnit": spec["binding"]["unit"],
            "role": spec["role"],
            "writeFromUnreal": False,
        })
    return {
        "schemaVersion": 1,
        "kind": "cloudgarden_unreal_property_preview",
        "authority": "synthetic_fixture_only",
        "mode": "dry_run_no_editor",
        "plantId": snapshot["plantId"],
        "sceneRef": snapshot["sceneRef"],
        "registrySha256": file_hash(ROOT / "contracts/parameter_registry_v1.json"),
        "parameterSnapshotSha256": file_hash(ROOT / "fixtures/parameter_snapshot_v1.json"),
        "assignments": assignments,
    }


if __name__ == "__main__":
    scene = load("fixtures/scene_demo_v0.json")
    registry = load("contracts/parameter_registry_v1.json")
    snapshot = load("fixtures/parameter_snapshot_v1.json")
    print(json.dumps(build_projection(scene, registry, snapshot), sort_keys=True, indent=2))
