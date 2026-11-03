#!/usr/bin/env python3
"""One-file Unreal Editor bootstrap bundle from strictly synthetic source contracts.

A local UE agent can load this JSON and map it into approved DataAssets/PCG.
This command CANNOT start Unreal, update real accounts, or authorize ecology.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from hashlib import sha256
import json
from pathlib import Path
import sys

from plan_tile_experiments import plan
from project_parameter_tiles import project_tiles
from project_unreal_parameters import build_projection
from tile_research_journal import build_journal, canonical_json, digest, replay_journal
from validate_fixture import validate, validate_camera_journey
from validate_gene_physics import ROOT, load

SOURCE_FILES = (
    "fixtures/scene_demo_v0.json",
    "fixtures/camera_journey_v0.json",
    "fixtures/tile_genomes_v1.json",
    "fixtures/tile_seed_profiles_v1.json",
    "fixtures/parameter_snapshot_v1.json",
    "contracts/parameter_registry_v1.json",
    "contracts/parameter_tiles_v1.json",
    "contracts/tile_research_activation_v1.json",
)


def source_digests() -> dict[str, str]:
    return {name: sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def build_bootstrap() -> dict:
    scene = load("fixtures/scene_demo_v0.json")
    cameras = load("fixtures/camera_journey_v0.json")
    genes = load("fixtures/tile_genomes_v1.json")
    profiles = load("fixtures/tile_seed_profiles_v1.json")
    snapshot = load("fixtures/parameter_snapshot_v1.json")
    registry = load("contracts/parameter_registry_v1.json")
    catalog = load("contracts/parameter_tiles_v1.json")
    activation = load("contracts/tile_research_activation_v1.json")
    with redirect_stdout(sys.stderr):
        validate(scene)
        validate_camera_journey(cameras, scene)
        assignments = build_projection(scene, registry, snapshot)
        projected = project_tiles(scene, catalog, genes)
        planned = plan(scene, catalog, genes, activation, profiles, registry, snapshot)
        journal = build_journal(planned)
        proof = replay_journal(journal, planned)
    if (planned["renderInstanceCount"] != len(projected["instances"])
            or planned["canonicalPlantCount"] != len(scene["plants"])
            or proof["releaseCount"] != len(planned["releases"])):
        raise ValueError("Incompatible view / seed plan / journal")
    bundle = {
        "schemaVersion": 1,
        "kind": "cloudgarden_ue5_synthetic_scene_bootstrap",
        "authority": "synthetic_fixture_only_no_editor_or_real_trading",
        "sceneRef": catalog["sceneRef"],
        "sourceFileSha256": source_digests(),
        "sourcePlants": scene["plants"],
        "sparseResearchGenes": genes["plants"],
        "geneAxisCatalog": catalog["axisDefinitions"],
        "parameterTiles": catalog["tiles"],
        "tileDecisions": projected["decisions"],
        "tileVisualInstances": projected["instances"],
        "plantToVisualInstances": projected["plantToInstances"],
        "unrealPropertyPreview": assignments,
        "tileCapitalAndEvolutionRequests": planned["tileStates"],
        "seedResearchPlan": planned["releases"],
        "seedResearchSkipped": planned["skipped"],
        "referenceEventJournal": journal,
        "cameraJourney": cameras,
        "totals": {
            "canonicalPlants": planned["canonicalPlantCount"],
            "visualInstances": planned["renderInstanceCount"],
            "independentResearchReleases": planned["totalIndependentReleases"],
            "referenceSeedOutcomes": planned["totalSimulatedSeedOutcomes"],
            "journalEvents": journal["eventCount"],
            "paperBudgetRequestedUsdt": planned["requestedPaperBudgetTotalUsdt"],
            "paperFundsActuallyBookedUsdt": "0.00",
            "liveOrders": 0,
        },
        "permissions": {
            "unrealRuntimeMayModifySourceGenome": False,
            "unrealRuntimeMayBookCapital": False,
            "unrealRuntimeMaySendExchangeOrders": False,
            "unrealRuntimeMayAdoptResearchCandidates": False,
            "unrealEditorMayImportAllowlistedAssetsAfterLocalApproval": True,
            "modeSwitchesRequireBackendAcceptance": True,
        },
        "disclaimer": "Preview-only JSON for local agent; source docs/fixtures are synthetic, not historical markets or Chaos. Bundle digest detects changes, not authenticates a source.",
    }
    bundle["bundleSha256"] = digest(bundle)
    return bundle


def verify_bootstrap(bundle: dict) -> dict:
    if bundle.get("schemaVersion") != 1 or bundle.get("kind") != "cloudgarden_ue5_synthetic_scene_bootstrap":
        raise ValueError("Unsupported bootstrap")
    before = {k: v for k, v in bundle.items() if k != "bundleSha256"}
    if bundle.get("bundleSha256") != digest(before):
        raise ValueError("Bootstrap integrity digest mismatch")
    if bundle != build_bootstrap():
        raise ValueError("Bootstrap does not match source fixtures")
    if (bundle["permissions"]["unrealRuntimeMayBookCapital"] is not False
            or bundle["permissions"]["unrealRuntimeMaySendExchangeOrders"] is not False
            or bundle["totals"]["paperFundsActuallyBookedUsdt"] != "0.00"):
        raise ValueError("Unreal bootstrap cannot own financial state")
    return bundle["totals"]


if __name__ == "__main__":
    data = build_bootstrap()
    verify_bootstrap(data)
    print(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False))
