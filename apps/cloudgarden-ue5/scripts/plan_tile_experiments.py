#!/usr/bin/env python3
"""Independent tile seed experiments + separate virtual paper-mode planning.

One canonical plant can emit seed experiments in EACH eligible 2-gene tile.
Emitted seed positions generate UNAPPROVED new child gene-axis proposals.
No trading engine, money bookings, Unreal Chaos or live OMS calls occur here.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import math
import re

from project_parameter_tiles import project_tiles, validate_catalog
from validate_gene_physics import load

MONEY = re.compile(r"^(?:0|[1-9][0-9]*)\.[0-9]{2}$")
MAX_SEEDS = 30
G = 9.81


def validate_money(text: object) -> Decimal:
    if not isinstance(text, str) or not MONEY.fullmatch(text):
        raise ValueError("Budgets must be exact two-decimal strings in USDT")
    try:
        n = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("Invalid monetary amount") from exc
    if n < 0 or n > Decimal("1000000000.00"):
        raise ValueError("Monetary request outside policy range")
    return n


def validate_activation(config: dict, tiles: dict) -> dict[str, dict]:
    if config.get("schemaVersion") != 1 or config.get("kind") != "cloudgarden_tile_research_activation":
        raise ValueError("Unsupported tile activation contract")
    if config.get("authority") != "synthetic_research_fixture_only" or config.get("sceneRef") != tiles["sceneRef"]:
        raise ValueError("Untrusted or stale activation contract")
    policy = config["policy"]
    if (policy["evolutionIndependentOfCapital"] is not True
            or policy["eachTileIndependentSeedExperiment"] is not True
            or policy["liveMode"] != "blocked_pending_backend_risk_and_venue_authorization"
            or policy["defaultCapitalMode"] != "off"
            or policy["simulationStepRule"] != "future_day_only"
            or policy["maxSeedsPerRelease"] != MAX_SEEDS
            or policy["sourceAccount"] != "none_synthetic_fixture"
            or policy["paperBudgetMeaning"] != "virtual_cap_request_not_funds_booked"
            or policy["capitalModeOptions"] != ["off", "paper", "live"]):
        raise ValueError("Unsafe activation/evolution policy")
    global_budget = config["globalResearchBudget"]
    if (global_budget["currency"] != "USDT"
            or global_budget["cashAccountingWrites"] is not False
            or validate_money(global_budget["liveBudgetAuthorized"]) != 0):
        raise ValueError("No cash/account/live budget is allowed in this preflight")
    limit = validate_money(global_budget["paperBudgetPoolCeiling"])
    rows = config["tiles"]
    ids = [row["tileId"] for row in rows]
    expected_ids = {t["id"] for t in tiles["tiles"]}
    if len(set(ids)) != len(ids) or set(ids) != expected_ids:
        raise ValueError("Tile activation must specify each registered tile exactly once")
    total = Decimal("0.00")
    for row in rows:
        if type(row["evolutionEnabled"]) is not bool:
            raise ValueError("Evolution flag must be bool")
        mode = row["capitalMode"]
        budget = validate_money(row["paperBudgetRequestedUsdt"])
        feed = row["paperFeedMode"]
        if mode == "live":
            raise ValueError("Live trading must fail closed: backend authorization is not implemented")
        if mode == "off":
            if budget != 0 or feed != "none":
                raise ValueError("Off mode must not request capital or paper feed")
        elif mode == "paper":
            if budget <= 0 or feed not in ("historical", "realtime"):
                raise ValueError("Paper mode requires positive virtual budget and explicit feed")
            total += budget
        else:
            raise ValueError("Unknown tile capital mode")
    if total > limit:
        raise ValueError("Virtual paper requests exceed global paper pool ceiling")
    return {row["tileId"]: row for row in rows}


def validate_profiles(profiles: dict, scene: dict, registry: dict, reference: dict) -> dict[str, dict]:
    if (profiles.get("schemaVersion") != 1
            or profiles.get("kind") != "cloudgarden_tile_seed_launch_profiles"
            or profiles.get("authority") != "synthetic_research_fixture_only"
            or profiles.get("sceneRef") != {"seed": scene["scene"]["seed"], "dayIndex": scene["scene"]["dayIndex"]}
            or "No profile = no seed release" not in profiles.get("rule", "")):
        raise ValueError("Invalid or stale emission profiles")
    definitions = {p["id"]: p for p in registry["parameters"]}
    rules = {
        "heightM": "eco.seed_release_height_m",
        "azimuthDeg": "eco.seed_azimuth_deg",
        "elevationDeg": "eco.seed_elevation_deg",
        "speedMps": "eco.seed_launch_speed_mps",
        "spreadDeg": "eco.seed_spread_deg",
        "seedCount": "eco.seed_count",
    }
    source_ids = {p["id"] for p in scene["plants"]}
    result = {}
    for profile in profiles["profiles"]:
        plant_id = profile["plantId"]
        if (plant_id not in source_ids or plant_id in result
                or not isinstance(profile["profileRevision"], str) or not profile["profileRevision"]):
            raise ValueError("Duplicate/unknown/unversioned launch profile")
        for field, name in rules.items():
            if name not in definitions:
                raise ValueError("Unregistered ecology field")
            d, value = definitions[name], profile[field]
            if d["kind"] == "integer":
                if type(value) is not int:
                    raise ValueError("Seed count must be int")
            elif type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Launch property must be a finite number")
            if not (d["min"] <= value <= d["max"]):
                raise ValueError("Launch property outside gene range")
        if not 1 <= profile["seedCount"] <= MAX_SEEDS:
            raise ValueError("Release too large")
        # Preserve earlier synthetic fixture's plant-a gene inputs, do not
        # secretly overwrite one source with a different profile.
        if plant_id == reference["plantId"]:
            for field, name in rules.items():
                if profile[field] != reference["values"][name]:
                    raise ValueError("Seed profile disagrees with the source parameter snapshot")
        result[plant_id] = profile
    return result


def xorshift32(state: int) -> int:
    state ^= (state << 13) & 0xffffffff
    state ^= (state >> 17)
    state ^= (state << 5) & 0xffffffff
    return state & 0xffffffff


def quantize(raw: float, axis: dict) -> int | float:
    step = axis["step"]
    # explicit half-up quantization; avoid Python round-to-even
    units = math.floor((raw - axis["min"]) / step + 0.5)
    value = min(axis["max"], max(axis["min"], axis["min"] + units * step))
    return int(value) if axis["kind"] == "integer" else round(value, 8)


def generate_seed_experiment(instance: dict, tile: dict, profile: dict, axes: dict,
                             scene_ref: dict) -> dict:
    # Even two presentations of the same parent have distinct release IDs,
    # per-tile positions, deterministic RNG, lineage and future gene mutations.
    event = f"seed-release:{scene_ref['seed']}:{scene_ref['dayIndex']}:{tile['id']}:{instance['plantId']}:0001"
    seed = int(sha256(event.encode("utf-8")).hexdigest()[:8], 16) or 1
    sx, sy, sz = instance["worldPositionM"]
    azimuth = profile["azimuthDeg"]
    elev = math.radians(profile["elevationDeg"])
    vh = profile["speedMps"] * math.cos(elev)
    vz = profile["speedMps"] * math.sin(elev)
    t = (vz + math.sqrt(vz * vz + 2 * G * profile["heightM"])) / G
    axes_info = [axes[tile["axes"]["x"]], axes[tile["axes"]["y"]]]
    impact_results = []
    for i in range(profile["seedCount"]):
        seed = xorshift32(seed)
        u = seed / 4294967296
        yaw = azimuth + (2 * u - 1) * profile["spreadDeg"]
        px = sx + vh * math.cos(math.radians(yaw)) * t
        py = sy + vh * math.sin(math.radians(yaw)) * t
        nx = (px - tile["originM"][0]) / tile["sizeM"][0]
        ny = (py - tile["originM"][1]) / tile["sizeM"][1]
        inside = 0 <= nx <= 1 and 0 <= ny <= 1
        seed_id = f"{event}:seed:{i+1}"
        outcome = {
            "seedId": seed_id,
            "sourcePlantId": instance["plantId"],
            "tileId": tile["id"],
            "releaseEventId": event,
            "impactWorldM": [round(px, 6), round(py, 6), round(sz, 6)],
            "flightSeconds": round(t, 6),
            "impactMode": "analytic_reference_not_chaos",
            "status": "unapproved_offspring_gene_patch" if inside else "outside_tile_unassigned",
            "notBeforeDay": scene_ref["dayIndex"] + 1,
            "newCanonicalPlantCreated": False,
            "moneyBooked": False,
        }
        if inside:
            proposal = {}
            for param_id, normalized, axis in zip([tile["axes"]["x"], tile["axes"]["y"]],
                                                    [nx, ny], axes_info):
                raw = axis["min"] + normalized * (axis["max"] - axis["min"])
                proposal[param_id] = quantize(raw, axis)
            outcome["proposedAxes"] = proposal
            outcome["baseGenomeRevision"] = instance["canonicalPlantRevision"]
        impact_results.append(outcome)
    return {
        "eventId": event, "tileId": tile["id"], "plantId": instance["plantId"],
        "experimentId": instance["experimentId"],
        "rng": {"algorithm": "xorshift32_v1", "startState": int(sha256(event.encode("utf-8")).hexdigest()[:8],16) or 1},
        "releaseProfileRevision": profile["profileRevision"],
        "simulation": "analytic_reference_not_chaos",
        "outcomes": impact_results,
    }


def plan(scene: dict, catalog: dict, source: dict, config: dict,
         profiles: dict, registry: dict, reference_snapshot: dict) -> dict:
    projection = project_tiles(scene, catalog, source)
    by_axes = validate_catalog(catalog, scene)
    activated = validate_activation(config, catalog)
    by_profile = validate_profiles(profiles, scene, registry, reference_snapshot)
    by_tile = {t["id"]: t for t in catalog["tiles"]}
    releases = []
    blocked = []
    for instance in projection["instances"]:
        tile_id, plant_id = instance["tileId"], instance["plantId"]
        settings = activated[tile_id]
        if not settings["evolutionEnabled"]:
            blocked.append({"tileId":tile_id,"plantId":plant_id,"reason":"evolution_paused"})
            continue
        if plant_id not in by_profile:
            blocked.append({"tileId":tile_id,"plantId":plant_id,"reason":"missing_explicit_seed_profile"})
            continue
        releases.append(generate_seed_experiment(instance, by_tile[tile_id],
                                                  by_profile[plant_id], by_axes, catalog["sceneRef"]))
    virtual_total = sum((validate_money(a["paperBudgetRequestedUsdt"]) for a in activated.values()),
                        Decimal("0.00"))
    return {
        "schemaVersion":1,"kind":"cloudgarden_per_tile_seed_and_capital_plan",
        "authority":"synthetic_research_fixture_only","sourceScene":catalog["sceneRef"],
        "canonicalPlantCount":projection["canonicalPlantCount"],
        "renderInstanceCount":projection["renderInstanceCount"],
        "tileStates":[{
            "tileId": t["id"],"evolutionEnabled": activated[t["id"]]["evolutionEnabled"],
            "capitalMode": activated[t["id"]]["capitalMode"],
            "paperFeedMode": activated[t["id"]]["paperFeedMode"],
            "paperBudgetRequestedUsdt":activated[t["id"]]["paperBudgetRequestedUsdt"],
        } for t in catalog["tiles"]],
        "requestedPaperBudgetTotalUsdt": f"{virtual_total:.2f}",
        "liveAccountOrOrderWrites":False,
        "virtualBudgetRequestsAreFunded":False,
        "physicsReceiptAuthority":"reference_only_untrusted_until_approved",
        "releases":releases,"skipped":blocked,
        "totalIndependentReleases":len(releases),
        "totalSimulatedSeedOutcomes":sum(len(x["outcomes"]) for x in releases),
        "resultPolicy":"proposed_genes_only_no_automatic_child_birth_no_trade_execution",
    }


def default_plan() -> dict:
    return plan(load("fixtures/scene_demo_v0.json"),
                load("contracts/parameter_tiles_v1.json"),
                load("fixtures/tile_genomes_v1.json"),
                load("contracts/tile_research_activation_v1.json"),
                load("fixtures/tile_seed_profiles_v1.json"),
                load("contracts/parameter_registry_v1.json"),
                load("fixtures/parameter_snapshot_v1.json"))


if __name__ == "__main__":
    print(json.dumps(default_plan(),sort_keys=True,indent=2))
