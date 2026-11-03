#!/usr/bin/env python3
"""Read-only plant inspector for one source organism with N visual tile placements.

The miniature is a mesh/render-target preview, never a cloned canonical plant.
Per-plant equity only appears with a reconciled EXPLICIT fictional series.
"""
from __future__ import annotations

from decimal import Decimal

from validate_gene_physics import load


def build_plant_inspectors(bootstrap: dict, finance: dict | None) -> dict[str, dict]:
    from build_hud_snapshot import dec, money, pct, drawdown_percent

    source_plants = bootstrap["sourcePlants"]
    scene_day = bootstrap["sceneRef"]["dayIndex"]
    ids = [p["id"] for p in source_plants]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate canonical plants in inspector")
    by_finance = {p["plantId"]: p for p in finance["plantResults"]} if finance else {}
    projected = bootstrap["plantToVisualInstances"]

    result: dict[str, dict] = {}
    for p in source_plants:
        pid = p["id"]
        born = p["bornDay"]
        if type(born) is not int or born > scene_day:
            raise ValueError("Inspector source birth-day invalid")
        generation = 0 if p["role"] == "parent" else 1
        preview_source = {
            "kind": "isolated_visual_mesh_preview",
            "sourcePlantId": pid,
            "sourceModel": "same_visual_genome_reference",
            "renderTarget": "RT_CloudgardenPlantFocus",
            "spawnNewCanonicalEntity": False,
            "addToSourcePopulation": False,
            "enablePhysics": False,
            "enableSeedEmitters": False,
            "enableTrading": False,
        }
        genetics = {
            "family": p["family"],
            "phenotype": p["phenotype"],
            "appearance": p["appearance"],
            "currentSourceGenes": p["genome"],
            "researchOptionalAxes": next(
                (g["geneValues"] for g in bootstrap["sparseResearchGenes"]
                 if g["plantId"] == pid), {}),
        }
        metrics = {
            "individualEquityUsdt": None,
            "netPnlUsdt": None,
            "returnPct": None,
            "highWaterMarkUsdt": None,
            "maxDrawdownPct": None,
            "peakToCurrentDrawdownPct": None,
            "closedTrades": None,
            "winRatePct": None,
            "biologicalBiomass": None,
        }
        equity_points = []
        financial_source = "unavailable_no_plant_ledger"
        if pid in by_finance:
            source = by_finance[pid]
            equity_points = source["plantNavHistory"]
            if not equity_points:
                raise ValueError("Plant individual equity history missing")
            days = [item["dayIndex"] for item in equity_points]
            max_day = p.get("archivedDay", scene_day)
            if (days != sorted(set(days)) or days[0] != born
                    or days[-1] > max_day or days[-1] > scene_day
                    or any(type(day) is not int for day in days)):
                raise ValueError("Plant equity series is temporally invalid")
            equity = [dec(row["equityUsdt"]) for row in equity_points]
            basis = dec(source["notionalBasisUsdt"])
            net = dec(source["netPnlUsdt"])
            if (basis <= 0 or equity[0] != basis or equity[-1] != basis + net
                    or any(value <= 0 for value in equity)):
                raise ValueError("Plant equity series does not reconcile with the source")
            peak = max(equity)
            end = equity[-1]
            trades = source["closedTrades"]
            win = source["winningTrades"]
            metrics = {
                "individualEquityUsdt": money(end),
                "netPnlUsdt": money(net),
                "returnPct": pct(net * 100 / basis),
                "highWaterMarkUsdt": money(peak),
                "maxDrawdownPct": pct(drawdown_percent([
                    {"navUsdt": item["equityUsdt"]} for item in equity_points])),
                "peakToCurrentDrawdownPct": pct((peak - end) * 100 / peak),
                "closedTrades": trades,
                "winRatePct": pct(Decimal(100 * win) / trades) if trades else None,
                "biologicalBiomass": None,
            }
            financial_source = "fictional_per_plant_equity_only"
        result[pid] = {
            "plantId": pid,
            "name": p["name"],
            "status": p["status"],
            "bornDay": born,
            "ageDays": scene_day - born,
            "generation": generation,
            "parentIds": p["parentIds"],
            "tileInstanceIds": projected[pid],
            "visualPreview": preview_source,
            "genes": genetics,
            "metrics": metrics,
            "equitySeries": [
                {"dayIndex": entry["dayIndex"], "equityUsdt": entry["equityUsdt"]}
                for entry in equity_points],
            "financialSource": financial_source,
        }
    return result


def make_focus_interaction(bootstrap: dict) -> dict:
    """One physical meadow, soft camera focus and offscreen mesh thumbnail."""
    return {
        "schemaVersion": 1,
        "selectionKey": "canonical_plant_id",
        "tileInstanceSelection": "resolve_tileId_plantId_to_one_source_plantId",
        "samePlantInAllTiles": True,
        "world": "single_continuous_garden",
        "hudPlantPreview": {
            "anchor": "lower_left",
            "layout": "miniature_left_timeseries_and_metrics_to_right",
            "renderMode": "UE_SceneCaptureComponent2D_to_transparent_UTextureRenderTarget2D",
            "renderTargetName": "RT_CloudgardenPlantFocus",
            "targetAspectRatio": 0.8,
            "separate_preview_stage": True,
            "no_CanonicalPlantActor_spawn": True,
            "disableSimulationPhysics": True,
            "disableSeedRelease": True,
            "reuse_appearance_from_source": True,
            "render_on_selection_or_source_revision_only": True,
        },
        "mainCamera": {
            "response": "spring_ease_to_selected_plant",
            "blendSeconds": 0.65,
            "dollyFactor": 0.12,
            "maxFocusOffsetMeters": 2.5,
            "preserveWorldAndCameraMode": True,
            "preserveResearchDay": True,
            "preserveSelectionOnAtlasExploreTransition": True,
            "noTeleport": True,
            "noHardCameraCut": True,
        },
        "dismiss": {
            "closeInspectorDoesNotDeselectSource": True,
            "restorePreviousCameraFramingOnEscape": True,
        },
        "sideEffects": {
            "canonicalPlantBirths": 0,
            "extraStrategies": 0,
            "marketOrders": 0,
            "capitalBookings": 0,
        },
    }
