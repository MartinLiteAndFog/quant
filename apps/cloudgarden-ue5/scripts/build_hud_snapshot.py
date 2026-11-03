#!/usr/bin/env python3
"""Cloudgarden game HUD view model: finance and ecology are distinct domains.

Default: genuinely unavailable finance stays NULL; no fabricated NAV or leaderboards.
Optional synthetic finance fixture is for UI and regression testing ONLY.
Never use visual tile instances or requested paper budgets as portfolio value.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
from typing import Any

from export_unreal_bootstrap import build_bootstrap
from build_plant_inspector import build_plant_inspectors, make_focus_interaction
from tile_research_journal import digest
from validate_gene_physics import load

STATUS = "synthetic_only"
MONEY_FIELDS = ("navUsdt", "startingNavUsdt", "realizedNetPnlUsdt")
MAX_ROWS = 5


def dec(value: object) -> Decimal:
    if type(value) not in (str, int):
        raise ValueError("Financial values must be exact decimal strings or integer counters")
    if type(value) is str and (not value or value.strip() != value or "e" in value.lower()):
        raise ValueError("Invalid decimal representation")
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("Invalid financial decimal") from exc
    if not result.is_finite():
        raise ValueError("Non-finite financial amount")
    return result


def money(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.01')):.2f}"


def pct(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.01')):.2f}"


def metric(label: str, value: str | None, unit: str, source: str, *,
           availability: str = "available") -> dict:
    if value is None:
        availability = "unavailable"
    return {"label": label, "value": value, "unit": unit,
            "availability": availability, "source": source}


def validate_mock_finance(fin: dict, bootstrap: dict) -> dict:
    if (fin.get("schemaVersion") != 1 or fin.get("kind") != "cloudgarden_hud_mock_finance"
            or fin.get("authority") != "fictional_ui_test_only"
            or fin.get("sceneRef") != bootstrap["sceneRef"]
            or fin.get("allowBrokerUse") is not False):
        raise ValueError("HUD finance data is not a validated synthetic UI fixture")
    plant_ids = {p["id"] for p in bootstrap["sourcePlants"]}
    if len(fin["plantResults"]) > len(plant_ids):
        raise ValueError("More performance entries than source plants")
    ids = [p["plantId"] for p in fin["plantResults"]]
    if len(ids) != len(set(ids)) or not set(ids).issubset(plant_ids):
        raise ValueError("Duplicate or unknown ranking source plant")
    nav = dec(fin["navUsdt"])
    starting_nav = dec(fin["startingNavUsdt"])
    pnl = dec(fin["realizedNetPnlUsdt"])
    if nav <= 0 or starting_nav <= 0 or nav != starting_nav + pnl:
        raise ValueError("Fictional portfolio NAV must reconcile")
    results_total = Decimal("0")
    for row in fin["plantResults"]:
        value = dec(row["netPnlUsdt"])
        allocated = dec(row["notionalBasisUsdt"])
        if allocated <= 0:
            raise ValueError("Return denominator must be positive")
        if (type(row["closedTrades"]) is not int or type(row["winningTrades"]) is not int
                or row["closedTrades"] < 0
                or not 0 <= row["winningTrades"] <= row["closedTrades"]):
            raise ValueError("Invalid trade evidence")
        results_total += value
    if results_total != pnl:
        raise ValueError("Plant results must reconcile to fictional NAV, no hidden PnL")
    history = fin["navHistory"]
    days = [h["dayIndex"] for h in history]
    if (not history or days != sorted(set(days))
            or days[0] != 0 or days[-1] != bootstrap["sceneRef"]["dayIndex"]
            or dec(history[0]["navUsdt"]) != starting_nav
            or dec(history[-1]["navUsdt"]) != nav
            or any(dec(h["navUsdt"]) <= 0 for h in history)):
        raise ValueError("Fictional NAV history must start and end on validated days")
    return fin


def drawdown_percent(history: list[dict]) -> Decimal:
    peak = Decimal("0")
    worst = Decimal("0")
    for h in history:
        nav = dec(h["navUsdt"])
        peak = max(peak, nav)
        worst = max(worst, (peak - nav) * 100 / peak)
    return worst


def rank_rows(fin: dict, plants: dict, *, winners: bool) -> list[dict]:
    rows = []
    for source in fin["plantResults"]:
        plant_id = source["plantId"]
        net = dec(source["netPnlUsdt"])
        if (net > 0) != winners or net == 0:
            continue
        basis = dec(source["notionalBasisUsdt"])
        rate = net * 100 / basis
        name = plants[plant_id]["name"]
        rows.append({
            "plantId": plant_id, "plantName": name,
            "status": plants[plant_id]["status"],
            "netPnlUsdt": money(net),
            "returnPct": pct(rate),
            "closedTrades": source["closedTrades"],
            "winningTrades": source["winningTrades"],
            "source": STATUS,
            "tileId": None,  # unique canonical plant ranking, never duplicate by tile
        })
    # Default is absolute NET PnL, not eye-catching ROI on tiny stakes.
    # Negative ranking is most negative first; positive ranking is largest first.
    rows.sort(key=lambda r: (-dec(r["netPnlUsdt"]) if winners else dec(r["netPnlUsdt"]),
                             r["plantId"]))
    return rows[:MAX_ROWS]


def build_hud_snapshot(bootstrap: dict, fin: dict | None = None) -> dict:
    if bootstrap["kind"] != "cloudgarden_ue5_synthetic_scene_bootstrap":
        raise ValueError("HUD can only ingest the validated synthetic bootstrap in this version")
    if (bootstrap["permissions"]["unrealRuntimeMayBookCapital"] is not False
            or bootstrap["permissions"]["unrealRuntimeMaySendExchangeOrders"] is not False):
        raise ValueError("Unreal HUD source cannot mutate capital or venue")
    plants = {p["id"]: p for p in bootstrap["sourcePlants"]}
    if len(plants) != len(bootstrap["sourcePlants"]):
        raise ValueError("Duplicated canonical plant")
    instances = bootstrap["tileVisualInstances"]
    active_ids = {pid for pid, p in plants.items() if p["status"] == "active"}
    if any(i["plantId"] not in active_ids for i in instances):
        raise ValueError("An archived plant was rendered as active")
    tile_ids = {t["id"] for t in bootstrap["parameterTiles"]}
    states = bootstrap["tileCapitalAndEvolutionRequests"]
    if {x["tileId"] for x in states} != tile_ids:
        raise ValueError("Tile mode table does not match catalog")
    # Never produce NAV from historical price conjecture, summed plant biomass,
    # multiple render instances, mock virtual requests or an unapproved journal.
    financial = {
        "totalBiomassPortfolioUsdt": metric("Gesamtbiomasse · Portfolio-NAV", None, "USDT", "authoritative_ledger_missing"),
        "portfolioReturnPct": metric("Portfolio-Wachstum", None, "%", "authoritative_nav_history_missing"),
        "netPnlUsdt": metric("Netto-PnL", None, "USDT", "authoritative_ledger_missing"),
        "maxDrawdownPct": metric("Max. Drawdown", None, "%", "authoritative_nav_history_missing"),
        "winRatePct": metric("Trade-Trefferquote", None, "%", "closed_trades_missing"),
    }
    winners: list[dict] = []
    losers: list[dict] = []
    financial_source = "unavailable_no_ledger"
    if fin is not None:
        validate_mock_finance(fin, bootstrap)
        financial_source = STATUS
        available = {"source": STATUS, "availability": "mock_only"}
        starting = dec(fin["startingNavUsdt"])
        nav = dec(fin["navUsdt"])
        pnl = dec(fin["realizedNetPnlUsdt"])
        trades = sum(x["closedTrades"] for x in fin["plantResults"])
        wins = sum(x["winningTrades"] for x in fin["plantResults"])
        financial = {
            "totalBiomassPortfolioUsdt": metric("Gesamtbiomasse · Portfolio-NAV", money(nav), "USDT", **available),
            "portfolioReturnPct": metric("Portfolio-Wachstum", pct((nav / starting - 1) * 100), "%", **available),
            "netPnlUsdt": metric("Netto-PnL", money(pnl), "USDT", **available),
            "maxDrawdownPct": metric("Max. Drawdown", pct(drawdown_percent(fin["navHistory"])), "%", **available),
            "winRatePct": metric("Trade-Trefferquote", pct(Decimal(100 * wins) / trades) if trades else None,
                                 "%", STATUS, availability="mock_only"),
        }
        if not trades:
            financial["winRatePct"]["availability"] = "unavailable"
        winners = rank_rows(fin, plants, winners=True)
        losers = rank_rows(fin, plants, winners=False)
    pending = sum(len(r["outcomes"]) for r in bootstrap["seedResearchPlan"])
    candidate_count = sum(1 for r in bootstrap["seedResearchPlan"]
                          for o in r["outcomes"] if o["status"] == "unapproved_offspring_gene_patch")
    states_by_id = {x["tileId"]: x for x in states}
    tile_cards = []
    for tile in bootstrap["parameterTiles"]:
        tile_id = tile["id"]
        cfg = states_by_id[tile_id]
        tile_cards.append({
            "tileId": tile_id,
            "label": tile["label"],
            "axes": [tile["axes"]["x"], tile["axes"]["y"]],
            "visiblePlantInstances": sum(1 for i in instances if i["tileId"] == tile_id),
            "evolutionEnabled": cfg["evolutionEnabled"],
            "capitalMode": cfg["capitalMode"],
            "paperFeedMode": cfg["paperFeedMode"],
            "paperBudgetRequestedUsdt": cfg["paperBudgetRequestedUsdt"],
            "paperBudgetIsFunded": False,
            "liveEnabled": False,
        })
    nav_history = (fin["navHistory"] if fin else [])
    inspectors = build_plant_inspectors(bootstrap, fin)
    return {
        "schemaVersion": 1,
        "kind": "cloudgarden_ue5_hud_snapshot",
        "authority": "synthetic_hud_preview_only",
        "sceneRef": bootstrap["sceneRef"],
        "bootstrapDigest": bootstrap["bundleSha256"],
        "sourceBanner": "DEMO · SYNTHETISCHE DATEN · KEIN LIVE-HANDEL",
        "financialSource": financial_source,
        "hudMode": "atlas",  # the same data remains valid for Explore
        "leaderboard": {
            "defaultToggle": "winners",
            "toggleOptions": ["winners", "losers"],
            "rankBy": "netPnlUsdt",
            "limit": 5,
            "winnerRows": winners,
            "loserRows": losers,
            "scope": "canonical_plant_unique_only",
            "evidence": STATUS if fin else "missing_verified_plant_results",
        },
        "financialMetrics": financial,
        "portfolioNavSeries": [
            {"dayIndex": x["dayIndex"], "navUsdt": x["navUsdt"]} for x in nav_history
        ],
        "ecologyMetrics": {
            "canonicalPlants": len(plants),
            "activePlants": len(active_ids),
            "archivedPlants": len(plants) - len(active_ids),
            "visibleTileRepresentations": len(instances),
            "tilesTotal": len(tile_ids),
            "tilesEvolving": sum(1 for x in states if x["evolutionEnabled"]),
            "paperTilesRequested": sum(1 for x in states if x["capitalMode"] == "paper"),
            "liveTilesApproved": 0,
            "independentSeedReleases": len(bootstrap["seedResearchPlan"]),
            "referenceSeedLandings": pending,
            "proposedOffspringCandidates": candidate_count,
            "newCanonicalBirthsApproved": 0,
            "source": STATUS,
        },
        "paperBudget": {
            "requestedTotalUsdt": bootstrap["totals"]["paperBudgetRequestedUsdt"],
            "fundedTotalUsdt": "0.00",
            "source": "virtual_requests_not_booked",
        },
        "tileCards": tile_cards,
        "plantInspectors": inspectors,
        "plantFocusInteraction": make_focus_interaction(bootstrap),
        "focus": {
            "selectedPlantId": "plant-a",
            "selectedTileId": "sol-imbase",
            "plantInstanceIds": bootstrap["plantToVisualInstances"]["plant-a"],
        },
        "hudControls": {
            "atlasVsExplore": ["atlas", "explore"],
            "exploreStartsCompact": True,
            "rankToggle": True,
            "tileFilter": True,
            "chronologyReadOnly": True,
            "capitalModeAction": "preview_only_requires_backend_acceptance",
            "liveModeClickAction": "blocked",
        },
        "recentResearchEvents": [
            {"type": e["type"], "tileId": e["tileId"], "plantId": e["plantId"],
             "dayIndex": e["dayIndex"], "source": "synthetic_reference_only"}
            for e in bootstrap["referenceEventJournal"]["events"]
            if e["dayIndex"] <= bootstrap["sceneRef"]["dayIndex"]
        ][-5:],
        "risk": {
            "source": "authoritative_risk_backend_missing",
            "dailyLossLimitPct": None,
            "totalLossLimitPct": None,
            "usedRiskPct": None,
            "aggregateExposureUsdt": None,
            "killSwitchStatus": "unknown_not_connected",
            "noGreenSafetyBadgeWithoutBackend": True,
        },
        "alerts": [
            {"severity": "info", "code": "NOT_LIVE", "text": "Demo-Modus: keine Orders"},
            {"severity": "info", "code": "PAPER_UNFUNDED", "text": "Paper-Kapital ist angefragt, nicht gebucht"},
            *([{"severity": "warning", "code": "NAV_UNAVAILABLE", "text": "Kein bestätigter Portfolio-NAV"}]
              if fin is None else []),
        ],
    }


def default_hud(*, mock_finance: bool = False) -> dict:
    bootstrap = build_bootstrap()
    fin = load("fixtures/hud_mock_finance_v1.json") if mock_finance else None
    return build_hud_snapshot(bootstrap, fin)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock-finance", action="store_true",
                        help="Inject EXPLICITLY FICTIONAL finance series for HUD design only")
    args = parser.parse_args()
    print(json.dumps(default_hud(mock_finance=args.mock_finance),
                     indent=2, sort_keys=True, ensure_ascii=False))
