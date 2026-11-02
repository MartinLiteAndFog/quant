"""Cross-language ECO-01 / Cloudgarden adapter and ledger-invariant tests."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
from pydantic import ValidationError

from quant.ecosystem.cloudgarden_bridge import WireObserverJournal, dump_wire, load_wire

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "ecosystem" / "cloudgarden-bridge-v1.json"
NODE_ENTRY = ROOT / "apps" / "cloudgarden" / "src" / "ecoWireBridge.cli.ts"


def golden() -> dict:
    return json.loads(FIXTURE.read_text())


def next_day(doc: dict) -> None:
    doc["day_index"] = 1
    doc["event_id"] = "golden-demo:d1:wire-snapshot"
    doc["as_of"] = "2026-01-02T00:00:00.000Z"
    doc["known_at"] = doc["as_of"]
    doc["groundwater"]["as_of"] = doc["as_of"]
    doc["groundwater"]["accounts"][0]["as_of"] = doc["as_of"]


def negative_nav(doc: dict) -> None:
    water = doc["groundwater"]
    water.update(equity="-2.00000", reserve="0.00000", distributable="0.00000",
                 risk_breach=True, allocations_frozen=True)
    water["accounts"][0].update(equity="-2.00000", free_margin="0.00000")
    water["entitlements"] = [{
        **water["entitlements"][0], "amount": "0.0000",
        "used_margin": "10.0000", "realizable_margin": "0.0000",
    }]
    doc["plants"][0]["allocation"] = "0.0000"
    doc["seed_entitlements"][0] = {"id": 1, "status": "used", "plant_id": "parent-01"}
    doc["capital_accounting"] = {
        "deployed_allocation": "0.0000", "seed_reserved_allocation": "0.0000",
        "free_allocation": "0.0000", "rounding_holdback": "0.00000",
        "used_margin": "10.0000", "reserved_order_margin": "0.0000", "free_margin": "0.00000",
    }
    doc["booked"]["net_pnl"] = "-102.00000"
    doc["booked"]["reconciliation_delta"] = "-102.00000"


def test_python_golden_roundtrip_preserves_decimal_strings_and_utc():
    wire = load_wire(golden())
    dumped = json.loads(dump_wire(wire))
    assert dumped["groundwater"]["equity"] == "100.00000"
    assert dumped["groundwater"]["reserve"] == "10.00000"
    assert dumped["as_of"] == dumped["groundwater"]["as_of"]
    assert dumped["as_of"] == dumped["groundwater"]["accounts"][0]["as_of"]
    assert dump_wire(load_wire(dump_wire(wire))) == dump_wire(wire)
    assert wire.plants[0].habitat.fitness == 73
    assert wire.plants[0].empirical is None


def test_python_to_typescript_to_python_golden():
    node = shutil.which("node")
    tsx = ROOT / "apps" / "cloudgarden" / "node_modules" / "tsx"
    if node is None or not tsx.exists():
        pytest.skip("Node/tsx missing; mandatory in bridge CI")
    first = dump_wire(load_wire(golden()))
    result = subprocess.run(
        [node, "--import", "tsx", str(NODE_ENTRY)],
        input=first, text=True, capture_output=True, check=True,
        cwd=ROOT / "apps" / "cloudgarden",
        env={**os.environ, "NODE_ENV": "test"},
    )
    second = dump_wire(load_wire(result.stdout))
    assert second == first


def test_conserved_seed_transfer_is_not_a_new_deposit():
    doc = golden()
    next_day(doc)
    doc["seed_entitlements"][0] = {"id": 1, "status": "used", "plant_id": "seedling-01"}
    doc["groundwater"]["entitlements"] = [
        e for e in doc["groundwater"]["entitlements"] if e["strategy_id"] != "seedbank:01"
    ]
    doc["groundwater"]["entitlements"].append({
        **doc["groundwater"]["entitlements"][0],
        "strategy_id": "seedling-01", "amount": "1.5000", "realizable_margin": "1.5000",
    })
    doc["plants"].append({
        **doc["plants"][0], "strategy_id": "seedling-01", "allocation": "1.5000",
        "lineage_parent_ids": ["parent-01", "parent-01"],
    })
    doc["capital_accounting"]["deployed_allocation"] = "21.5000"
    doc["capital_accounting"]["seed_reserved_allocation"] = "0.0000"
    assert load_wire(doc).groundwater.equity == 100
    journal = WireObserverJournal()
    assert journal.observe(golden()) == "observed"
    assert journal.observe(doc) == "observed"
    assert journal.observe(doc) == "duplicate"
    assert journal.latest is not None and journal.latest.day_index == 1
    conflict = copy.deepcopy(doc)
    conflict["booked"]["reconciliation_delta"] = "1.00000"
    conflict["booked"]["price_pnl"] = "-1.00000"
    with pytest.raises(ValueError, match="conflicting"):
        journal.observe(conflict)


def test_external_cashflow_is_not_mistaken_for_trading_profit():
    doc = golden()
    next_day(doc)
    doc["booked"]["cashflow"] = "10.0000"
    water = doc["groundwater"]
    water.update(equity="110.00000", reserve="11.00000", distributable="99.00000")
    water["accounts"][0].update(equity="110.00000", free_margin="110.00000")
    doc["capital_accounting"].update(free_margin="110.00000", free_allocation="77.5000")
    assert load_wire(doc).booked.cashflow == 10
    doc["booked"]["cashflow"] = "0.0000"
    with pytest.raises(ValidationError):
        load_wire(doc)


def test_negative_nav_can_be_reported_but_must_freeze_budget():
    doc = golden()
    negative_nav(doc)
    wire = load_wire(doc)
    assert wire.groundwater.equity == -2
    assert wire.groundwater.distributable == 0
    assert wire.groundwater.entitlements[0].used_margin == 10
    doc["groundwater"]["allocations_frozen"] = False
    with pytest.raises(ValidationError):
        load_wire(doc)


@pytest.mark.parametrize("mutate", [
    lambda d: d["plants"][0].update(empirical={"twr": 0.2, "oos_credibility": 1}),
    lambda d: d["groundwater"].update(equity=100.0),
    lambda d: d["groundwater"].update(reserve="9.00000"),
    lambda d: d["groundwater"]["entitlements"].append(d["groundwater"]["entitlements"][0]),
    lambda d: d["groundwater"]["entitlements"][0].update(amount="90.0000"),
    lambda d: d["groundwater"]["entitlements"][0].update(used_margin="21.0000"),
    lambda d: d["groundwater"]["entitlements"][0].update(reserved_order_margin="21.0000"),
    lambda d: d["capital_accounting"].update(rounding_holdback="0.00006"),
    lambda d: d.update(known_at="2026-01-02T00:00:00.000Z"),
    lambda d: d.update(capital_writer="python-ecosystem"),
    lambda d: d["seed_entitlements"][0].update(status="used"),
])
def test_rejects_fabricated_evidence_double_budget_and_noncausal_data(mutate):
    doc = golden()
    mutate(doc)
    with pytest.raises((ValueError, ValidationError)):
        load_wire(doc)


def test_observer_rejects_stale_day_or_new_archive_identity():
    journal = WireObserverJournal()
    journal.observe(golden())
    next_snapshot = golden()
    next_day(next_snapshot)
    journal.observe(next_snapshot)
    third = copy.deepcopy(next_snapshot)
    third["day_index"] = 2
    third["event_id"] = "golden-demo:d2:wire-snapshot"
    third["dataset_fingerprint"] = "a" * 64
    with pytest.raises(ValueError, match="stale replay or second capital writer"):
        journal.observe(third)
