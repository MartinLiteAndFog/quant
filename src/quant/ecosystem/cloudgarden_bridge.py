"""ECO-01 / Cloudgarden read-only wire bridge, v1.

This module verifies Cloudgarden paper snapshots using the canonical ECO-01
GroundwaterState and never writes to paper, broker or account ledgers.
"""
from __future__ import annotations

import json
import re
from decimal import Decimal
from datetime import datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from quant.ecosystem.contracts import Contract, GroundwaterState, Money

BRIDGE_VERSION = "eco-cloudgarden/1.0"
_HASH = re.compile(r"^[a-f0-9]{64}$")
_UTC = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{3})?Z$")
_ZERO = Decimal("0")
_ONE_TENTH = Decimal("0.10")


def _timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not _UTC.fullmatch(value):
        raise ValueError("a fully qualified UTC timestamp is required")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.utcoffset().total_seconds() != 0:
        raise ValueError("UTC timestamp required")
    return result


def _exact_money(value: Decimal) -> bool:
    return value.is_finite() and value.as_tuple().exponent >= -5


class BookedPnl(Contract):
    opening_nav: Money
    cashflow: Money
    price_pnl: Money
    fees: Money
    funding: Money
    net_pnl: Money
    reconciliation_delta: Money


class CapitalAccounting(Contract):
    deployed_allocation: Money
    seed_reserved_allocation: Money
    free_allocation: Money
    used_margin: Money
    reserved_order_margin: Money
    free_margin: Money
    rounding_holdback: Money


class HabitatMetrics(Contract):
    fitness: float = Field(ge=0, le=100)
    vigor: float = Field(ge=0, le=100)
    evidence: float = Field(ge=0, le=100)
    evidence_days: int = Field(ge=0)


class WirePlant(Contract):
    strategy_id: str
    status: Literal["active", "archived"]
    allocation: Money
    genome_version: Literal[1]
    strategy_code_version: Literal["cloudgarden-source-v1", "cloudgarden-demo-v1"]
    genome_kind: Literal["source_candle_research", "synthetic_demo"]
    source_genome: dict[str, Any] | None
    visual_genes: dict[str, float]
    trading_traits: dict[str, int]
    lineage_parent_ids: tuple[str, ...]
    habitat: HabitatMetrics
    empirical: None

    @model_validator(mode="after")
    def valid_genome(self):
        source = self.source_genome
        if source is None:
            if self.genome_kind != "synthetic_demo" or self.strategy_code_version != "cloudgarden-demo-v1":
                raise ValueError("synthetic visual DNA is not historical trading DNA")
        else:
            if self.genome_kind != "source_candle_research" or self.strategy_code_version != "cloudgarden-source-v1":
                raise ValueError("source genome and code version must agree")
            required = {
                "schemaVersion", "family", "channelLookback", "pivotLength", "atrLength",
                "ewMinMoveAtr", "ewRetraceMin", "ewRetraceMax", "ewFilter", "ewExit", "stopLossBps",
            }
            if not required.issubset(source) or source["schemaVersion"] != 1 or source["family"] not in {
                "imba_trend", "imba_countertrend", "imba_ew"
            } or type(source["stopLossBps"]) not in (int, float) or not 50 <= source["stopLossBps"] <= 500:
                raise ValueError("mandatory bounded historical close stop/source genome missing")
        if self.allocation < 0 or (self.status == "archived" and self.allocation != 0):
            raise ValueError("archived plants cannot hold paper entitlements")
        if not self.strategy_id:
            raise ValueError("empty plant ID")
        return self


class SeedSlot(Contract):
    id: int = Field(gt=0)
    status: Literal["available", "used"]
    plant_id: str | None


class CloudgardenWire(Contract):
    schema_version: Literal["eco-cloudgarden/1.0"]
    ecosystem_schema_version: Literal["1.0"]
    simulation_id: str
    dataset_fingerprint: str
    source_kind: Literal["historical", "synthetic"]
    capital_writer: Literal["cloudgarden-paper"]
    access: Literal["read_only"]
    account_id: str
    currency: Literal["USDT"]
    as_of: str
    known_at: str
    day_index: int = Field(ge=0)
    event_id: str
    booked: BookedPnl
    groundwater: GroundwaterState
    capital_accounting: CapitalAccounting
    seed_entitlements: tuple[SeedSlot, ...]
    plants: tuple[WirePlant, ...]

    @model_validator(mode="after")
    def conserved(self):
        if not self.simulation_id or self.account_id != "cloudgarden-paper:" + self.simulation_id:
            raise ValueError("a single Cloudgarden account writer is mandatory")
        if not _HASH.fullmatch(self.dataset_fingerprint) or (
            self.source_kind == "historical" and self.dataset_fingerprint == "0" * 64
        ):
            raise ValueError("historical data fingerprint must identify a nonzero archive")
        as_of = _timestamp(self.as_of)
        known_at = _timestamp(self.known_at)
        if known_at > as_of or self.event_id != f"{self.simulation_id}:d{self.day_index}:wire-snapshot":
            raise ValueError("future event or duplicate-unsafe event ID")
        water = self.groundwater
        if water.currency != self.currency or water.as_of != as_of:
            raise ValueError("account currency/day differs from ECO-01 GroundwaterState")
        if len(water.accounts) != 1:
            raise ValueError("exactly one authoritative account is permitted")
        account = water.accounts[0]
        if account.account_id != self.account_id or account.free_margin != self.capital_accounting.free_margin:
            raise ValueError("foreign writer or mismatched account free margin")
        positive = max(_ZERO, water.equity)
        if positive != positive.quantize(Decimal("0.0001")):
            raise ValueError("Cloudgarden NAV is limited to 0.0001 USDT")
        if self.groundwater.reserve != positive * _ONE_TENTH or (
            self.groundwater.distributable != positive - self.groundwater.reserve
        ):
            raise ValueError("90/10 must conserve exact money without rounding upward")
        if water.risk_breach != water.allocations_frozen or (
            water.equity < 0 and not (water.risk_breach and water.allocations_frozen)
        ):
            raise ValueError("risk breach must freeze allocations")
        if water.equity >= 0 and account.free_margin > water.equity:
            raise ValueError("account free margin exceeds NAV")
        for amount in [
            self.booked.opening_nav, self.booked.cashflow, self.booked.price_pnl,
            self.booked.fees, self.booked.funding, self.booked.net_pnl,
            self.booked.reconciliation_delta, self.capital_accounting.deployed_allocation,
            self.capital_accounting.seed_reserved_allocation,
            self.capital_accounting.free_allocation, self.capital_accounting.used_margin,
            self.capital_accounting.reserved_order_margin, self.capital_accounting.free_margin,
            self.capital_accounting.rounding_holdback,
        ]:
            if not _exact_money(amount):
                raise ValueError("wire money must have no more than five decimal places")
        if self.booked.fees < 0 or (
            self.booked.opening_nav + self.booked.cashflow + self.booked.net_pnl != water.equity
        ) or (
            self.booked.price_pnl - self.booked.fees + self.booked.funding
            + self.booked.reconciliation_delta != self.booked.net_pnl
        ):
            raise ValueError("cashflow, realized PnL, fees and NAV do not reconcile")
        claims = {e.strategy_id: e for e in water.entitlements}
        if len(claims) != len(water.entitlements):
            raise ValueError("double-funded entitlement")
        capital = self.capital_accounting
        if any(v < 0 for v in (
            capital.deployed_allocation, capital.seed_reserved_allocation, capital.free_allocation,
            capital.used_margin, capital.reserved_order_margin, capital.free_margin,
            capital.rounding_holdback
        )):
            raise ValueError("negative available allocations or margins")
        if capital.rounding_holdback > Decimal("0.00005"):
            raise ValueError("rounding holdback exceeds one half-unit at 4dp")
        if sum((e.amount for e in claims.values()), _ZERO) != (
            capital.deployed_allocation + capital.seed_reserved_allocation
        ) or sum((e.amount for e in claims.values()), _ZERO) + capital.free_allocation > water.distributable:
            raise ValueError("allocation/seed transfers exceed the exact 90% ceiling")
        if sum((e.used_margin for e in claims.values()), _ZERO) != capital.used_margin or (
            sum((e.reserved_order_margin for e in claims.values()), _ZERO) != capital.reserved_order_margin
        ):
            raise ValueError("locked order margin was misclassified as free capital")
        plants = {p.strategy_id: p for p in self.plants}
        if not plants or len(plants) != len(self.plants):
            raise ValueError("plant IDs must be unique")
        if self.source_kind == "historical" and any(p.source_genome is None for p in plants.values()):
            raise ValueError("historical trading requires real source genomes")
        deployed = _ZERO
        for plant_id, plant in plants.items():
            if plant.status == "active":
                if plant_id not in claims or claims[plant_id].amount != plant.allocation:
                    raise ValueError("unfunded active plant")
                deployed += plant.allocation
            elif plant_id in claims:
                raise ValueError("archived plant retains an entitlement")
        if deployed != capital.deployed_allocation:
            raise ValueError("deployed amount differs from active plant claims")
        seen_ids: set[int] = set()
        used_plants: set[str] = set()
        reserved = _ZERO
        if not self.seed_entitlements:
            raise ValueError("seed bank missing")
        for slot in self.seed_entitlements:
            if slot.id in seen_ids:
                raise ValueError("duplicate seed bank ID")
            seen_ids.add(slot.id)
            key = f"seedbank:{slot.id:02d}"
            if slot.status == "available":
                if slot.plant_id is not None or key not in claims:
                    raise ValueError("available seed must have exactly one reserved claim")
                reserved += claims[key].amount
            else:
                if slot.plant_id not in plants or slot.plant_id in used_plants or key in claims:
                    raise ValueError("used seed was never transferred or is counted twice")
                used_plants.add(slot.plant_id)
        if reserved != capital.seed_reserved_allocation:
            raise ValueError("seedbank reserve does not reconcile")
        return self


def load_wire(document: str | dict[str, Any]) -> CloudgardenWire:
    """Validate with ECO-01 Decimal and GroundwaterState; does not book anything."""
    value = json.loads(document) if isinstance(document, str) else document
    return CloudgardenWire.model_validate(value)


def dump_wire(value: CloudgardenWire) -> str:
    return json.dumps(value.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False)


class WireObserverJournal:
    """In-memory idempotent observer; intentionally no write/transfer operation."""

    def __init__(self) -> None:
        self._events: dict[str, str] = {}
        self._latest: CloudgardenWire | None = None

    def observe(self, document: str | dict[str, Any]) -> Literal["observed", "duplicate"]:
        wire = load_wire(document)
        canonical = dump_wire(wire)
        existing = self._events.get(wire.event_id)
        if existing is not None:
            if existing != canonical:
                raise ValueError("conflicting payload reuses event_id")
            return "duplicate"
        if self._latest is not None:
            prior = self._latest
            if (
                wire.account_id != prior.account_id or wire.simulation_id != prior.simulation_id
                or wire.dataset_fingerprint != prior.dataset_fingerprint
                or wire.source_kind != prior.source_kind
                or wire.day_index <= prior.day_index
                or _timestamp(wire.known_at) < _timestamp(prior.known_at)
            ):
                raise ValueError("stale replay or second capital writer")
        self._events[wire.event_id] = canonical
        self._latest = wire
        return "observed"

    @property
    def latest(self) -> CloudgardenWire | None:
        return self._latest
