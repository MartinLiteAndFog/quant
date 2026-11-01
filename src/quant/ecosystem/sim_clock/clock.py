"""Deterministic, side-effect-free UTC day clock for simulated trading ecosystems.

The clock does not run strategy logic, place orders, or invent fills. External
paper execution supplies *incremental NAV changes* at their actual timestamps.
Biology, breeding and lifecycle phases use optional pure functional reducers.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import random
from typing import Any, Callable, Literal, Mapping, Sequence

PHASES = ("execution", "metrics", "capital", "ecology", "breeding", "lifecycle", "snapshot")
TICK_VERSION = "1.0"
KIND = Literal["signal", "pnl_delta", "fee", "funding", "external_cashflow"]
UTC = timezone.utc
ZERO = Decimal(0)


class ClockError(ValueError):
    """Unsafe, ambiguous or non-reproducible paper event/snapshot."""


def _money(value: str | Decimal, label: str) -> Decimal:
    if not isinstance(value, (str, Decimal)) or isinstance(value, bool):
        raise ClockError(f"{label} must use a decimal string/Decimal")
    try:
        d = Decimal(value)
    except (InvalidOperation, ValueError):
        raise ClockError(f"invalid {label}") from None
    if not d.is_finite():
        raise ClockError(f"non-finite {label}")
    return d


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ClockError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ClockError(f"non-JSON state: {exc}") from exc


@dataclass(frozen=True)
class IntradayEvent:
    event_id: str
    occurred_at: datetime
    available_at: datetime
    kind: KIND
    amount: Decimal = ZERO
    strategy_id: str = ""

    def __post_init__(self):
        if not self.event_id or not self.event_id.strip():
            raise ClockError("event_id required")
        a, b = _utc(self.occurred_at, "occurred_at"), _utc(self.available_at, "available_at")
        if b < a:
            raise ClockError("signal/event data available before occurrence")
        if self.kind not in ("signal", "pnl_delta", "fee", "funding", "external_cashflow"):
            raise ClockError("unknown paper event kind")
        value = _money(self.amount, "paper amount")
        if self.kind == "signal" and value != ZERO:
            raise ClockError("signal events cannot create profit")
        if self.kind == "fee" and value > ZERO:
            raise ClockError("fees cannot increase NAV")
        object.__setattr__(self, "occurred_at", a)
        object.__setattr__(self, "available_at", b)
        object.__setattr__(self, "amount", value)


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    simulation_id: str
    day_index: int
    sequence: int
    phase: str
    occurred_at: str
    known_at: str
    kind: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class SimulationState:
    simulation_id: str
    seed: int
    as_of: date
    day_index: int
    equity: Decimal
    reserve: Decimal
    distributable: Decimal
    seen_event_ids: tuple[str, ...] = ()
    state_json: str = "{}"
    audit: tuple[AuditEvent, ...] = ()
    version: str = TICK_VERSION

    @property
    def ecology(self) -> dict[str, Any]:
        return json.loads(self.state_json)

    def to_json(self) -> str:
        obj = {
            "version": self.version, "simulation_id": self.simulation_id,
            "seed": self.seed, "as_of": self.as_of.isoformat(), "day_index": self.day_index,
            "equity": str(self.equity), "reserve": str(self.reserve),
            "distributable": str(self.distributable),
            "seen_event_ids": list(self.seen_event_ids),
            "state": json.loads(self.state_json),
            "audit": [vars(a) for a in self.audit],
        }
        return _canonical(obj)

    @classmethod
    def from_json(cls, raw: str) -> "SimulationState":
        try:
            obj = json.loads(raw)
            if set(obj) != {"version", "simulation_id", "seed", "as_of", "day_index",
                            "equity", "reserve", "distributable", "seen_event_ids", "state", "audit"}:
                raise ClockError("unexpected snapshot fields")
            if obj["version"] != TICK_VERSION:
                raise ClockError("unsupported tick version")
            audits = tuple(AuditEvent(**entry) for entry in obj["audit"])
            state = cls(
                version=obj["version"], simulation_id=obj["simulation_id"],
                seed=obj["seed"], as_of=date.fromisoformat(obj["as_of"]),
                day_index=obj["day_index"], equity=_money(obj["equity"], "equity"),
                reserve=_money(obj["reserve"], "reserve"),
                distributable=_money(obj["distributable"], "distributable"),
                seen_event_ids=tuple(obj["seen_event_ids"]),
                state_json=_canonical(obj["state"]), audit=audits,
            )
            _check_state(state)
            return state
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ClockError):
                raise
            raise ClockError(f"invalid checkpoint: {exc}") from exc


@dataclass(frozen=True)
class DayResult:
    state: SimulationState
    day_events: tuple[AuditEvent, ...]
    daily_net_pnl: Decimal
    external_cashflow: Decimal


# Reducers are pure paper logic. Input is a fresh JSON dict and a local seeded RNG.
# Output is a replacement JSON dict; no implicit I/O, no access to future bars.
PhaseReducer = Callable[[dict[str, Any], random.Random], Mapping[str, Any]]


def _check_state(state: SimulationState) -> None:
    if state.version != TICK_VERSION or not state.simulation_id:
        raise ClockError("invalid simulation identity/version")
    if type(state.seed) is not int or type(state.day_index) is not int or state.day_index < -1:
        raise ClockError("invalid seed/day_index")
    if len(set(state.seen_event_ids)) != len(state.seen_event_ids):
        raise ClockError("duplicate persisted event IDs")
    if _canonical(json.loads(state.state_json)) != state.state_json:
        raise ClockError("noncanonical paper-state JSON")
    reserve = max(ZERO, state.equity) * Decimal("0.10")
    if state.reserve != reserve or state.distributable != max(ZERO, state.equity) - reserve:
        raise ClockError("90/10 NAV conservation violated")
    phases = [PHASES.index(e.phase) for e in state.audit]
    if not all(e.simulation_id == state.simulation_id for e in state.audit):
        raise ClockError("foreign audit")
    if len(set(e.event_id for e in state.audit)) != len(state.audit):
        raise ClockError("duplicate audit event IDs")
    if any(e.sequence != i for i, e in enumerate(state.audit)):
        raise ClockError("non-contiguous audit sequence")
    if any(state.audit[i].day_index > state.audit[i+1].day_index
           or (state.audit[i].day_index == state.audit[i+1].day_index
               and phases[i] > phases[i+1]) for i in range(len(phases)-1)):
        raise ClockError("out-of-order audit")
    if any(e.day_index > state.day_index for e in state.audit):
        raise ClockError("audit from future day")


def new_simulation(
    simulation_id: str, equity: str | Decimal, first_day: date, *, seed: int = 1
) -> SimulationState:
    amount = _money(equity, "starting equity")
    if amount < ZERO or not simulation_id or type(seed) is not int:
        raise ClockError("starting NAV must be nonnegative and simulation ID/seed valid")
    if not isinstance(first_day, date) or isinstance(first_day, datetime):
        raise ClockError("first_day must be UTC date")
    reserve = amount * Decimal("0.10")
    state = SimulationState(simulation_id, seed, first_day - timedelta(days=1),
                            -1, amount, reserve, amount - reserve)
    _check_state(state)
    return state


def _phase_rng(seed: int, day_index: int, phase: str) -> random.Random:
    digest = sha256(f"{TICK_VERSION}|{seed}|{day_index}|{phase}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def advance_day(
    previous: SimulationState,
    day: date,
    intraday: Sequence[IntradayEvent] = (),
    reducers: Mapping[str, PhaseReducer] | None = None,
) -> DayResult:
    """Advance exactly one UTC calendar day, or raise without mutating previous.

    A period's event inputs represent *incremental* mark-to-market paper NAV
    changes already computed by execution; avoid supplying both full cumulative
    unrealized PnL and its daily delta or realized trades will double-count.
    """
    _check_state(previous)
    if not isinstance(day, date) or isinstance(day, datetime):
        raise ClockError("day must be date")
    if day != previous.as_of + timedelta(days=1):
        raise ClockError("day must be the next consecutive UTC date")
    rules = dict(reducers or {})
    if any(phase not in PHASES[1:-1] for phase in rules):
        raise ClockError("reducers permitted only for metrics/capital/ecology/breeding/lifecycle")
    cutoff = datetime.combine(day, time.max, tzinfo=UTC)
    sorted_inputs = sorted(intraday, key=lambda e: (e.available_at, e.occurred_at, e.event_id))
    if len({e.event_id for e in sorted_inputs}) != len(sorted_inputs):
        raise ClockError("duplicate event in day")
    if set(e.event_id for e in sorted_inputs) & set(previous.seen_event_ids):
        raise ClockError("event was already incorporated in a prior day")
    for event in sorted_inputs:
        if event.occurred_at.date() != day or event.available_at > cutoff:
            raise ClockError("event is outside day or future-known")
    today = previous.day_index + 1
    out: list[AuditEvent] = []
    equity = previous.equity
    net_pnl = external = ZERO
    extras = previous.ecology
    n = len(previous.audit)

    def record(phase: str, kind: str, payload: Mapping[str, Any], *, event_id: str | None = None,
               when: datetime = cutoff) -> None:
        _canonical(dict(payload))
        out.append(AuditEvent(
            event_id=event_id or f"{previous.simulation_id}:{today}:{phase}:{kind}:{len(out)}",
            simulation_id=previous.simulation_id, day_index=today, sequence=n+len(out),
            phase=phase, occurred_at=when.isoformat(), known_at=when.isoformat(),
            kind=kind, payload=dict(payload),
        ))

    for event in sorted_inputs:
        if event.kind == "external_cashflow":
            external += event.amount
        elif event.kind != "signal":
            net_pnl += event.amount
        equity += event.amount
        record("execution", event.kind, {
            "source_event_id": event.event_id, "strategy_id": event.strategy_id,
            "delta_nav": str(event.amount), "available_at": event.available_at.isoformat(),
        }, event_id=f"input:{previous.simulation_id}:{event.event_id}", when=event.available_at)

    reserve = max(ZERO, equity) * Decimal("0.10")
    distributable = max(ZERO, equity) - reserve
    for phase in PHASES[1:]:
        if phase == "metrics":
            record("metrics", "daily-accounting", {
                "net_pnl": str(net_pnl), "external_cashflow": str(external),
            })
        if phase == "capital":
            record("capital", "groundwater-90-10", {
                "equity": str(equity), "reserve": str(reserve),
                "distributable": str(distributable), "risk_frozen": equity < ZERO,
            })
        if phase in rules:
            # Reducers receive only data known by UTC close; each works with
            # an isolated copied state and deterministic phase-specific RNG.
            payload = {
                "as_of": day.isoformat(), "day_index": today,
                "equity": str(equity), "reserve": str(reserve),
                "distributable": str(distributable), "net_pnl": str(net_pnl),
                "external_cashflow": str(external), "state": deepcopy(extras),
            }
            value = rules[phase](deepcopy(payload), _phase_rng(previous.seed, today, phase))
            if not isinstance(value, Mapping):
                raise ClockError(f"{phase} reducer must return a JSON mapping")
            extras = json.loads(_canonical(dict(value)))
        if phase in ("ecology", "breeding", "lifecycle"):
            record(phase, "phase-complete", {"updated": phase in rules})
        if phase == "snapshot":
            record(phase, "phase-complete", {
                "state_digest": sha256(_canonical(extras).encode()).hexdigest(),
            })
    state = SimulationState(
        previous.simulation_id, previous.seed, day, today, equity, reserve,
        distributable,
        previous.seen_event_ids + tuple(e.event_id for e in sorted_inputs),
        _canonical(extras), previous.audit + tuple(out),
    )
    _check_state(state)
    return DayResult(state, tuple(out), net_pnl, external)


def advance_days(
    state: SimulationState, days: Sequence[tuple[date, Sequence[IntradayEvent]]],
    reducers: Mapping[str, PhaseReducer] | None = None,
) -> SimulationState:
    for day, events in days:
        state = advance_day(state, day, events, reducers).state
    return state
