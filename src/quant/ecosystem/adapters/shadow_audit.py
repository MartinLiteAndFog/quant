"""ECO-13 review gate: immutable, read-only pre-live safety checklist.

Do not confuse a claim about a check with independent verification. This module
does NOT fetch artifacts, validate signatures, contact brokers, send orders,
activate alerts, write strategy registries, or authorize production cutover.
Even when the submitted checklist is complete the answer remains NO_GO.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from typing import Literal, Sequence

from .shadow_bridge import StrategyConfigSnapshot

SCHEMA = "eco13.shadow-audit.v1"
GATES: tuple[str, ...] = (
    "original_market_data_identity_and_license",
    "instrument_and_contract_multiplier",
    "tradingview_indicator_and_signal_parity",
    "independent_out_of_sample_and_multiplicity",
    "fee_slippage_spread_and_funding",
    "causal_net_pnl_and_mark_to_market",
    "hedge_mode_read_only_check",
    "single_authoritative_order_writer",
    "reduce_only_tp_sl_owned_by_strategy",
    "no_cross_bot_cancel_all",
    "minimum_lot_and_step_size",
    "fill_order_and_position_reconciliation",
    "crash_recovery_event_idempotence",
    "global_loss_margin_and_90_10_reserve",
    "shadow_only_event_replay",
    "separate_operator_cutover_procedure",
)
DIGEST_RE = re.compile(r"^[a-f0-9]{64}$")
STATUS = ("pass_claimed", "fail", "unknown")


class AuditError(ValueError):
    """Invalid attestations are not accepted as safety evidence."""


def _utc(dt: datetime, name: str) -> datetime:
    if not isinstance(dt, datetime) or dt.tzinfo is None or dt.utcoffset() is None:
        raise AuditError(f"{name} must be an aware UTC timestamp")
    return dt.astimezone(timezone.utc)


def snapshot_digest(snapshot: StrategyConfigSnapshot) -> str:
    """Hash a *canonical frozen* candidate; no mutable genome auto-approval."""
    if not isinstance(snapshot, StrategyConfigSnapshot):
        raise AuditError("versioned shadow StrategyConfigSnapshot required")
    if (snapshot.execution_status != "shadow" or
        snapshot.requires_human_registration is not True):
        raise AuditError("only unregistered shadow candidates can be reviewed")
    try:
        serialized = json.dumps(snapshot.as_dict(), sort_keys=True,
                                separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise AuditError("snapshot is not deterministic JSON") from exc
    return sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GateAttestation:
    """CLAIM + external artifact pointer; artifact bytes are not verified here."""

    gate_id: str
    status: Literal["pass_claimed", "fail", "unknown"]
    snapshot_sha256: str
    checked_at: datetime
    reviewer: str
    artifact_sha256: str | None = None
    artifact_ref: str | None = None

    def __post_init__(self):
        if self.gate_id not in GATES:
            raise AuditError("unknown required gate")
        if self.status not in STATUS:
            raise AuditError("invalid gate status")
        if not DIGEST_RE.fullmatch(self.snapshot_sha256):
            raise AuditError("gate must be bound to full candidate snapshot SHA256")
        object.__setattr__(self, "checked_at", _utc(self.checked_at, "checked_at"))
        if not isinstance(self.reviewer, str) or not self.reviewer.strip() or len(self.reviewer) > 128:
            raise AuditError("named human reviewer required")
        if self.status == "pass_claimed":
            if not isinstance(self.artifact_sha256, str) or not DIGEST_RE.fullmatch(self.artifact_sha256):
                raise AuditError("pass_claimed requires SHA256 of separate evidence artifact")
            if (not isinstance(self.artifact_ref, str) or
                not self.artifact_ref.strip() or len(self.artifact_ref)>1024):
                raise AuditError("pass_claimed requires artifact reference")
        elif self.artifact_sha256 and not DIGEST_RE.fullmatch(self.artifact_sha256):
            raise AuditError("invalid optional artifact fingerprint")


@dataclass(frozen=True)
class ShadowAuditReport:
    schema: str
    strategy_id: str
    snapshot_sha256: str
    as_of: str
    status: str
    claimed_pass: tuple[str, ...]
    failed: tuple[str, ...]
    unknown: tuple[str, ...]
    absent: tuple[str, ...]
    attestation_count: int
    artifact_hashes_unverified: bool = True
    independent_market_oos_verified_by_this_tool: bool = False
    live_routing_authorized: bool = False
    order_sent: bool = False
    cutover: Literal["NO_GO"] = "NO_GO"

    def as_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, indent=2)


def review_shadow_gates(
    snapshot: StrategyConfigSnapshot,
    attestations: Sequence[GateAttestation],
    *,
    as_of: datetime,
    max_attestation_age_days: int = 90,
) -> ShadowAuditReport:
    """Review checklist completeness without interpreting claims as real approval.

    Fails closed on unknown/duplicate/gene-version-mismatched/future/stale
    evidence. Evidence read from an external source should be verified by a
    separately authorized human/broker audit system. Even if all 16 gate
    *claims* pass, this module never returns a GO or routes an order.
    """
    now=_utc(as_of,"as_of")
    if type(max_attestation_age_days) is not int or not 0<max_attestation_age_days<=365:
        raise AuditError("invalid evidence freshness limit")
    candidate_hash=snapshot_digest(snapshot)
    observed:dict[str,GateAttestation]={}
    for gate in attestations:
        if not isinstance(gate, GateAttestation):
            raise AuditError("typed gate attestations required")
        if gate.gate_id in observed:
            raise AuditError("duplicate safety gate; conflicting claims not allowed")
        if gate.snapshot_sha256 != candidate_hash:
            raise AuditError("gate evidence is tied to a different genome/snapshot")
        if gate.checked_at>now:
            raise AuditError("gate evidence is future-dated")
        observed[gate.gate_id]=gate
    from datetime import timedelta
    cutoff=now-timedelta(days=max_attestation_age_days)
    claimed=tuple(sorted(k for k,v in observed.items()
                         if v.status=="pass_claimed" and v.checked_at>=cutoff))
    stale=tuple(sorted(k for k,v in observed.items()
                       if v.status=="pass_claimed" and v.checked_at<cutoff))
    failed=tuple(sorted(k for k,v in observed.items() if v.status=="fail"))
    unknown=tuple(sorted(
        {k for k,v in observed.items() if v.status=="unknown"}|set(stale)
    ))
    absent=tuple(sorted(set(GATES)-set(observed)))
    complete=not (failed or unknown or absent) and len(claimed)==len(GATES)
    # Completing the *claimed* checklist is never a verified go-live check.
    state="claims_complete_needs_independent_review" if complete else "NO_GO_missing_or_failed_checks"
    return ShadowAuditReport(
        schema=SCHEMA,strategy_id=snapshot.strategy_id,
        snapshot_sha256=candidate_hash,
        as_of=now.isoformat(),status=state,
        claimed_pass=claimed,failed=failed,unknown=unknown,absent=absent,
        attestation_count=len(observed),
    )
