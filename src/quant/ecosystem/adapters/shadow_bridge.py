"""Pure adapter from verified research candidates to manual shadow config.

No registry, database, network, TradingView, or venue client is imported here.
Functions validate data and return snapshots for human review only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping
from types import MappingProxyType

SCHEMA_VERSION = "1.0"
FAMILY_WEBHOOK_PATH = "/webhook/family"
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")
_LIFECYCLE = {
    "proposed", "screened", "backtested", "out_of_sample_verified",
    "paper_forward", "eligible_for_sowing", "capitalized", "dormant",
    "closing", "archived",
}
_ACTIONS = {"entry", "exit", "flip", "tp1", "tp2", "sl"}


@dataclass(frozen=True)
class StrategyConfigSnapshot:
    """Versioned proposal suitable for manual strategy-registry review."""

    schema_version: str
    family_id: str
    strategy_id: str
    parent_strategy_id: str | None
    candidate_id: str
    genome_id: str
    genome_version: int
    strategy_code_version: str
    parent_genome_ids: tuple[str, ...]
    benchmark_id: str
    verdict: str
    configuration: Mapping[str, Any]
    execution_status: str = "shadow"
    requires_human_registration: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "family_id": self.family_id,
            "strategy_id": self.strategy_id,
            "parent_strategy_id": self.parent_strategy_id,
            "candidate_id": self.candidate_id,
            "genome_id": self.genome_id,
            "genome_version": self.genome_version,
            "strategy_code_version": self.strategy_code_version,
            "parent_genome_ids": list(self.parent_genome_ids),
            "benchmark_id": self.benchmark_id,
            "verdict": self.verdict,
            "configuration": _thaw(self.configuration),
            "execution_status": self.execution_status,
            "requires_human_registration": self.requires_human_registration,
        }


@dataclass(frozen=True)
class ManualAlertDraft:
    """Secret-free fields a human may copy into TradingView."""

    strategy_id: str
    webhook_path: str
    message: Mapping[str, str]
    manual_only: bool = True
    contains_secret: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "webhook_path": self.webhook_path,
            "message": dict(self.message),
            "manual_only": self.manual_only,
            "contains_secret": self.contains_secret,
        }


@dataclass(frozen=True)
class ShadowAlert:
    """Validated shadow signal fixture; signal action never means an order."""

    strategy_id: str
    event_id: str
    signal_action: str
    side: str
    symbol: str
    status: str = "recorded_no_execution"
    execution_action: str = "none"
    order_sent: bool = False


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _required_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _strategy_id(candidate_id: str) -> str:
    # Stable across replays and disjoint from manually managed pilot IDs.
    digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:16]
    return f"eco-{digest}"


def _validate_genome(genome: Mapping[str, Any]):
    genome_id = _required_text(genome.get("genome_id"), "genome.genome_id")
    version = genome.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("genome.version must be a positive integer")
    code_version = _required_text(genome.get("strategy_code_version"), "genome.strategy_code_version")
    parent_ids_raw = genome.get("parent_genome_ids", ())
    if not isinstance(parent_ids_raw, (list, tuple)) or any(not str(x).strip() for x in parent_ids_raw):
        raise ValueError("genome.parent_genome_ids must be a sequence of non-empty IDs")
    parent_ids = tuple(str(x) for x in parent_ids_raw)
    genes, policies = genome.get("genes"), genome.get("policies")
    if not isinstance(genes, Mapping) or not isinstance(policies, (list, tuple)):
        raise ValueError("genome genes and policies are required")
    required = {"protective_stop", "max_leverage", "max_margin_fraction", "reproduction_budget"}
    if not required.issubset(genes) or genes.get("protective_stop") is not True:
        raise ValueError("mandatory safety and financial genes are missing or disabled")
    policy_by_gene = {}
    for policy in policies:
        if not isinstance(policy, Mapping):
            raise ValueError("each gene policy must be an object")
        gene_id = _required_text(policy.get("gene_id"), "policy.gene_id")
        if gene_id in policy_by_gene:
            raise ValueError("duplicate gene policy")
        policy_by_gene[gene_id] = policy
    if set(policy_by_gene) != set(genes):
        raise ValueError("every genome gene must have exactly one policy")
    if policy_by_gene["protective_stop"].get("kind") != "immutable_safety_policy":
        raise ValueError("protective_stop must use an immutable safety policy")
    for name in required - {"protective_stop"}:
        if policy_by_gene[name].get("kind") != "required_bounded_financial":
            raise ValueError(f"{name} must use a required bounded financial policy")
    for name, value in genes.items():
        if isinstance(value, bool):
            numeric_value = None
        elif isinstance(value, (int, float)):
            numeric_value = float(value)
            if numeric_value != numeric_value or numeric_value in (float("inf"), float("-inf")):
                raise ValueError(f"{name} must be finite")
        else:
            raise ValueError(f"{name} must be a boolean or finite number")
        policy = policy_by_gene[name]
        if numeric_value is not None:
            minimum, maximum = policy.get("minimum"), policy.get("maximum")
            if minimum is not None and numeric_value < minimum:
                raise ValueError(f"{name} is below its authorized minimum")
            if maximum is not None and numeric_value > maximum:
                raise ValueError(f"{name} exceeds its authorized maximum")
    if genes["max_leverage"] < 1:
        raise ValueError("max_leverage must be at least one")
    if not 0 < genes["max_margin_fraction"] <= 1:
        raise ValueError("max_margin_fraction must be within (0, 1]")
    if not 0 <= genes["reproduction_budget"] <= 1:
        raise ValueError("reproduction_budget must be within [0, 1]")
    try:
        normalized_genes = json.loads(json.dumps(dict(genes), allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError("genome genes must be finite JSON data") from exc
    return genome_id, version, code_version, parent_ids, normalized_genes


def _candidate_mapping(candidate: Any) -> Mapping[str, Any]:
    if isinstance(candidate, Mapping):
        return candidate
    dump = getattr(candidate, "model_dump", None)
    if callable(dump):
        result = dump(mode="json")
        if isinstance(result, Mapping):
            return result
    raise ValueError("candidate must be a mapping or a versioned contract model")


def prepare_strategy_snapshot(
    candidate: Any,
    *,
    family_id: str = "blaupunkt",
    parent_strategy_id: str | None = None,
) -> StrategyConfigSnapshot:
    """Validate an ECO-01 ResearchCandidate and return a non-persisted snapshot."""
    candidate = _candidate_mapping(candidate)
    candidate_id = _required_text(candidate.get("candidate_id"), "candidate_id")
    if not _ID_RE.fullmatch(family_id):
        raise ValueError("invalid family_id")
    if parent_strategy_id is not None and not _ID_RE.fullmatch(parent_strategy_id):
        raise ValueError("invalid parent_strategy_id")
    stage = candidate.get("stage")
    if stage not in _LIFECYCLE:
        raise ValueError("unknown candidate lifecycle stage")
    if stage != "eligible_for_sowing":
        raise ValueError("candidate must be eligible_for_sowing")
    if candidate.get("verdict") != "beats":
        raise ValueError("candidate must beat its named benchmark")
    benchmark_id = _required_text(candidate.get("benchmark_id"), "benchmark_id")
    evidence = candidate.get("evidence")
    if not isinstance(evidence, Mapping) or evidence.get("out_of_sample") is not True:
        raise ValueError("candidate requires out-of-sample evidence")
    genome = candidate.get("genome")
    if not isinstance(genome, Mapping):
        raise ValueError("candidate genome is required")
    genome_id, version, code_version, parent_genome_ids, genes = _validate_genome(genome)
    try:
        evidence_snapshot = json.loads(json.dumps(dict(evidence), allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate evidence must be finite JSON data") from exc
    config = {
        "genes": genes,
        "evidence": evidence_snapshot,
        "lifecycle_at_snapshot": stage,
        "source": "research_candidate",
        "no_arbitrary_code_execution": True,
        "manual_cutover_required": True,
    }
    return StrategyConfigSnapshot(
        schema_version=SCHEMA_VERSION,
        family_id=family_id,
        strategy_id=_strategy_id(candidate_id),
        parent_strategy_id=parent_strategy_id,
        candidate_id=candidate_id,
        genome_id=genome_id,
        genome_version=version,
        strategy_code_version=code_version,
        parent_genome_ids=parent_genome_ids,
        benchmark_id=benchmark_id,
        verdict="beats",
        configuration=_freeze(config),
    )


def prepare_manual_alert(snapshot: StrategyConfigSnapshot) -> ManualAlertDraft:
    """Prepare a human-operated alert draft without creating a TV alert."""
    return ManualAlertDraft(
        strategy_id=snapshot.strategy_id,
        webhook_path=FAMILY_WEBHOOK_PATH,
        message=_freeze({
            "strategy_id": snapshot.strategy_id,
            "event_id": "{{UNIQUE_SIGNAL_EVENT_ID}}",
            "token": "{{OPERATOR_CONFIGURES_SHARED_FAMILY_WEBHOOK_TOKEN}}",
            "action": "{{entry|exit|flip|tp1|tp2|sl}}",
            "side": "{{buy|sell}}",
            "symbol": "SOL-USDT",
        }),
    )


def validate_shadow_alert(payload: Mapping[str, Any], *, expected_strategy_id: str) -> ShadowAlert:
    """Validate one alert fixture; do not register, route, or execute it."""
    if not _ID_RE.fullmatch(expected_strategy_id):
        raise ValueError("invalid expected_strategy_id")
    strategy_id = _required_text(payload.get("strategy_id"), "strategy_id").lower()
    if strategy_id != expected_strategy_id:
        raise ValueError("alert strategy_id does not match the snapshot")
    event_id = _required_text(payload.get("event_id"), "event_id")
    if len(event_id) > 128:
        raise ValueError("event_id exceeds 128 characters")
    action = _required_text(payload.get("action"), "action").lower()
    if action not in _ACTIONS:
        raise ValueError("unsupported shadow signal action")
    side = str(payload.get("side") or "").strip().lower()
    if action in {"entry", "flip"} and side not in {"buy", "sell"}:
        raise ValueError("buy/sell side required for entry and flip signals")
    if action not in {"entry", "flip"} and side not in {"", "buy", "sell"}:
        raise ValueError("invalid signal side")
    symbol = str(payload.get("symbol") or "SOL-USDT").upper().replace("-", "").replace("/", "").replace("_", "")
    if symbol not in {"SOLUSDT", "SOLUSDTM"}:
        raise ValueError("only the existing SOL family shadow symbol is supported")
    return ShadowAlert(strategy_id, event_id, action, side, "SOLUSDTM")


class InMemoryShadowInbox:
    """Replay fixture with the ingress' strategy_id/event_id idempotency key."""

    def __init__(self) -> None:
        self._events: dict[tuple[str, str], tuple[str, str, str]] = {}

    def record(self, alert: ShadowAlert) -> tuple[ShadowAlert, bool]:
        key = (alert.strategy_id, alert.event_id)
        fingerprint = (alert.signal_action, alert.side, alert.symbol)
        previous = self._events.get(key)
        if previous is not None:
            if previous != fingerprint:
                raise ValueError("event_id reused for a different signal")
            return alert, True
        self._events[key] = fingerprint
        return alert, False
