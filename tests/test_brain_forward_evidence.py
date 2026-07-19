from __future__ import annotations

import copy
import hashlib
import unittest
from datetime import timedelta, timezone
from pathlib import Path

from quant.brain_forward.evidence import ForwardProtocol, build_checkpoint_report


UTC = timezone.utc


def _protocol_dict() -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol_id": "test-epoch",
        "candidate_id": "candidate-a",
        "symbol": "SOL-USDT",
        "source": "binance_spot_klines",
        "artifact_sha256": "a" * 64,
        "candidate_spec_sha256": "b" * 64,
        "observer_code_sha256": {
            "runtime.py": "c" * 64,
            "service.py": "d" * 64,
            "store.py": "e" * 64,
        },
        "warmup_start": "2026-01-01T00:00:00Z",
        "evidence_start": "2026-01-02T00:00:00Z",
        "evidence_end": "2026-01-02T00:10:00Z",
        "checkpoint_at": "2026-01-02T00:15:00Z",
        "outcome_maturity_minutes": 5,
        "base_cost_bps": 14.0,
        "stress_cost_bps": 22.0,
        "minimum_formal_trades": 2,
        "minimum_bar_coverage": 0.9,
        "maximum_drawdown_bps": 1000.0,
        "minimum_mean_net_bps": 0.0,
        "minimum_lcb95_net_bps": 0.0,
        "minimum_stress_mean_net_bps": 0.0,
        "promotion_scope": "shadow_champion_review_only",
        "live_orders_permitted": False,
    }


def _trade(protocol: ForwardProtocol, minute: int, gross_bps: float) -> dict[str, object]:
    event = protocol.evidence_start + timedelta(minutes=minute)
    return {
        "decision_id": f"decision-{minute}",
        "event_ts": event.isoformat(),
        "entry_ts": (event + timedelta(minutes=1)).isoformat(),
        "exit_ts": (event + timedelta(minutes=2)).isoformat(),
        "gross_bps": gross_bps,
        "net_bps": gross_bps - protocol.base_cost_bps,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.protocol_sha256,
        "artifact_sha256": protocol.artifact_sha256,
        "symbol": protocol.symbol,
        "source": protocol.source,
        "evidence_phase": "formal",
    }


def _decision(protocol: ForwardProtocol, minute: int) -> dict[str, object]:
    event = protocol.evidence_start + timedelta(minutes=minute)
    return {
        "decision_id": f"decision-{minute}",
        "event_ts": event.isoformat(),
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.protocol_sha256,
        "artifact_sha256": protocol.artifact_sha256,
        "symbol": protocol.symbol,
        "source": protocol.source,
        "evidence_phase": "formal",
    }


class BrainForwardEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = ForwardProtocol.from_dict(_protocol_dict())
        self.bars = [
            self.protocol.evidence_start + timedelta(minutes=minute)
            for minute in range(15)
        ]
        self.decisions = [_decision(self.protocol, 0), _decision(self.protocol, 4)]
        self.trades = [_trade(self.protocol, 0, 60.0), _trade(self.protocol, 4, 60.0)]

    def test_protocol_hash_is_canonical_and_phase_boundaries_are_fixed(self) -> None:
        reordered = dict(reversed(list(_protocol_dict().items())))
        self.assertEqual(self.protocol.protocol_sha256, ForwardProtocol.from_dict(reordered).protocol_sha256)
        self.assertEqual(self.protocol.phase_at("2026-01-01T12:00:00Z"), "warmup")
        self.assertEqual(self.protocol.phase_at("2026-01-02T00:00:00Z"), "formal")
        self.assertEqual(self.protocol.phase_at("2026-01-02T00:10:00Z"), "post_epoch")

    def test_runtime_identity_mismatch_fails_before_observation(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "artifact_sha256"):
            self.protocol.assert_runtime(
                symbol=self.protocol.symbol,
                source=self.protocol.source,
                artifact_sha256="0" * 64,
            )

    def test_repository_protocol_matches_frozen_artifact_and_candidate_spec(self) -> None:
        protocol = ForwardProtocol.load()
        artifact = Path("src/quant/brain_forward/frozen_utility_memory.json")
        candidate = Path("research/1000_brains_multihorizon_barriers_20260717/candidate_spec.json")
        self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), protocol.artifact_sha256)
        self.assertEqual(hashlib.sha256(candidate.read_bytes()).hexdigest(), protocol.candidate_spec_sha256)
        protocol.assert_runtime(
            symbol=protocol.symbol,
            source=protocol.source,
            artifact_sha256=protocol.artifact_sha256,
        )

    def test_observer_code_mutation_fails_runtime_identity(self) -> None:
        mutated = _protocol_dict()
        mutated["observer_code_sha256"] = {
            "runtime.py": "0" * 64,
            "service.py": "0" * 64,
            "store.py": "0" * 64,
        }
        protocol = ForwardProtocol.from_dict(mutated)
        with self.assertRaisesRegex(RuntimeError, "observer_code_sha256:runtime.py"):
            protocol.assert_runtime(
                symbol=protocol.symbol,
                source=protocol.source,
                artifact_sha256=protocol.artifact_sha256,
            )

    def test_checkpoint_is_locked_before_preregistered_time(self) -> None:
        report = build_checkpoint_report(
            self.protocol, self.decisions, self.trades, self.bars, as_of="2026-01-02T00:14:59Z"
        )
        self.assertEqual(report["verdict"], "locked_before_checkpoint")
        self.assertFalse(report["gates"]["checkpoint_open"])
        self.assertFalse(report["live_orders_permitted"])

    def test_complete_positive_epoch_is_only_shadow_review_eligible(self) -> None:
        report = build_checkpoint_report(
            self.protocol, self.decisions, self.trades, self.bars, as_of=self.protocol.checkpoint_at
        )
        self.assertEqual(report["verdict"], "eligible_for_shadow_champion_review")
        self.assertTrue(all(report["gates"].values()))
        self.assertEqual(len(report["report_sha256"]), 64)
        repeated = build_checkpoint_report(
            self.protocol, self.decisions, self.trades, self.bars, as_of=self.protocol.checkpoint_at
        )
        self.assertEqual(report, repeated)

    def test_cost_or_protocol_mutation_fails_ledger_integrity(self) -> None:
        bad = copy.deepcopy(self.trades)
        bad[0]["net_bps"] = float(bad[0]["net_bps"]) + 1.0
        bad[1]["protocol_sha256"] = "0" * 64
        report = build_checkpoint_report(
            self.protocol, self.decisions, bad, self.bars, as_of=self.protocol.checkpoint_at
        )
        self.assertEqual(report["verdict"], "retain_research_candidate")
        self.assertFalse(report["gates"]["ledger_integrity"])
        self.assertTrue(any("base cost mismatch" in item for item in report["integrity_failures"]))
        self.assertTrue(any("protocol identity mismatch" in item for item in report["integrity_failures"]))

    def test_missing_bar_coverage_fails_closed(self) -> None:
        report = build_checkpoint_report(
            self.protocol, self.decisions, self.trades, self.bars[:8], as_of=self.protocol.checkpoint_at
        )
        self.assertEqual(report["verdict"], "retain_research_candidate")
        self.assertFalse(report["gates"]["bar_coverage"])

    def test_overlapping_trades_fail_ledger_integrity(self) -> None:
        overlapping = [_trade(self.protocol, 0, 60.0), _trade(self.protocol, 1, 60.0)]
        report = build_checkpoint_report(
            self.protocol, [_decision(self.protocol, 0), _decision(self.protocol, 1)], overlapping, self.bars, as_of=self.protocol.checkpoint_at
        )
        self.assertFalse(report["gates"]["ledger_integrity"])
        self.assertIn("overlapping formal trades", report["integrity_failures"])

    def test_trade_without_recorded_decision_fails_traceability(self) -> None:
        report = build_checkpoint_report(
            self.protocol, self.decisions[:1], self.trades, self.bars,
            as_of=self.protocol.checkpoint_at,
        )
        self.assertFalse(report["gates"]["ledger_integrity"])
        self.assertTrue(any("trade without formal decision" in item for item in report["integrity_failures"]))


if __name__ == "__main__":
    unittest.main()
