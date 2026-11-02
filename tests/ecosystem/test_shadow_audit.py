"""ECO-13 gate claims must never become live authorization or orders."""
from __future__ import annotations

import ast
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
import unittest

from quant.ecosystem.adapters.shadow_audit import (
    AuditError, GateAttestation, GATES, review_shadow_gates, snapshot_digest,
)
from quant.ecosystem.adapters.shadow_bridge import (
    prepare_strategy_snapshot,
)

UTC=timezone.utc
NOW=datetime(2026,11,2,12,tzinfo=UTC)
ARTEFACT=sha256(b"external evidence test fixture, not independently verified").hexdigest()


def snapshot(*, rsi=14, genome_version=7):
    genes={
        "protective_stop":True,
        "max_leverage":1.5,
        "max_margin_fraction":.25,
        "reproduction_budget":.02,
        "rsi_period":rsi,
    }
    policies=[
        {"gene_id":"protective_stop","kind":"immutable_safety_policy"},
        {"gene_id":"max_leverage","kind":"required_bounded_financial","minimum":1,"maximum":3},
        {"gene_id":"max_margin_fraction","kind":"required_bounded_financial","minimum":0,"maximum":1},
        {"gene_id":"reproduction_budget","kind":"required_bounded_financial","minimum":0,"maximum":.1},
        {"gene_id":"rsi_period","kind":"optional_strategy_gene","minimum":2,"maximum":100},
    ]
    return prepare_strategy_snapshot({
        "candidate_id":"research-handoff-2026",
        "stage":"eligible_for_sowing","benchmark_id":"research-benchmark-v1",
        "verdict":"beats","evidence":{"out_of_sample":True,"confidence":.9},
        "genome":{
            "genome_id":"research-genome-a","version":genome_version,
            "strategy_code_version":"research-module-v3",
            "parent_genome_ids":["genome-parent-1"],
            "genes":genes,"policies":policies,
        },
    })


def claimed(gate_id, s, **patch):
    data={
        "gate_id":gate_id,
        "status":"pass_claimed",
        "snapshot_sha256":snapshot_digest(s),
        "checked_at":NOW,"reviewer":"manual-review-fixture",
        "artifact_sha256":ARTEFACT,
        "artifact_ref":"research-test-artifact://evidence",
    }
    data.update(patch)
    return GateAttestation(**data)


class ShadowGateTests(unittest.TestCase):
    def setUp(self):
        self.snap=snapshot()

    def test_no_attestations_means_no_go_and_nothing_executed(self):
        report=review_shadow_gates(self.snap,[],as_of=NOW)
        self.assertEqual(report.cutover,"NO_GO")
        self.assertEqual(report.status,"NO_GO_missing_or_failed_checks")
        self.assertEqual(set(report.absent),set(GATES))
        self.assertEqual(report.attestation_count,0)
        self.assertTrue(report.artifact_hashes_unverified)
        self.assertFalse(report.independent_market_oos_verified_by_this_tool)
        self.assertFalse(report.live_routing_authorized)
        self.assertFalse(report.order_sent)
        self.assertIn('"cutover": "NO_GO"',report.to_json())

    def test_all_sixteen_claims_still_not_real_authorization(self):
        items=[claimed(g,self.snap) for g in GATES]
        report=review_shadow_gates(self.snap,items,as_of=NOW)
        self.assertEqual(len(GATES),16)
        self.assertEqual(report.status,"claims_complete_needs_independent_review")
        self.assertEqual(report.absent,())
        self.assertEqual(report.failed,())
        self.assertEqual(report.unknown,())
        self.assertEqual(set(report.claimed_pass),set(GATES))
        self.assertFalse(report.live_routing_authorized)
        self.assertFalse(report.order_sent)
        self.assertEqual(report.cutover,"NO_GO")
        self.assertTrue(report.artifact_hashes_unverified)

    def test_gene_mutation_invalidates_every_prior_attestation(self):
        original=claimed(GATES[0],self.snap)
        changed=snapshot(rsi=20)
        self.assertNotEqual(snapshot_digest(self.snap),snapshot_digest(changed))
        with self.assertRaisesRegex(AuditError,"different genome/snapshot"):
            review_shadow_gates(changed,[original],as_of=NOW)
        version_changed=snapshot(genome_version=8)
        with self.assertRaisesRegex(AuditError,"different genome/snapshot"):
            review_shadow_gates(version_changed,[original],as_of=NOW)

    def test_future_dated_and_stale_claims_do_not_pass(self):
        late=claimed(GATES[0],self.snap,checked_at=NOW+timedelta(minutes=1))
        with self.assertRaisesRegex(AuditError,"future"):
            review_shadow_gates(self.snap,[late],as_of=NOW)
        stale=claimed(GATES[0],self.snap,checked_at=NOW-timedelta(days=100))
        report=review_shadow_gates(self.snap,[stale],as_of=NOW)
        self.assertIn(GATES[0],report.unknown)
        self.assertNotIn(GATES[0],report.claimed_pass)
        self.assertEqual(report.cutover,"NO_GO")

    def test_missing_artifact_or_no_reviewer_never_qualifies(self):
        with self.assertRaisesRegex(AuditError,"SHA256"):
            claimed(GATES[0],self.snap,artifact_sha256=None)
        with self.assertRaisesRegex(AuditError,"artifact reference"):
            claimed(GATES[0],self.snap,artifact_ref="")
        with self.assertRaisesRegex(AuditError,"reviewer"):
            claimed(GATES[0],self.snap,reviewer="  ")

    def test_unknown_status_and_unbound_attestations_rejected(self):
        with self.assertRaisesRegex(AuditError,"unknown required gate"):
            claimed("cancel_everyones_stop_orders",self.snap)
        with self.assertRaisesRegex(AuditError,"invalid gate status"):
            claimed(GATES[0],self.snap,status="independently_verified")
        with self.assertRaisesRegex(AuditError,"full candidate snapshot"):
            claimed(GATES[0],self.snap,snapshot_sha256="xyz")
        with self.assertRaisesRegex(AuditError,"duplicate"):
            review_shadow_gates(self.snap,[claimed(GATES[0],self.snap),
                                           claimed(GATES[0],self.snap)],as_of=NOW)

    def test_failed_checks_refuse_to_be_overridden_by_other_green_flags(self):
        evidence=[claimed(g,self.snap) for g in GATES]
        failed=replace(evidence[6],status="fail",artifact_sha256=None,artifact_ref=None)
        evidence[6]=failed
        result=review_shadow_gates(self.snap,evidence,as_of=NOW)
        self.assertIn(GATES[6],result.failed)
        self.assertEqual(result.status,"NO_GO_missing_or_failed_checks")
        self.assertEqual(result.cutover,"NO_GO")

    def test_aware_dates_and_no_order_imports(self):
        with self.assertRaisesRegex(AuditError,"aware UTC timestamp"):
            review_shadow_gates(self.snap,[],as_of=datetime(2026,11,2,12))
        with self.assertRaisesRegex(AuditError,"aware UTC timestamp"):
            claimed(GATES[0],self.snap,checked_at=datetime(2026,11,2,12))
        source=Path(__file__).parents[2]/"src/quant/ecosystem/adapters/shadow_audit.py"
        tree=ast.parse(source.read_text())
        forbidden=("requests","httpx","socket","urllib","kucoin",
                   "quant.execution","quant.brokers","quant.oms")
        imported=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                imported.extend(x.name for x in node.names)
            if isinstance(node,ast.ImportFrom) and node.module:
                imported.append(node.module)
        self.assertFalse([x for x in imported
                          if any(x==a or x.startswith(a+".") for a in forbidden)])


if __name__=="__main__":
    unittest.main()
