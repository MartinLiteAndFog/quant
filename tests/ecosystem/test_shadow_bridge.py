from __future__ import annotations

import ast
from pathlib import Path
import unittest
from unittest.mock import Mock

from quant.ecosystem.adapters.shadow_bridge import (
    InMemoryShadowInbox,
    prepare_manual_alert,
    prepare_strategy_snapshot,
    validate_shadow_alert,
)


def candidate():
    genes = {
        "protective_stop": True,
        "max_leverage": 1.5,
        "max_margin_fraction": 0.25,
        "reproduction_budget": 0.02,
        "rsi_period": 14,
    }
    policies = [
        {"gene_id": "protective_stop", "kind": "immutable_safety_policy"},
        {"gene_id": "max_leverage", "kind": "required_bounded_financial", "minimum": 1, "maximum": 3},
        {"gene_id": "max_margin_fraction", "kind": "required_bounded_financial", "minimum": 0, "maximum": 1},
        {"gene_id": "reproduction_budget", "kind": "required_bounded_financial", "minimum": 0, "maximum": 0.1},
        {"gene_id": "rsi_period", "kind": "optional_strategy_gene", "minimum": 2, "maximum": 100},
    ]
    return {
        "candidate_id": "research-abc-123",
        "stage": "eligible_for_sowing",
        "benchmark_id": "frozen-sol-2021-2025-v1",
        "verdict": "beats",
        "evidence": {
            "out_of_sample": True,
            "effective_trades": 120,
            "observed_days": 400,
            "confidence": 0.91,
            "data_quality": 0.99,
        },
        "genome": {
            "genome_id": "genome-child-v7",
            "version": 7,
            "strategy_code_version": "strategy-2.3.1",
            "parent_genome_ids": ["genome-parent-a", "genome-parent-b"],
            "genes": genes,
            "policies": policies,
        },
    }


class ShadowBridgeTests(unittest.TestCase):
    def test_snapshot_preserves_identity_lineage_and_versions(self):
        snapshot = prepare_strategy_snapshot(candidate(), parent_strategy_id="blaupunkt-p02")
        result = snapshot.as_dict()
        self.assertTrue(result["strategy_id"].startswith("eco-"))
        self.assertEqual(result["candidate_id"], "research-abc-123")
        self.assertEqual(result["genome_id"], "genome-child-v7")
        self.assertEqual(result["genome_version"], 7)
        self.assertEqual(result["strategy_code_version"], "strategy-2.3.1")
        self.assertEqual(result["parent_genome_ids"], ["genome-parent-a", "genome-parent-b"])
        self.assertEqual(result["parent_strategy_id"], "blaupunkt-p02")
        self.assertEqual(result["benchmark_id"], "frozen-sol-2021-2025-v1")
        self.assertEqual(result["execution_status"], "shadow")
        self.assertTrue(result["requires_human_registration"])
        self.assertEqual(prepare_strategy_snapshot(candidate()).strategy_id, snapshot.strategy_id)

    def test_accepts_versioned_contract_model_without_importing_eco01(self):
        class ResearchCandidateModel:
            def model_dump(self, *, mode):
                self.asserted_mode = mode
                return candidate()

        model = ResearchCandidateModel()
        snapshot = prepare_strategy_snapshot(model)
        self.assertEqual(snapshot.candidate_id, "research-abc-123")

    def test_snapshot_is_deeply_immutable_and_gene_bounds_are_enforced(self):
        snapshot = prepare_strategy_snapshot(candidate())
        with self.assertRaises(TypeError):
            snapshot.configuration["genes"]["rsi_period"] = 20

        item = candidate()
        item["genome"]["genes"]["max_leverage"] = 4
        with self.assertRaisesRegex(ValueError, "authorized maximum"):
            prepare_strategy_snapshot(item)

    def test_candidate_must_pass_lifecycle_and_benchmark_gates(self):
        for field, value in (
            ("stage", "paper_forward"),
            ("stage", "capitalized"),
            ("verdict", "does_not_beat"),
            ("verdict", "not_comparable"),
        ):
            item = candidate()
            item[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                prepare_strategy_snapshot(item)

    def test_candidate_requires_oos_and_protected_financial_policies(self):
        item = candidate()
        item["evidence"]["out_of_sample"] = False
        with self.assertRaisesRegex(ValueError, "out-of-sample"):
            prepare_strategy_snapshot(item)

        item = candidate()
        item["genome"]["genes"]["protective_stop"] = False
        with self.assertRaisesRegex(ValueError, "missing or disabled"):
            prepare_strategy_snapshot(item)

        item = candidate()
        item["genome"]["policies"][0]["kind"] = "optional_strategy_gene"
        with self.assertRaisesRegex(ValueError, "immutable"):
            prepare_strategy_snapshot(item)

    def test_manual_alert_is_secret_free_and_requires_human_action(self):
        draft = prepare_manual_alert(prepare_strategy_snapshot(candidate())).as_dict()
        self.assertTrue(draft["manual_only"])
        self.assertFalse(draft["contains_secret"])
        self.assertEqual(draft["webhook_path"], "/webhook/family")
        self.assertIn("OPERATOR_CONFIGURES_SHARED_FAMILY_WEBHOOK_TOKEN", draft["message"]["token"])
        self.assertEqual(draft["message"]["event_id"], "{{UNIQUE_SIGNAL_EVENT_ID}}")

    def test_shadow_fixture_never_expresses_an_execution_action(self):
        snapshot = prepare_strategy_snapshot(candidate())
        alert = validate_shadow_alert(
            {
                "strategy_id": snapshot.strategy_id,
                "event_id": "tv:sol:1m:2026-11-01T10:00:00Z",
                "action": "entry",
                "side": "buy",
                "symbol": "SOL-USDT",
            },
            expected_strategy_id=snapshot.strategy_id,
        )
        self.assertEqual(alert.signal_action, "entry")
        self.assertEqual(alert.execution_action, "none")
        self.assertFalse(alert.order_sent)
        self.assertEqual(alert.status, "recorded_no_execution")

    def test_shadow_event_ids_are_idempotent_and_conflicts_are_rejected(self):
        snapshot = prepare_strategy_snapshot(candidate())
        inbox = InMemoryShadowInbox()
        payload = {
            "strategy_id": snapshot.strategy_id,
            "event_id": "stable-event-001",
            "action": "entry",
            "side": "buy",
            "symbol": "SOL-USDT",
        }
        alert = validate_shadow_alert(payload, expected_strategy_id=snapshot.strategy_id)
        first, duplicate_first = inbox.record(alert)
        second, duplicate_second = inbox.record(alert)
        self.assertEqual(first, second)
        self.assertFalse(duplicate_first)
        self.assertTrue(duplicate_second)

        changed = validate_shadow_alert(
            {**payload, "action": "exit", "side": ""},
            expected_strategy_id=snapshot.strategy_id,
        )
        with self.assertRaisesRegex(ValueError, "reused for a different signal"):
            inbox.record(changed)

    def test_event_id_scope_includes_strategy_id(self):
        inbox = InMemoryShadowInbox()
        one = validate_shadow_alert(
            {"strategy_id": "strategy-one", "event_id": "same", "action": "entry", "side": "buy"},
            expected_strategy_id="strategy-one",
        )
        two = validate_shadow_alert(
            {"strategy_id": "strategy-two", "event_id": "same", "action": "entry", "side": "buy"},
            expected_strategy_id="strategy-two",
        )
        self.assertFalse(inbox.record(one)[1])
        self.assertFalse(inbox.record(two)[1])

    def test_adapter_ignores_venue_objects_and_makes_no_live_calls(self):
        venue = Mock()
        item = candidate()
        item["venue"] = venue
        snapshot = prepare_strategy_snapshot(item)
        prepare_manual_alert(snapshot)
        venue.create_order.assert_not_called()
        venue.cancel_all.assert_not_called()
        venue.transfer.assert_not_called()
        venue.set_position_mode.assert_not_called()
        self.assertNotIn("venue", snapshot.as_dict())

    def test_adapter_has_no_network_or_execution_imports(self):
        source = Path(__file__).parents[2] / "src/quant/ecosystem/adapters/shadow_bridge.py"
        tree = ast.parse(source.read_text())
        forbidden = ("requests", "httpx", "urllib", "socket", "quant.execution", "kucoin")
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        blocked = [name for name in imports if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)]
        self.assertFalse(blocked, blocked)


if __name__ == "__main__":
    unittest.main()
