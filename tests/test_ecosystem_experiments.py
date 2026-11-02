import unittest
import json
from pathlib import Path

from quant.ecosystem.experiments import (
    ExperimentConfig,
    MarketData,
    build_split_plan,
    make_synthetic_market,
    run_experiment,
)


def quick_config():
    return ExperimentConfig(
        seeds=(4, 5),
        walk_forward_folds=2,
        purge_bars=2,
        population_size=4,
        generations=2,
        evaluations_per_arm_fold=8,
        minimum_seeds_for_verdict=2,
    )


class EcosystemExperimentTests(unittest.TestCase):
    def test_split_plan_purges_between_train_and_validation_and_reserves_tail(self):
        config = quick_config()
        plan = build_split_plan(160, config)
        for fold in plan.folds:
            self.assertEqual(fold.validation_start - fold.train_end, config.purge_bars)
            self.assertLessEqual(fold.validation_end, plan.development_end)
        self.assertEqual(plan.final_start, plan.development_end)
        self.assertEqual(plan.final_end, 160)
        self.assertGreater(plan.final_start, plan.folds[-1].validation_end)

    def test_synthetic_run_is_replayable_and_never_claims_trading_evidence(self):
        market = make_synthetic_market(160, seed=19)
        first = run_experiment(quick_config(), market)
        second = run_experiment(quick_config(), market)
        self.assertEqual(first.arms, second.arms)
        self.assertEqual(first.verdicts, {"B_local": "not_comparable", "C_hive": "not_comparable", "D_hybrid": "not_comparable"})
        for row in first.arms:
            self.assertEqual(row["evaluations"], 2 * 8)
            self.assertEqual(len(row["walk_forward_metrics"]), 2)
        self.assertIn("synthetic", first.dataset["source_kind"])
        self.assertIn("not a profitability claim", first.to_markdown())

    def test_self_labelled_empirical_source_cannot_mint_significant_verdict(self):
        # Same historical holdout reused across seeds is pseudoreplication.
        # Even an external evaluator ID and user-declared "empirical"/
        # "verified" flags are insufficient to claim independent OOS.
        from quant.ecosystem.experiments.core import SyntheticEvaluator
        class PretendLiveEvaluator(SyntheticEvaluator):
            evaluator_id = "pretend_live_v1"
        synthetic = make_synthetic_market(160, seed=19)
        purported_real = MarketData(
            synthetic.returns, "user-supplied-real", "empirical",
            "self-described verified trading feed", signal_parity="verified",
        )
        report = run_experiment(quick_config(), purported_real, PretendLiveEvaluator())
        self.assertTrue(report.arms)
        self.assertEqual(set(report.verdicts.values()), {"not_comparable"})
        self.assertIn("Evidence firewall", report.to_markdown())
        for comp in report.comparisons.values():
            self.assertEqual(comp["independent_market_holdouts"], 1)
            self.assertEqual(comp["statistical_unit"], "one_shared_market_holdout")
            self.assertIsNone(comp["p_value_two_sided"])
            self.assertIsNone(comp["p_value_holm"])
            self.assertIsNone(comp["confidence_interval_95"])
            self.assertIsNone(comp["familywise_interval_95"])
            self.assertFalse(comp["inferential_interval_valid"])
            self.assertEqual(comp["seeds"], len(quick_config().seeds))
        self.assertIn("ONE final market holdout", " ".join(report.limitations))

    def test_descriptive_spread_not_confused_with_market_confidence(self):
        market = make_synthetic_market(160, seed=25)
        first = run_experiment(quick_config(), market)
        second = run_experiment(quick_config(), market)
        self.assertEqual(first.comparisons, second.comparisons)
        self.assertEqual(first.verdicts, second.verdicts)
        for comp in first.comparisons.values():
            self.assertIn("seed_spread_95_descriptive", comp)
            self.assertNotIn("beats", comp["verdict"])
            self.assertIsNone(comp["p_value_holm"])

    def test_unavailable_data_abstains_without_fabricating_results(self):
        report = run_experiment(quick_config(), MarketData((), "none", "unavailable", "No data configured"))
        self.assertFalse(report.arms)
        self.assertIsNone(report.split_plan)
        self.assertEqual(set(report.verdicts.values()), {"not_comparable"})
        self.assertIn("no market observations", " ".join(report.limitations).lower())

    def test_cost_budget_and_seed_contracts_are_validated(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(seeds=(1, 1))
        with self.assertRaises(ValueError):
            ExperimentConfig(population_size=10, generations=2, evaluations_per_arm_fold=15)

    def test_example_config_loads_as_a_versioned_protocol(self):
        path = Path(__file__).parents[1] / "src/quant/ecosystem/experiments/experiment.example.json"
        config = ExperimentConfig.from_mapping(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(config.protocol_version, "eco12.v1.1")
        self.assertEqual(len(config.seeds), config.minimum_seeds_for_verdict)
        self.assertEqual(config.ablations.gene_cost_weight, 0.0)


if __name__ == "__main__":
    unittest.main()
