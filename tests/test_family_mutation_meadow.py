"""Evolution meadow candidate sampling; never creates TradingView alerts or orders."""
import math
import unittest

from quant.execution.family_monthly_evaluation import MonthlyMetrics
from quant.execution.family_mutation_meadow import (
    MutationBounds, MutationError, propose_sowing,
)


class EvolutionMeadowTests(unittest.TestCase):
    def metrics(self, bot="blaupunkt", net="0.21", dd="0.04", trades=45, days=31):
        return MonthlyMetrics(
            strategy_id=bot, monthly_net_twr=net, max_drawdown=dd,
            closed_trades=trades, observed_days=days,
        )

    def test_losing_bot_produces_no_seeds(self):
        result = propose_sowing("blaupunkt", sensitivity=17.2, ttp_pct=0.65,
                                month="2026-10", metrics=self.metrics(net="-0.10"))
        self.assertEqual(result.seed_count_requested, 0)
        self.assertEqual(result.proposals, ())

    def test_trading_sample_gates_reproduction(self):
        result = propose_sowing("blaupunkt", sensitivity=17.2, ttp_pct=0.65,
                                month="2026-10", metrics=self.metrics(trades=2, days=4))
        self.assertEqual(result.proposals, ())

    def test_higher_profit_means_more_and_wider_seed_cloud(self):
        low = propose_sowing("blaupunkt", sensitivity=17.2, ttp_pct=0.65,
                             month="2026-10", metrics=self.metrics(net="0.07", dd="0.02"))
        high = propose_sowing("blaupunkt", sensitivity=17.2, ttp_pct=0.65,
                              month="2026-10", metrics=self.metrics(net="0.45", dd="0.02"))
        self.assertGreater(high.seed_count_requested, low.seed_count_requested)
        self.assertGreater(high.radius_normalized, low.radius_normalized)
        self.assertGreaterEqual(low.seed_count_requested, 5)
        self.assertLessEqual(high.seed_count_requested, 15)

    def test_all_seeds_are_independent_inbounds_and_deterministic(self):
        args = {
            "sensitivity": 17.2, "ttp_pct": 0.65,
            "month": "2026-10", "metrics": self.metrics(),
        }
        a = propose_sowing("blaupunkt", **args)
        b = propose_sowing("blaupunkt", **args)
        self.assertEqual(a, b)
        self.assertTrue(a.proposals)
        self.assertEqual(len(a.proposals), len({(p.sensitivity, p.ttp_pct) for p in a.proposals}))
        for p in a.proposals:
            self.assertEqual(p.parent_strategy_id, "blaupunkt")
            self.assertTrue(12 <= p.sensitivity <= 25)
            self.assertTrue(0.25 <= p.ttp_pct <= 1.65)
            self.assertFalse(p.execution_enabled)
            self.assertEqual(p.signal_source, "TRADINGVIEW_MANUAL_ALERT_REQUIRED")

    def test_existing_configs_are_not_recreated(self):
        known = {(17.2, 0.65), (17.3, 0.65), (17.2, 0.7)}
        s = propose_sowing("blaupunkt", sensitivity=17.2, ttp_pct=0.65,
                           month="2026-10", metrics=self.metrics(),
                           known_configs=known)
        self.assertFalse(any((p.sensitivity, p.ttp_pct) in known for p in s.proposals))

    def test_incompatible_metrics_identity_is_rejected(self):
        with self.assertRaises(MutationError):
            propose_sowing("blaupunkt-p02", sensitivity=17.2, ttp_pct=0.65,
                           month="2026-10", metrics=self.metrics())

    def test_invalid_bounds_and_parameter_nan_are_rejected(self):
        with self.assertRaises(MutationError):
            propose_sowing("blaupunkt", sensitivity=float("nan"), ttp_pct=0.65,
                           month="2026-10", metrics=self.metrics())
        with self.assertRaises(MutationError):
            propose_sowing("blaupunkt", sensitivity=17.2, ttp_pct=0.65,
                           month="2026-10", metrics=self.metrics(),
                           bounds=MutationBounds(sensitivity_step=0))

    def test_radial_sectors_roughly_cover_all_directions(self):
        # Very profitable parent -> ~15 seeds. Circumference should have
        # at least one candidate in every quadrant, absent boundaries.
        plan = propose_sowing("blaupunkt", sensitivity=18.5, ttp_pct=0.95,
                              month="2026-10",
                              metrics=self.metrics(net="1.0", dd="0.01", trades=100))
        directions = {(math.copysign(1, p.sensitivity - 18.5),
                       math.copysign(1, p.ttp_pct - 0.95))
                      for p in plan.proposals
                      if p.sensitivity != 18.5 and p.ttp_pct != 0.95}
        self.assertGreaterEqual(len(directions), 3)


if __name__ == "__main__":
    unittest.main()
