"""Monthly ratings and family budget proposals: absolutely no exchange calls."""
import unittest
from decimal import Decimal

from quant.execution.family_monthly_evaluation import (
    EvaluationError,
    MonthlyMetrics,
    propose_family_rebalance,
    rate_month,
)


class FamilyMonthlyEvaluationTests(unittest.TestCase):
    def metrics(self, id, return_month="0.05", dd="0.10", trades=40, days=30, **kwargs):
        return MonthlyMetrics(
            strategy_id=id, monthly_net_twr=return_month,
            max_drawdown=dd, closed_trades=trades,
            observed_days=days, **kwargs,
        )

    def test_net_profit_is_primary_but_drawdown_penalizes(self):
        high_return = rate_month(self.metrics("a", return_month="0.20", dd="0.08"))
        low_return = rate_month(self.metrics("b", return_month="0.05", dd="0.08"))
        self.assertGreater(high_return.score, low_return.score)
        riskier = rate_month(self.metrics("c", return_month="0.20", dd="0.30"))
        self.assertLess(riskier.score, high_return.score)

    def test_low_sample_is_shrunk_and_does_not_qualify_for_reallocation(self):
        new = rate_month(self.metrics("new", trades=1, days=3))
        established = rate_month(self.metrics("old", trades=80, days=100))
        self.assertFalse(new.eligible_for_reallocation)
        self.assertTrue(established.eligible_for_reallocation)
        self.assertLess(new.credibility, established.credibility)

    def test_cashflow_and_mark_to_market_required(self):
        for kwargs in (
            {"cashflows_adjusted": False},
            {"marked_to_market": False},
            {"fees_and_funding_included": False},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(EvaluationError):
                rate_month(self.metrics("bad", **kwargs))

    def test_ineligible_daughters_keep_two_percent_seed(self):
        # 300 USDT root budget -> 282 USDT parent and 3 x 6 daughters.
        current = {
            "parent": Decimal("282"), "d1": Decimal("6"),
            "d2": Decimal("6"), "d3": Decimal("6"),
        }
        ratings = {
            "parent": rate_month(self.metrics("parent", return_month="0.01")),
            "d1": rate_month(self.metrics("d1", return_month="0.50", trades=2, days=10)),
            "d2": rate_month(self.metrics("d2", trades=1, days=4)),
            "d3": rate_month(self.metrics("d3", trades=0, days=1)),
        }
        out = propose_family_rebalance(family_id="blaupunkt", current_budgets=current, ratings=ratings)
        self.assertEqual(out.proposed_budgets, current)
        self.assertEqual(out.execution, "PREVIEW_ONLY_NO_ORDERS_NO_TRANSFERS")
        self.assertTrue(out.conserves_capital)

    def test_verified_high_score_bot_grows_within_family(self):
        current = {"parent": Decimal("282"), "d1": Decimal("6"), "d2": Decimal("6"), "d3": Decimal("6")}
        ratings = {
            "parent": rate_month(self.metrics("parent", return_month="-0.05", dd="0.20")),
            "d1": rate_month(self.metrics("d1", return_month="0.20", dd="0.02")),
            "d2": rate_month(self.metrics("d2", trades=1, days=2)),
            "d3": rate_month(self.metrics("d3", trades=1, days=2)),
        }
        plan = propose_family_rebalance(family_id="blaupunkt", current_budgets=current, ratings=ratings)
        self.assertGreater(plan.proposed_budgets["d1"], Decimal("6"))
        self.assertLess(plan.proposed_budgets["parent"], Decimal("282"))
        self.assertEqual(plan.proposed_budgets["d2"], Decimal("6"))
        self.assertEqual(plan.proposed_budgets["d3"], Decimal("6"))
        self.assertEqual(sum(plan.proposed_budgets.values()), Decimal("300"))
        self.assertTrue(plan.conserves_capital)

    def test_cannot_move_money_across_parent_families(self):
        a = propose_family_rebalance(
            family_id="parent-a",
            current_budgets={"parent-a": "282", "daughter-a": "18"},
            ratings={},
        )
        b = propose_family_rebalance(
            family_id="parent-b",
            current_budgets={"parent-b": "282", "daughter-b": "18"},
            ratings={},
        )
        self.assertEqual(a.family_budget, Decimal("300"))
        self.assertEqual(b.family_budget, Decimal("300"))
        self.assertEqual(a.current_budgets, a.proposed_budgets)
        self.assertEqual(b.current_budgets, b.proposed_budgets)

    def test_invalid_score_input_rejected(self):
        for raw in ("NaN", "Infinity", "-1.5"):
            with self.subTest(raw=raw), self.assertRaises(EvaluationError):
                rate_month(self.metrics("x", return_month=raw))

    def test_no_random_absolute_cap_on_number_of_bots(self):
        current = {f"bot{i}": Decimal("0.01") for i in range(120)}
        plan = propose_family_rebalance(family_id="many", current_budgets=current, ratings={})
        self.assertEqual(len(plan.proposed_budgets), 120)
        self.assertTrue(plan.conserves_capital)


if __name__ == "__main__":
    unittest.main()
