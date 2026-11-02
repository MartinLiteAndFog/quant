"""Tests for parent-family and nested daughter split (no broker or Postgres)."""
import unittest
from decimal import Decimal

from quant.execution.family_tree_allocation import (
    AllocationError,
    StrategyNode,
    allocate_tree,
)


class FamilyTreeAllocationTests(unittest.TestCase):
    def test_one_parent_gets_ninety_percent(self):
        a = allocate_tree(1000, [StrategyNode("blaupunkt")])
        self.assertEqual(a.bot_budgets, {"blaupunkt": Decimal("900")})
        self.assertEqual(a.reserve, Decimal("100"))
        self.assertTrue(a.audit())

    def test_three_roots_share_equally(self):
        a = allocate_tree(1000, [
            StrategyNode("blaupunkt"),
            StrategyNode("parent-2"),
            StrategyNode("parent-3"),
        ])
        self.assertEqual(a.family_budgets, {
            "blaupunkt": Decimal("300"),
            "parent-2": Decimal("300"),
            "parent-3": Decimal("300"),
        })
        self.assertEqual(a.bot_budgets, a.family_budgets)
        self.assertEqual(a.reserve, Decimal("100"))

    def test_six_daughters_each_take_two_percent_of_family(self):
        nodes = [StrategyNode("a"), StrategyNode("b"), StrategyNode("c")]
        nodes.extend(StrategyNode(f"a-child-{i}", "a") for i in range(6))
        a = allocate_tree(1000, nodes)
        self.assertEqual(a.bot_budgets["a"], Decimal("264"))
        for i in range(6):
            self.assertEqual(a.bot_budgets[f"a-child-{i}"], Decimal("6"))
        self.assertEqual(a.bot_budgets["b"], Decimal("300"))
        self.assertEqual(a.bot_budgets["c"], Decimal("300"))
        self.assertEqual(a.reserve, Decimal("100"))

    def test_granddaughter_takes_from_daughters_budget_not_root(self):
        nodes = [StrategyNode("a"), StrategyNode("b", "a"), StrategyNode("c", "b")]
        plan = allocate_tree(1000, nodes)
        self.assertEqual(plan.family_budgets["a"], Decimal("900"))
        self.assertEqual(plan.bot_budgets["a"], Decimal("882"))
        self.assertEqual(plan.bot_budgets["b"], Decimal("17.64"))
        self.assertEqual(plan.bot_budgets["c"], Decimal("0.36"))
        self.assertTrue(plan.audit())

    def test_angel_pool_does_not_receive_evolution_budget(self):
        a = allocate_tree(1000, [StrategyNode("a"), StrategyNode("fishing", pool="angel")])
        self.assertEqual(a.bot_budgets, {"a": Decimal("900")})
        self.assertEqual(a.reserve, Decimal("100"))

    def test_no_active_parent_keeps_all_capital(self):
        a = allocate_tree(500, [StrategyNode("off", status="disabled")])
        self.assertEqual(a.reserve, Decimal("500"))
        self.assertEqual(a.bot_budgets, {})

    def test_inactive_parent_cannot_fund_active_child(self):
        with self.assertRaises(AllocationError):
            allocate_tree(1000, [
                StrategyNode("parent", status="disabled"),
                StrategyNode("child", "parent"),
            ])

    def test_duplicate_strategies_rejected(self):
        with self.assertRaises(AllocationError):
            allocate_tree(1, [StrategyNode("x"), StrategyNode("x")])

    def test_cycle_rejected(self):
        with self.assertRaises(AllocationError):
            allocate_tree(1, [StrategyNode("a", "b"), StrategyNode("b", "a")])

    def test_more_than_50_children_at_2pct_each_rejected(self):
        nodes = [StrategyNode("root")] + [
            StrategyNode(f"kid-{n}", "root") for n in range(51)
        ]
        with self.assertRaises(AllocationError):
            allocate_tree(1000, nodes)

    def test_fractional_cents_are_not_invented(self):
        a = allocate_tree("0.03", [StrategyNode("a"), StrategyNode("b"), StrategyNode("c")])
        self.assertTrue(a.audit())
        self.assertLessEqual(sum(a.bot_budgets.values()), a.equity)

    def test_invalid_equity_rejected(self):
        for value in ("NaN", "-1", "Infinity", "string"):
            with self.subTest(value=value), self.assertRaises(AllocationError):
                allocate_tree(value, [StrategyNode("a")])


if __name__ == "__main__":
    unittest.main()
