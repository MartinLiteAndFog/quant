"""ECO-05 pure groundwater tests, including conservation and unsafe commitments."""
from decimal import Decimal as D
import unittest

from quant.ecosystem.capital.groundwater import (
    Account, Demand, GroundwaterError, allocate_groundwater, split_seed_budget,
)


class GroundwaterTests(unittest.TestCase):
    def setUp(self):
        self.a = Account("kucoin", "USDT", D("100"), D("45"))

    def test_three_parents_twenty_children_and_reserve(self):
        bots = [Demand("parent-" + str(i), "kucoin", D("1")) for i in range(3)]
        plan = allocate_groundwater([self.a], bots)
        self.assertEqual(plan.equity, D("100"))
        self.assertEqual(plan.reserve, D("10"))
        self.assertEqual(plan.distributable, D("90"))
        for i in range(20):
            parent = "parent-" + str(i % 3)
            plan = split_seed_budget(plan, parent, "child-" + str(i), "0.02", "45")
        self.assertEqual(len(plan.entitlements), 23)
        self.assertEqual(sum((e.amount for e in plan.entitlements), D("0")), D("90"))
        self.assertLessEqual(sum((e.entry_capacity for e in plan.entitlements), D("0")), D("45"))

    def test_account_isolation_and_no_double_count(self):
        b = Account("second", "USDT", D("50"), D("15"))
        requests = [Demand("s1", "kucoin", D("1")),
                    Demand("s2", "second", D("999"))]
        plan = allocate_groundwater([self.a, b], requests)
        self.assertEqual(plan.equity, D("150"))
        self.assertEqual(plan.reserve, D("15"))
        self.assertEqual(plan.by_strategy("s1").amount, D("90"))
        self.assertEqual(plan.by_strategy("s2").amount, D("45"))
        with self.assertRaisesRegex(GroundwaterError, "duplicate account"):
            allocate_groundwater([self.a, self.a], requests)
        with self.assertRaisesRegex(GroundwaterError, "same strategy"):
            allocate_groundwater([self.a], [requests[0], requests[0]])

    def test_cross_currency_refused_without_fx(self):
        with self.assertRaisesRegex(GroundwaterError, "FX conversion"):
            allocate_groundwater(
                [self.a, Account("fx", "USD", D("50"), D("20"))], [])

    def test_weighted_distribution_and_cap_reallocation(self):
        demands = [
            Demand("tiny", "kucoin", D("100"), D("0.1")),
            Demand("large", "kucoin", D("1")),
        ]
        p = allocate_groundwater([self.a], demands)
        self.assertEqual(p.by_strategy("tiny").amount, D("9"))
        self.assertEqual(p.by_strategy("large").amount, D("81"))
        self.assertEqual(p.unallocated, D("0"))

    def test_dormant_can_keep_locked_but_not_siphon_new_money(self):
        p = allocate_groundwater([self.a], [
            Demand("dormant", "kucoin", D("0"), D("1"), D("10"), eligible=False),
            Demand("active", "kucoin", D("1")),
        ])
        self.assertEqual(p.by_strategy("dormant").amount, D("10"))
        self.assertEqual(p.by_strategy("active").amount, D("80"))
        self.assertEqual(p.by_strategy("dormant").entry_capacity, D("0"))

    def test_locked_margin_drawdown_freezes_and_does_not_flatten(self):
        p = allocate_groundwater([Account("a", "USDT", D("20"), D("0"))],
                                [Demand("old", "a", D("1"), used_margin=D("25"))])
        self.assertTrue(p.risk_breach)
        self.assertTrue(p.allocations_frozen)
        self.assertEqual(p.by_strategy("old").used_margin, D("25"))
        self.assertEqual(p.by_strategy("old").entry_capacity, D("0"))
        with self.assertRaisesRegex(GroundwaterError, "risk-frozen"):
            split_seed_budget(p, "old", "new", "0.02", "0")

    def test_child_cannot_steal_locked_budget(self):
        p = allocate_groundwater([self.a], [
            Demand("old", "kucoin", D("1"), used_margin=D("89"))])
        with self.assertRaisesRegex(GroundwaterError, "occupied"):
            split_seed_budget(p, "old", "baby", "0.02", "45")
        q = split_seed_budget(p, "old", "baby", "0.01", "45")
        self.assertEqual(q.by_strategy("baby").amount, D("0.9"))
        self.assertEqual(sum((e.amount for e in q.entitlements), D("0")), D("90"))
        with self.assertRaisesRegex(GroundwaterError, "duplicate child"):
            split_seed_budget(q, "old", "baby", "0.01", "45")

    def test_uneconomic_or_ineligible_no_forced_capital_usage(self):
        p = allocate_groundwater([self.a], [
            Demand("idle", "kucoin", D("0"), eligible=False)])
        self.assertEqual(p.unallocated, D("90"))
        self.assertEqual(p.by_strategy("idle").amount, D("0"))

    def test_non_finite_or_float_input_rejected(self):
        for invalid in ("NaN", "Infinity", "-1", 3.2):
            with self.subTest(invalid=invalid):
                with self.assertRaises(GroundwaterError):
                    Demand("x", "kucoin", invalid)


if __name__ == "__main__":
    unittest.main()
