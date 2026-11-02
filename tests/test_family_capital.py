"""No-trade tests of 90% equity allocation across Blaupunkt parent families."""
import unittest
from decimal import Decimal

from quant.execution.family_capital import (
    CapitalAllocationError,
    allocate_account,
)


class CapitalAllocationTests(unittest.TestCase):
    def test_one_parent_receives_ninety_percent(self):
        plan = allocate_account("1000", ["blaupunkt"])
        self.assertEqual(plan.tradable_equity, Decimal("900"))
        self.assertEqual(plan.reserve, Decimal("100"))
        self.assertEqual(plan.families["blaupunkt"].parent_budget, Decimal("900"))

    def test_three_parents_each_receive_three_hundred(self):
        plan = allocate_account("1000", ["a", "b", "c"])
        self.assertEqual([x.family_budget for x in plan.families.values()], [Decimal("300")] * 3)
        self.assertEqual(plan.reserve, Decimal("100"))
        self.assertEqual(plan.allocated, Decimal("900"))

    def test_daughters_take_two_percent_within_their_own_family(self):
        plan = allocate_account(
            "1000", ["a", "b", "c"],
            {"a": {f"a-daughter-{i}": "0.02" for i in range(6)}},
        )
        a = plan.families["a"]
        self.assertEqual(a.family_budget, Decimal("300"))
        self.assertEqual(a.parent_budget, Decimal("264"))
        self.assertEqual(list(a.daughter_budgets.values()), [Decimal("6")] * 6)
        self.assertEqual(plan.families["b"].parent_budget, Decimal("300"))
        self.assertEqual(plan.families["c"].parent_budget, Decimal("300"))
        self.assertEqual(plan.reserve, Decimal("100"))

    def test_all_inactive_stays_in_reserve(self):
        plan = allocate_account("500", [])
        self.assertEqual(plan.tradable_equity, Decimal("0"))
        self.assertEqual(plan.reserve, Decimal("500"))
        self.assertEqual(plan.families, {})

    def test_parent_list_must_be_unique(self):
        with self.assertRaises(CapitalAllocationError):
            allocate_account("100", ["same", "same"])

    def test_daughters_cannot_claim_more_than_one_family(self):
        with self.assertRaises(CapitalAllocationError):
            allocate_account("100", ["a"], {"a": {"x": "0.6", "y": "0.6"}})

    def test_daughter_must_belong_to_active_parent(self):
        with self.assertRaises(CapitalAllocationError):
            allocate_account("100", ["a"], {"b": {"x": "0.02"}})

    def test_invalid_numbers_fail_closed(self):
        for val in ("NaN", "-1", "Infinity", "oops"):
            with self.subTest(val=val):
                with self.assertRaises(CapitalAllocationError):
                    allocate_account(val, ["a"])

    def test_allocation_never_uses_more_than_equity(self):
        plan = allocate_account("123.45", ["a", "b"], {"a": {"x": "0.02"}})
        self.assertEqual(plan.allocated + plan.reserve, plan.equity)

    def test_fishing_bots_can_be_separate(self):
        # The pool type is managed by the registry, not counted as extra
        # parents unless explicitly included in active_parents.
        plan = allocate_account("1000", ["evolution-parent"])
        self.assertEqual(plan.families["evolution-parent"].family_budget, Decimal("900"))


if __name__ == "__main__":
    unittest.main()
