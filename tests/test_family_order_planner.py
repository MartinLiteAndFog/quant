"""Unit tests for bot-identified KuCoin order previews; never send orders."""
import unittest
from decimal import Decimal

from quant.execution.family_ledger import BotIntent, FamilyLedger
from quant.execution.family_order_planner import PlanningError, plan_signal


class OrderPlannerTests(unittest.TestCase):
    def setUp(self):
        self.ledger = FamilyLedger("blaupunkt", "0.1")

    def _plan(self, bot="a", event="e-1", action="entry", side="buy", **kwargs):
        args = {
            "ledger": self.ledger,
            "strategy_id": bot, "event_id": event, "symbol": "SOLUSDTM",
            "action": action, "side": side, "bot_margin_budget": "30",
            "free_margin": "100", "reserve_floor": "10",
            "mark_price": "100", "leverage": "3", "contract_multiplier": "0.1",
        }
        args.update(kwargs)
        return plan_signal(**args)

    def _fill(self, oid, bot, side, contracts, entry_price):
        self.ledger.register(
            BotIntent(oid, bot, "SOLUSDTM", side, "OPEN", contracts)
        )
        self.ledger.apply_fill(oid, oid + "-fill", contracts, entry_price)

    def test_one_percent_parent_entry_sized_with_budget(self):
        plan = self._plan()
        self.assertEqual(plan.status, "planned_no_execution")
        self.assertEqual(plan.intent.contracts, 9)
        self.assertEqual(plan.intent.position_side, "LONG")
        self.assertEqual(plan.intent.order_fields()["side"], "buy")
        self.assertFalse(plan.as_dict()["order_sent"])

    def test_free_margin_and_reserve_also_cap_order(self):
        plan = self._plan(free_margin="20", reserve_floor="10")
        self.assertEqual(plan.intent.contracts, 3)

    def test_same_signal_produces_same_client_oid(self):
        self.assertEqual(
            self._plan().intent.client_oid,
            self._plan().intent.client_oid,
        )
        self.assertNotEqual(
            self._plan(bot="a").intent.client_oid,
            self._plan(bot="b").intent.client_oid,
        )

    def test_too_small_bot_has_no_order(self):
        plan = self._plan(bot_margin_budget="0.01")
        self.assertIsNone(plan.intent)
        self.assertEqual(plan.status, "below_venue_contract_minimum")

    def test_sl_reduces_only_this_bots_long(self):
        self._fill("a-buy", "a", "LONG", 2, 100)
        self._fill("b-buy", "b", "LONG", 5, 100)
        plan = self._plan(action="sl", side="")
        self.assertEqual(plan.intent.contracts, 2)
        self.assertEqual(plan.intent.operation, "CLOSE")
        self.assertTrue(plan.intent.order_fields()["reduceOnly"])
        self.assertEqual(self.ledger.position("b", "SOLUSDTM", "LONG").contracts, 5)

    def test_tp1_reduces_half_of_own_short(self):
        self._fill("a-short", "a", "SHORT", 7, 100)
        plan = self._plan(action="tp1", side="")
        self.assertEqual(plan.intent.position_side, "SHORT")
        self.assertEqual(plan.intent.order_fields()["side"], "buy")
        self.assertEqual(plan.intent.contracts, 3)

    def test_flip_emits_only_close_until_confirmed_fill(self):
        self._fill("a-buy", "a", "LONG", 3, 100)
        plan = self._plan(action="flip", side="sell")
        self.assertEqual(plan.intent.operation, "CLOSE")
        self.assertEqual(plan.intent.position_side, "LONG")
        self.assertEqual(plan.intent.contracts, 3)
        self.assertEqual(plan.deferred_side, "SHORT")
        self.assertEqual(plan.status, "flip_close_only_wait_for_confirmed_fill")

    def test_entry_does_not_reverse_existing_position(self):
        self._fill("a-buy", "a", "LONG", 3, 100)
        plan = self._plan(action="entry", side="sell")
        self.assertIsNone(plan.intent)
        self.assertEqual(plan.status, "entry_cannot_reverse_without_flip")

    def test_same_side_duplicate_does_not_increase_exposure(self):
        self._fill("a-buy", "a", "LONG", 2, 100)
        plan = self._plan(action="entry", side="buy")
        self.assertIsNone(plan.intent)
        self.assertEqual(plan.status, "already_on_requested_side")

    def test_bot_with_both_sides_requires_reconciliation(self):
        self._fill("a-buy", "a", "LONG", 1, 100)
        self._fill("a-sell", "a", "SHORT", 1, 100)
        with self.assertRaises(PlanningError):
            self._plan()

    def test_missing_margin_never_opens_a_position(self):
        for val in ("-1", "NaN", "Infinity", "x"):
            with self.subTest(val=val), self.assertRaises(PlanningError):
                self._plan(free_margin=val)

    def test_broker_contract_multiplier_mismatch_is_blocked(self):
        with self.assertRaises(PlanningError):
            self._plan(contract_multiplier="1")

    def test_missing_side_on_entry_is_blocked(self):
        with self.assertRaises(PlanningError):
            self._plan(side="")


if __name__ == "__main__":
    unittest.main()
