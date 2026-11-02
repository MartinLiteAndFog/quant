"""Real PostgreSQL integration tests for family fill idempotency and ownership.

Uses only a CI-local throwaway Postgres database. Set
FAMILY_TEST_DATABASE_URL and optional DATABASE_URL for connection. Never run
these against production credentials.
"""
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from uuid import uuid4

from quant.execution.family_ledger import BotIntent
from quant.execution.family_durable_ledger import (
    DurableLedgerError,
    apply_confirmed_fill,
    confirm_cancel,
    ensure_family_ledger_schema,
    get_strategy_position,
    mark_submitted,
    reconcile_side_totals,
    reserve_intent,
)


@unittest.skipUnless(
    os.getenv("FAMILY_TEST_DATABASE_URL"),
    "requires separate FAMILY_TEST_DATABASE_URL (never production)",
)
class DurableLedgerPostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Crucial: never connect an integration test to user production DB.
        expected = os.environ["FAMILY_TEST_DATABASE_URL"]
        current = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
        if current != expected:
            raise RuntimeError("test database URL does not match FAMILY_TEST_DATABASE_URL")
        ensure_family_ledger_schema()

    def setUp(self):
        self.family = "test-" + uuid4().hex
        self.symbol = "SOLUSDTM"

    def intent(self, oid, bot, side, operation, size):
        return BotIntent(oid, bot, self.symbol, side, operation, size)

    def reserve(self, oid, bot, side, operation, size):
        return reserve_intent(
            self.family, self.intent(oid, bot, side, operation, size),
            contract_multiplier="0.1",
        )

    def fill(self, oid, fill, size, price, fee="0"):
        return apply_confirmed_fill(self.family, oid, fill, size, price, fee)

    def test_two_longs_one_short_and_bot_specific_close(self):
        self.reserve("a1", "a", "LONG", "OPEN", 2)
        self.reserve("b1", "b", "LONG", "OPEN", 3)
        self.reserve("c1", "c", "SHORT", "OPEN", 4)
        self.fill("a1", "af", 2, "100", "0.02")
        self.fill("b1", "bf", 3, "100")
        self.fill("c1", "cf", 4, "100")
        self.assertEqual(
            reconcile_side_totals(
                self.family, self.symbol, venue_long=5, venue_short=4
            ), {"long": 5, "short": 4}
        )
        self.reserve("a-close", "a", "LONG", "CLOSE", 2)
        self.fill("a-close", "a-close-f", 2, "110", "0.02")
        self.assertEqual(get_strategy_position(self.family, "a", self.symbol, "LONG")["net_realized"], Decimal("1.96"))
        self.assertEqual(get_strategy_position(self.family, "b", self.symbol, "LONG")["contracts"], 3)
        self.assertEqual(get_strategy_position(self.family, "c", self.symbol, "SHORT")["contracts"], 4)
        reconcile_side_totals(self.family, self.symbol, venue_long=3, venue_short=4)

    def test_short_profit_is_signed_correctly(self):
        self.reserve("a", "a", "SHORT", "OPEN", 5)
        self.fill("a", "f1", 5, 110)
        self.reserve("close", "a", "SHORT", "CLOSE", 3)
        self.fill("close", "f2", 3, 100)
        p = get_strategy_position(self.family, "a", self.symbol, "SHORT")
        self.assertEqual(p["contracts"], 2)
        self.assertEqual(p["gross_realized"], Decimal("3"))

    def test_partial_fill_and_idempotent_replay(self):
        self.reserve("a", "a", "LONG", "OPEN", 3)
        mark_submitted(self.family, "a")
        self.assertTrue(self.fill("a", "f1", 1, 100, "0.01"))
        self.assertFalse(self.fill("a", "f1", 1, 100, "0.01"))
        self.assertTrue(self.fill("a", "f2", 2, 110, "0.02"))
        p = get_strategy_position(self.family, "a", self.symbol, "LONG")
        self.assertEqual(p["contracts"], 3)
        self.assertAlmostEqual(float(p["average_entry"]), 106.666666667)
        self.assertEqual(p["fees"], Decimal("0.03"))
        with self.assertRaises(DurableLedgerError):
            self.fill("a", "f1", 1, 101, "0.01")

    def test_double_oid_cannot_change_order(self):
        self.assertTrue(self.reserve("oid", "a", "LONG", "OPEN", 1))
        self.assertFalse(self.reserve("oid", "a", "LONG", "OPEN", 1))
        with self.assertRaises(DurableLedgerError):
            self.reserve("oid", "b", "LONG", "OPEN", 1)

    def test_pending_same_bot_entry_is_blocked(self):
        self.reserve("oid", "a", "LONG", "OPEN", 2)
        with self.assertRaises(DurableLedgerError):
            self.reserve("oid2", "a", "SHORT", "OPEN", 1)

    def test_pending_close_reservation_and_cancel(self):
        self.reserve("open", "a", "LONG", "OPEN", 3)
        self.fill("open", "f1", 3, 100)
        self.reserve("close-a", "a", "LONG", "CLOSE", 2)
        with self.assertRaises(DurableLedgerError):
            self.reserve("close-b", "a", "LONG", "CLOSE", 2)
        confirm_cancel(self.family, "close-a")
        self.reserve("close-b", "a", "LONG", "CLOSE", 2)
        self.assertEqual(get_strategy_position(self.family, "a", self.symbol, "LONG")["contracts"], 3)

    def test_late_fill_after_cancel_blocks(self):
        self.reserve("oid", "a", "LONG", "OPEN", 1)
        confirm_cancel(self.family, "oid")
        with self.assertRaises(DurableLedgerError):
            self.fill("oid", "late", 1, 100)

    def test_reconciliation_mismatch_blocks(self):
        self.reserve("oid", "a", "LONG", "OPEN", 1)
        self.fill("oid", "f", 1, 100)
        with self.assertRaises(DurableLedgerError):
            reconcile_side_totals(self.family, self.symbol, venue_long=2, venue_short=0)

    def test_concurrent_closes_cannot_both_reserve_same_lot(self):
        self.reserve("open", "a", "LONG", "OPEN", 1)
        self.fill("open", "f", 1, 100)
        def claim(oid):
            try:
                self.reserve(oid, "a", "LONG", "CLOSE", 1)
                return True
            except DurableLedgerError:
                return False
        with ThreadPoolExecutor(max_workers=2) as pool:
            out = list(pool.map(claim, ("close1", "close2")))
        self.assertEqual(sorted(out), [False, True])


if __name__ == "__main__":
    unittest.main()
