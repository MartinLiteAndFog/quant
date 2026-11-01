"""Pure accounting tests: never touch KuCoin, Railway, or trading credentials."""
import unittest
from decimal import Decimal

from quant.execution.family_ledger import BotIntent, FamilyLedger, FamilyLedgerError


class FamilyLedgerTests(unittest.TestCase):
    def setUp(self):
        self.ledger = FamilyLedger("family-a", "0.1")

    def intent(self, oid, bot, side, action, qty):
        return BotIntent(oid, bot, "SOLUSDTM", side, action, qty)

    def fill(self, oid, bot, side, action, qty, px, fee="0"):
        self.ledger.register(self.intent(oid, bot, side, action, qty))
        self.ledger.apply_fill(oid, oid + "-fill", qty, px, fee)

    def test_parallel_long_and_short_with_own_bot_positions(self):
        a = self.intent("a-open", "a", "LONG", "OPEN", 2)
        b = self.intent("b-open", "b", "LONG", "OPEN", 3)
        c = self.intent("c-open", "c", "SHORT", "OPEN", 4)
        self.assertEqual(self.ledger.register(a)["positionSide"], "LONG")
        self.assertEqual(self.ledger.register(b)["side"], "buy")
        self.assertEqual(self.ledger.register(c)["side"], "sell")
        for i in (a, b, c):
            self.ledger.apply_fill(i.client_oid, i.client_oid + "-f", i.contracts, 100)
        self.ledger.reconcile("SOLUSDTM", venue_long=5, venue_short=4)
        self.fill("a-close", "a", "LONG", "CLOSE", 2, 110)
        self.assertEqual(self.ledger.position("a", "SOLUSDTM", "LONG").contracts, 0)
        self.assertEqual(self.ledger.position("b", "SOLUSDTM", "LONG").contracts, 3)
        self.ledger.reconcile("SOLUSDTM", venue_long=3, venue_short=4)

    def test_close_cannot_consume_another_bots_exposure(self):
        self.fill("buy1", "a", "LONG", "OPEN", 1, 100)
        self.fill("buy2", "b", "LONG", "OPEN", 10, 100)
        with self.assertRaises(FamilyLedgerError):
            self.ledger.register(self.intent("a-overclose", "a", "LONG", "CLOSE", 2))

    def test_reservations_prevent_duplicate_close(self):
        self.fill("buy", "a", "LONG", "OPEN", 3, 100)
        self.ledger.register(self.intent("close-1", "a", "LONG", "CLOSE", 2))
        with self.assertRaises(FamilyLedgerError):
            self.ledger.register(self.intent("close-2", "a", "LONG", "CLOSE", 2))
        self.ledger.apply_fill("close-1", "fill1", 1, 105)
        with self.assertRaises(FamilyLedgerError):
            self.ledger.register(self.intent("close-3", "a", "LONG", "CLOSE", 2))
        self.ledger.confirm_cancel("close-1")
        self.ledger.register(self.intent("close-4", "a", "LONG", "CLOSE", 2))

    def test_partial_fills_and_dedup(self):
        self.ledger.register(self.intent("entry", "a", "LONG", "OPEN", 2))
        self.assertTrue(self.ledger.apply_fill("entry", "trade1", 1, "100", "0.01"))
        self.assertFalse(self.ledger.apply_fill("entry", "trade1", 1, "100", "0.01"))
        with self.assertRaises(FamilyLedgerError):
            self.ledger.apply_fill("entry", "trade1", 1, "101", "0.01")
        self.ledger.apply_fill("entry", "trade2", 1, "110", "0.01")
        p = self.ledger.position("a", "SOLUSDTM", "LONG")
        self.assertEqual(p.contracts, 2)
        self.assertEqual(p.average_entry, Decimal("105"))
        self.assertEqual(p.fees, Decimal("0.02"))

    def test_short_pnl_and_fee(self):
        self.fill("short", "a", "SHORT", "OPEN", 5, "110", "0.05")
        fields = self.ledger.register(self.intent("cover", "a", "SHORT", "CLOSE", 3))
        self.assertEqual(fields["side"], "buy")
        self.assertTrue(fields["reduceOnly"])
        self.ledger.apply_fill("cover", "cover-fill", 3, "100", "0.03")
        p = self.ledger.position("a", "SOLUSDTM", "SHORT")
        self.assertEqual(p.contracts, 2)
        self.assertEqual(p.gross_realized, Decimal("3"))
        self.assertEqual(p.net_realized, Decimal("2.92"))

    def test_mismatched_exchange_exposure_blocks(self):
        self.fill("buy", "a", "LONG", "OPEN", 1, 100)
        with self.assertRaises(FamilyLedgerError):
            self.ledger.reconcile("SOLUSDTM", venue_long=2, venue_short=0)

    def test_cancellation_and_unknown_fill_fail_closed(self):
        self.ledger.register(self.intent("buy", "a", "LONG", "OPEN", 2))
        with self.assertRaises(FamilyLedgerError):
            self.ledger.apply_fill("unknown", "f", 1, 100)
        self.ledger.confirm_cancel("buy")
        with self.assertRaises(FamilyLedgerError):
            self.ledger.apply_fill("buy", "late", 1, 100)

    def test_rejects_bad_modes(self):
        with self.assertRaises(FamilyLedgerError):
            self.intent("bad", "a", "BOTH", "OPEN", 1)
        with self.assertRaises(FamilyLedgerError):
            self.intent("bad", "a", "LONG", "OPEN", 0)


if __name__ == "__main__":
    unittest.main()
