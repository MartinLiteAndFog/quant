"""Tests for KuCoin stop-market placement and stop verification.

Incident 2026-07-25. `place_stop_market` POSTed to `/api/v1/st-orders` — the
*attached* TP/SL endpoint, which keys off `triggerStopUpPrice` /
`triggerStopDownPrice`. Handed `stop` / `stopPrice` instead, it dropped the
trigger and accepted the remainder: a plain reduce-only market order. That
filled instantly, flattening the position ~300 ms after entry, on every trade:

    imba5   19:05:35.868 buy 163  entry
            19:05:36.179 sell 163 clientOid=quant-tv-emergency_sl-… stop='' type=market
    imbatp  20:10:07.565 buy  65  flip_entry
            20:10:07.865 sell  65 clientOid=quant-tv-emergency_sl-… stop='' type=market

Both bots therefore round-tripped on fees and ran with no stop protection.
Conditional stops belong on `POST /api/v1/orders`.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from quant.execution import tv_signal_executor as ktv
from quant.execution.kucoin_futures import KucoinFuturesBroker


class _Recorder(KucoinFuturesBroker):
    """Captures the request instead of sending it."""

    def __init__(self):  # deliberately skips the real __init__ (no credentials)
        self.calls = []
        self._order_leverage = 10.0

    def _req(self, method, path, body=None):
        self.calls.append({"method": method, "path": path, "body": body})
        return {"orderId": "stop-order-1"}

    def _order_margin_mode_candidates(self, symbol):
        return ["ISOLATED"]


class PlaceStopMarketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.broker = _Recorder()

    def _place(self, side="sell"):
        return self.broker.place_stop_market(
            symbol="SOL-USDT",
            side=side,
            qty=163,
            stop_price=73.1923,
            reduce_only=True,
            client_id="quant:tv:emergency_sl:1",
        )

    def test_uses_the_orders_endpoint_not_st_orders(self) -> None:
        self._place()
        call = self.broker.calls[0]
        self.assertEqual(call["path"], "/api/v1/orders")
        self.assertNotEqual(
            call["path"], "/api/v1/st-orders",
            "st-orders silently drops the trigger and fills at market",
        )
        self.assertEqual(call["method"], "POST")

    def test_carries_the_trigger_fields(self) -> None:
        self._place()
        body = self.broker.calls[0]["body"]
        self.assertEqual(body["stop"], "down")
        self.assertEqual(body["stopPrice"], "73.1923")
        self.assertEqual(body["stopPriceType"], "TP")
        self.assertTrue(body["reduceOnly"])
        self.assertEqual(body["size"], 163)

    def test_stop_direction_follows_the_side(self) -> None:
        # A long is protected by a sell that triggers on the way down; a short
        # by a buy that triggers on the way up. Wrong direction = instant fill.
        self._place(side="sell")
        self.assertEqual(self.broker.calls[-1]["body"]["stop"], "down")
        self._place(side="buy")
        self.assertEqual(self.broker.calls[-1]["body"]["stop"], "up")

    def test_returns_the_order_id(self) -> None:
        self.assertEqual(self._place(), "stop-order-1")


class _StubBroker:
    def __init__(self, order):
        self._order = order
        self.asked = []

    def get_order(self, order_id):
        self.asked.append(order_id)
        return self._order


class VerifyStopRegisteredTests(unittest.TestCase):
    def test_accepts_a_resting_stop(self) -> None:
        b = _StubBroker({"stop": "down", "status": "open", "stopPrice": "73.19"})
        self.assertTrue(ktv._verify_stop_registered(b, "oid-1", 73.19))

    def test_rejects_an_order_with_no_trigger(self) -> None:
        # Exactly what the broken endpoint produced.
        b = _StubBroker({"stop": "", "status": "done", "size": 163})
        self.assertFalse(ktv._verify_stop_registered(b, "oid-1", 73.19))

    def test_rejects_a_stop_that_already_filled(self) -> None:
        b = _StubBroker({"stop": "down", "status": "done"})
        self.assertFalse(ktv._verify_stop_registered(b, "oid-1", 73.19))

    def test_missing_order_id_is_false_not_an_error(self) -> None:
        self.assertFalse(ktv._verify_stop_registered(_StubBroker({}), None, 73.19))

    def test_lookup_failure_never_raises(self) -> None:
        class _Boom:
            def get_order(self, order_id):
                raise RuntimeError("kucoin down")

        # A failed check must not undo a good entry.
        self.assertFalse(ktv._verify_stop_registered(_Boom(), "oid-1", 73.19))


class EmergencySlPriceTests(unittest.TestCase):
    """The stop is a percentage of price, not of the account."""

    def test_long_stop_sits_below_entry_price(self) -> None:
        placed = {}

        class _B:
            def place_stop_market(self, **kw):
                placed.update(kw)
                return "oid"

            def get_order(self, order_id):
                return {"stop": "down", "status": "open"}

        ktv._place_emergency_sl(_B(), "SOL-USDT", "long", 163, 74.69, 0.02)
        self.assertEqual(placed["side"], "sell")
        self.assertAlmostEqual(placed["stop_price"], round(74.69 * 0.98, 4), places=4)
        self.assertTrue(placed["reduce_only"])

    def test_short_stop_sits_above_entry_price(self) -> None:
        placed = {}

        class _B:
            def place_stop_market(self, **kw):
                placed.update(kw)
                return "oid"

            def get_order(self, order_id):
                return {"stop": "up", "status": "open"}

        ktv._place_emergency_sl(_B(), "SOL-USDT", "short", 65, 74.44, 0.02)
        self.assertEqual(placed["side"], "buy")
        self.assertAlmostEqual(placed["stop_price"], round(74.44 * 1.02, 4), places=4)

    def test_zero_pct_disables_placement(self) -> None:
        class _B:
            def place_stop_market(self, **kw):
                raise AssertionError("must not place a stop when disabled")

        self.assertIsNone(ktv._place_emergency_sl(_B(), "SOL-USDT", "long", 163, 74.69, 0.0))



class StrategyStopOverrideTests(unittest.TestCase):
    """The alert may carry the strategy's own stop, which can be wider than the
    percentage backstop. Use it — but never if it sits on the wrong side."""

    def _place(self, side, mid, strategy_sl):
        placed = {}

        class _B:
            def place_stop_market(self, **kw):
                placed.update(kw)
                return "oid"

            def get_order(self, order_id):
                return {"stop": "down", "status": "open"}

        ktv._place_emergency_sl(_B(), "SOL-USDT", side, 100, mid, 0.025, strategy_sl)
        return placed

    def test_uses_a_wider_strategy_stop_for_a_long(self) -> None:
        # 4% away, wider than the 2.5% backstop — the strategy's level wins.
        placed = self._place("long", 74.69, 71.70)
        self.assertAlmostEqual(placed["stop_price"], 71.70, places=4)

    def test_uses_a_tighter_strategy_stop_for_a_long(self) -> None:
        placed = self._place("long", 74.69, 74.00)
        self.assertAlmostEqual(placed["stop_price"], 74.00, places=4)

    def test_uses_strategy_stop_for_a_short(self) -> None:
        placed = self._place("short", 74.44, 78.00)
        self.assertAlmostEqual(placed["stop_price"], 78.00, places=4)

    def test_rejects_a_long_stop_above_price(self) -> None:
        # Would trigger the instant it is placed — fall back to the backstop.
        placed = self._place("long", 74.69, 80.00)
        self.assertAlmostEqual(placed["stop_price"], round(74.69 * 0.975, 4), places=4)

    def test_rejects_a_short_stop_below_price(self) -> None:
        placed = self._place("short", 74.44, 70.00)
        self.assertAlmostEqual(placed["stop_price"], round(74.44 * 1.025, 4), places=4)

    def test_falls_back_when_no_strategy_stop_given(self) -> None:
        placed = self._place("long", 74.69, None)
        self.assertAlmostEqual(placed["stop_price"], round(74.69 * 0.975, 4), places=4)


class ParseStrategyStopTests(unittest.TestCase):
    def test_reads_common_stop_field_names(self) -> None:
        from quant.execution.tv_signal_executor import parse_tv_signal

        for key in ("sl_price", "sl", "stop_price", "stop", "stoploss", "stop_loss"):
            with self.subTest(key=key):
                sig = parse_tv_signal(
                    {"action": "entry", "side": "buy", "symbol": "SOL-USDT", key: 71.7}
                )
                self.assertAlmostEqual(sig.sl_price, 71.7)

    def test_absent_or_junk_stop_is_none(self) -> None:
        from quant.execution.tv_signal_executor import parse_tv_signal

        for payload in ({}, {"sl": ""}, {"sl": "abc"}, {"sl": 0}, {"sl": -5}, {"sl": True}):
            with self.subTest(payload=payload):
                base = {"action": "entry", "side": "buy", "symbol": "SOL-USDT"}
                base.update(payload)
                self.assertIsNone(parse_tv_signal(base).sl_price)

if __name__ == "__main__":
    unittest.main()
