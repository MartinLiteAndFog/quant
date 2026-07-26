"""Tests for the exit-after-open guard in the TradingView executor.

Incident 2026-07-25 20:10Z (imbatp / sol-pilot-pc3axis). TradingView sent the
`flip` and its matching `exit` 101 ms apart. `_do_flip` already closes the old
position and opens the new one, so the `exit` arrived a moment later and
flattened the position the flip had just created:

    20:10:07.300  flip/buy  accepted  -> buy  65 @ 74.440
    20:10:07.401  exit      accepted  -> sell 65 @ 74.439   (221 ms later)

Round-trip at the same price, holding nothing, paying entry+exit fees — about
1.08% of a $54 account each time it happens. An `exit` is computed against the
position that existed when the alert fired; one that lands milliseconds after an
open cannot have been meant for that new position.
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from quant.execution import tv_signal_executor as ktv
from quant.execution.tv_signal_executor import TVCache, TVExecConfig, TVSignal

SYMBOL = "SOL-USDT"


def _config(guard: float = 5.0) -> TVExecConfig:
    return TVExecConfig(
        symbol=SYMBOL,
        pos_pct=0.5,
        leverage=10.0,
        tp1_close_pct=0.5,
        dry_run=False,
        gate_mode="countertrend",
        cache_sec=10.0,
        cache_max_age_sec=60.0,
        emergency_sl_pct=0.023,
        flip_delay_sec=2.0,
        exit_after_open_guard_sec=guard,
        sl_liq_buffer_frac=0.25,
    )


def _cache(side: str = "long", position: float = 65.0) -> TVCache:
    return TVCache(
        position=position,
        current_side=side,
        equity=54.0,
        mid_price=74.44,
        bid=74.43,
        ask=74.45,
        contract_multiplier=0.1,
        qty=65,
        gate_on=1,
        gate_allows_entry=True,
        gate_source="test",
        updated_at=time.time(),
    )


class _FakeBroker:
    def __init__(self):
        self.orders = []

    def place_market(self, symbol, side, qty, reduce_only, client_id):
        self.orders.append({"side": side, "qty": qty, "reduce_only": reduce_only})
        return "order-1"

    def get_position(self, symbol):
        return 0.0

    def get_best_bid_ask(self, symbol):
        return 74.43, 74.45

    def cancel_all_stop_orders(self, symbol):
        return {"ok": True}


class ExitAfterOpenGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        with ktv._open_legs_lock:
            ktv._open_legs.clear()
        self.broker = _FakeBroker()
        self._patches = [
            patch.object(ktv, "_broker", self.broker),
            patch.object(ktv, "_cache", _cache()),
            patch.object(ktv, "_log_bg", lambda *a, **k: None),
            patch.object(ktv, "_append_tv_closed_trade", lambda **k: None),
            patch.object(ktv, "_refresh_position_in_cache", lambda *a, **k: None),
            patch.object(ktv, "_cancel_emergency_sl", lambda *a, **k: None),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()

    @staticmethod
    def _exit_signal() -> TVSignal:
        return TVSignal(action="exit", side="", symbol=SYMBOL)

    def test_exit_right_after_open_is_ignored(self) -> None:
        # This is the incident: the flip just opened, then the exit lands.
        ktv._record_open_leg(symbol=SYMBOL, side="long", qty=65.0, entry_price=74.44)
        result = ktv._execute_locked(self._exit_signal(), _config(guard=5.0))
        self.assertEqual(result["reason"], "exit_after_open_guard")
        self.assertTrue(result["ok"])
        self.assertEqual(self.broker.orders, [], "must not have placed a closing order")

    def test_exit_after_the_guard_window_closes_normally(self) -> None:
        ktv._record_open_leg(symbol=SYMBOL, side="long", qty=65.0, entry_price=74.44)
        # Age the position past the window.
        with ktv._open_legs_lock:
            ktv._open_legs[SYMBOL]["opened_monotonic"] = time.monotonic() - 60.0
        result = ktv._execute_locked(self._exit_signal(), _config(guard=5.0))
        self.assertNotEqual(result.get("reason"), "exit_after_open_guard")
        self.assertEqual(len(self.broker.orders), 1, "a genuine exit must still close")
        self.assertTrue(self.broker.orders[0]["reduce_only"])

    def test_guard_can_be_disabled(self) -> None:
        ktv._record_open_leg(symbol=SYMBOL, side="long", qty=65.0, entry_price=74.44)
        result = ktv._execute_locked(self._exit_signal(), _config(guard=0.0))
        self.assertNotEqual(result.get("reason"), "exit_after_open_guard")
        self.assertEqual(len(self.broker.orders), 1)

    def test_untracked_position_is_not_blocked(self) -> None:
        # No open leg recorded (e.g. the position predates this process, or the
        # leg was reconstructed from execution_events without a monotonic stamp).
        # The guard must fail open: a real exit is more important than the edge
        # case it protects against.
        result = ktv._execute_locked(self._exit_signal(), _config(guard=5.0))
        self.assertNotEqual(result.get("reason"), "exit_after_open_guard")
        self.assertEqual(len(self.broker.orders), 1)

    def test_leg_without_monotonic_stamp_is_not_blocked(self) -> None:
        with ktv._open_legs_lock:
            ktv._open_legs[SYMBOL] = {"side": "long", "qty": 65.0, "entry_price": 74.44}
        self.assertIsNone(ktv._seconds_since_open(SYMBOL))
        result = ktv._execute_locked(self._exit_signal(), _config(guard=5.0))
        self.assertNotEqual(result.get("reason"), "exit_after_open_guard")
        self.assertEqual(len(self.broker.orders), 1)

    def test_flat_position_short_circuits_before_the_guard(self) -> None:
        with patch.object(ktv, "_cache", _cache(side="flat", position=0.0)):
            result = ktv._execute_locked(self._exit_signal(), _config(guard=5.0))
        self.assertEqual(result["reason"], "already_flat")
        self.assertEqual(self.broker.orders, [])


class SecondsSinceOpenTests(unittest.TestCase):
    def setUp(self) -> None:
        with ktv._open_legs_lock:
            ktv._open_legs.clear()

    def test_none_when_no_leg(self) -> None:
        self.assertIsNone(ktv._seconds_since_open(SYMBOL))

    def test_fresh_open_is_near_zero(self) -> None:
        ktv._record_open_leg(symbol=SYMBOL, side="long", qty=1.0, entry_price=74.0)
        age = ktv._seconds_since_open(SYMBOL)
        self.assertIsNotNone(age)
        self.assertLess(age, 1.0)

    def test_reopening_resets_the_clock(self) -> None:
        ktv._record_open_leg(symbol=SYMBOL, side="long", qty=1.0, entry_price=74.0)
        with ktv._open_legs_lock:
            ktv._open_legs[SYMBOL]["opened_monotonic"] = time.monotonic() - 60.0
        self.assertGreater(ktv._seconds_since_open(SYMBOL), 30.0)
        ktv._record_open_leg(symbol=SYMBOL, side="short", qty=1.0, entry_price=74.0)
        self.assertLess(ktv._seconds_since_open(SYMBOL), 1.0)


if __name__ == "__main__":
    unittest.main()
