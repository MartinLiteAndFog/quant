"""Margin mode must be ISOLATED for configured leverage to apply on KuCoin."""
import unittest
from unittest.mock import MagicMock
from quant.execution.kucoin_futures import KucoinFuturesBroker

def mk(current, position):
    b = KucoinFuturesBroker.__new__(KucoinFuturesBroker)
    b._position_margin_mode = lambda s: current
    b.get_position = lambda s: position
    b.set_margin_mode = MagicMock(return_value={"marginMode": "ISOLATED"})
    return b

class T(unittest.TestCase):
    def test_switches_when_flat(self):
        b = mk("CROSS", 0.0)
        self.assertEqual(b.ensure_margin_mode("SOL-USDT", "ISOLATED"), "ISOLATED")
        b.set_margin_mode.assert_called_once()

    def test_refuses_with_open_position(self):
        b = mk("CROSS", 4.0)
        self.assertEqual(b.ensure_margin_mode("SOL-USDT", "ISOLATED"), "CROSS")
        b.set_margin_mode.assert_not_called()

    def test_noop_when_already_correct(self):
        b = mk("ISOLATED", 4.0)
        self.assertEqual(b.ensure_margin_mode("SOL-USDT", "ISOLATED"), "ISOLATED")
        b.set_margin_mode.assert_not_called()

    def test_rejects_bad_mode(self):
        b = KucoinFuturesBroker.__new__(KucoinFuturesBroker)
        with self.assertRaises(ValueError):
            b.set_margin_mode("SOL-USDT", "hedged")

if __name__ == "__main__":
    unittest.main()
