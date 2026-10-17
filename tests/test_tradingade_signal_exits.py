import unittest
import pandas as pd

from quant.backtest.tradingade_signal_exits import (
    EntryFilter, SignalExitParams, SignalOnlyTrailing, is_entry_allowed,
    confirmed_30m_rsi, attach_last_confirmed_rsi,
)


class TestTradingadeSignalOnly(unittest.TestCase):
    def test_rsi_gate_is_directional_and_strict(self):
        self.assertTrue(is_entry_allowed(1, 60))
        self.assertFalse(is_entry_allowed(1, 55))
        self.assertTrue(is_entry_allowed(-1, 70))
        self.assertFalse(is_entry_allowed(-1, 75))
        self.assertFalse(is_entry_allowed(-1, float("nan")))

    def test_30m_rsi_only_available_after_complete_candle(self):
        times = pd.date_range("2026-01-01", periods=1000, freq="min", tz="UTC")
        bars = pd.DataFrame({"ts": times, "close": [100 + n/10 for n in range(len(times))]})
        rsi = confirmed_30m_rsi(bars)
        self.assertEqual(rsi["available_at"].iloc[0], pd.Timestamp("2026-01-01T00:30Z"))
        decisions = pd.DataFrame({
            "decision_ts": ["2026-01-01T07:29Z", "2026-01-01T07:30Z"]})
        matched = attach_last_confirmed_rsi(decisions, rsi)
        self.assertEqual(matched["available_at"].iloc[0], pd.Timestamp("2026-01-01T07:00Z"))
        self.assertEqual(matched["available_at"].iloc[1], pd.Timestamp("2026-01-01T07:30Z"))
        self.assertEqual(matched["rsi30"].iloc[1], 100.0)

    def test_long_signal_trailing_after_confirmed_close(self):
        s = SignalOnlyTrailing(1, 100, SignalExitParams(.011, 0, .008))
        self.assertIsNone(s.mark(101))
        self.assertEqual(s.mark(100.15), "TRAILING_TP_SIGNAL")

    def test_short_signal_trailing_after_confirmed_close(self):
        s = SignalOnlyTrailing(-1, 100, SignalExitParams(.011, 0, .008))
        self.assertIsNone(s.mark(99))
        self.assertEqual(s.mark(99.81), "TRAILING_TP_SIGNAL")

    def test_stop_is_only_confirmed_chart_signal_not_broker_trigger(self):
        s = SignalOnlyTrailing(1, 100, SignalExitParams(.011, .02, .008))
        self.assertIsNone(s.mark(100.5))
        self.assertEqual(s.mark(98.8), "SL_SIGNAL")

    def test_no_intrabar_1m_stop_injection(self):
        s = SignalOnlyTrailing(1, 100)
        self.assertIsNone(s.mark(100.1))
        self.assertFalse(s.closed)


if __name__ == "__main__":
    unittest.main()
