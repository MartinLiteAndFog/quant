from __future__ import annotations

import unittest

import pandas as pd

from quant.strategies.flip_engine import FlipParams, run_flip_state_machine


class FlipEngineProfileTests(unittest.TestCase):
    def _run(self, *, reverse_on_wait_sl: bool):
        bars = pd.DataFrame(
            {
                "ts": pd.date_range("2026-01-01", periods=5, freq="min", tz="UTC"),
                "open": [100.0, 110.0, 105.0, 103.0, 106.0],
                "high": [100.0, 110.0, 105.0, 103.0, 106.0],
                "low": [100.0, 110.0, 105.0, 103.0, 106.0],
                "close": [100.0, 110.0, 105.0, 103.0, 106.0],
            }
        )
        signals = pd.DataFrame(
            {
                "ts": [pd.Timestamp("2026-01-01T00:00:00Z")],
                "signal": [1],
            }
        )
        params = FlipParams(
            ttp_trail_pct=0.05,
            min_sl_pct=0.01,
            max_sl_pct=0.01,
            swing_lookback=0,
        )
        return run_flip_state_machine(
            bars,
            signals,
            params,
            regime_on=None,
            regime_forces_flat=False,
            reverse_on_wait_sl=reverse_on_wait_sl,
        )

    def test_wait_stop_remains_flat_by_default(self) -> None:
        _, events, terminal = self._run(reverse_on_wait_sl=False)
        self.assertEqual(events.iloc[-1]["event"], "sl_exit")
        self.assertEqual(terminal["pos"], 0)

    def test_wait_stop_reverses_for_opt_in_profile(self) -> None:
        _, events, terminal = self._run(reverse_on_wait_sl=True)
        self.assertEqual(events.iloc[-1]["event"], "sl_reverse_exit")
        self.assertEqual(terminal["pos"], 1)
        self.assertEqual(terminal["mode"], "WAIT")
