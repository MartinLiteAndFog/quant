"""Tests for the Kraken bot state machine — pure logic, no API calls."""
import unittest
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pandas as pd

from quant.execution.kraken_bot import (
    BotState,
    FlipParams,
    TP2Params,
    _load_signals_from_postgres,
    _terminal_to_actions,
    compute_swing_sl,
    compute_target_size,
    run_once_logic,
)
from quant.strategies.flip_engine import FlipParams as SharedFlipParams, run_flip_state_machine

FLIP = FlipParams()
TP2 = TP2Params()
SIZE = 1.0


def _tick(state, gate_on=0, signal=0, signal_ts="", mark=100.0,
          swing_low=95.0, swing_high=105.0, target_size=SIZE,
          flip_p=FLIP, tp2_p=TP2):
    return run_once_logic(
        state=state, gate_on=gate_on, signal=signal, signal_ts=signal_ts,
        mark=mark, swing_low=swing_low, swing_high=swing_high,
        target_size=target_size, flip_p=flip_p, tp2_p=tp2_p,
    )


class TestFlatEntry(unittest.TestCase):
    def test_flat_long_signal_enters_flip_ttp(self):
        s = BotState(engine="flip", gate_on=1)
        ns, acts = _tick(s, gate_on=1, signal=1, signal_ts="t1", mark=100.0)
        self.assertEqual(ns.mode, "FLIP_TTP")
        self.assertEqual(ns.pos_side, 1)
        self.assertEqual(ns.entry_px, 100.0)
        self.assertEqual(len(acts), 1)
        self.assertEqual(acts[0]["action"], "enter_long")

    def test_flat_short_signal_enters_tp2_open(self):
        s = BotState(engine="tp2", gate_on=0)
        ns, acts = _tick(s, gate_on=0, signal=-1, signal_ts="t1", mark=100.0)
        self.assertEqual(ns.mode, "TP2_OPEN")
        self.assertEqual(ns.pos_side, -1)
        self.assertEqual(acts[0]["action"], "enter_short")

    def test_flat_no_signal_stays_flat(self):
        s = BotState()
        ns, acts = _tick(s, gate_on=1, signal=0, mark=100.0)
        self.assertEqual(ns.mode, "FLAT")
        self.assertEqual(len(acts), 0)

    def test_duplicate_signal_ignored(self):
        s = BotState(engine="flip", gate_on=1, last_signal_ts="t1")
        ns, acts = _tick(s, gate_on=1, signal=1, signal_ts="t1", mark=100.0)
        self.assertEqual(ns.mode, "FLAT")
        self.assertEqual(len(acts), 0)


class TestGateTransition(unittest.TestCase):
    def test_gate_change_regime_exit(self):
        s = BotState(mode="FLIP_TTP", engine="flip", pos_side=1, entry_px=100.0,
                      best_fav=105.0, gate_on=1, size_full=1.0, size_rem=1.0)
        ns, acts = _tick(s, gate_on=0, mark=103.0)
        self.assertEqual(ns.mode, "FLAT")
        self.assertEqual(ns.pos_side, 0)
        self.assertEqual(acts[0]["reason"], "regime_exit")

    def test_gate_change_while_flat_no_action(self):
        s = BotState(mode="FLAT", gate_on=1)
        ns, acts = _tick(s, gate_on=0, mark=100.0)
        self.assertEqual(ns.mode, "FLAT")
        self.assertEqual(len(acts), 0)


class TestFlipTTP(unittest.TestCase):
    def test_ttp_trail_triggers_flip_to_wait(self):
        s = BotState(mode="FLIP_TTP", engine="flip", pos_side=1, entry_px=100.0,
                      best_fav=105.0, gate_on=1, size_full=1.0, size_rem=1.0)
        # TTP stop: 105 * (1 - 0.012) = 103.74. Mark at 103.0 → triggers.
        ns, acts = _tick(s, gate_on=1, mark=103.0)
        self.assertEqual(ns.mode, "FLIP_WAIT")
        self.assertEqual(ns.pos_side, -1)  # flipped to short
        self.assertEqual(len(acts), 2)  # close_all + enter_short
        self.assertEqual(acts[0]["reason"], "ttp_flip")

    def test_ttp_trail_not_triggered(self):
        s = BotState(mode="FLIP_TTP", engine="flip", pos_side=1, entry_px=100.0,
                      best_fav=105.0, gate_on=1, size_full=1.0, size_rem=1.0)
        # TTP stop: 103.74. Mark at 104.5 → no trigger.
        ns, acts = _tick(s, gate_on=1, mark=104.5)
        self.assertEqual(ns.mode, "FLIP_TTP")
        self.assertEqual(ns.best_fav, 105.0)  # unchanged (104.5 < 105.0)
        self.assertEqual(len(acts), 0)

    def test_best_fav_updates(self):
        s = BotState(mode="FLIP_TTP", engine="flip", pos_side=1, entry_px=100.0,
                      best_fav=105.0, gate_on=1, size_full=1.0, size_rem=1.0)
        ns, acts = _tick(s, gate_on=1, mark=107.0)
        self.assertEqual(ns.best_fav, 107.0)

    def test_opposite_signal_flips(self):
        s = BotState(mode="FLIP_TTP", engine="flip", pos_side=1, entry_px=100.0,
                      best_fav=105.0, gate_on=1, size_full=1.0, size_rem=1.0)
        ns, acts = _tick(s, gate_on=1, signal=-1, signal_ts="t2", mark=103.0)
        self.assertEqual(ns.mode, "FLIP_TTP")
        self.assertEqual(ns.pos_side, -1)
        self.assertEqual(len(acts), 2)  # close + enter_short
        self.assertEqual(acts[0]["reason"], "signal_flip")


class TestFlipWait(unittest.TestCase):
    def test_swing_sl_exits_flat(self):
        s = BotState(mode="FLIP_WAIT", engine="flip", pos_side=1, entry_px=100.0,
                      best_fav=100.0, gate_on=1, size_full=1.0, size_rem=1.0)
        # swing SL for long: entry * (1 - clamp((entry-swing_low)/entry, 0.015, 0.030))
        # (100 - 95)/100 = 0.05, clamped to 0.030 → SL = 100 * 0.97 = 97.0
        ns, acts = _tick(s, gate_on=1, mark=96.5, swing_low=95.0)
        self.assertEqual(ns.mode, "FLAT")
        self.assertEqual(acts[0]["reason"], "swing_sl")

    def test_signal_rearms_ttp(self):
        s = BotState(mode="FLIP_WAIT", engine="flip", pos_side=1, entry_px=100.0,
                      best_fav=100.0, gate_on=1, size_full=1.0, size_rem=1.0)
        ns, acts = _tick(s, gate_on=1, signal=1, signal_ts="t3", mark=102.0)
        self.assertEqual(ns.mode, "FLIP_TTP")
        self.assertEqual(ns.best_fav, 102.0)


class TestTP2Open(unittest.TestCase):
    def test_tp2_full_exit(self):
        s = BotState(mode="TP2_OPEN", engine="tp2", pos_side=1, entry_px=100.0,
                      gate_on=0, size_full=1.0, size_rem=1.0)
        # TP2 at 100 * 1.030 = 103.0. Mark at 103.5 → triggers.
        ns, acts = _tick(s, gate_on=0, mark=103.5)
        self.assertEqual(ns.mode, "FLAT")
        self.assertEqual(acts[0]["reason"], "tp2_exit")

    def test_tp1_partial_exit(self):
        s = BotState(mode="TP2_OPEN", engine="tp2", pos_side=1, entry_px=100.0,
                      gate_on=0, size_full=1.0, size_rem=1.0)
        # TP1 at 100 * 1.015 = 101.5. Mark at 101.8 → triggers TP1 (not TP2).
        ns, acts = _tick(s, gate_on=0, mark=101.8)
        self.assertEqual(ns.mode, "TP2_BE")
        self.assertTrue(ns.tp1_done)
        self.assertAlmostEqual(ns.size_rem, 0.5)
        self.assertEqual(acts[0]["action"], "close_partial")
        self.assertEqual(acts[0]["reason"], "tp1_exit")

    def test_swing_sl_exit(self):
        s = BotState(mode="TP2_OPEN", engine="tp2", pos_side=1, entry_px=100.0,
                      gate_on=0, size_full=1.0, size_rem=1.0)
        # TP2 SL: clamp((100-90)/100, 0.030, 0.080) = 0.080 → SL = 92.0
        ns, acts = _tick(s, gate_on=0, mark=91.0, swing_low=90.0)
        self.assertEqual(ns.mode, "FLAT")
        self.assertEqual(acts[0]["reason"], "swing_sl")

    def test_opposite_signal_close_and_flip(self):
        s = BotState(mode="TP2_OPEN", engine="tp2", pos_side=1, entry_px=100.0,
                      gate_on=0, size_full=1.0, size_rem=1.0)
        ns, acts = _tick(s, gate_on=0, signal=-1, signal_ts="t4", mark=100.0)
        self.assertEqual(ns.pos_side, -1)
        self.assertEqual(ns.mode, "TP2_OPEN")  # flipped into new TP2
        self.assertEqual(len(acts), 2)  # close + enter_short

    def test_opposite_signal_no_flip(self):
        tp2_noflip = TP2Params(flip_on_opposite=False)
        s = BotState(mode="TP2_OPEN", engine="tp2", pos_side=1, entry_px=100.0,
                      gate_on=0, size_full=1.0, size_rem=1.0)
        ns, acts = _tick(s, gate_on=0, signal=-1, signal_ts="t5", mark=100.0, tp2_p=tp2_noflip)
        self.assertEqual(ns.pos_side, 0)
        self.assertEqual(ns.mode, "FLAT")
        self.assertEqual(len(acts), 1)


class TestTP2BE(unittest.TestCase):
    def test_be_exit(self):
        s = BotState(mode="TP2_BE", engine="tp2", pos_side=1, entry_px=100.0,
                      gate_on=0, size_full=1.0, size_rem=0.5, tp1_done=True)
        ns, acts = _tick(s, gate_on=0, mark=99.5)
        self.assertEqual(ns.mode, "FLAT")
        self.assertEqual(acts[0]["reason"], "be_exit")

    def test_tp2_remainder_exit(self):
        s = BotState(mode="TP2_BE", engine="tp2", pos_side=1, entry_px=100.0,
                      gate_on=0, size_full=1.0, size_rem=0.5, tp1_done=True)
        ns, acts = _tick(s, gate_on=0, mark=103.5)
        self.assertEqual(ns.mode, "FLAT")
        self.assertEqual(acts[0]["reason"], "tp2_exit")


class TestTargetSize(unittest.TestCase):
    def test_90pct_equity_5x_leverage(self):
        # 111 USD * 0.90 * 5 / 85.0 = 5.8 → floored to 5.8
        size = compute_target_size(111.0, 85.0, leverage=5.0, equity_pct=0.90)
        self.assertAlmostEqual(size, 5.8)

    def test_50pct_equity(self):
        # 111 * 0.50 * 5 / 85 = 3.26 → floored to 3.2
        size = compute_target_size(111.0, 85.0, leverage=5.0, equity_pct=0.50)
        self.assertAlmostEqual(size, 3.2)

    def test_zero_mark_returns_zero(self):
        self.assertEqual(compute_target_size(111.0, 0.0, 5.0, 0.90), 0.0)

    def test_zero_equity_returns_zero(self):
        self.assertEqual(compute_target_size(0.0, 85.0, 5.0, 0.90), 0.0)


class TestSwingSL(unittest.TestCase):
    def test_long_sl_clamped(self):
        sl = compute_swing_sl(1, 100.0, 90.0, 110.0, 0.015, 0.030)
        self.assertAlmostEqual(sl, 97.0)  # (100-90)/100=0.10, clamped to 0.030

    def test_short_sl_clamped(self):
        sl = compute_swing_sl(-1, 100.0, 90.0, 110.0, 0.015, 0.030)
        self.assertAlmostEqual(sl, 103.0)  # (110-100)/100=0.10, clamped to 0.030

    def test_long_sl_within_range(self):
        sl = compute_swing_sl(1, 100.0, 98.0, 102.0, 0.015, 0.030)
        self.assertAlmostEqual(sl, 98.0)  # (100-98)/100=0.020, within [0.015, 0.030]


class TestLoadSignalsFiltersStrategy(unittest.TestCase):
    """Verify _load_signals_from_postgres filters by strategy column."""

    @patch("quant.execution.kraken_bot.get_conn")
    def test_default_filters_countertrend(self, mock_get_conn):
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            (pd.Timestamp("2026-03-13T10:00Z"), -1),
        ]
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value = mock_conn

        df = _load_signals_from_postgres("SOL-USDT")
        sql_arg = mock_cur.execute.call_args[0][0]
        params_arg = mock_cur.execute.call_args[0][1]

        self.assertIn("strategy = %s", sql_arg)
        self.assertEqual(params_arg[1], "countertrend")

    @patch("quant.execution.kraken_bot.get_conn")
    def test_explicit_strategy_passed_to_query(self, mock_get_conn):
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value = mock_conn

        _load_signals_from_postgres("SOL-USDT", strategy="trendfollower")
        params_arg = mock_cur.execute.call_args[0][1]
        self.assertEqual(params_arg[1], "trendfollower")


class TestMixedSignalsCauseMissedSell(unittest.TestCase):
    """Demonstrate that mixing countertrend and trendfollower signals
    into the flip engine causes missed sells, and filtering fixes it."""

    def _make_bars(self, n=10, start_px=100.0):
        ts = pd.date_range("2026-03-13T10:00:00Z", periods=n, freq="min", tz="UTC")
        close = [start_px + i * 0.5 for i in range(n)]
        return pd.DataFrame({
            "ts": ts,
            "open": close,
            "high": [c + 0.3 for c in close],
            "low": [c - 0.3 for c in close],
            "close": close,
        })

    def test_sell_signal_visible_when_only_countertrend(self):
        bars = self._make_bars(6)
        ct_signals = pd.DataFrame({
            "ts": [bars["ts"].iloc[1], bars["ts"].iloc[4]],
            "signal": [1, -1],
        })

        _, events_df, terminal = run_flip_state_machine(
            bars=bars, signals_df=ct_signals,
            params=SharedFlipParams(ttp_trail_pct=0.10, min_sl_pct=0.015, max_sl_pct=0.030),
        )
        self.assertEqual(terminal["pos"], -1, "flip engine should be short after sell signal")

    def test_sell_signal_can_be_swallowed_by_mixed_signals(self):
        bars = self._make_bars(6)
        ct_sell_ts = bars["ts"].iloc[4]
        mixed_signals = pd.DataFrame({
            "ts": [
                bars["ts"].iloc[1], bars["ts"].iloc[1],
                ct_sell_ts, ct_sell_ts,
            ],
            "signal": [1, -1, -1, 1],
        })
        mixed_signals = (
            mixed_signals.sort_values("ts")
            .drop_duplicates("ts", keep="last")
            .reset_index(drop=True)
        )

        _, events_df, terminal = run_flip_state_machine(
            bars=bars, signals_df=mixed_signals,
            params=SharedFlipParams(ttp_trail_pct=0.10, min_sl_pct=0.015, max_sl_pct=0.030),
        )
        self.assertEqual(
            terminal["pos"], 1,
            "with mixed signals (keep=last gives trendfollower), "
            "the sell signal is swallowed and position stays long",
        )


class TestTerminalToActions(unittest.TestCase):
    def test_long_to_short_generates_close_and_enter(self):
        acts = _terminal_to_actions(terminal_pos=-1, current_pos_side=1, target_size=2.0, mark=100.0)
        self.assertEqual(len(acts), 2)
        self.assertEqual(acts[0]["action"], "close_all")
        self.assertEqual(acts[1]["action"], "enter_short")
        self.assertEqual(acts[1]["side"], "sell")

    def test_same_position_no_action(self):
        acts = _terminal_to_actions(terminal_pos=1, current_pos_side=1, target_size=2.0, mark=100.0)
        self.assertEqual(len(acts), 0)

    def test_flat_to_short(self):
        acts = _terminal_to_actions(terminal_pos=-1, current_pos_side=0, target_size=2.0, mark=100.0)
        self.assertEqual(len(acts), 1)
        self.assertEqual(acts[0]["action"], "enter_short")


if __name__ == "__main__":
    unittest.main()
