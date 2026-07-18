from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from quant.execution.live_executor import (
    ExecutorState, run_once, _apply_live_ttp_guard, _live_order_qty,
)


class _DummyBroker:
    def __init__(self, pos: float, bid: float, ask: float, multiplier: float = 1.0, equity: float = 1000.0) -> None:
        self._pos = float(pos)
        self._bid = float(bid)
        self._ask = float(ask)
        self._multiplier = float(multiplier)
        self._equity = float(equity)

    def get_best_bid_ask(self, symbol: str):
        return (self._bid, self._ask)

    def get_position(self, symbol: str) -> float:
        return self._pos

    def get_contract_multiplier(self, symbol: str) -> float:
        return self._multiplier

    def get_account_balance(self, currency: str = "USDT"):
        return {"equity": self._equity, "available": self._equity, "margin": 0.0, "unrealised_pnl": 0.0}


class _Res:
    def __init__(self, ok: bool = True) -> None:
        self.ok = bool(ok)


class _DummyOms:
    def __init__(self) -> None:
        self.enter_calls = []
        self.flip_calls = []
        self.exit_calls = []
        self.tp1_partial_calls = []

    def enter(self, symbol: str, side: str, qty: float):
        self.enter_calls.append((symbol, side, float(qty)))
        return _Res(True)

    def enter_market(self, symbol: str, side: str, qty: float):
        return self.enter(symbol, side, qty)

    def exit_tp_or_flip(self, symbol: str, side: str, qty: float, flip_to: str | None = None):
        self.flip_calls.append((symbol, side, float(qty), flip_to))
        return _Res(True)

    def flatten_market(self, symbol: str, side: str, qty: float):
        self.flip_calls.append((symbol, side, float(qty), None))
        return _Res(True)

    def exit_sl(self, symbol: str, side: str, qty: float):
        self.exit_calls.append((symbol, side, float(qty)))
        return _Res(True)

    def partial_tp1_market(self, symbol: str, side: str, qty: float):
        self.tp1_partial_calls.append((symbol, side, float(qty)))
        return _Res(True)


class LiveExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.signals_root = self.root / "signals"
        self.symbol_dir = self.signals_root / "SOL-USDT"
        self.symbol_dir.mkdir(parents=True, exist_ok=True)
        self.renko_path = self.root / "renko.parquet"

        bars = pd.DataFrame(
            {
                "ts": pd.to_datetime(
                    [
                        "2026-02-25T10:00:00Z",
                        "2026-02-25T10:01:00Z",
                        "2026-02-25T10:02:00Z",
                    ],
                    utc=True,
                ),
                "open": [100.0, 100.0, 120.0],
                "high": [100.0, 120.0, 120.0],
                "low": [100.0, 100.0, 107.0],
                "close": [100.0, 120.0, 107.0],
            }
        )
        bars.to_parquet(self.renko_path, index=False)

        rec = {"ts": "2026-02-25T10:00:00Z", "signal": 1}
        (self.symbol_dir / "20260225.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")

        os.environ["LIVE_RENKO_PATH"] = str(self.renko_path)
        os.environ["LIVE_FLIP_TTP_TRAIL_PCT"] = "0.10"
        os.environ["LIVE_FLIP_MIN_SL_PCT"] = "0.015"
        os.environ["LIVE_FLIP_MAX_SL_PCT"] = "0.030"
        os.environ["LIVE_EXECUTOR_POS_PCT"] = "1.0"

    def tearDown(self) -> None:
        self.tmp.cleanup()
        for key in ("LIVE_RENKO_PATH", "LIVE_FLIP_TTP_TRAIL_PCT",
                     "LIVE_FLIP_MIN_SL_PCT", "LIVE_FLIP_MAX_SL_PCT",
                     "LIVE_EXECUTOR_POS_PCT", "LIVE_EXECUTOR_MAX_MARGIN_USDT",
                     "LIVE_EXECUTOR_MAX_CONTRACTS"):
            os.environ.pop(key, None)

    def test_micro_pilot_sizing_caps_margin_and_contracts(self) -> None:
        with patch.dict(
            os.environ,
            {"LIVE_EXECUTOR_MAX_MARGIN_USDT": "5", "LIVE_EXECUTOR_MAX_CONTRACTS": "1"},
            clear=False,
        ):
            qty = _live_order_qty(
                equity=1000,
                pos_pct=1.0,
                leverage=3.0,
                mid_price=75.0,
                contract_multiplier=0.1,
            )
        self.assertEqual(qty, 1)

    def test_terminal_state_short_flips_long_broker_position(self) -> None:
        """Flip engine ends in short (pos=-1) after TTP exit.  Broker is
        long -> executor must flip to short."""
        broker = _DummyBroker(pos=2.0, bid=106.9, ask=107.1)
        oms = _DummyOms()
        st = ExecutorState()

        st = run_once(
            broker=broker,
            oms=oms,
            symbol="SOL-USDT",
            signals_root=self.signals_root,
            state=st,
            live_enabled=True,
            dry_run=False,
            leverage=1.0,
        )

        self.assertEqual(st.last_action, "flip_to_short")
        self.assertEqual(len(oms.flip_calls), 1)
        sym, side, qty, flip_to = oms.flip_calls[0]
        self.assertEqual(sym, "SOL-USDT")
        self.assertEqual(side, "long")
        self.assertEqual(int(qty), 2)
        self.assertIsNone(flip_to)


    def test_sizing_uses_contract_multiplier(self) -> None:
        broker = _DummyBroker(pos=0.0, bid=100.0, ask=100.0, multiplier=0.1)
        oms = _DummyOms()
        st = ExecutorState()

        st = run_once(
            broker=broker,
            oms=oms,
            symbol="SOL-USDT",
            signals_root=self.signals_root,
            state=st,
            live_enabled=True,
            dry_run=False,
            leverage=1.0,
        )

        self.assertEqual(st.last_action, "enter_short")
        self.assertEqual(len(oms.enter_calls), 1)
        sym, side, qty = oms.enter_calls[0]
        self.assertEqual(sym, "SOL-USDT")
        self.assertEqual(side, "short")
        self.assertEqual(int(qty), 100)

    def test_last_event_is_idempotent(self) -> None:
        """After flipping to short and the fill landing, the next poll must
        not produce another flip (terminal state unchanged, position matches)."""
        broker = _DummyBroker(pos=2.0, bid=106.9, ask=107.1)
        oms = _DummyOms()
        st = ExecutorState()

        st = run_once(
            broker=broker,
            oms=oms,
            symbol="SOL-USDT",
            signals_root=self.signals_root,
            state=st,
            live_enabled=True,
            dry_run=False,
            leverage=1.0,
        )
        self.assertEqual(len(oms.flip_calls), 1)

        # After the flip fills, broker now reports short.
        broker._pos = -9.0
        st = run_once(
            broker=broker,
            oms=oms,
            symbol="SOL-USDT",
            signals_root=self.signals_root,
            state=st,
            live_enabled=True,
            dry_run=False,
            leverage=1.0,
        )
        self.assertEqual(len(oms.flip_calls), 1, "No additional flip after fill")

    def test_apply_live_ttp_guard_short_caps_stale_ttp(self) -> None:
        terminal = {"side": "short", "mode": "TTP", "ttp": 83.996}
        out = _apply_live_ttp_guard(
            terminal,
            live_pos=-30.0,
            live_mid=82.70,
            ttp_trail_pct=0.012,
        )
        # 82.70 * 1.012 = 83.6924 ; stale higher ttp must be capped.
        self.assertAlmostEqual(float(out["ttp"]), 83.6924, places=4)

    def test_terminal_state_drives_entry_when_flat(self) -> None:
        """Flip engine ends in short position.  Broker is flat.
        Executor must enter short based on terminal state."""
        broker = _DummyBroker(pos=0.0, bid=84.9, ask=85.1)
        oms = _DummyOms()

        st = run_once(
            broker=broker,
            oms=oms,
            symbol="SOL-USDT",
            signals_root=self.signals_root,
            state=ExecutorState(),
            live_enabled=True,
            dry_run=False,
            leverage=1.0,
        )

        self.assertEqual(st.last_action, "enter_short",
                         "Terminal state short + broker flat must enter short")
        self.assertEqual(len(oms.enter_calls), 1)
        self.assertEqual(oms.enter_calls[0][1], "short")

    def test_terminal_short_with_long_broker_flips(self) -> None:
        """Terminal state is short but broker is long -> must flip."""
        broker = _DummyBroker(pos=5.0, bid=84.9, ask=85.1)
        oms = _DummyOms()

        st = run_once(
            broker=broker,
            oms=oms,
            symbol="SOL-USDT",
            signals_root=self.signals_root,
            state=ExecutorState(),
            live_enabled=True,
            dry_run=False,
            leverage=1.0,
        )

        self.assertEqual(st.last_action, "flip_to_short",
                         "Terminal short + broker long must trigger flip")
        self.assertEqual(len(oms.flip_calls), 1)
        self.assertEqual(oms.flip_calls[0][1], "long")

    def test_exact_match_signal_processed_by_flip_engine(self) -> None:
        """Signal with exact bar timestamp is processed by the flip engine."""
        bars = pd.DataFrame({
            "ts": pd.to_datetime([
                "2026-02-25T10:00:00Z",
                "2026-02-25T10:01:00Z",
                "2026-02-25T10:02:00Z",
            ], utc=True),
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
        })
        renko_path = self.root / "renko_exact.parquet"
        bars.to_parquet(renko_path, index=False)
        os.environ["LIVE_RENKO_PATH"] = str(renko_path)

        rec = {"ts": "2026-02-25T10:00:00Z", "signal": 1}
        exact_dir = self.signals_root / "SOL-USDT"
        exact_dir.mkdir(parents=True, exist_ok=True)
        (exact_dir / "20260225_exact.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")

        broker = _DummyBroker(pos=0.0, bid=99.9, ask=100.1)
        oms = _DummyOms()
        st = run_once(
            broker=broker,
            oms=oms,
            symbol="SOL-USDT",
            signals_root=self.signals_root,
            state=ExecutorState(),
            live_enabled=True,
            dry_run=False,
            leverage=1.0,
        )
        self.assertIn(st.last_action, ("enter_long", "enter_short", "flip_to_long", "flip_to_short"),
                       "Exact-match signal must result in a trade action")
        self.assertTrue(len(oms.enter_calls) > 0 or len(oms.flip_calls) > 0,
                        "At least one OMS call must have been made")

    def test_terminal_state_idempotent_no_whipsaw(self) -> None:
        """After entering short, the same terminal state on the next poll
        must NOT produce another action (no whipsaw sell-then-buy)."""
        broker = _DummyBroker(pos=-20.0, bid=89.9, ask=90.1)
        oms = _DummyOms()

        # First run: terminal is short, broker is short -> hold
        st = run_once(
            broker=broker,
            oms=oms,
            symbol="SOL-USDT",
            signals_root=self.signals_root,
            state=ExecutorState(),
            live_enabled=True,
            dry_run=False,
            leverage=1.0,
        )
        self.assertEqual(st.last_action, "hold")

        # Second run: same state -> no action at all
        oms2 = _DummyOms()
        st = run_once(
            broker=broker,
            oms=oms2,
            symbol="SOL-USDT",
            signals_root=self.signals_root,
            state=st,
            live_enabled=True,
            dry_run=False,
            leverage=1.0,
        )
        self.assertEqual(len(oms2.enter_calls), 0, "No enter calls on idempotent hold")
        self.assertEqual(len(oms2.flip_calls), 0, "No flip calls on idempotent hold")
        self.assertEqual(len(oms2.exit_calls), 0, "No exit calls on idempotent hold")

    @patch("quant.execution.live_executor._read_live_gate_from_redis", side_effect=AssertionError("redis bypass should not be used"))
    @patch(
        "quant.execution.live_executor.get_live_gate_state",
        return_value={"gate_on": 1, "gate_countertrend_on": 1, "gate_trend_on": 0, "source": "forced_countertrend"},
    )
    def test_run_once_uses_canonical_gate_provider(self, mock_gate, mock_redis) -> None:
        broker = _DummyBroker(pos=0.0, bid=84.9, ask=85.1)
        oms = _DummyOms()

        st = run_once(
            broker=broker,
            oms=oms,
            symbol="SOL-USDT",
            signals_root=self.signals_root,
            state=ExecutorState(),
            live_enabled=True,
            dry_run=True,
            leverage=1.0,
        )

        self.assertEqual(st.last_action, "enter_short")
        mock_gate.assert_called_once()

    @patch(
        "quant.execution.live_executor.get_live_gate_state",
        return_value={"gate_on": 1, "gate_countertrend_on": 1, "gate_trend_on": 0, "source": "forced_countertrend"},
    )
    @patch("quant.execution.live_executor._latest_backtest_event")
    def test_new_imba_signal_overwrites_open_trade_regime_without_forced_flatten(
        self, mock_latest_backtest, mock_gate
    ) -> None:
        ev = {
            "ts": pd.Timestamp("2026-02-25T10:00:00Z"),
            "event": "signal_exit",
            "side": -1,
            "seq": 9,
            "note": "Opposite signal -> close",
        }
        terminal = {
            "pos": -1,
            "side": "short",
            "mode": "TTP",
            "entry_px": 99.5,
            "sl": 101.0,
            "ttp": 98.0,
            "tp1": None,
            "tp2": None,
            "be_px": None,
            "be_armed": False,
            "tp1_done": False,
            "size_rem": 1.0,
            "entry_bar_ts": pd.Timestamp("2026-02-25T10:00:00Z"),
            "leg_id": "flip-leg",
            "tp1_frac": 0.5,
            "tp1_hit_ts": None,
            "tp1_hit_px": None,
            "best_fav": None,
        }
        mock_latest_backtest.return_value = (ev, terminal)

        broker = _DummyBroker(pos=10.0, bid=99.9, ask=100.1, equity=1000.0)
        oms = _DummyOms()
        st = ExecutorState(latched_exit_engine="tp2", last_signal_ts="2026-02-25T09:59:00+00:00")

        st = run_once(
            broker=broker,
            oms=oms,
            symbol="SOL-USDT",
            signals_root=self.signals_root,
            state=st,
            live_enabled=True,
            dry_run=True,
            leverage=1.0,
        )

        self.assertEqual(st.latched_exit_engine, "flip")
        self.assertEqual(st.last_action, "hold")
        self.assertEqual(len(oms.flip_calls), 0)
        self.assertEqual(len(oms.exit_calls), 0)
        mock_gate.assert_called_once()

    @patch(
        "quant.execution.live_executor.get_live_gate_state",
        return_value={"gate_on": 0, "gate_countertrend_on": 0, "gate_trend_on": 1, "source": "forced_trend"},
    )
    @patch("quant.execution.live_executor.run_follow_tp2_state_machine")
    def test_tp1_done_terminal_still_triggers_live_partial_once(self, mock_tp2, mock_gate) -> None:
        terminal = {
            "pos": 1,
            "side": "long",
            "mode": "TP2",
            "entry_px": 99.0,
            "sl": 97.0,
            "tp1": 103.0,
            "tp2": 108.0,
            "be_px": 99.0,
            "be_armed": True,
            "tp1_done": True,
            "size_rem": 0.5,
            "entry_bar_ts": pd.Timestamp("2026-02-25T10:02:00Z"),
            "leg_id": "tp2-leg-1",
            "tp1_frac": 0.5,
            "tp1_hit_ts": pd.Timestamp("2026-02-25T10:04:00Z"),
            "tp1_hit_px": 103.0,
            "ttp": None,
            "best_fav": None,
        }
        events_df = pd.DataFrame(
            [
                {
                    "ts": pd.Timestamp("2026-02-25T10:04:00Z"),
                    "event": "tp1_hit",
                    "side": 1,
                    "price": 103.0,
                    "pnl_pct": 0.0,
                    "note": "TP1 hit",
                    "seq": 9,
                    "size": 0.5,
                }
            ]
        )
        mock_tp2.return_value = (pd.Series([0]), events_df, terminal)

        broker = _DummyBroker(pos=10.0, bid=100.0, ask=100.0)
        oms = _DummyOms()
        st = ExecutorState(latched_exit_engine="tp2")

        st = run_once(
            broker=broker,
            oms=oms,
            symbol="SOL-USDT",
            signals_root=self.signals_root,
            state=st,
            live_enabled=True,
            dry_run=False,
            leverage=1.0,
        )

        self.assertEqual(st.last_action, "tp1_partial")
        self.assertEqual(len(oms.tp1_partial_calls), 1)
        self.assertEqual(oms.tp1_partial_calls[0][0], "SOL-USDT")
        self.assertEqual(oms.tp1_partial_calls[0][1], "long")
        self.assertGreater(oms.tp1_partial_calls[0][2], 0.0)
        self.assertTrue(st.tp2_tp1_done)
        self.assertFalse(st.tp2_tp1_pending)

    @patch(
        "quant.execution.live_executor.get_live_gate_state",
        return_value={"gate_on": 0, "gate_countertrend_on": 0, "gate_trend_on": 1, "source": "forced_trend"},
    )
    @patch("quant.execution.live_executor.run_follow_tp2_state_machine")
    def test_tp1_done_with_consumed_hit_reconciles_oversized_live_position(self, mock_tp2, mock_gate) -> None:
        terminal = {
            "pos": 1,
            "side": "long",
            "mode": "TP2",
            "entry_px": 99.0,
            "sl": 97.0,
            "tp1": 103.0,
            "tp2": 108.0,
            "be_px": 99.0,
            "be_armed": True,
            "tp1_done": True,
            "size_rem": 0.5,
            "entry_bar_ts": pd.Timestamp("2026-02-25T10:02:00Z"),
            "leg_id": "tp2-leg-1",
            "tp1_frac": 0.5,
            "tp1_hit_ts": pd.Timestamp("2026-02-25T10:04:00Z"),
            "tp1_hit_px": 103.0,
            "ttp": None,
            "best_fav": None,
        }
        events_df = pd.DataFrame(
            [
                {
                    "ts": pd.Timestamp("2026-02-25T10:04:00Z"),
                    "event": "be_armed",
                    "side": 1,
                    "price": 103.0,
                    "pnl_pct": 0.0,
                    "note": "TP1 consumed in replay",
                    "seq": 9,
                    "size": 0.5,
                }
            ]
        )
        mock_tp2.return_value = (pd.Series([0]), events_df, terminal)

        broker = _DummyBroker(pos=10.0, bid=100.0, ask=100.0)
        oms = _DummyOms()
        st = ExecutorState(
            latched_exit_engine="tp2",
            tp2_leg_id="tp2-leg-1",
            tp2_leg_side="long",
            tp2_tp1_done=True,
            tp2_tp1_pending=False,
            tp2_size_rem=0.5,
            tp2_remaining_qty_abs=8.0,
            tp2_tp1_hit_ts="2026-02-25T10:04:00+00:00",
            tp2_last_consumed_tp1_hit_ts="2026-02-25T10:04:00+00:00",
        )

        st = run_once(
            broker=broker,
            oms=oms,
            symbol="SOL-USDT",
            signals_root=self.signals_root,
            state=st,
            live_enabled=True,
            dry_run=False,
            leverage=1.0,
        )

        self.assertEqual(st.last_action, "tp1_partial")
        self.assertEqual(len(oms.tp1_partial_calls), 1)
        self.assertEqual(oms.tp1_partial_calls[0][0], "SOL-USDT")
        self.assertEqual(oms.tp1_partial_calls[0][1], "long")
        self.assertAlmostEqual(oms.tp1_partial_calls[0][2], 5.0)
        self.assertTrue(st.tp2_tp1_done)
        self.assertFalse(st.tp2_tp1_pending)

    @patch(
        "quant.execution.live_executor.get_live_gate_state",
        return_value={"gate_on": 1, "gate_countertrend_on": 1, "gate_trend_on": 0, "source": "forced_countertrend"},
    )
    @patch("quant.execution.live_executor.run_follow_tp2_state_machine")
    def test_tp2_imba_flip_overwrites_regime_without_flatten(self, mock_tp2, mock_gate) -> None:
        terminal = {
            "pos": -1,
            "side": "short",
            "mode": "TP2",
            "entry_px": 99.0,
            "sl": 101.0,
            "tp1": 95.0,
            "tp2": 90.0,
            "be_px": None,
            "be_armed": False,
            "tp1_done": False,
            "size_rem": 1.0,
            "entry_bar_ts": pd.Timestamp("2026-02-25T10:02:00Z"),
            "leg_id": "flip-leg-1",
            "tp1_frac": 0.5,
            "tp1_hit_ts": None,
            "tp1_hit_px": None,
            "ttp": None,
            "best_fav": None,
        }
        events_df = pd.DataFrame(
            [
                {
                    "ts": pd.Timestamp("2026-02-25T10:02:00Z"),
                    "event": "entry",
                    "side": -1,
                    "price": 99.0,
                    "pnl_pct": 0.0,
                    "note": "Flip: open opposite on same bar",
                    "seq": 7,
                    "size": 1.0,
                }
            ]
        )
        mock_tp2.return_value = (pd.Series([0]), events_df, terminal)

        broker = _DummyBroker(pos=10.0, bid=99.9, ask=100.1)
        oms = _DummyOms()
        st = ExecutorState(latched_exit_engine="tp2")

        st = run_once(
            broker=broker,
            oms=oms,
            symbol="SOL-USDT",
            signals_root=self.signals_root,
            state=st,
            live_enabled=True,
            dry_run=True,
            leverage=1.0,
        )

        self.assertEqual(st.last_action, "hold")
        self.assertEqual(st.latched_exit_engine, "flip")
        self.assertEqual(len(oms.flip_calls), 0)
        self.assertEqual(len(oms.exit_calls), 0)
        mock_gate.assert_called_once()

    def test_apply_live_ttp_guard_short_does_not_loosen(self) -> None:
        terminal = {"side": "short", "mode": "TTP", "ttp": 83.50}
        out = _apply_live_ttp_guard(
            terminal,
            live_pos=-30.0,
            live_mid=82.70,
            ttp_trail_pct=0.012,
        )
        # Existing tighter stop must remain.
        self.assertAlmostEqual(float(out["ttp"]), 83.50, places=6)


if __name__ == "__main__":
    unittest.main()
