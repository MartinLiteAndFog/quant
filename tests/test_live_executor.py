from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant.execution.live_executor import (
    ExecutorState, run_once, _apply_live_ttp_guard, _snap_signals_to_bars,
)


class _DummyBroker:
    def __init__(self, pos: float, bid: float, ask: float, multiplier: float = 1.0) -> None:
        self._pos = float(pos)
        self._bid = float(bid)
        self._ask = float(ask)
        self._multiplier = float(multiplier)

    def get_best_bid_ask(self, symbol: str):
        return (self._bid, self._ask)

    def get_position(self, symbol: str) -> float:
        return self._pos

    def get_contract_multiplier(self, symbol: str) -> float:
        return self._multiplier


class _Res:
    def __init__(self, ok: bool = True) -> None:
        self.ok = bool(ok)


class _DummyOms:
    def __init__(self) -> None:
        self.enter_calls = []
        self.flip_calls = []
        self.exit_calls = []

    def enter(self, symbol: str, side: str, qty: float):
        self.enter_calls.append((symbol, side, float(qty)))
        return _Res(True)

    def exit_tp_or_flip(self, symbol: str, side: str, qty: float, flip_to: str | None = None):
        self.flip_calls.append((symbol, side, float(qty), flip_to))
        return _Res(True)

    def exit_sl(self, symbol: str, side: str, qty: float):
        self.exit_calls.append((symbol, side, float(qty)))
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

        os.environ["LIVE_EXECUTOR_RENKO_PARQUET"] = str(self.renko_path)
        os.environ["LIVE_FLIP_TTP_TRAIL_PCT"] = "0.10"
        os.environ["LIVE_FLIP_MIN_SL_PCT"] = "0.015"
        os.environ["LIVE_FLIP_MAX_SL_PCT"] = "0.030"

    def tearDown(self) -> None:
        self.tmp.cleanup()

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
            max_eur=1000.0,
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
            max_eur=1000.0,
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
            max_eur=1000.0,
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
            max_eur=1000.0,
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
            max_eur=1000.0,
            leverage=1.0,
        )

        self.assertEqual(st.last_action, "enter_short",
                         "Terminal state short + broker flat must enter short")
        self.assertEqual(len(oms.enter_calls), 1)
        self.assertEqual(oms.enter_calls[0][1], "short")

    def test_snap_signals_to_bars_exact_match_unchanged(self) -> None:
        """Signals already matching bar timestamps are not modified."""
        bars = pd.DataFrame({"ts": pd.to_datetime(["2026-02-25T10:00:00Z", "2026-02-25T10:01:00Z"], utc=True)})
        sigs = pd.DataFrame({"ts": pd.to_datetime(["2026-02-25T10:00:00Z"], utc=True), "signal": [1]})
        out = _snap_signals_to_bars(sigs, bars)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["ts"], pd.Timestamp("2026-02-25T10:00:00Z"))

    def test_snap_signals_to_bars_snaps_within_tolerance(self) -> None:
        """Signal 30s off from a bar is snapped to the nearest bar."""
        bars = pd.DataFrame({"ts": pd.to_datetime(["2026-02-25T10:00:00Z", "2026-02-25T10:02:00Z"], utc=True)})
        sigs = pd.DataFrame({"ts": pd.to_datetime(["2026-02-25T10:00:30Z"], utc=True), "signal": [-1]})
        out = _snap_signals_to_bars(sigs, bars)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["ts"], pd.Timestamp("2026-02-25T10:00:00Z"))
        self.assertEqual(int(out.iloc[0]["signal"]), -1)

    def test_snap_signals_to_bars_outside_tolerance_kept(self) -> None:
        """Signal outside tolerance window keeps its original timestamp."""
        bars = pd.DataFrame({"ts": pd.to_datetime(["2026-02-25T10:00:00Z"], utc=True)})
        sigs = pd.DataFrame({"ts": pd.to_datetime(["2026-02-25T11:00:00Z"], utc=True), "signal": [1]})
        out = _snap_signals_to_bars(sigs, bars, tolerance=pd.Timedelta(minutes=5))
        self.assertEqual(out.iloc[0]["ts"], pd.Timestamp("2026-02-25T11:00:00Z"))

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
            max_eur=1000.0,
            leverage=1.0,
        )

        self.assertEqual(st.last_action, "flip_to_short",
                         "Terminal short + broker long must trigger flip")
        self.assertEqual(len(oms.flip_calls), 1)
        self.assertEqual(oms.flip_calls[0][1], "long")

    def test_snapped_signal_processed_by_flip_engine(self) -> None:
        """Signal with slightly off timestamp gets snapped to a bar and
        processed by the flip engine (not just fallback)."""
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
        renko_path = self.root / "renko_snap.parquet"
        bars.to_parquet(renko_path, index=False)
        os.environ["LIVE_EXECUTOR_RENKO_PARQUET"] = str(renko_path)

        rec = {"ts": "2026-02-25T10:00:17Z", "signal": 1}
        snap_dir = self.signals_root / "SOL-USDT"
        snap_dir.mkdir(parents=True, exist_ok=True)
        (snap_dir / "20260225_snap.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")

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
            max_eur=1000.0,
            leverage=1.0,
        )
        self.assertIn(st.last_action, ("enter_long", "enter_short", "flip_to_long", "flip_to_short"),
                       "Snapped signal must result in a trade action, not be ignored")
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
            max_eur=1000.0,
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
            max_eur=1000.0,
            leverage=1.0,
        )
        self.assertEqual(len(oms2.enter_calls), 0, "No enter calls on idempotent hold")
        self.assertEqual(len(oms2.flip_calls), 0, "No flip calls on idempotent hold")
        self.assertEqual(len(oms2.exit_calls), 0, "No exit calls on idempotent hold")

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
