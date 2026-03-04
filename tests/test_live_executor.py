from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant.execution.live_executor import ExecutorState, run_once, _apply_live_ttp_guard


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
        os.environ["LIVE_REGIME_MODE"] = "none"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_tp_exit_event_flips_without_new_opposite_signal(self) -> None:
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
