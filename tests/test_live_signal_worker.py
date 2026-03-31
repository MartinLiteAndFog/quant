from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from quant.execution.live_signal_worker import WorkerState, run_once
from quant.regime import RegimeDecision, RegimeService, RegimeStore


class _DummyBroker:
    def get_position(self, symbol: str) -> float:
        return 0.0


class LiveSignalWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        os.environ["REGIME_DB_PATH"] = str(self.root / "regime.db")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @patch("quant.execution.live_signal_worker.get_live_gate_state", create=True)
    @patch("quant.execution.live_signal_worker.compute_imba_signals")
    @patch("quant.execution.live_signal_worker._load_shared_renko")
    def test_run_once_prefers_daily_gate_over_stale_regime_store(
        self,
        mock_load_shared_renko,
        mock_compute_imba_signals,
        mock_get_live_gate_state,
    ) -> None:
        regime_store = RegimeStore()
        RegimeService(regime_store).upsert_decision(
            RegimeDecision(
                ts="2026-03-30T00:00:00Z",
                symbol="SOL-USDT",
                gate_on=1,
                regime_state="countertrend",
                regime_score=1.0,
                confidence=0.8,
                reason_code="seed",
            )
        )

        mock_get_live_gate_state.return_value = {
            "gate_on": 0,
            "gate_off": 1,
            "gate_countertrend_on": 0,
            "gate_trend_on": 1,
            "source": "postgres_daily_gate",
        }
        mock_load_shared_renko.return_value = pd.DataFrame(
            {
                "ts": pd.date_range("2026-03-31", periods=300, freq="min", tz="UTC"),
                "open": [100.0] * 300,
                "high": [101.0] * 300,
                "low": [99.0] * 300,
                "close": [100.5] * 300,
            }
        )
        mock_compute_imba_signals.return_value = pd.DataFrame(
            {
                "ts": [pd.Timestamp("2026-03-31T08:42:00Z")],
                "signal": [-1],
                "position": [-1],
                "sl": [83.05],
            }
        )

        signals_dir = self.root / "signals"
        state = run_once(
            _DummyBroker(),
            symbol="SOL-USDT",
            renko_parquet=self.root / "renko.parquet",
            lookback=250,
            sl_abs=1.5,
            signals_dir=signals_dir,
            regime_store=regime_store,
            default_gate_on=1,
            state=WorkerState(),
        )

        self.assertIsNotNone(state.last_signal_ts)

        sym_dir = signals_dir / "SOL-USDT"
        root_file = sym_dir / f"{pd.Timestamp.now('UTC').strftime('%Y%m%d')}.jsonl"
        rows = [json.loads(line) for line in root_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(rows[-1]["strategy_mode"], "trendfollower")
        self.assertEqual(int(rows[-1]["gate_on"]), 0)

        latest = regime_store.get_latest_state(symbol="SOL-USDT") or {}
        self.assertEqual(int(latest.get("gate_on", -1)), 0)
        self.assertEqual(latest.get("regime_state"), "trendfollower")
