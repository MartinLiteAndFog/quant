from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from quant.regime import RegimeDecision, RegimeService, RegimeStore


class RegimeStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "regime.db")
        self.store = RegimeStore(db_path=self.db_path)
        self.svc = RegimeService(self.store)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_upsert_and_latest(self) -> None:
        self.svc.upsert_decision(
            RegimeDecision(
                ts="2026-02-20T00:00:00Z",
                symbol="SOL-USDT",
                gate_on=1,
                regime_state="trend",
                regime_score=0.9,
                confidence=0.8,
                reason_code="test",
            )
        )
        latest = self.store.get_latest_state("SOL-USDT")
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(int(latest["gate_on"]), 1)
        self.assertEqual(str(latest["regime_state"]), "trend")

    def test_transition_is_recorded(self) -> None:
        self.svc.upsert_decision(
            RegimeDecision(
                ts="2026-02-20T00:00:00Z",
                symbol="SOL-USDT",
                gate_on=1,
                regime_state="trend",
                regime_score=0.8,
                confidence=0.75,
                reason_code="seed",
            )
        )
        self.svc.upsert_decision(
            RegimeDecision(
                ts="2026-02-21T00:00:00Z",
                symbol="SOL-USDT",
                gate_on=0,
                regime_state="countertrend",
                regime_score=-0.7,
                confidence=0.72,
                reason_code="flip",
            )
        )
        transitions = self.store.get_recent_transitions("SOL-USDT", limit=10)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["prev_state"], "trend")
        self.assertEqual(transitions[0]["new_state"], "countertrend")


if __name__ == "__main__":
    unittest.main()
