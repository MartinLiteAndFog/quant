from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant.brain_forward.runtime import BrainDecision, completed_paper_trades


def _bars(count: int = 1600) -> pd.DataFrame:
    ts = pd.date_range("2026-07-01", periods=count, freq="min", tz="UTC")
    close = 100.0 + np.arange(count) * 0.01
    return pd.DataFrame(
        {
            "ts": ts,
            "open": close - 0.01,
            "high": close + 0.02,
            "low": close - 0.02,
            "close": close,
            "volume": np.full(count, 10.0),
            "taker_base": np.full(count, 4.0),
        }
    )


class _EventModel:
    def __init__(self, event_ts: pd.Timestamp) -> None:
        self.event_ts = event_ts

    def decision_from_feature_row(self, row: pd.DataFrame) -> BrainDecision | None:
        value = row.iloc[0]
        if value["ts"] != self.event_ts:
            return None
        return BrainDecision(
            event_ts=value["ts"],
            event_close=float(value["close"]),
            candle_range=0.03,
            expected_net_bps=5.0,
            active_memories=1,
            shock_z=3.2,
            close_position=0.1,
            volatility_ratio=2.2,
            flow_imbalance=-0.2,
        )


class BrainForwardRuntimeContinuityTests(unittest.TestCase):
    def test_contiguous_path_keeps_next_open_target_trade(self) -> None:
        raw = _bars()
        event_index = 1590
        raw.loc[event_index + 1, "high"] = raw.loc[event_index + 1, "open"] + 0.04

        decisions, trades = completed_paper_trades(
            raw, _EventModel(raw.loc[event_index, "ts"])
        )

        self.assertEqual(len(decisions), 1)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["exit_reason"], "target")

    def test_gap_inside_horizon_fails_closed_without_trade(self) -> None:
        raw = _bars()
        event_index = 1590
        raw.loc[event_index + 2 :, "ts"] += pd.Timedelta(minutes=1)
        raw.loc[event_index + 1, "high"] = raw.loc[event_index + 1, "open"] + 0.04

        decisions, trades = completed_paper_trades(
            raw, _EventModel(raw.loc[event_index, "ts"])
        )

        self.assertEqual(len(decisions), 1)
        self.assertEqual(trades, [])


if __name__ == "__main__":
    unittest.main()
