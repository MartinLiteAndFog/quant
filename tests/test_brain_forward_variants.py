from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant.brain_forward.runtime import BrainDecision, completed_paper_trades
from quant.brain_forward.variants import (
    IMMEDIATE,
    PREVIOUS_HIGH_CONFIRMATION,
    STOP_COOLDOWN_3M,
    evaluate_paper_variants,
)


def _bars(count: int = 1600) -> pd.DataFrame:
    ts = pd.date_range("2026-08-01", periods=count, freq="min", tz="UTC")
    close = np.full(count, 100.0)
    return pd.DataFrame({
        "ts": ts,
        "open": close.copy(),
        "high": close + 0.10,
        "low": close - 0.10,
        "close": close.copy(),
        "volume": np.full(count, 10.0),
        "taker_base": np.full(count, 4.0),
    })


class _EventModel:
    def __init__(self, event_times: list[pd.Timestamp], candle_range: float = 1.0) -> None:
        self.event_times = set(event_times)
        self.candle_range = candle_range

    def decision_from_feature_row(self, row: pd.DataFrame) -> BrainDecision | None:
        value = row.iloc[0]
        if value["ts"] not in self.event_times:
            return None
        return BrainDecision(
            event_ts=value["ts"],
            event_close=float(value["close"]),
            candle_range=self.candle_range,
            expected_net_bps=5.0,
            active_memories=2,
            shock_z=3.2,
            close_position=0.1,
            volatility_ratio=2.2,
            flow_imbalance=-0.2,
        )


class BrainForwardVariantTests(unittest.TestCase):
    def test_immediate_variant_matches_frozen_baseline_execution(self) -> None:
        raw = _bars()
        event_index = 1590
        raw.loc[event_index + 1, "high"] = 101.0
        model = _EventModel([raw.loc[event_index, "ts"]])

        _, baseline = completed_paper_trades(raw, model)
        evaluated = evaluate_paper_variants(raw, model)
        immediate = [trade for trade in evaluated.trades if trade["variant_id"] == IMMEDIATE]

        self.assertEqual(len(baseline), 1)
        self.assertEqual(len(immediate), 1)
        for field in ("entry_ts", "exit_ts", "entry_price", "exit_price", "exit_reason", "net_bps"):
            self.assertEqual(immediate[0][field], baseline[0][field])

    def test_cooldown_suppresses_only_three_minutes_after_stop(self) -> None:
        raw = _bars()
        first = 1590
        raw.loc[first + 1, "low"] = 99.0
        raw.loc[first + 4, "high"] = 101.0
        event_times = [raw.loc[index, "ts"] for index in (first, first + 1, first + 2, first + 3)]

        evaluated = evaluate_paper_variants(raw, _EventModel(event_times))
        events = {
            event["event_ts"]: event
            for event in evaluated.events
            if event["variant_id"] == STOP_COOLDOWN_3M
        }

        self.assertEqual(events[event_times[0]]["status"], "triggered")
        self.assertEqual(events[event_times[1]]["reason"], "stop_cooldown")
        self.assertEqual(events[event_times[2]]["reason"], "stop_cooldown")
        self.assertEqual(events[event_times[3]]["status"], "triggered")

    def test_confirmation_enters_on_break_of_immediately_previous_high(self) -> None:
        raw = _bars()
        event_index = 1590
        raw.loc[event_index, ["open", "high", "low", "close"]] = [100.0, 101.0, 99.0, 100.0]
        raw.loc[event_index + 1, ["open", "high", "low", "close"]] = [100.0, 100.5, 99.8, 100.2]
        raw.loc[event_index + 2, ["open", "high", "low", "close"]] = [100.2, 100.8, 100.1, 100.7]
        model = _EventModel([raw.loc[event_index, "ts"]], candle_range=2.0)

        evaluated = evaluate_paper_variants(raw, model)
        event = next(
            item for item in evaluated.events
            if item["variant_id"] == PREVIOUS_HIGH_CONFIRMATION
        )
        trade = next(
            item for item in evaluated.trades
            if item["variant_id"] == PREVIOUS_HIGH_CONFIRMATION
        )

        self.assertEqual(event["status"], "confirmed")
        self.assertEqual(event["reason"], "previous_high_intrabar_break")
        self.assertEqual(event["entry_ts"], raw.loc[event_index + 2, "ts"])
        self.assertEqual(float(trade["entry_price"]), 100.5)

    def test_confirmation_without_break_is_logged_as_suppressed(self) -> None:
        raw = _bars()
        event_index = 1590
        for index in range(event_index, event_index + 6):
            raw.loc[index, "high"] = 101.0 - (index - event_index) * 0.1
        model = _EventModel([raw.loc[event_index, "ts"]])

        evaluated = evaluate_paper_variants(raw, model)
        event = next(
            item for item in evaluated.events
            if item["variant_id"] == PREVIOUS_HIGH_CONFIRMATION
        )

        self.assertEqual(event["status"], "suppressed")
        self.assertEqual(event["reason"], "no_previous_high_break")


if __name__ == "__main__":
    unittest.main()
