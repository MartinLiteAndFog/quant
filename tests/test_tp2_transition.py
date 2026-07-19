from __future__ import annotations

import pytest
import pandas as pd

from quant.strategies.follow_tp2_engine import TP2Params, run_follow_tp2_state_machine
from quant.strategies.tp2_transition import (
    SAME_BAR_PRIORITY,
    TP2Observation,
    TP2TradeState,
    TP2TransitionPolicy,
    transition_tp2,
)


POLICY = TP2TransitionPolicy(
    tp1_pct=0.04,
    tp2_pct=0.08,
    tp1_frac=0.5,
    be_after_tp1=True,
)


def _long(**overrides: object) -> TP2TradeState:
    values = {
        "side": 1,
        "entry_price": 100.0,
        "initial_qty": 10.0,
        "remaining_qty": 10.0,
        "stop_price": 95.0,
        "leg_id": "leg-1",
    }
    values.update(overrides)
    return TP2TradeState(**values)


def _obs(**overrides: object) -> TP2Observation:
    values = {"close": 100.0, "high": 100.0, "low": 100.0, "timestamp": "t1"}
    values.update(overrides)
    return TP2Observation(**values)


def test_same_bar_priority_is_public_and_protective() -> None:
    assert SAME_BAR_PRIORITY == (
        "regime_exit",
        "be_exit",
        "sl_exit",
        "tp2_exit",
        "tp1_exit",
        "signal_exit",
    )

    result = transition_tp2(
        _long(),
        _obs(close=90.0, high=110.0, low=90.0, signal=-1),
        POLICY,
    )

    assert [event.kind for event in result.events] == ["sl_exit"]
    assert result.events[0].price == 95.0
    assert result.state.is_flat


def test_prearmed_break_even_beats_tp2_on_same_bar() -> None:
    state = _long(remaining_qty=5.0, tp1_filled_qty=5.0, be_armed=True)

    result = transition_tp2(state, _obs(high=110.0, low=99.0), POLICY)

    assert [event.kind for event in result.events] == ["be_exit"]
    assert result.events[0].price == 100.0


def test_tp2_beats_tp1_on_same_bar_before_break_even_is_armed() -> None:
    result = transition_tp2(_long(), _obs(high=109.0, low=100.0), POLICY)

    assert [event.kind for event in result.events] == ["tp2_exit"]
    assert result.events[0].qty == 10.0


def test_partial_tp1_fills_accumulate_before_arming_break_even() -> None:
    first = transition_tp2(
        _long(),
        _obs(high=105.0, low=100.0, tp1_fill_qty=2.0, timestamp="t1"),
        POLICY,
    )

    assert [event.kind for event in first.events] == ["tp1_exit"]
    assert first.state.remaining_qty == 8.0
    assert first.state.tp1_filled_qty == 2.0
    assert not first.state.be_armed

    second = transition_tp2(
        first.state,
        _obs(high=105.0, low=100.0, tp1_fill_qty=3.0, timestamp="t2"),
        POLICY,
    )

    assert [event.kind for event in second.events] == ["tp1_exit", "be_armed"]
    assert second.state.remaining_qty == 5.0
    assert second.state.tp1_filled_qty == 5.0
    assert second.state.be_armed


def test_tp1_arms_break_even_only_for_next_observation() -> None:
    first = transition_tp2(_long(), _obs(high=105.0, low=99.0), POLICY)

    assert [event.kind for event in first.events] == ["tp1_exit", "be_armed"]
    assert not first.state.is_flat

    second = transition_tp2(first.state, _obs(high=101.0, low=99.0, timestamp="t2"), POLICY)

    assert [event.kind for event in second.events] == ["be_exit"]
    assert second.state.is_flat


def test_opposite_signal_closes_then_opens_new_leg() -> None:
    result = transition_tp2(
        _long(),
        _obs(signal=-1, entry_qty=7.0, entry_leg_id="leg-2"),
        POLICY,
    )

    assert [event.kind for event in result.events] == ["signal_exit", "entry"]
    assert result.state.side == -1
    assert result.state.remaining_qty == 7.0
    assert result.state.leg_id == "leg-2"


def test_regime_exit_preempts_every_other_event() -> None:
    result = transition_tp2(
        _long(),
        _obs(
            close=90.0,
            high=110.0,
            low=90.0,
            signal=-1,
            regime_on=False,
            regime_forces_flat=True,
        ),
        POLICY,
    )

    assert [event.kind for event in result.events] == ["regime_exit"]
    assert result.events[0].price == 90.0


def test_close_event_exposes_per_leg_fees_and_cash_accounting() -> None:
    policy = TP2TransitionPolicy(
        tp1_pct=0.04,
        tp2_pct=0.08,
        tp1_frac=0.5,
        entry_fee_rate=0.001,
        exit_fee_rate=0.002,
    )

    result = transition_tp2(_long(), _obs(high=109.0), policy)
    event = result.events[0]

    assert event.kind == "tp2_exit"
    assert event.gross_pnl == pytest.approx(80.0)
    assert event.allocated_entry_fee == pytest.approx(1.0)
    assert event.exit_fee == pytest.approx(2.16)
    assert event.net_pnl == pytest.approx(76.84)
    assert event.cash_flow == pytest.approx(77.84)


def test_follow_adapter_preserves_tp1_be_and_legacy_roundtrip_fee_contract() -> None:
    timestamps = pd.date_range("2026-01-01", periods=3, freq="min", tz="UTC")
    bars = pd.DataFrame(
        {
            "ts": timestamps,
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 105.0, 101.0],
            "low": [100.0, 99.0, 99.0],
            "close": [100.0, 103.0, 100.0],
        }
    )
    signals = pd.DataFrame({"ts": [timestamps[0]], "signal": [1]})

    positions, events, terminal = run_follow_tp2_state_machine(
        bars,
        signals,
        TP2Params(fee_bps=12.0, tp1_pct=0.04, tp2_pct=0.08, tp1_frac=0.5),
    )

    assert positions.tolist() == [1, 1, 0]
    assert events["event"].tolist() == ["entry", "tp1_exit", "be_armed", "be_exit"]
    assert events.loc[1, "pnl_pct"] == pytest.approx(0.0194)
    assert events.loc[3, "pnl_pct"] == pytest.approx(-0.0006)
    assert events.loc[1, "allocated_entry_fee"] == pytest.approx(0.06)
    assert terminal["pos"] == 0
