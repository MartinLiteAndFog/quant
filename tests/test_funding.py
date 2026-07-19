from __future__ import annotations

import pandas as pd
import pytest

from quant.backtest.funding import (
    FundingRate,
    PositionInterval,
    calculate_funding_cashflows,
    funding_cashflows_frame,
)


def test_funding_matches_exact_half_open_position_interval_and_side_sign() -> None:
    positions = [
        PositionInterval(
            symbol="SOLUSDT",
            entry_ts="2026-01-01T00:00:00Z",
            exit_ts="2026-01-01T02:00:00Z",
            side=1,
            quantity=2.0,
            entry_price=90.0,
            position_id="long",
        ),
        PositionInterval(
            symbol="SOLUSDT",
            entry_ts="2026-01-01T01:00:00Z",
            exit_ts="2026-01-01T03:00:00Z",
            side=-1,
            quantity=1.0,
            entry_price=90.0,
            position_id="short",
        ),
    ]
    rates = [
        FundingRate("SOLUSDT", pd.Timestamp("2026-01-01T00:00:00Z"), 0.001, 100.0),
        FundingRate("SOLUSDT", pd.Timestamp("2026-01-01T01:00:00Z"), 0.002, 110.0),
        FundingRate("SOLUSDT", pd.Timestamp("2026-01-01T02:00:00Z"), 0.003, 120.0),
    ]

    flows = calculate_funding_cashflows(positions, rates)

    assert [(flow.position_id, flow.timestamp.hour) for flow in flows] == [
        ("long", 0),
        ("long", 1),
        ("short", 1),
        ("short", 2),
    ]
    assert [flow.cashflow for flow in flows] == pytest.approx([-0.2, -0.44, 0.22, 0.36])


def test_negative_funding_reverses_payer_and_frame_is_stable() -> None:
    position = PositionInterval("BTCUSDT", "2026-01-01", "2026-01-02", 1, 0.1, 50_000.0)
    rate = FundingRate("BTCUSDT", "2026-01-01T08:00:00Z", -0.0001)

    frame = funding_cashflows_frame([position], [rate])

    assert list(frame.columns) == [
        "position_id",
        "symbol",
        "timestamp",
        "side",
        "quantity",
        "mark_price",
        "rate",
        "notional",
        "cashflow",
    ]
    assert frame.loc[0, "mark_price"] == 50_000.0
    assert frame.loc[0, "cashflow"] == pytest.approx(0.5)


def test_funding_inputs_reject_invalid_intervals_and_rates() -> None:
    with pytest.raises(ValueError, match="exit_ts"):
        PositionInterval("SOL", "2026-01-02", "2026-01-01", 1, 1, 100)
    with pytest.raises(ValueError, match="finite"):
        FundingRate("SOL", "2026-01-01", float("nan"), 100)
