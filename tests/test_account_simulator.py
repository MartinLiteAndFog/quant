from __future__ import annotations

import pandas as pd
import pytest

from quant.backtest.account_simulator import (
    AccountPolicy,
    AccountSimulator,
    ContractSpec,
    MarginMode,
    simulate_execution_trades,
)
from quant.backtest.fill_model import ExecutionCostScenario, FillModelParams, apply_execution_scenario
from quant.backtest.funding import FundingRate


def _contract(*, maintenance: float = 0.05, min_notional: float = 5.0) -> ContractSpec:
    return ContractSpec(
        symbol="SOLUSDT",
        tick_size=0.1,
        lot_size=0.01,
        min_notional=min_notional,
        maintenance_margin_rate=maintenance,
        max_leverage=20.0,
    )


def _account(
    *,
    cash: float = 1_000.0,
    leverage: float = 2.0,
    mode: MarginMode = MarginMode.CROSS,
    maintenance: float = 0.05,
    liquidation_fraction: float = 0.5,
    liquidation_fee: float = 0.0,
) -> AccountSimulator:
    return AccountSimulator(
        initial_cash=cash,
        contracts=[_contract(maintenance=maintenance)],
        policy=AccountPolicy(
            margin_mode=mode,
            leverage=leverage,
            position_size_fraction=1.0,
            liquidation_fraction=liquidation_fraction,
            liquidation_fee_rate=liquidation_fee,
        ),
    )


def test_sizing_rounding_per_fill_fees_and_auditable_ledger() -> None:
    account = _account(leverage=2.0)

    # Buys round up to the price tick; quantity floors to the lot size.  Sizing
    # reserves the entry fee, so an all-cash order remains admissible.
    quantity = account.size_order("SOLUSDT", 100.01, side=1, fee_rate=0.001)
    assert quantity == pytest.approx(19.94)
    entry = account.execute_fill(
        timestamp="2026-01-01T00:00:00Z",
        symbol="SOLUSDT",
        quantity=quantity,
        price=100.01,
        fee_rate=0.001,
        note="entry",
    )
    assert entry.price == 100.1
    assert entry.fee == pytest.approx(19.94 * 100.1 * 0.001)
    assert account.locked_margin == pytest.approx(19.94 * 100.1 / 2.0)
    assert account.available_cash >= -1e-9

    exit_entry = account.execute_fill(
        timestamp="2026-01-01T01:00:00Z",
        symbol="SOLUSDT",
        quantity=-quantity,
        price=110.09,
        fee_rate=0.002,
        note="exit",
    )
    assert exit_entry.price == 110.0
    assert exit_entry.realized_pnl == pytest.approx(quantity * (110.0 - 100.1))
    assert account.fees_total == pytest.approx(entry.fee + exit_entry.fee)
    assert account.open_notional == 0.0
    assert list(account.ledger_frame()["sequence"]) == [1, 2]
    assert set(account.ledger_frame()["event_type"]) == {"fill"}


def test_tick_lot_min_notional_and_leverage_constraints_are_explicit() -> None:
    account = _account(cash=10.0, leverage=2.0)

    assert account.normalize_price("SOLUSDT", 100.01, side=1) == 100.1
    assert account.normalize_price("SOLUSDT", 100.09, side=-1) == 100.0
    assert account.normalize_quantity("SOLUSDT", -1.239) == pytest.approx(-1.23)
    assert account.size_order("SOLUSDT", 10_000.0, side=1) == 0.0
    with pytest.raises(ValueError, match="max"):
        _account(leverage=21.0)


def test_contract_symbols_must_be_unique() -> None:
    with pytest.raises(ValueError, match="unique"):
        AccountSimulator(initial_cash=1_000.0, contracts=[_contract(), _contract()])


def test_cross_policy_partially_liquidates_until_account_is_healthy() -> None:
    account = _account(leverage=5.0, maintenance=0.10, liquidation_fraction=0.5)
    account.execute_fill(
        timestamp="2026-01-01T00:00:00Z",
        symbol="SOLUSDT",
        quantity=40.0,
        price=100.0,
    )

    events = account.mark(timestamp="2026-01-01T01:00:00Z", symbol="SOLUSDT", price=80.0)

    assert [event.event_type for event in events] == ["partial_liquidation"]
    assert account.positions["SOLUSDT"].quantity == pytest.approx(20.0)
    assert account.equity > account.maintenance_margin
    assert account.liquidation_count == 1


def test_isolated_policy_checks_position_margin_and_emits_partial_then_full_events() -> None:
    account = _account(
        leverage=5.0,
        mode=MarginMode.ISOLATED,
        maintenance=0.10,
        liquidation_fraction=0.5,
    )
    account.execute_fill(
        timestamp="2026-01-01T00:00:00Z",
        symbol="SOLUSDT",
        quantity=40.0,
        price=100.0,
    )

    events = account.mark(timestamp="2026-01-01T01:00:00Z", symbol="SOLUSDT", price=88.0)

    assert events[0].event_type == "partial_liquidation"
    assert events[-1].event_type == "liquidation"
    assert "SOLUSDT" not in account.positions
    assert all(event.note == "isolated" for event in events)


def test_account_funding_uses_mark_notional_and_can_trigger_margin_check() -> None:
    account = _account(leverage=2.0)
    account.execute_fill(
        timestamp="2026-01-01T00:00:00Z",
        symbol="SOLUSDT",
        quantity=-2.0,
        price=100.0,
    )

    entry = account.apply_funding(
        FundingRate("SOLUSDT", "2026-01-01T01:00:00Z", 0.001, 110.0)
    )

    assert entry is not None
    assert entry.funding == pytest.approx(0.22)
    assert account.wallet_balance == pytest.approx(1_000.22)
    assert account.funding_total == pytest.approx(0.22)


def test_execution_scenario_adapter_reconstructs_per_fill_costs_and_funding() -> None:
    account = AccountSimulator(
        initial_cash=1_000.0,
        contracts=[_contract()],
        policy=AccountPolicy(leverage=1.0, position_size_fraction=0.5),
    )
    raw_trades = pd.DataFrame(
        {
            "entry_ts": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T03:00:00Z"]),
            "exit_ts": pd.to_datetime(["2026-01-01T02:00:00Z", "2026-01-01T04:00:00Z"]),
            "side": [1, -1],
            "entry_px": [100.0, 100.0],
            "exit_px": [110.0, 90.0],
            "exit_event": ["tp_exit", "sl_exit"],
        }
    )
    scenario = ExecutionCostScenario(
        name="stress",
        seed=11,
        params=FillModelParams(
            l1_bps=0.0,
            l2_bps=0.0,
            entry_fallback_bps=0.0,
            tp_maker_bps=0.0,
            tp_fallback_bps=0.0,
            sl_taker_bps=0.0,
            fee_bps_roundtrip=0.002,
        ),
    )
    trades = apply_execution_scenario(raw_trades, scenario)
    trades.loc[1, "entry_fill_status"] = "missed"
    trades.loc[1, "entry_fill_fraction"] = 0.0
    # This intentionally impossible value proves the account adapter rebuilds
    # economics from fills instead of double-counting pnl_pct.
    trades["pnl_pct_adj"] = -999.0
    rates = [
        FundingRate("SOLUSDT", "2026-01-01T00:00:00Z", 0.001, 100.0),
        FundingRate("SOLUSDT", "2026-01-01T02:00:00Z", 0.001, 110.0),
    ]

    result = simulate_execution_trades(
        trades,
        account=account,
        symbol="SOLUSDT",
        funding_rates=rates,
    )

    assert list(result.trades["status"]) == ["closed", "missed"]
    assert list(result.ledger["event_type"]) == ["fill", "funding", "fill"]
    # Entry-time funding is included, exit-time funding is excluded.
    assert result.summary["funding"] < 0.0
    assert result.summary["fees"] > 0.0
    assert result.summary["equity"] > 1_000.0


def test_execution_adapter_rejects_overlapping_positions_per_symbol() -> None:
    account = _account(leverage=1.0)
    trades = pd.DataFrame(
        {
            "entry_ts": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"]),
            "exit_ts": pd.to_datetime(["2026-01-01T02:00:00Z", "2026-01-01T03:00:00Z"]),
            "side": [1, 1],
            "entry_px": [100.0, 100.0],
            "exit_px": [101.0, 101.0],
        }
    )

    with pytest.raises(ValueError, match="non-overlapping"):
        simulate_execution_trades(trades, account=account, symbol="SOLUSDT")
