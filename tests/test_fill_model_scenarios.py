from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import scripts.walk_forward_ct_renko as walk_forward
from quant.backtest.fill_model import (
    ExecutionCostScenario,
    FillModelParams,
    apply_execution_scenario,
    apply_fill_model,
    fixed_total_cost_scenarios,
)


def _trades(n: int = 2) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "entry_px": [100.0, 100.0],
            "exit_px": [110.0, 90.0],
            "side": [1, -1],
            "exit_event": ["tp_exit", "sl_exit"],
        }
    )
    return pd.concat([base] * ((n + 1) // 2), ignore_index=True).iloc[:n].copy()


def test_fixed_cost_grid_subtracts_exact_total_cost() -> None:
    costs = [12, 14, 16, 20, 24]

    for cost, scenario in zip(costs, fixed_total_cost_scenarios(costs)):
        adjusted = apply_execution_scenario(_trades(), scenario)
        np.testing.assert_allclose(adjusted["pnl_pct_adj"], 0.10 - cost / 10_000.0)
        np.testing.assert_allclose(adjusted["execution_cost_pct"], cost / 10_000.0)
        assert scenario.name == f"fixed_{cost}bps"
        assert scenario.source == "fixed_total_cost_grid"


def test_default_seeded_fill_sequence_remains_backwards_compatible() -> None:
    seed = 7
    params = FillModelParams()
    expected_rng = np.random.default_rng(seed)
    entry_draws = expected_rng.random(8)
    exit_draws = expected_rng.random(8)
    expected_entry = np.where(
        entry_draws < params.p_entry_l1,
        "L1",
        np.where(entry_draws < params.p_entry_l1 + params.p_entry_l2, "L2", "FB"),
    )
    tp_mask = np.array([True, False] * 4)
    expected_exit = np.where(
        ~tp_mask,
        "SL_MK",
        np.where(exit_draws < params.p_tp_maker, "PO", "FB"),
    )

    adjusted = apply_fill_model(_trades(8), params=params, seed=seed)

    np.testing.assert_array_equal(adjusted["fill_mode_entry"], expected_entry)
    np.testing.assert_array_equal(adjusted["fill_mode_exit"], expected_exit)


def test_seeded_participation_is_reproducible_and_records_missed_partial() -> None:
    scenario = ExecutionCostScenario(
        name="calibrated",
        seed=123,
        params=FillModelParams(
            p_entry_missed=0.20,
            p_entry_partial=0.50,
            partial_fill_fraction=0.40,
        ),
    )

    first = apply_execution_scenario(_trades(500), scenario)
    second = apply_execution_scenario(_trades(500), scenario)
    different = apply_execution_scenario(_trades(500), scenario, seed=124)

    pd.testing.assert_frame_equal(first, second)
    assert not first["entry_fill_status"].equals(different["entry_fill_status"])
    assert {"missed", "partial", "filled"} <= set(first["entry_fill_status"])
    assert first.loc[first["entry_fill_status"] == "missed", "pnl_pct_adj"].isna().all()
    np.testing.assert_allclose(
        first.loc[first["entry_fill_status"] == "partial", "entry_fill_fraction"],
        0.40,
    )


def test_partial_fill_scales_account_return_and_missed_fill_is_excluded() -> None:
    partial_params = FillModelParams(
        l1_bps=0.0,
        l2_bps=0.0,
        entry_fallback_bps=0.0,
        tp_maker_bps=0.0,
        tp_fallback_bps=0.0,
        sl_taker_bps=0.0,
        fee_bps_roundtrip=0.0012,
        p_entry_partial=1.0,
        partial_fill_fraction=0.25,
    )
    missed_params = FillModelParams(p_entry_missed=1.0)

    partial = apply_fill_model(_trades(), partial_params, seed=1)
    missed = apply_fill_model(_trades(), missed_params, seed=1)

    np.testing.assert_allclose(partial["pnl_pct_adj"], (0.10 - 0.0012) * 0.25)
    assert missed["pnl_pct_adj"].isna().all()


def test_maker_taker_mix_selects_configured_roundtrip_fee() -> None:
    params = FillModelParams(
        l1_bps=0.0,
        l2_bps=0.0,
        entry_fallback_bps=0.0,
        tp_maker_bps=0.0,
        tp_fallback_bps=0.0,
        sl_taker_bps=0.0,
        p_entry_l1=1.0,
        p_entry_l2=0.0,
        p_entry_fb=0.0,
        p_tp_maker=1.0,
        p_tp_fb=0.0,
        fee_maker_bps_roundtrip=0.0004,
        fee_taker_bps_roundtrip=0.0012,
    )

    adjusted = apply_fill_model(_trades(), params=params, seed=1)

    assert adjusted.loc[0, "fill_mode_entry"] == "L1"
    assert adjusted.loc[0, "fill_mode_exit"] == "PO"
    assert adjusted.loc[0, "fee_pct_roundtrip"] == pytest.approx(0.0004)
    # Maker entry plus taker stop exit uses the midpoint of roundtrip costs.
    assert adjusted.loc[1, "fill_mode_exit"] == "SL_MK"
    assert adjusted.loc[1, "fee_pct_roundtrip"] == pytest.approx(0.0008)


def test_scenario_json_roundtrips_assumptions(tmp_path) -> None:
    scenario_file = tmp_path / "execution.json"
    scenario_file.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "name": "observed_conservative",
                        "seed": 99,
                        "source": "live_fill_calibration_2026q2",
                        "params": {
                            "p_entry_l1": 0.2,
                            "p_entry_l2": 0.3,
                            "p_entry_fb": 0.5,
                            "p_entry_missed": 0.1,
                            "p_entry_partial": 0.2,
                            "partial_fill_fraction": 0.6,
                            "fee_maker_bps_roundtrip": 0.0004,
                            "fee_taker_bps_roundtrip": 0.0012,
                            "slippage_bps_roundtrip": 0.0002,
                        },
                    }
                ]
            }
        )
    )

    scenarios = walk_forward.load_execution_scenarios("12,24", str(scenario_file))

    assert [scenario.name for scenario in scenarios] == [
        "fixed_12bps",
        "fixed_24bps",
        "observed_conservative",
    ]
    assert scenarios[-1].to_dict()["params"]["p_entry_missed"] == 0.1
    assert scenarios[-1].to_dict()["source"] == "live_fill_calibration_2026q2"


def test_run_fold_emits_one_row_per_cost_scenario_without_double_fee(monkeypatch) -> None:
    train_start = pd.Timestamp("2026-01-01", tz="UTC")
    test_start = train_start + pd.Timedelta(days=2)
    test_end = test_start + pd.Timedelta(days=2)
    bar_ts = pd.date_range(train_start, test_end, freq="1min")
    bars = pd.DataFrame(
        {
            "ts": bar_ts,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": np.linspace(100.0, 101.0, len(bar_ts)),
        }
    )
    brick_ts = pd.date_range(train_start, test_end, periods=2500)
    bricks = pd.DataFrame(
        {
            "ts": brick_ts,
            "open": 100.0,
            "close": 100.1,
        }
    )

    monkeypatch.setattr(walk_forward, "_load_bars", lambda _: bars)
    monkeypatch.setattr(walk_forward, "build_renko_fixed", lambda *args, **kwargs: bricks)
    monkeypatch.setattr(
        walk_forward,
        "fast_imba_signals",
        lambda *args, **kwargs: (np.array([1]), np.array([1])),
    )

    def fake_trades(*args, **kwargs):
        assert kwargs["fee_bps"] == 0.0
        return {
            "entry_idx": np.array([200, 800, 1600, 2200]),
            "exit_idx": np.array([300, 900, 1700, 2300]),
            "side": np.array([1, -1, 1, -1]),
            "entry_px": np.full(4, 100.0),
            "exit_px": np.array([110.0, 90.0, 110.0, 90.0]),
            "pnl_pct": np.full(4, 0.10),
            "exit_code": np.array([0, 2, 1, 2]),
        }

    monkeypatch.setattr(walk_forward, "fast_flip_trades", fake_trades)

    rows = walk_forward.run_fold(
        3,
        "unused.parquet",
        train_start,
        test_start,
        test_end,
        [5.0],
        [220],
        fee_bps=999.0,
        ttp_trail_pct=0.01,
        min_sl_pct=0.0095,
        max_sl_pct=0.03,
        swing_lookback=50,
        execution_scenarios=fixed_total_cost_scenarios([12, 24]),
    )

    assert [row["execution_scenario"] for row in rows] == ["fixed_12bps", "fixed_24bps"]
    assert all(row["train_n_trades"] == 2 for row in rows)
    assert all(row["test_n_trades"] == 2 for row in rows)
    assert rows[0]["train_total_return_pct"] == pytest.approx(((1.0 + 0.10 - 0.0012) ** 2 - 1.0) * 100.0)
    assert rows[1]["test_total_return_pct"] == pytest.approx(((1.0 + 0.10 - 0.0024) ** 2 - 1.0) * 100.0)


@pytest.mark.parametrize(
    "params",
    [
        FillModelParams(p_entry_l1=0.8, p_entry_l2=0.3, p_entry_fb=0.0),
        FillModelParams(p_entry_missed=0.7, p_entry_partial=0.4),
        FillModelParams(partial_fill_fraction=1.0),
    ],
)
def test_invalid_probability_assumptions_fail_fast(params: FillModelParams) -> None:
    with pytest.raises(ValueError):
        apply_fill_model(_trades(), params=params, seed=1)
