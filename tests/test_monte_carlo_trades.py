from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.monte_carlo_trades import (
    bootstrap_indices,
    bootstrap_paths,
    path_stats,
    run_monte_carlo,
    summarize_results,
)


def _trade_frame(n: int = 24) -> pd.DataFrame:
    trade_id = np.arange(n)
    return pd.DataFrame(
        {
            "trade_id": trade_id,
            "pnl_pct": np.where(trade_id % 3 == 0, 0.02, -0.005),
            "execution_cost_bps": trade_id * 10 + 3,
            "maker_filled": trade_id % 2 == 0,
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="6h", tz="UTC"),
        }
    )


def test_iid_bootstrap_matches_legacy_rng_choice() -> None:
    returns = np.array([-0.03, 0.01, 0.04, -0.02, 0.005])

    actual = bootstrap_paths(returns, n_paths=7, seed=123)
    expected = np.random.default_rng(123).choice(returns, size=(7, len(returns)), replace=True)

    np.testing.assert_array_equal(actual, expected)


def test_moving_block_is_deterministic_and_preserves_serial_blocks() -> None:
    n_obs = 11
    block_length = 4

    first = bootstrap_indices(n_obs, n_paths=8, seed=17, method="moving-block", block_length=block_length)
    second = bootstrap_indices(n_obs, n_paths=8, seed=17, method="moving_block", block_length=block_length)

    np.testing.assert_array_equal(first, second)
    for row in first:
        for start in range(0, n_obs, block_length):
            block = row[start : start + block_length]
            if len(block) > 1:
                np.testing.assert_array_equal(np.diff(block) % n_obs, np.ones(len(block) - 1))


def test_sampled_indices_keep_return_cost_and_fill_fields_aligned() -> None:
    trades = _trade_frame(24)
    sampled_indices = bootstrap_indices(24, n_paths=6, seed=4, method="moving_block", block_length=5)
    sampled = trades.iloc[sampled_indices.ravel()].reset_index(drop=True)

    np.testing.assert_array_equal(
        sampled["execution_cost_bps"].to_numpy(),
        sampled["trade_id"].to_numpy() * 10 + 3,
    )
    np.testing.assert_array_equal(
        sampled["maker_filled"].to_numpy(),
        sampled["trade_id"].to_numpy() % 2 == 0,
    )


def test_grouped_bootstrap_samples_complete_contiguous_runs() -> None:
    labels = np.repeat(["fold-a", "fold-b", "fold-c"], 4)
    expected_runs = {tuple(range(0, 4)), tuple(range(4, 8)), tuple(range(8, 12))}

    sampled = bootstrap_indices(12, n_paths=10, seed=8, method="grouped", groups=labels)

    for row in sampled:
        assert {tuple(row[start : start + 4]) for start in range(0, 12, 4)} <= expected_runs


def test_run_is_deterministic_derives_day_groups_and_emits_risk_summary(tmp_path) -> None:
    trades_path = tmp_path / "trades.parquet"
    _trade_frame().to_parquet(trades_path, index=False)

    first = run_monte_carlo(
        trades_path,
        n_paths=30,
        seed=31,
        method="grouped",
        group_by="day",
        scenario_cost_bps_range=(2.0, 7.0),
    )
    second = run_monte_carlo(
        trades_path,
        n_paths=30,
        seed=31,
        method="grouped",
        group_by="day",
        scenario_cost_bps_range=(2.0, 7.0),
    )

    pd.testing.assert_frame_equal(first, second)
    assert first.attrs["method"] == "grouped"
    assert first.attrs["group_by"] == "day(timestamp)"
    assert first["scenario_cost_bps"].between(2.0, 7.0).all()
    assert "the same sampled indices apply to all trade-row fields" in first.attrs["assumptions"]
    summary = summarize_results(first)
    assert set(summary) == {
        "return_p5_pct",
        "return_p50_pct",
        "return_p95_pct",
        "drawdown_p5_pct",
        "drawdown_p50_pct",
        "drawdown_p95_pct",
        "probability_of_ruin",
        "probability_of_positive_return",
    }
    assert 0.0 <= summary["probability_of_ruin"] <= 1.0
    assert 0.0 <= summary["probability_of_positive_return"] <= 1.0


def test_default_run_preserves_iid_sampling_and_fixed_cost_stress(tmp_path) -> None:
    trades = _trade_frame()
    trades_path = tmp_path / "trades.parquet"
    trades.to_parquet(trades_path, index=False)

    baseline = run_monte_carlo(trades_path, n_paths=5, seed=9)
    stressed = run_monte_carlo(
        trades_path,
        n_paths=5,
        seed=9,
        scenario_cost_bps_range=(10.0, 10.0),
    )
    expected_paths = bootstrap_paths(trades["pnl_pct"].to_numpy(), n_paths=5, seed=9)
    expected_first = path_stats(expected_paths[0])

    assert baseline.attrs["method"] == "iid"
    assert baseline.attrs["block_length"] == 1
    assert baseline.loc[0, "total_return_pct"] == pytest.approx(expected_first["total_return_pct"])
    assert baseline.loc[0, "max_drawdown_pct"] == pytest.approx(expected_first["max_drawdown_pct"])
    np.testing.assert_array_equal(stressed["scenario_cost_bps"], np.full(5, 10.0))
    assert (stressed["total_return_pct"] < baseline["total_return_pct"]).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_obs": 0}, "at least one observation"),
        ({"n_obs": 3, "n_paths": 0}, "n_paths must be positive"),
        ({"n_obs": 3, "block_length": 0}, "block_length must be positive"),
        ({"n_obs": 3, "method": "unknown"}, "Unknown bootstrap method"),
        ({"n_obs": 3, "method": "grouped"}, "requires group labels"),
        ({"n_obs": 3, "method": "grouped", "groups": ["a", "b"]}, "length must match"),
        ({"n_obs": 3, "method": "grouped", "groups": ["a", None, "b"]}, "missing values"),
    ],
)
def test_bootstrap_rejects_invalid_inputs(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        bootstrap_indices(**kwargs)


def test_run_rejects_too_few_returns_and_missing_group(tmp_path) -> None:
    short_path = tmp_path / "short.parquet"
    _trade_frame(19).to_parquet(short_path, index=False)
    with pytest.raises(ValueError, match="minimum 20"):
        run_monte_carlo(short_path)

    full_path = tmp_path / "full.parquet"
    _trade_frame().drop(columns="timestamp").to_parquet(full_path, index=False)
    with pytest.raises(ValueError, match="No fold/day/regime"):
        run_monte_carlo(full_path, method="grouped", group_by="auto")


def test_path_stats_counts_initial_loss_in_drawdown_and_validates_returns() -> None:
    stats = path_stats(np.array([-0.25, 0.10]), ruin_equity_fraction=0.80)

    assert stats["max_drawdown_pct"] == pytest.approx(-25.0)
    assert stats["ruined"] is True
    with pytest.raises(ValueError, match="below -100%"):
        path_stats(np.array([-1.01]))
