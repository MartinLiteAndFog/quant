from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pandas as pd


def _load_subject():
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_ct_multi_asset.py"
    spec = importlib.util.spec_from_file_location("validate_ct_multi_asset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


subject = _load_subject()


def _config(root: str = "data/raw") -> dict:
    return {
        "schema_version": 1,
        "required_assets": ["SOL-USDT", "BTC-USDT", "ETH-USDT"],
        "data": {
            "root": root,
            "timeframe": "1m",
            "required_columns": ["ts", "open", "high", "low", "close", "volume"],
            "required_start": "2024-01-01T00:00:00Z",
            "required_end": "2024-01-01T00:09:00Z",
            "minimum_coverage_ratio": 1.0,
            "maximum_gap_minutes": 1,
        },
        "methodology": {
            "development_start": "2024-01-01T00:00:00Z",
            "outer_holdout_start": "2025-01-01T00:00:00Z",
            "outer_holdout_end": "2025-06-29T23:59:00Z",
            "train_days": 120,
            "test_days": 30,
            "step_days": 30,
            "box_bps_grid": [4.0, 5.33],
            "lookback_grid": [220],
            "selection_rule": "best_development_oos_compounded_return",
            "kill_switch": "ma100_25",
            "kill_switch_publication_lag_days": 1,
            "ttp_trail_pct": 0.01,
            "min_sl_pct": 0.0095,
            "max_sl_pct": 0.03,
            "swing_lookback": 50,
        },
        "execution": {"roundtrip_cost_bps": [12.0, 20.0], "selection_cost_bps": 12.0},
        "pass_criteria": {
            "minimum_outer_folds": 6,
            "minimum_total_return_pct_each_cost": 0.0,
            "minimum_max_drawdown_pct_each_cost": -60.0,
        },
        "output_root": "outputs",
    }


def _write_canonical(root: Path, asset: str, filename: str = "bars.parquet") -> Path:
    path = root / "data" / "raw" / "exchange=test" / f"symbol={asset}" / "timeframe=1m" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = pd.date_range("2024-01-01T00:00:00Z", periods=10, freq="1min")
    close = pd.Series(range(100, 110), dtype=float)
    pd.DataFrame(
        {
            "ts": ts,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close + 0.5,
            "volume": 10.0,
        }
    ).to_parquet(path, index=False)
    return path


def test_data_gate_discovers_and_validates_canonical_input(tmp_path, monkeypatch):
    monkeypatch.setattr(subject, "_REPO", tmp_path)
    path = _write_canonical(tmp_path, "SOL-USDT")

    discovered = subject.discover_asset_input("SOL-USDT", _config()["data"])
    validated = subject.validate_asset_input(discovered, _config()["data"])

    assert discovered["status"] == "discovered"
    assert validated["status"] == "passed"
    assert validated["coverage_ratio"] == 1.0
    assert validated["sha256"] == subject.sha256_file(path)


def test_data_gate_fails_closed_on_missing_or_ambiguous_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(subject, "_REPO", tmp_path)
    data_config = _config()["data"]
    assert subject.discover_asset_input("BTC-USDT", data_config)["status"] == "missing"

    _write_canonical(tmp_path, "ETH-USDT", "one.parquet")
    _write_canonical(tmp_path, "ETH-USDT", "two.parquet")
    result = subject.discover_asset_input("ETH-USDT", data_config)
    assert result["status"] == "ambiguous"
    assert len(result["eligible_paths"]) == 2


def test_development_selection_is_complete_and_deterministic():
    metrics = pd.DataFrame(
        [
            {"fold": 0, "box_bps": 4.0, "lookback": 220, "test_total_return_pct": 10.0},
            {"fold": 1, "box_bps": 4.0, "lookback": 220, "test_total_return_pct": -5.0},
            {"fold": 0, "box_bps": 5.33, "lookback": 220, "test_total_return_pct": 2.0},
            {"fold": 1, "box_bps": 5.33, "lookback": 220, "test_total_return_pct": 3.0},
        ]
    )
    metrics["selection_cost_bps"] = 12.0
    metrics["kill_switch"] = "ma100_25"
    selected, summary = subject._select_development_config(
        metrics, 2, _config()["methodology"], 12.0
    )
    assert selected["box_bps"] == 5.33
    assert selected["lookback"] == 220
    assert selected["selection_rule"] == "best_development_oos_compounded_return"
    assert len(summary) == 2


def test_outer_windows_are_non_overlapping_and_exclude_development():
    windows = subject.make_windows(
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-03-01T23:59:00Z"),
        train_days=120,
        test_days=30,
        step_days=30,
    )
    assert len(windows) == 2
    assert windows[0][2] + pd.Timedelta(minutes=1) == windows[1][1]
    assert windows[0][0] == pd.Timestamp("2024-09-03T00:00:00Z")


def test_aggregate_pass_requires_every_cost_criterion():
    passing = {
        "12": {"total_return_pct": 10.0, "max_drawdown_pct": -20.0},
        "20": {"total_return_pct": 1.0, "max_drawdown_pct": -59.0},
    }
    ok, failures = subject._asset_passes(passing, 6, _config()["pass_criteria"])
    assert ok and not failures
    passing["20"]["total_return_pct"] = -0.01
    ok, failures = subject._asset_passes(passing, 6, _config()["pass_criteria"])
    assert not ok
    assert any("20bps return" in failure for failure in failures)


def test_outer_results_cannot_influence_development_selected_config(tmp_path, monkeypatch):
    monkeypatch.setattr(subject, "_REPO", tmp_path)
    development_metrics = pd.DataFrame(
        [
            {
                "fold": fold,
                "box_bps": box,
                "lookback": 220,
                "test_total_return_pct": result,
                "selection_cost_bps": 12.0,
                "kill_switch": "ma100_25",
            }
            for fold, results in enumerate(((10.0, 1.0), (5.0, 2.0)))
            for box, result in zip((4.0, 5.33), results)
        ]
    )
    monkeypatch.setattr(subject, "_inner_rows", lambda *args, **kwargs: development_metrics.copy())
    outer_return = {"value": 100.0}

    def fake_outer(*args, **kwargs):
        rows = pd.DataFrame(
            [
                {"fold": 0, "roundtrip_cost_bps": 12.0},
                {"fold": 0, "roundtrip_cost_bps": 20.0},
            ]
        )
        metrics = {
            cost: {
                "n_trades": 1,
                "total_return_pct": outer_return["value"],
                "max_drawdown_pct": -1.0,
                "win_rate": 1.0,
            }
            for cost in ("12", "20")
        }
        return rows, metrics

    monkeypatch.setattr(subject, "_outer_evaluation", fake_outer)
    config = _config()
    config["methodology"]["outer_holdout_start"] = "2024-06-29T00:00:00Z"
    config["methodology"]["outer_holdout_end"] = "2024-12-25T23:59:00Z"
    first = subject.run_asset(
        "SOL-USDT", {"path": "unused.parquet"}, config, tmp_path / "first", workers=1
    )
    outer_return["value"] = -100.0
    second = subject.run_asset(
        "SOL-USDT", {"path": "unused.parquet"}, config, tmp_path / "second", workers=1
    )

    assert first["selected_config"] == second["selected_config"]
    assert first["selected_config"]["box_bps"] == 4.0
    assert first["status"] == "passed"
    assert second["status"] == "failed_criteria"


def test_main_writes_not_validated_manifest_when_btc_eth_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(subject, "_REPO", tmp_path)
    _write_canonical(tmp_path, "SOL-USDT")
    config = _config()
    config["methodology"]["outer_holdout_start"] = "2024-01-01T00:01:00Z"
    config["methodology"]["outer_holdout_end"] = "2024-01-01T00:09:00Z"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"

    exit_code = subject.main(
        ["--config", str(config_path), "--mode", "check", "--manifest", str(manifest_path)]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert exit_code == 2
    assert manifest["status"] == "blocked_data"
    assert manifest["aggregate"]["status"] == "not_validated"
    assert manifest["assets"]["SOL-USDT"]["status"] == "data_ready"
    assert manifest["assets"]["BTC-USDT"]["status"] == "blocked_data"
    assert manifest["assets"]["ETH-USDT"]["status"] == "blocked_data"
