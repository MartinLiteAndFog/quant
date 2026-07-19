from __future__ import annotations

import unittest
import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from quant.brain_forward.runtime import BrainDecision, completed_paper_trades, parse_binance_klines
from quant.brain_forward.evidence import ForwardProtocol
from quant.brain_forward import service


def bars(count: int = 1460) -> pd.DataFrame:
    ts = pd.date_range("2026-07-01", periods=count, freq="min", tz="UTC")
    close = 100.0 + np.arange(count) * 0.01
    return pd.DataFrame({
        "ts": ts, "open": close - 0.01, "high": close + 0.02, "low": close - 0.02,
        "close": close, "volume": np.full(count, 10.0), "taker_base": np.full(count, 4.0),
    })


class FakeModel:
    def __init__(self, event_ts: pd.Timestamp) -> None:
        self.event_ts = event_ts
        self.artifact_sha256 = "a" * 64

    def decision_from_feature_row(self, row: pd.DataFrame) -> BrainDecision | None:
        value = row.iloc[0]
        if value["ts"] != self.event_ts:
            return None
        return BrainDecision(
            event_ts=value["ts"], event_close=float(value["close"]), candle_range=0.03,
            expected_net_bps=5.0, active_memories=1, shock_z=3.2,
            close_position=0.1, volatility_ratio=2.2, flow_imbalance=-0.2,
        )


def protocol_for(model: FakeModel) -> ForwardProtocol:
    code_dir = Path("src/quant/brain_forward")
    return ForwardProtocol.from_dict({
        "schema_version": 1, "protocol_id": "test-forward", "candidate_id": "test-candidate",
        "symbol": "SOL-USDT", "source": "binance_spot_klines",
        "artifact_sha256": model.artifact_sha256, "candidate_spec_sha256": "1" * 64,
        "observer_code_sha256": {
            name: hashlib.sha256((code_dir / name).read_bytes()).hexdigest()
            for name in ("runtime.py", "service.py", "store.py")
        },
        "warmup_start": "2026-06-30T00:00:00Z", "evidence_start": "2026-07-01T00:00:00Z",
        "evidence_end": "2026-07-03T00:00:00Z", "checkpoint_at": "2026-07-03T00:05:00Z",
        "outcome_maturity_minutes": 5,
        "base_cost_bps": 14, "stress_cost_bps": 22, "minimum_formal_trades": 2,
        "minimum_bar_coverage": 0.99, "maximum_drawdown_bps": 1000,
        "minimum_mean_net_bps": 0, "minimum_lcb95_net_bps": 0,
        "minimum_stress_mean_net_bps": 0, "promotion_scope": "shadow_champion_review_only",
        "live_orders_permitted": False,
    })


class BrainForwardRuntimeTests(unittest.TestCase):
    def test_parse_binance_excludes_open_candle(self) -> None:
        now = pd.Timestamp("2026-07-01T00:02:00Z")
        rows = [
            [int(pd.Timestamp("2026-07-01T00:00:00Z").timestamp() * 1000), "1", "2", "0.5", "1.5", "10", int(pd.Timestamp("2026-07-01T00:00:59Z").timestamp() * 1000), "0", "0", "4"],
            [int(pd.Timestamp("2026-07-01T00:01:00Z").timestamp() * 1000), "1.5", "2", "1", "1.7", "11", int(pd.Timestamp("2026-07-01T00:02:00Z").timestamp() * 1000), "0", "0", "5"],
        ]
        out = parse_binance_klines(rows, now)
        self.assertEqual(len(out), 1)
        self.assertEqual(float(out.iloc[0]["taker_base"]), 4.0)

    def test_completed_trade_uses_target_and_cost(self) -> None:
        raw = bars(1600)
        event_index = 1590
        raw.loc[event_index + 1, "high"] = raw.loc[event_index + 1, "open"] + 0.04
        decisions, trades = completed_paper_trades(raw, FakeModel(raw.loc[event_index, "ts"]))
        self.assertEqual(len(decisions), 1)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["exit_reason"], "target")
        self.assertAlmostEqual(float(trades[0]["net_bps"]), float(trades[0]["gross_bps"]) - 14.0)

    def test_run_once_only_writes_paper_ledger(self) -> None:
        raw = bars(1600)
        event_index = 1590
        raw.loc[event_index + 1, "high"] = raw.loc[event_index + 1, "open"] + 0.04
        model = FakeModel(raw.loc[event_index, "ts"])
        protocol = protocol_for(model)
        with patch.object(service, "fetch_closed_binance_bars", return_value=raw), \
             patch.object(service, "register_protocol") as register, \
             patch.object(service, "upsert_minute_bars", return_value=len(raw)) as write_bars, \
             patch.object(service, "upsert_decisions", return_value=1) as write_decisions, \
             patch.object(service, "upsert_trades", return_value=1) as write_trades:
            result = service.run_once("SOL-USDT", model, protocol)
        self.assertEqual(result, {"bars": len(raw), "decisions": 1, "trades": 1})
        self.assertEqual(write_bars.call_args.args[:2], ("SOL-USDT", "binance_spot_klines"))
        self.assertEqual(write_decisions.call_args.args[:2], ("SOL-USDT", "binance_spot_klines"))
        self.assertEqual(write_trades.call_args.args[:2], ("SOL-USDT", "binance_spot_klines"))
        register.assert_called_once_with(protocol)

    def test_once_mode_exits_after_one_successful_refresh(self) -> None:
        with patch.object(service, "run_once", return_value={"bars": 5, "decisions": 0, "trades": 0}) as refresh:
            self.assertEqual(service.main(["--once"]), 0)
        refresh.assert_called_once_with("SOL-USDT")


if __name__ == "__main__":
    unittest.main()
