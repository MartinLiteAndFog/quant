from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from quant.execution.bot_profiles import (
    resolve_profile_gate,
    reverse_on_wait_sl,
    strategy_config_hash,
    strategy_instance_id,
)
from quant.execution.railway_bot import configure_environment


class BotProfileTests(unittest.TestCase):
    def tearDown(self) -> None:
        for key in (
            "BOT_PROFILE",
            "BOT_INSTANCE_ID",
            "BOT_DATA_ROOT",
            "PC3AXIS_STATE_SPACE_PATH",
            "SIGNALS_DIR",
            "LIVE_SIGNAL_STATE",
            "LIVE_EXECUTOR_STATE",
            "LIVE_IMBA_LOOKBACK",
            "LIVE_FLIP_TTP_TRAIL_PCT",
            "LIVE_FLIP_MIN_SL_PCT",
            "LIVE_FLIP_MAX_SL_PCT",
            "LIVE_FLIP_SWING_LOOKBACK",
            "EVENTS_DIR",
            "MICRO_PILOT_MODE",
            "LIVE_EXECUTOR_MAX_MARGIN_USDT",
            "LIVE_EXECUTOR_MAX_CONTRACTS",
            "LIVE_EXECUTOR_MAX_LEVERAGE",
            "LIVE_EXECUTOR_LEVERAGE",
            "KUCOIN_FUTURES_ORDER_LEVERAGE",
            "KUCOIN_FUTURES_MARGIN_MODE",
            "KUCOIN_FUTURES_STRICT_MARGIN_MODE",
            "LIVE_EXECUTOR_POS_PCT",
        ):
            os.environ.pop(key, None)

    def test_countertrend_profiles_force_gate_on(self) -> None:
        for profile, reverse_expected in (
            ("countertrend", False),
            ("countertrend_sl_reverse", True),
        ):
            with self.subTest(profile=profile), patch.dict(
                os.environ, {"BOT_PROFILE": profile}, clear=False
            ):
                gate = resolve_profile_gate(
                    {"gate_on": 0, "gate_countertrend_on": 0, "gate_trend_on": 1}
                )
                self.assertEqual(gate["gate_on"], 1)
                self.assertEqual(gate["gate_countertrend_on"], 1)
                self.assertEqual(gate["regime_state"], "countertrend")
                self.assertEqual(reverse_on_wait_sl(), reverse_expected)

    def test_unconfigured_profile_preserves_canonical_gate(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            gate = resolve_profile_gate({"gate_on": 0, "source": "postgres_daily_gate"})
        self.assertEqual(gate, {"gate_on": 0, "source": "postgres_daily_gate"})

    def test_pc3axis_uses_strict_last_backtest_quantiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state_space.parquet"
            n = 100
            frame = pd.DataFrame(
                {
                    "ts": pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC"),
                    "X_raw": list(range(1, n)) + [1],
                    "Y_res": list(range(1, n + 1)),
                    "Z_res": list(range(1, n)) + [1],
                }
            )
            frame.to_parquet(path, index=False)
            with patch.dict(
                os.environ,
                {
                    "BOT_PROFILE": "pc3axis",
                    "PC3AXIS_STATE_SPACE_PATH": str(path),
                },
                clear=False,
            ):
                gate = resolve_profile_gate({"gate_on": 0})

            self.assertEqual(gate["source"], "bot_profile:pc3axis_state_space")
            self.assertEqual(gate["gate_countertrend_on"], 1)
            self.assertEqual(gate["pc3axis"]["mode"], "strict_3of3")
            self.assertEqual(gate["pc3axis"]["drift_abs_q"], 0.55)
            self.assertEqual(gate["pc3axis"]["elasticity_q"], 0.25)
            self.assertEqual(gate["pc3axis"]["instability_q"], 0.35)

    def test_pc3axis_falls_back_to_canonical_gate_when_cache_missing(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BOT_PROFILE": "pc3axis",
                "PC3AXIS_STATE_SPACE_PATH": "/definitely/missing/state_space.parquet",
            },
            clear=False,
        ):
            gate = resolve_profile_gate({"gate_on": 1, "source": "postgres_daily_gate"})
        self.assertEqual(gate["gate_on"], 1)
        self.assertEqual(gate["source"], "postgres_daily_gate")
        self.assertIn("profile_gate_error", gate)

    def test_launcher_sets_isolated_paths_and_backtest_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BOT_PROFILE": "countertrend_sl_reverse",
                "BOT_INSTANCE_ID": "SL Reverse #1",
                "BOT_DATA_ROOT": "/tmp/bots",
            },
            clear=False,
        ):
            profile, instance = configure_environment()
            self.assertEqual(profile, "countertrend_sl_reverse")
            self.assertEqual(instance, "sl-reverse-1")
            self.assertEqual(os.environ["SIGNALS_DIR"], "/tmp/bots/sl-reverse-1/signals")
            self.assertEqual(os.environ["LIVE_IMBA_LOOKBACK"], "150")
            self.assertEqual(os.environ["LIVE_FLIP_TTP_TRAIL_PCT"], "0.0025")
            self.assertEqual(os.environ["LIVE_FLIP_MIN_SL_PCT"], "0.010")
            self.assertEqual(os.environ["LIVE_FLIP_MAX_SL_PCT"], "0.080")
            self.assertEqual(os.environ["LIVE_FLIP_SWING_LOOKBACK"], "180")
            self.assertEqual(os.environ["EVENTS_DIR"], "/tmp/bots/sl-reverse-1/events")

    def test_canonical_profile_skips_countertrend_backtest_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BOT_PROFILE": "canonical",
                "BOT_INSTANCE_ID": "canonical",
                "BOT_DATA_ROOT": "/tmp/bots",
            },
            clear=False,
        ):
            profile, instance = configure_environment()
            self.assertEqual(profile, "canonical")
            self.assertEqual(instance, "canonical")
            self.assertNotIn("LIVE_IMBA_LOOKBACK", os.environ)

    def test_strategy_instance_helpers(self) -> None:
        with patch.dict(
            os.environ,
            {"BOT_INSTANCE_ID": "pc3axis", "BOT_PROFILE": "pc3axis"},
            clear=False,
        ):
            self.assertEqual(strategy_instance_id(), "pc3axis")
            self.assertEqual(strategy_config_hash(), "pc3axis_pc3axis_v1")

    def test_micro_pilot_sets_fail_closed_risk_limits(self) -> None:
        with patch.dict(
            os.environ,
            {"BOT_PROFILE": "canonical", "BOT_INSTANCE_ID": "micro-a", "MICRO_PILOT_MODE": "1"},
            clear=False,
        ):
            configure_environment()
            self.assertEqual(os.environ["LIVE_EXECUTOR_MAX_MARGIN_USDT"], "5")
            self.assertEqual(os.environ["LIVE_EXECUTOR_MAX_CONTRACTS"], "1")
            self.assertEqual(os.environ["LIVE_EXECUTOR_LEVERAGE"], "3")
            self.assertEqual(os.environ["KUCOIN_FUTURES_MARGIN_MODE"], "isolated")
            self.assertEqual(os.environ["KUCOIN_FUTURES_STRICT_MARGIN_MODE"], "1")

    def test_micro_pilot_rejects_cross_margin(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BOT_PROFILE": "canonical",
                "BOT_INSTANCE_ID": "micro-a",
                "MICRO_PILOT_MODE": "1",
                "KUCOIN_FUTURES_MARGIN_MODE": "cross",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "isolated margin"):
                configure_environment()
