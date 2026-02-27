from __future__ import annotations
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from quant.predictive_coding.config import PCConfig

class TestPCConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = PCConfig()
        self.assertEqual(cfg.d_latent, 32)
        self.assertEqual(cfg.horizons, [1, 5, 15, 60])
        self.assertAlmostEqual(cfg.total_cost, (cfg.fee_bps + cfg.slippage_bps) / 10_000)

    def test_override(self):
        cfg = PCConfig(d_latent=64, fee_bps=10.0)
        self.assertEqual(cfg.d_latent, 64)
        self.assertAlmostEqual(cfg.fee_bps, 10.0)


class TestTargets(unittest.TestCase):
    def _make_close(self, n=500):
        np.random.seed(42)
        return 100.0 * np.exp(np.cumsum(np.random.randn(n) * 0.001))

    def test_build_obs_features_shape(self):
        from quant.predictive_coding.targets import build_obs_features
        close = self._make_close(500)
        obs = build_obs_features(close)
        self.assertEqual(obs.shape, (500, 5))
        self.assertTrue(np.isnan(obs[0, 0]))
        self.assertFalse(np.isnan(obs[60, :]).any())

    def test_build_targets_shape(self):
        from quant.predictive_coding.targets import build_targets
        close = self._make_close(500)
        targets = build_targets(close, horizons=[1, 5, 15, 60])
        self.assertEqual(targets.shape, (500, 4))
        self.assertTrue(np.isnan(targets[-1, 3]))
        self.assertFalse(np.isnan(targets[-2, 0]))

    def test_valid_mask(self):
        from quant.predictive_coding.targets import get_valid_range
        close = self._make_close(500)
        start, end = get_valid_range(close, horizons=[1, 5, 15, 60], obs_lookback=60)
        self.assertEqual(start, 60)
        self.assertEqual(end, 500 - 60)


class TestTemporalPCModel(unittest.TestCase):
    def test_init_shapes(self):
        from quant.predictive_coding.model import TemporalPCModel
        from quant.predictive_coding.config import PCConfig
        cfg = PCConfig(d_latent=16)
        m = TemporalPCModel(cfg)
        self.assertEqual(m.x.shape, (16,))
        self.assertEqual(m.A.shape, (16, 16))
        self.assertEqual(len(m.W), 4)
        self.assertEqual(m.W[1].shape, (16,))
        self.assertEqual(m.C.shape, (5, 16))

    def test_step_returns_mu_and_sigma(self):
        from quant.predictive_coding.model import TemporalPCModel
        from quant.predictive_coding.config import PCConfig
        cfg = PCConfig(d_latent=16, warmup_bars=0)
        m = TemporalPCModel(cfg)
        obs = np.random.randn(5) * 0.01
        targets = {1: 0.001, 5: 0.002, 15: 0.003, 60: 0.005}
        mu, sigma = m.step(obs, targets, is_warmup=False)
        self.assertEqual(len(mu), 4)
        self.assertEqual(len(sigma), 4)
        for s in sigma.values():
            self.assertGreater(s, 0)

    def test_variance_updates_during_warmup(self):
        from quant.predictive_coding.model import TemporalPCModel
        from quant.predictive_coding.config import PCConfig
        cfg = PCConfig(d_latent=8)
        m = TemporalPCModel(cfg)
        v_before = {h: m.v_h[h] for h in cfg.horizons}
        obs = np.random.randn(5) * 0.01
        targets = {1: 0.01, 5: 0.02, 15: 0.03, 60: 0.05}
        m.step(obs, targets, is_warmup=True)
        changed = any(m.v_h[h] != v_before[h] for h in cfg.horizons)
        self.assertTrue(changed)

    def test_weights_frozen_during_warmup(self):
        from quant.predictive_coding.model import TemporalPCModel
        from quant.predictive_coding.config import PCConfig
        cfg = PCConfig(d_latent=8)
        m = TemporalPCModel(cfg)
        A_before = m.A.copy()
        obs = np.random.randn(5) * 0.01
        targets = {1: 0.001, 5: 0.002, 15: 0.003, 60: 0.005}
        m.step(obs, targets, is_warmup=True)
        np.testing.assert_array_equal(m.A, A_before)


class TestProbabilityLayer(unittest.TestCase):
    def test_compute_basics(self):
        from quant.predictive_coding.probability import compute_probabilities
        mu = {1: 0.001, 5: -0.002, 15: 0.0, 60: 0.005}
        sigma = {1: 0.001, 5: 0.001, 15: 0.001, 60: 0.002}
        price = 100.0
        result = compute_probabilities(mu, sigma, price)
        self.assertGreater(result[1]["p_up"], 0.5)
        self.assertLess(result[5]["p_up"], 0.5)
        self.assertAlmostEqual(result[15]["p_up"], 0.5, places=3)
        self.assertAlmostEqual(result[1]["price_level"], 100.0 * np.exp(0.001), places=4)

    def test_bands(self):
        from quant.predictive_coding.probability import compute_probabilities
        mu = {1: 0.001}
        sigma = {1: 0.002}
        result = compute_probabilities(mu, sigma, 100.0)
        self.assertGreater(result[1]["price_upper"], result[1]["price_level"])
        self.assertLess(result[1]["price_lower"], result[1]["price_level"])


class TestTradeDecisionLayer(unittest.TestCase):
    def test_no_trade_during_cooldown(self):
        from quant.predictive_coding.trade_logic import TradeDecisionLayer
        from quant.predictive_coding.config import PCConfig
        cfg = PCConfig(cooldown_bars=3)
        tdl = TradeDecisionLayer(cfg)
        tdl.position = 0
        tdl.cooldown_remaining = 3
        probs = {h: {"mu": 0.01, "sigma": 0.001, "z": 10.0, "p_up": 0.99,
                      "price_level": 101.0, "price_upper": 102.0, "price_lower": 100.0}
                 for h in [1, 5, 15, 60]}
        signal, events = tdl.update(probs, 100.0, bar_idx=10)
        self.assertEqual(signal, 0)
        self.assertEqual(tdl.cooldown_remaining, 2)

    def test_long_entry(self):
        from quant.predictive_coding.trade_logic import TradeDecisionLayer
        from quant.predictive_coding.config import PCConfig
        cfg = PCConfig(margin=0.02, z_min=0.1, min_edge_bps=0.0, fee_bps=0.0, slippage_bps=0.0)
        tdl = TradeDecisionLayer(cfg)
        probs = {}
        for h in [1, 5, 15, 60]:
            probs[h] = {"mu": 0.005, "sigma": 0.001, "z": 5.0, "p_up": 0.99,
                        "price_level": 100.5, "price_upper": 101.0, "price_lower": 100.0}
        signal, events = tdl.update(probs, 100.0, bar_idx=10)
        self.assertEqual(signal, 1)
        self.assertEqual(tdl.position, 1)

    def test_short_entry(self):
        from quant.predictive_coding.trade_logic import TradeDecisionLayer
        from quant.predictive_coding.config import PCConfig
        cfg = PCConfig(margin=0.02, z_min=0.1, min_edge_bps=0.0, fee_bps=0.0, slippage_bps=0.0)
        tdl = TradeDecisionLayer(cfg)
        probs = {}
        for h in [1, 5, 15, 60]:
            probs[h] = {"mu": -0.005, "sigma": 0.001, "z": -5.0, "p_up": 0.01,
                        "price_level": 99.5, "price_upper": 100.0, "price_lower": 99.0}
        signal, events = tdl.update(probs, 100.0, bar_idx=10)
        self.assertEqual(signal, -1)
        self.assertEqual(tdl.position, -1)

    def test_stop_loss(self):
        from quant.predictive_coding.trade_logic import TradeDecisionLayer
        from quant.predictive_coding.config import PCConfig
        cfg = PCConfig(sl_pct=0.01, fee_bps=0.0, slippage_bps=0.0, margin=0.0, z_min=0.0, min_edge_bps=0.0)
        tdl = TradeDecisionLayer(cfg)
        tdl.position = 1
        tdl.entry_price = 100.0
        tdl.entry_bar = 0
        tdl.chosen_horizon = 5
        probs = {h: {"mu": -0.02, "sigma": 0.01, "z": -2.0, "p_up": 0.02,
                      "price_level": 98.0, "price_upper": 99.0, "price_lower": 97.0}
                 for h in [1, 5, 15, 60]}
        signal, events = tdl.update(probs, 98.5, bar_idx=5)
        self.assertEqual(tdl.position, 0)
        self.assertTrue(any(e["event"] == "sl_exit" for e in events))

    def test_take_profit(self):
        from quant.predictive_coding.trade_logic import TradeDecisionLayer
        from quant.predictive_coding.config import PCConfig
        cfg = PCConfig(tp_pct=0.02, fee_bps=0.0, slippage_bps=0.0, margin=0.0, z_min=0.0, min_edge_bps=0.0)
        tdl = TradeDecisionLayer(cfg)
        tdl.position = 1
        tdl.entry_price = 100.0
        tdl.entry_bar = 0
        tdl.chosen_horizon = 5
        probs = {h: {"mu": 0.03, "sigma": 0.01, "z": 3.0, "p_up": 0.99,
                      "price_level": 103.0, "price_upper": 104.0, "price_lower": 102.0}
                 for h in [1, 5, 15, 60]}
        signal, events = tdl.update(probs, 103.0, bar_idx=3)
        self.assertEqual(tdl.position, 0)
        self.assertTrue(any(e["event"] == "tp_exit" for e in events))

    def test_timeout_exit(self):
        from quant.predictive_coding.trade_logic import TradeDecisionLayer
        from quant.predictive_coding.config import PCConfig
        cfg = PCConfig(fee_bps=0.0, slippage_bps=0.0, margin=0.0, z_min=0.0, min_edge_bps=0.0)
        tdl = TradeDecisionLayer(cfg)
        tdl.position = 1
        tdl.entry_price = 100.0
        tdl.entry_bar = 0
        tdl.chosen_horizon = 5
        probs = {h: {"mu": 0.001, "sigma": 0.01, "z": 0.1, "p_up": 0.54,
                      "price_level": 100.1, "price_upper": 101.0, "price_lower": 99.0}
                 for h in [1, 5, 15, 60]}
        signal, events = tdl.update(probs, 100.1, bar_idx=5)
        self.assertEqual(tdl.position, 0)
        self.assertTrue(any(e["event"] == "timeout_exit" for e in events))


class TestIntegration(unittest.TestCase):
    def test_end_to_end_synthetic(self):
        """Full pipeline on synthetic trending data should produce trades."""
        from quant.predictive_coding.config import PCConfig
        from quant.predictive_coding.model import TemporalPCModel
        from quant.predictive_coding.probability import compute_probabilities
        from quant.predictive_coding.targets import build_obs_features, build_targets, get_valid_range
        from quant.predictive_coding.trade_logic import TradeDecisionLayer

        np.random.seed(42)
        n = 2000
        trend = np.cumsum(np.random.randn(n) * 0.001 + 0.0001)
        close = 100.0 * np.exp(trend)

        cfg = PCConfig(
            d_latent=8,
            n_inference_steps=3,
            warmup_bars=100,
            fee_bps=0.0,
            slippage_bps=0.0,
            margin=0.01,
            z_min=0.05,
            min_edge_bps=0.0,
        )
        obs_all = build_obs_features(close)
        targets_all = build_targets(close, cfg.horizons)
        start, end = get_valid_range(close, cfg.horizons)

        model = TemporalPCModel(cfg)
        tdl = TradeDecisionLayer(cfg)
        n_signals = 0

        for t in range(start, end):
            obs = obs_all[t]
            tgt = {h: targets_all[t, i] for i, h in enumerate(cfg.horizons)}
            if np.any(np.isnan(obs)) or any(np.isnan(v) for v in tgt.values()):
                continue
            is_warmup = t < start + cfg.warmup_bars
            mu, sigma = model.step(obs, tgt, is_warmup=is_warmup)
            if is_warmup:
                continue
            probs = compute_probabilities(mu, sigma, float(close[t]))
            signal, _ = tdl.update(probs, float(close[t]), t)
            if signal != 0:
                n_signals += 1

        self.assertGreater(n_signals, 0)

    def test_run_backtest_function(self):
        """Test the run_backtest function directly with synthetic data."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

        from quant.predictive_coding.config import PCConfig
        from quant.predictive_coding.model import TemporalPCModel
        from quant.predictive_coding.probability import compute_probabilities
        from quant.predictive_coding.targets import build_obs_features, build_targets, get_valid_range
        from quant.predictive_coding.trade_logic import TradeDecisionLayer

        np.random.seed(123)
        n = 1000
        close = 100.0 * np.exp(np.cumsum(np.random.randn(n) * 0.002))
        ts = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
        df = pd.DataFrame({"ts": ts, "open": close, "high": close*1.001, "low": close*0.999, "close": close})

        cfg = PCConfig(d_latent=8, n_inference_steps=3, warmup_bars=50,
                       fee_bps=0.0, slippage_bps=0.0, margin=0.01, z_min=0.05, min_edge_bps=0.0)

        close_arr = df["close"].values.astype(np.float64)
        obs_all = build_obs_features(close_arr)
        targets_all = build_targets(close_arr, cfg.horizons)
        start, end = get_valid_range(close_arr, cfg.horizons, obs_lookback=60)

        model = TemporalPCModel(cfg)
        trade_logic = TradeDecisionLayer(cfg)

        pred_count = 0
        for t in range(start, end):
            obs = obs_all[t]
            tgt = {h: targets_all[t, i] for i, h in enumerate(cfg.horizons)}
            if np.any(np.isnan(obs)) or any(np.isnan(v) for v in tgt.values()):
                continue
            is_warmup = t < start + cfg.warmup_bars
            mu, sigma = model.step(obs, tgt, is_warmup=is_warmup)
            if is_warmup:
                continue
            probs = compute_probabilities(mu, sigma, float(close_arr[t]))
            signal, events = trade_logic.update(probs, float(close_arr[t]), t)
            pred_count += 1

        self.assertGreater(pred_count, 0)
