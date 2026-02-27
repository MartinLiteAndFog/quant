from __future__ import annotations
import unittest
import numpy as np
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
