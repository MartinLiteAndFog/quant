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
