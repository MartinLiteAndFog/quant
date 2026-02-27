from __future__ import annotations
import unittest
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
