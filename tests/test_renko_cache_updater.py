from __future__ import annotations

import unittest

import pandas as pd

from quant.execution.renko_cache_updater import _build_renko_ohlc


class RenkoCacheUpdaterOrderTests(unittest.TestCase):
    def test_build_renko_ohlc_preserves_intra_timestamp_order(self) -> None:
        ts0 = pd.Timestamp("2026-03-01T15:00:00Z")
        ts1 = pd.Timestamp("2026-03-01T15:01:00Z")
        # Mixed directions with identical ts; order must stay exactly as emitted.
        bricks = pd.DataFrame(
            [
                {"ts": ts0, "open": 84.0, "close": 84.1},
                {"ts": ts0, "open": 84.1, "close": 84.2},
                {"ts": ts0, "open": 84.2, "close": 84.1},
                {"ts": ts1, "open": 84.1, "close": 84.0},
                {"ts": ts1, "open": 84.0, "close": 83.9},
            ]
        )
        out = _build_renko_ohlc(bricks)
        self.assertEqual(len(out), len(bricks))
        # Continuity check: each next open must match previous close.
        for i in range(1, len(out)):
            self.assertAlmostEqual(float(out.iloc[i]["open"]), float(out.iloc[i - 1]["close"]), places=10)
        # Timestamps must be strictly increasing after ns-offset normalization.
        ts = pd.to_datetime(out["ts"], utc=True)
        self.assertTrue((ts.diff().dropna() > pd.Timedelta(0)).all())


if __name__ == "__main__":
    unittest.main()
