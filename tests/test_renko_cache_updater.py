from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from quant.execution.renko_cache_updater import _build_renko_ohlc, refresh_renko_cache


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

    @patch("quant.execution.renko_cache_updater.prune_live_renko_bricks_before")
    @patch("quant.execution.renko_cache_updater.upsert_live_renko_bricks")
    @patch("quant.execution.renko_cache_updater._publish_renko_to_redis", return_value={"ok": True})
    @patch(
        "quant.execution.renko_cache_updater.renko_from_close",
        return_value=pd.DataFrame(
            [
                {"ts": pd.Timestamp("2026-03-01T00:00:00Z"), "open": 100.0, "close": 100.1},
                {"ts": pd.Timestamp("2026-03-01T00:01:00Z"), "open": 100.1, "close": 100.2},
            ]
        ),
    )
    @patch(
        "quant.execution.renko_cache_updater._fetch_1m_close_paged",
        return_value=pd.DataFrame(
            {
                "ts": pd.date_range("2026-03-01T00:00:00Z", periods=10, freq="min", tz="UTC"),
                "close": [100.0 + (i * 0.1) for i in range(10)],
            }
        ),
    )
    @patch("quant.execution.renko_cache_updater.KucoinFuturesBroker")
    def test_refresh_renko_cache_persists_postgres_rows_and_prunes_old_history(
        self,
        _mock_broker,
        _mock_fetch,
        _mock_renko_from_close,
        _mock_publish,
        mock_upsert,
        mock_prune,
    ) -> None:
        info = refresh_renko_cache(
            symbol="SOL-USDT",
            box=0.1,
            days_back=14,
            step_hours=6,
            out_parquet="/tmp/quant-renko-test.parquet",
        )

        self.assertTrue(info["ok"])
        mock_upsert.assert_called_once()
        upsert_df = mock_upsert.call_args.kwargs["renko"]
        self.assertEqual(list(upsert_df.columns), ["ts", "open", "high", "low", "close"])
        mock_prune.assert_called_once()


if __name__ == "__main__":
    unittest.main()
