"""Synthetic algorithmic tests only. Requires no private trading data."""
import unittest
from verify_tv_renko_mtf_and_fills import (
    bars_from_rows, construct_traditional_renko, construct_traditional_renko_mixed,
    wilder_rsi, series_lookup, lookup, asof_first_open,
)


class MtfRenkoAuditTests(unittest.TestCase):
    def test_one_box_continuation_two_box_reversal_and_unique_ms(self):
        rows = [(0, 100.), (300, 100.4), (600, 100.3), (900, 100.2)]
        bricks = construct_traditional_renko(rows, box=.1, source_period_seconds=300)
        self.assertEqual([round(x[1], 2) for x in bricks], [100.1, 100.2, 100.3, 100.4, 100.2])
        self.assertEqual([x[0] for x in bricks[:4]], [300000, 300001, 300002, 300003])
        self.assertEqual(bricks[-1][0], 900000)

    def test_complete_bucket_rejects_gap(self):
        rows = [(0, 100., 100.), (60, 100., 101.), (180, 101., 102.), (240, 102., 103.),
                (300, 103., 104.), (360, 104., 105.), (420, 105., 106.),
                (480, 106., 107.), (540, 107., 108.)]
        r = bars_from_rows(rows, 300)
        self.assertEqual(list(r), [300])
        self.assertEqual(r[300], (108., 5))

    def test_rsi_uptrend_and_flat_uses_correct_seed(self):
        values = wilder_rsi([100 + .1 * i for i in range(40)], 14)
        self.assertTrue(all(x == 100 for x in values[14:]))
        self.assertEqual(wilder_rsi([100] * 30, 14)[-1], 50.)

    def test_same_source_multiple_bricks_not_available_early(self):
        series = [(600000, 41., 900000), (600001, 55., 900000), (600002, 65., 900000)]
        names, vals = series_lookup(series, False)
        known, confirmed = series_lookup(series, True)
        self.assertEqual(lookup(names, vals, 600001), 55.)
        self.assertIsNone(lookup(known, confirmed, 899999))
        self.assertEqual(lookup(known, confirmed, 900000), 65.)

    def test_price_reprice_must_wait_for_complete_source_bar(self):
        minute = {300: 101.0, 360: 102.0}
        stamp, price = asof_first_open(minute, 10000, source_minutes=5)
        self.assertEqual((stamp, price), (300, 101.0))
        self.assertEqual(asof_first_open(minute, 300000, source_minutes=5), (600, None))

    def test_mixed_historical_availability_uses_hour_then_five_min(self):
        p = construct_traditional_renko_mixed([(0, 100.), (3600, 100.4), (3900, 100.7)], 3900, .1)
        self.assertEqual(p[0][2], 7200000)
        self.assertEqual(p[-1][2], 4200000)


if __name__ == '__main__':
    unittest.main()