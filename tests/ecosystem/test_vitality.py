"""Synthetic, causal tests of slow/fast traders and aged-champion detection."""
import unittest
from datetime import date, datetime, timedelta, timezone
from math import isfinite

from quant.ecosystem.metrics.vitality import (
    MetricConfig, MetricError, NetDay, evaluate_strategy,
)


ORIGIN = date(2024, 1, 1)
UTC = timezone.utc


def sample(i: int, result: float, *, trades: int = 1, opps: int = 1,
           expected: int = 1) -> NetDay:
    day = ORIGIN + timedelta(days=i)
    known = datetime(day.year, day.month, day.day, 21, tzinfo=UTC)
    return NetDay(day, result, trades, opps, expected, known)


class VitalityEvidenceTests(unittest.TestCase):
    def test_two_year_winner_then_six_months_stale(self):
        old = [sample(i, .001) for i in range(730)]
        bad = [sample(730 + i, -.004) for i in range(180)]
        before = evaluate_strategy(old, oos_verified=True)
        after = evaluate_strategy(old + bad, oos_verified=True)
        self.assertGreater(before.current_vitality, .5)
        self.assertLess(after.current_vitality, .5)
        self.assertLess(after.current_vitality, before.current_vitality)
        self.assertLess(after.performance_drift, 0)
        self.assertGreater(after.sample_days, 900)
        self.assertGreater(after.max_drawdown, .4)
        self.assertGreater(after.underwater_days, 100)
        self.assertGreater(after.expected_shortfall_loss, 0)

    def test_cactus_ten_idle_months_does_not_die_just_for_waiting(self):
        active = [sample(i, .02 if i % 7 == 0 else -.001)
                  for i in range(100)]
        sleeping = [sample(100+i, 0, trades=0, opps=0, expected=0)
                    for i in range(300)]
        early = evaluate_strategy(active, oos_verified=True)
        late = evaluate_strategy(active+sleeping, oos_verified=True)
        self.assertAlmostEqual(early.current_vitality, late.current_vitality, places=10)
        self.assertEqual(early.effective_independent_samples,
                         late.effective_independent_samples)
        self.assertEqual(late.days_since_last_opportunity, 300)
        self.assertEqual(early.credibility, late.credibility)
        self.assertEqual(late.expected_signals, 100)

    def test_mark_to_market_loss_during_sleep_still_counts(self):
        active = [sample(i, .002) for i in range(100)]
        dormant = [sample(100+i, 0, trades=0, opps=0, expected=0) for i in range(80)]
        losing = dormant[:-1] + [sample(179, -.4, trades=0, opps=0, expected=0)]
        quiet = evaluate_strategy(active + dormant, oos_verified=True)
        bad = evaluate_strategy(active + losing, oos_verified=True)
        self.assertLess(bad.current_vitality, quiet.current_vitality)
        self.assertGreater(bad.current_drawdown, .3)
        self.assertEqual(bad.opportunity_active_days, 101)

    def test_confidence_eight_versus_800_trades(self):
        small = evaluate_strategy([sample(i, .01 if i%2 else -.005)
                                   for i in range(8)], oos_verified=True)
        large = evaluate_strategy([sample(i, .01 if i%2 else -.005)
                                   for i in range(800)], oos_verified=True)
        self.assertGreater(large.credibility, small.credibility)
        self.assertGreater(large.effective_independent_samples, small.effective_independent_samples)
        self.assertIsNotNone(large.lower_confidence_bound)
        self.assertIsNotNone(small.lower_confidence_bound)

    def test_positive_serial_dependence_reduces_effective_samples(self):
        blocks = [sample(i, .02 if i % 20 < 10 else -.02)
                  for i in range(400)]
        alternating = [sample(i, .02 if i % 2 else -.02)
                       for i in range(400)]
        a = evaluate_strategy(blocks, oos_verified=True)
        b = evaluate_strategy(alternating, oos_verified=True)
        self.assertLess(a.effective_independent_samples,
                        b.effective_independent_samples)
        self.assertLess(a.credibility,b.credibility)

    def test_cyclic_profitable_bush_without_forced_jagd_score(self):
        # Ten negative months and two unusually profitable periods can
        # still yield a positive annual return. Bark is distinct from
        # recent vitality, and security/position gates remain external.
        observations = [sample(i, .22 if i in (40, 260) else -.0005)
                        for i in range(365)]
        r = evaluate_strategy(observations, oos_verified=True)
        self.assertGreater(r.annualized_net_return, 0)
        self.assertGreater(r.bark_evidence, 0)
        self.assertGreaterEqual(r.current_vitality, 0)
        self.assertLessEqual(r.current_vitality, 1)

    def test_oos_data_quality_and_cashflow_contract(self):
        series = [sample(i, .003 if i%2 else -.001) for i in range(90)]
        raw = evaluate_strategy(series, oos_verified=False)
        verified = evaluate_strategy(series, oos_verified=True)
        poor = evaluate_strategy(series, oos_verified=True, config=MetricConfig(data_quality=.25))
        self.assertLess(raw.credibility, verified.credibility)
        self.assertLess(poor.credibility, verified.credibility)
        self.assertEqual(raw.opportunity_capture, 1)
        self.assertIsNone(evaluate_strategy(
            [sample(i, 0, trades=0,opps=0,expected=0) for i in range(30)]
        ).opportunity_capture)

    def test_correlation_is_optional_and_requires_overlap(self):
        series = [sample(i, .02 if i%2 else -.01) for i in range(120)]
        r = evaluate_strategy(series,portfolio_daily_returns=[
            .02 if i%2 else -.01 for i in range(120)],oos_verified=True)
        self.assertAlmostEqual(r.portfolio_correlation, 1)
        self.assertIsNone(evaluate_strategy(series[:12], portfolio_daily_returns=[
            .02 if i%2 else -.01 for i in range(12)]).portfolio_correlation)
        with self.assertRaisesRegex(MetricError, "series"):
            evaluate_strategy(series,portfolio_daily_returns=[0])

    def test_validation_no_future_lookahead_no_duplicate_days(self):
        s=sample(0,.001)
        with self.assertRaisesRegex(MetricError,"strictly increasing"):
            evaluate_strategy([s,s])
        day=ORIGIN
        future=datetime(day.year,day.month,day.day,21,tzinfo=UTC)+timedelta(days=1)
        with self.assertRaisesRegex(MetricError,"future-known"):
            NetDay(day,.01,1,1,1,future)
        with self.assertRaisesRegex(MetricError,"signals cannot exceed"):
            sample(1,.01,trades=1,opps=0,expected=1)
        with self.assertRaisesRegex(MetricError,"daily net TWR"):
            sample(1,-1)
        with self.assertRaisesRegex(MetricError,"at least one"):
            evaluate_strategy([])
        with self.assertRaisesRegex(MetricError,"weights"):
            MetricConfig(fitness_profit_weight=.8)

    def test_missing_days_reported_without_fake_zeros(self):
        s=[sample(0,.02),sample(3,-.01),sample(8,.03)]
        r=evaluate_strategy(s,oos_verified=True)
        self.assertEqual(r.missing_calendar_days,6)
        self.assertTrue(isfinite(r.current_vitality))
        self.assertEqual(r.sample_days,3)
        self.assertEqual(r.effective_independent_samples,3)


if __name__=="__main__":
    unittest.main()
