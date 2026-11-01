"""ECO-03: reproducible isolated UTC daily clock, never live execution."""
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal as D
import unittest

from quant.ecosystem.sim_clock.clock import (
    PHASES, ClockError, IntradayEvent, advance_day, advance_days,
    new_simulation, SimulationState,
)


UTC = timezone.utc


def moment(day, hour=12, minute=0):
    return datetime.combine(day, datetime.min.time(), UTC).replace(hour=hour, minute=minute)


def event(day, suffix, kind="pnl_delta", amount="2", hour=12):
    return IntradayEvent(suffix, moment(day, hour),
                         moment(day, hour, 1), kind, D(amount), "plant-1")


class DailyClockTests(unittest.TestCase):
    def setUp(self):
        self.start_day = date(2026, 1, 31)
        self.origin = new_simulation("paper-test", "100", self.start_day, seed=42)

    def test_day_profit_fees_funding_cashflow_accounting(self):
        d = self.start_day
        items = [
            event(d, "pnl", "pnl_delta", "20", 10),
            event(d, "fee", "fee", "-2", 11),
            event(d, "funding", "funding", "-1", 12),
            event(d, "deposit", "external_cashflow", "5", 13),
            event(d, "alert", "signal", "0", 14),
        ]
        run = advance_day(self.origin, d, list(reversed(items)))
        self.assertEqual(run.daily_net_pnl, D("17"))
        self.assertEqual(run.external_cashflow, D("5"))
        self.assertEqual(run.state.equity, D("122"))
        self.assertEqual(run.state.reserve, D("12.2"))
        self.assertEqual(run.state.distributable, D("109.8"))
        self.assertEqual([a.kind for a in run.day_events[:5]],
                         ["pnl_delta", "fee", "funding", "external_cashflow", "signal"])
        self.assertEqual([a.phase for a in run.day_events][-6:],
                         list(PHASES)[1:])
        self.assertEqual(run.state.day_index, 0)
        self.assertEqual(self.origin.equity, D("100"))

    def test_same_seed_same_events_produces_byte_identical_checkpoint(self):
        seq = [
            (date(2026, 1, 31), [event(date(2026, 1, 31), "a", amount="3")]),
            (date(2026, 2, 1), [event(date(2026, 2, 1), "b", amount="-2")]),
            (date(2026, 2, 2), []),
        ]
        def growth(snapshot, rng):
            obj = dict(snapshot["state"])
            obj["growth"] = obj.get("growth", 0) + 1
            obj["random_daily"] = rng.randrange(100000)
            return obj
        reducers = {"ecology": growth}
        direct = advance_days(self.origin, seq, reducers)
        again = advance_days(self.origin, seq, reducers)
        self.assertEqual(direct.to_json(), again.to_json())
        paused = advance_days(self.origin, seq[:1], reducers).to_json()
        restored = SimulationState.from_json(paused)
        resumed = advance_days(restored, seq[1:], reducers)
        self.assertEqual(direct.to_json(), resumed.to_json())
        self.assertEqual(resumed.ecology["growth"], 3)

    def test_no_trade_day_still_has_all_daily_phases(self):
        r = advance_day(self.origin, self.start_day)
        self.assertEqual([e.phase for e in r.day_events], list(PHASES)[1:])
        self.assertEqual(r.daily_net_pnl, D("0"))
        self.assertEqual(r.state.equity, D("100"))
        self.assertEqual(r.state.to_json(), SimulationState.from_json(r.state.to_json()).to_json())

    def test_duplicate_event_and_replayed_day_fail_closed(self):
        a = event(self.start_day, "x")
        with self.assertRaisesRegex(ClockError, "duplicate event"):
            advance_day(self.origin, self.start_day, [a, a])
        after = advance_day(self.origin, self.start_day, [a]).state
        with self.assertRaisesRegex(ClockError, "next consecutive"):
            advance_day(after, self.start_day, [a])
        with self.assertRaisesRegex(ClockError, "already incorporated"):
            advance_day(after, self.start_day + timedelta(days=1),
                        [event(self.start_day + timedelta(days=1), "x")])

    def test_noncausal_future_inputs_and_no_future_fee(self):
        d = self.start_day
        with self.assertRaisesRegex(ClockError, "before occurrence"):
            IntradayEvent("future", moment(d, 15), moment(d, 14), "signal")
        with self.assertRaisesRegex(ClockError, "outside day or future-known"):
            advance_day(self.origin, d, [IntradayEvent(
                "late", moment(d, 23), moment(d + timedelta(days=1), 0),
                "pnl_delta", D("1"))])
        with self.assertRaisesRegex(ClockError, "signal events cannot"):
            event(d, "not-profit", "signal", "42")
        with self.assertRaisesRegex(ClockError, "fees cannot"):
            event(d, "mint", "fee", "1")
        with self.assertRaisesRegex(ClockError, "decimal"):
            IntradayEvent("float", moment(d), moment(d), "pnl_delta", 1.5)

    def test_negative_nav_freezes_budget_not_implicitly_positions(self):
        d = self.start_day
        r = advance_day(self.origin, d, [event(d, "crash", amount="-130")])
        self.assertEqual(r.state.equity, D("-30"))
        self.assertEqual(r.state.reserve, D("0"))
        self.assertEqual(r.state.distributable, D("0"))
        self.assertTrue(any(e.payload.get("risk_frozen") is True for e in r.day_events))
        self.assertFalse(any("close" in e.kind for e in r.day_events))

    def test_reducer_error_does_not_mutate_origin_or_emit_snapshot(self):
        def broken(_, rng):
            rng.random()
            raise RuntimeError("strategy error")
        with self.assertRaisesRegex(RuntimeError, "strategy error"):
            advance_day(self.origin, self.start_day, reducers={"ecology": broken})
        self.assertEqual(self.origin.day_index, -1)
        self.assertEqual(self.origin.audit, ())

    def test_reject_unversioned_snapshot_and_non_json_state(self):
        blob = self.origin.to_json().replace('"version":"1.0"', '"version":"oops"')
        with self.assertRaisesRegex(ClockError, "unsupported tick version"):
            SimulationState.from_json(blob)
        with self.assertRaisesRegex(ClockError, "non-JSON"):
            advance_day(self.origin, self.start_day,
                        reducers={"ecology": lambda p, r: {"bad": {1, 2}}})

    def test_utc_calendar_and_timezone(self):
        with self.assertRaisesRegex(ClockError, "timezone-aware"):
            IntradayEvent("x", datetime(2026, 1, 31, 12),
                         datetime(2026, 1, 31, 12), "signal")
        start = date(2025, 12, 31)
        s = new_simulation("years", "1", start)
        s = advance_day(s, start).state
        s = advance_day(s, date(2026, 1, 1)).state
        self.assertEqual(s.day_index, 1)


if __name__ == "__main__":
    unittest.main()
