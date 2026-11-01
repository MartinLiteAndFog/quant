"""Causal market-data/fill -> ECO-03/ECO-17 bridge checks (no venue I/O)."""
from datetime import date, datetime, timedelta, timezone
from dataclasses import replace
from decimal import Decimal as D
import unittest

from quant.ecosystem.market_data import Provenance, normalize, Costs, timestamp
from quant.ecosystem.integration.market_replay import (
    TargetSignal, FundingTick, ReplayConfig, ReplayError,
    replay_market, apply_replay_to_evolution,
)
from quant.ecosystem.integration.evolution_cycle import (
    CycleConfig, start_evolution,
)
from quant.ecosystem.ecology.physiology import Weather
from quant.ecosystem.sim_clock.clock import new_simulation, advance_day


UTC=timezone.utc
START=date(2026,1,1)
FUTURES=Provenance("generated 2-day synthetic", "CC0 synthetic", "SOL-USDT",
                   "linear_futures", synthetic=True)
SPOT=Provenance("generated 2-day synthetic", "CC0 synthetic", "SOL-USDT",
                "spot", synthetic=True)


def time_on(day,hour,minute):
    return datetime(day.year,day.month,day.day,hour,minute,tzinfo=UTC)


def bars(days=2,missing=(),spot=False):
    start=time_on(START,0,0)
    missing=set(missing)
    rows=[]
    for i in range(days*1440):
        if i in missing:
            continue
        dt=start+timedelta(minutes=i)
        # Stable intraday fair price. At 23:58 the completed closing
        # candle is known only at 23:59, never earlier on the same day.
        close=D("110") if i==1438 else D("90") if i==2878 else D("100")
        high=max(close,D("100"))
        low=min(close,D("100"))
        rows.append(dict(timestamp=dt.isoformat(),open="100",
                         high=str(high),low=str(low),close=str(close),
                         volume="1000"))
    return normalize(rows,SPOT if spot else FUTURES)


def signal(sid,day,hour,minute,target,strategy="parent-00"):
    at=time_on(day,hour,minute)
    return TargetSignal(sid,strategy,D(target),at,at,at)


class MarketReplayTests(unittest.TestCase):
    def test_open_close_overnight_net_pnl_never_double_counts_spread_or_fees(self):
        data=bars()
        orders=[
            signal("open",START,9,0,"1"),
            signal("close",START+timedelta(days=1),9,0,"0"),
        ]
        run=replay_market(data,orders)
        self.assertEqual(len(run.fills),2)
        self.assertEqual(run.fills[0].executed_at,time_on(START,9,1))
        self.assertEqual(run.fills[0].price,D("100.02"))
        self.assertEqual(run.fills[1].price,D("99.98"))
        self.assertEqual([x.end_positions for x in run.days],
                         [(("parent-00",D("1")),),(("parent-00",D("0")),)])
        self.assertEqual([day.trade_counts().get("parent-00",0) for day in run.days],[0,1])
        self.assertEqual([d.net_pnl for d in run.days],
                         [D("9.919988"),D("-10.079988")])
        self.assertEqual(sum((d.net_pnl for d in run.days),D("0")),D("-0.160000"))
        self.assertTrue(run.synthetic)
        self.assertEqual(run.data_hash,data.fingerprint)
        self.assertEqual(run.events(),replay_market(data,orders).events())
        self.assertTrue(any(e.event_id.startswith("mark:") for e in run.events()))
        self.assertEqual(len({e.event_id for e in run.events()}),len(run.events()))
        # Direct integration with the current clock must produce same NAV.
        clock=new_simulation("market", "100", START)
        for day in run.days:
            clock=advance_day(clock,day.day,day.events).state
        self.assertEqual(clock.equity,D("99.840000"))
        self.assertEqual(clock.reserve,clock.equity*D(".10"))

    def test_short_futures_and_directional_funding(self):
        first=signal("short",START,9,0,"-1")
        second=signal("cover",START+timedelta(days=1),9,0,"0")
        settlement=FundingTick("settlement-1",time_on(START,16,0),
                               time_on(START,16,0),D(".001"),D("100"))
        run=replay_market(bars(),[first,second],funding=[settlement])
        self.assertEqual([e.amount for e in run.events() if e.kind=="funding"],[D(".100")])
        self.assertEqual(sum((d.net_pnl for d in run.days),D(0)),D("-0.060000"))
        self.assertEqual(len(run.fills),2)
        self.assertEqual(run.days[1].trade_counts()["parent-00"],1)
        self.assertEqual(run.days[1].end_positions[0][1],D("0"))

    def test_cross_day_funding_or_spot_short_are_refused(self):
        with self.assertRaisesRegex(ReplayError,"shorting"):
            replay_market(bars(spot=True),[signal("s",START,9,0,"-1")])
        with self.assertRaisesRegex(ReplayError,"spot data"):
            replay_market(bars(spot=True),[],funding=[
                FundingTick("f",time_on(START,10,0),time_on(START,10,0),
                            D(".001"),D("100"))])
        at=time_on(START,23,59)
        with self.assertRaisesRegex(ReplayError,"cross-day funding"):
            FundingTick("late",at,at+timedelta(minutes=1),D(".001"),D("100"))

    def test_next_open_never_bar_close_and_unavailable_signal_rejected(self):
        s=signal("first",START,9,0,"1")
        delayed=replace(s,available_at=time_on(START,9,2))
        run=replay_market(bars(),[delayed])
        self.assertEqual(run.fills[0].executed_at,time_on(START,9,3))
        self.assertNotEqual(run.fills[0].executed_at,s.signal_at)
        with self.assertRaisesRegex(ReplayError,"signal before dataset"):
            replay_market(bars(),[signal("pre",START-timedelta(days=1),9,0,"1")])
        with self.assertRaisesRegex(ReplayError,"signal after final"):
            replay_market(bars(),[signal("after",START+timedelta(days=2),9,0,"1")])
        with self.assertRaisesRegex(ReplayError,"duplicate signal IDs"):
            replay_market(bars(),[s,s])
        with self.assertRaisesRegex(ReplayError,"overlapping target changes"):
            replay_market(bars(),[s,replace(s,signal_id="same-open",target_quantity=D("2"))])

    def test_gap_during_open_position_never_interpolated(self):
        q=[signal("open",START,9,0,"1")]
        with self.assertRaisesRegex(ReplayError,"missing OHLCV while position open"):
            replay_market(bars(missing=[600]),q)
        # Missing history before an entry is recorded as a real data quality
        # limitation, but is not silently turned into pre-entry PnL.
        run=replay_market(bars(missing=[180]),q)
        self.assertGreater(run.days[0].net_pnl,D("0"))
        with self.assertRaisesRegex(ReplayError,"incomplete UTC calendar day"):
            replay_market(bars(missing=[0]),q)

    def test_full_day_policy_and_partial_fixture_optin(self):
        original=bars()
        reduced=normalize([dict(timestamp=b.timestamp.isoformat(),
                                open=str(b.open),high=str(b.high),low=str(b.low),
                                close=str(b.close),volume=str(b.volume))
                           for b in original.bars[:10]],FUTURES)
        q=[signal("tiny",START,0,1,"1")]
        with self.assertRaisesRegex(ReplayError,"incomplete UTC calendar day"):
            replay_market(reduced,q)
        short=replay_market(reduced,q,config=ReplayConfig(require_full_utc_days=False))
        self.assertEqual(len(short.days),1)
        self.assertTrue(short.days[0].end_positions[0][1]>0)
        self.assertEqual(short.days[0].net_pnl,D("-0.080012"))

    def test_refuse_excessive_targets_and_delayed_fills(self):
        large=signal("large",START,9,0,"200")
        with self.assertRaisesRegex(ReplayError,"quantity cap"):
            replay_market(bars(),[large])
        with self.assertRaisesRegex(ReplayError,"signal .*timely"):
            replay_market(bars(),[signal("late",START+timedelta(days=1),23,58,"1")])
        with self.assertRaisesRegex(ReplayError,"cross-day trade confirmation"):
            replay_market(
                bars(),[signal("edge",START,23,58,"1")],
                config=ReplayConfig(fill_confirmation_latency_seconds=60),
            )

    def test_clock_to_evolution_requires_independent_opportunity_and_margin(self):
        market=replay_market(bars(),[
            signal("open",START,9,0,"1"),
            signal("close",START+timedelta(days=1),9,0,"0"),
        ])
        cfg=CycleConfig(reproduction_every_days=100)
        world=start_evolution("market-evolution",START,"100",seed=2,config=cfg)
        dates=[START,START+timedelta(days=1)]
        weather={d:Weather(time_on(d,16,0),time_on(d,16,0),.5,True) for d in dates}
        evidence={d:{"parent-00":1} for d in dates}
        free={d:"20" for d in dates}
        with self.assertRaisesRegex(ReplayError,"explicit causal"):
            apply_replay_to_evolution(world,market,weather_by_day=weather,
                                      opportunities_by_day={},free_margin_by_day=free,
                                      config=cfg)
        # Signals are supplied externally and pre-specified; this only tests
        # the input event bridge, not profitable evolutionary strategy logic.
        final=apply_replay_to_evolution(world,market,weather_by_day=weather,
                                        opportunities_by_day=evidence,
                                        free_margin_by_day=free,config=cfg)
        self.assertEqual(final.equity,D("99.840000"))
        self.assertEqual(final.reserve,final.equity*D(".10"))
        self.assertEqual(final.day_index,1)
        self.assertEqual(final.ecology["metrics"]["parent-00"]["executed_trades"],1)
        self.assertEqual(final.to_json(),
                         apply_replay_to_evolution(world,market,weather_by_day=weather,
                            opportunities_by_day=evidence,free_margin_by_day=free,
                            config=cfg).to_json())


if __name__=="__main__":
    unittest.main()
