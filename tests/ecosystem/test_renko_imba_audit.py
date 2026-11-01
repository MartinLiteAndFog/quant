"""ECO-19b: verified 5m timing and explicit unverified TV/Pine signal status."""
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

from quant.ecosystem.market_data import Provenance
from quant.ecosystem.market_data.streaming import ingest_ohlcv
from quant.ecosystem.market_data.renko_imba_audit import (
    ReferenceAlert, SignalAuditError,
    source_5min_bars, candidate_imba_signals, compare_confirmed_alerts,
)

UTC=timezone.utc
START=datetime(2026,9,15,tzinfo=UTC)
PROV=Provenance(
    source="synthetic local fixture",license="CC0 synthetic",symbol="SOL-DEMO",
    market_type="linear_futures",interval_seconds=60,synthetic=True,
)


def price(i):
    group=i//5
    regime=(group//5)%2
    return D("100")+(D(".8") if regime else D("-.8")) + D(group%5)*D(".02")


def bars_csv(path, n=400, missing=()):
    omissions=set(missing)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=[
            "timestamp","open","high","low","close","volume"])
        w.writeheader()
        for i in range(n):
            if i in omissions:
                continue
            px=price(i)
            w.writerow({
                "timestamp":(START+timedelta(minutes=i)).isoformat(),
                "open":str(px),"high":str(px+D(".1")),
                "low":str(px-D(".1")),"close":str(px),"volume":"3",
            })


def refs(candidates):
    return [ReferenceAlert(f"tv-{i}",c.direction,c.available_at)
            for i,c in enumerate(candidates)]


class RenkoImbaAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.source=self.root/"bars.csv"
        self.index=self.root/"bars.sqlite"
        bars_csv(self.source)
        ingest_ohlcv(self.source,self.index,PROV)
        self.end=START+timedelta(minutes=400)
        self.trade=START+timedelta(minutes=100)

    def candidate(self,**kwargs):
        return candidate_imba_signals(
            str(self.index),
            start=START,end=self.end,
            trading_start=self.trade,
            strategy_id="research-imba",
            box="0.1",lookback=6,**kwargs,
        )

    def test_completed_5min_aggregation_preserves_hilo_volume_and_causal_close(self):
        bars=source_5min_bars(str(self.index),START,
                              START+timedelta(minutes=15))
        self.assertEqual(len(bars),3)
        self.assertEqual(bars[0].open_at,START)
        self.assertEqual(bars[0].known_at,START+timedelta(minutes=5))
        self.assertEqual(bars[0].volume,D("15"))
        self.assertEqual(bars[0].high,price(0)+D(".1"))
        self.assertEqual(bars[0].low,price(0)-D(".1"))
        self.assertEqual(bars[1].close,price(5))
        self.assertTrue(all(b.open_at+timedelta(minutes=5)==b.known_at for b in bars))

    def test_candidates_use_true_5m_source_close_not_renko_millisecond_timestamp(self):
        batch=self.candidate()
        self.assertTrue(batch.brick_count>6)
        self.assertGreater(len(batch.candidates),0)
        self.assertEqual(batch.renko_source_interval,300)
        self.assertFalse(batch.renko_wicks_reproduced)
        self.assertFalse(batch.tv_feed_identity_verified)
        self.assertFalse(batch.independent_oos_verified)
        self.assertFalse(batch.capital_risk_cleared)
        self.assertEqual(batch,batch.__class__(**batch.__dict__))
        self.assertEqual(len({c.available_at for c in batch.candidates}),
                         len(batch.candidates))
        for c in batch.candidates:
            self.assertGreaterEqual(c.available_at,self.trade)
            self.assertEqual(c.available_at,c.source_5m_open_at+
                             timedelta(minutes=5))
            self.assertEqual(int(c.available_at.timestamp())%300,0)
            self.assertGreater(c.available_at,c.renko_brick_at)
            self.assertIn(c.direction,(-1,1))
            self.assertEqual(c.box,D("0.1"))
            self.assertTrue(c.candidate_id.startswith("research-imba:"))

    def test_true_parent_lookback_incompatible_window_is_rejected(self):
        with self.assertRaisesRegex(SignalAuditError,"requested research window"):
            self.candidate(max_source_1min=399)
        with self.assertRaisesRegex(SignalAuditError,"insufficient warmup"):
            candidate_imba_signals(
                str(self.index),start=START,end=self.end,
                trading_start=self.trade,strategy_id="x",lookback=100000,
            )
        with self.assertRaisesRegex(SignalAuditError,"trading_start"):
            candidate_imba_signals(str(self.index),start=START,end=self.end,
                                   trading_start=START,strategy_id="x")

    def test_reference_comparison_never_claims_tv_wicks_or_oos_verified(self):
        batch=self.candidate()
        reference=refs(batch.candidates)
        correct=compare_confirmed_alerts(batch,reference,minimum_alerts=1)
        self.assertTrue(correct.matches_supplied_reference)
        self.assertEqual(correct.exact_matches,len(reference))
        self.assertEqual(correct.unpaired_references,0)
        self.assertEqual(correct.unpaired_candidates,0)
        self.assertFalse(correct.full_TradingView_parity_established)
        self.assertFalse(correct.safe_to_claim_oos_alpha)
        self.assertFalse(compare_confirmed_alerts(
            batch,reference,minimum_alerts=len(reference)+1,
        ).enough_reference_signals_for_screening)

        first=reference[0]
        reverse=[ReferenceAlert(first.reference_id,-first.direction,
                                first.confirmed_5m_close_at),*reference[1:]]
        mismatch=compare_confirmed_alerts(batch,reverse,minimum_alerts=1)
        self.assertEqual(mismatch.direction_mismatches,1)
        self.assertFalse(mismatch.matches_supplied_reference)

    def test_missing_or_shifted_tv_alerts_cannot_silently_pass(self):
        batch=self.candidate()
        reference=refs(batch.candidates)
        trimmed=compare_confirmed_alerts(batch,reference[:-1],minimum_alerts=1)
        self.assertGreaterEqual(trimmed.unpaired_candidates,1)
        self.assertFalse(trimmed.matches_supplied_reference)
        modified=[ReferenceAlert(reference[0].reference_id,
                                 reference[0].direction,
                                 reference[0].confirmed_5m_close_at+timedelta(minutes=300)),
                  *reference[1:]]
        shifted=compare_confirmed_alerts(batch,modified,minimum_alerts=1)
        self.assertFalse(shifted.matches_supplied_reference)
        self.assertGreater(shifted.timestamp_mismatches,0)
        with self.assertRaisesRegex(SignalAuditError,"duplicate TV alert"):
            compare_confirmed_alerts(batch,[reference[0],reference[0]],minimum_alerts=1)

    def test_source_gap_fails_even_if_ingest_reported_it(self):
        broken=self.root/"missing.csv"
        bars_csv(broken,400,missing=(51,))
        other=self.root/"missing.sqlite"
        r=ingest_ohlcv(broken,other,PROV)
        self.assertEqual(r["missing_bars"],1)
        with self.assertRaisesRegex(SignalAuditError,"missing/misaligned"):
            source_5min_bars(str(other),START,self.end)

    def test_csv_cli_exports_candidate_only_no_trade_order(self):
        output=self.root/"audit.json"
        command=[
            sys.executable,"scripts/ecosystem_renko_imba_audit.py",
            "--database",str(self.index),
            "--start",START.isoformat(),
            "--trading-start",self.trade.isoformat(),
            "--end",self.end.isoformat(),
            "--lookback","6","--box","0.1",
            "--strategy-id","research-imba",
            "--output",str(output),
        ]
        run=subprocess.run(command,capture_output=True,text=True,
                           env={**os.environ,"PYTHONPATH":"src"})
        self.assertEqual(run.returncode,0,run.stderr)
        report=json.loads(output.read_text())
        self.assertTrue(report["candidate_only"])
        self.assertTrue(report["renko_close_only_no_tv_wicks"])
        self.assertFalse(report["full_tradingview_parity_verified"])
        self.assertFalse(report["capital_risk_cleared"])
        self.assertFalse(report["independent_oos_verified"])
        self.assertEqual(report["trade_target_signals_exported"],0)
        self.assertEqual(report["live_orders_sent"],0)
        self.assertEqual(report["renko_source_timeframe_seconds"],300)
        self.assertTrue(report["source_data_synthetic"])
        self.assertGreater(report["candidate_imba_flip_count"],0)
        self.assertEqual(len(report["candidates"]),
                         report["candidate_imba_flip_count"])

    def test_not_1min_source_rejected_and_no_incomplete_group(self):
        prov5=Provenance(
            source="synthetic 5m",license="CC0",symbol="SOL-DEMO",
            market_type="linear_futures",interval_seconds=300,synthetic=True,
        )
        five=self.root/"five.csv"
        with five.open("w",newline="") as handle:
            w=csv.DictWriter(handle,fieldnames=[
                "timestamp","open","high","low","close","volume"])
            w.writeheader()
            for j in range(3):
                w.writerow({"timestamp":(START+timedelta(minutes=j*5)).isoformat(),
                            "open":"100","high":"101","low":"99","close":"100",
                            "volume":"100"})
        path=self.root/"five.sqlite"
        ingest_ohlcv(five,path,prov5)
        with self.assertRaisesRegex(SignalAuditError,"exact 1m source"):
            source_5min_bars(str(path),START,START+timedelta(minutes=15))
        with self.assertRaisesRegex(SignalAuditError,"align"):
            source_5min_bars(str(self.index),
                             START+timedelta(minutes=1),self.end)


if __name__ == "__main__":
    unittest.main()
