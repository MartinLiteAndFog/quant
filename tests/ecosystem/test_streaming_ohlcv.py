"""ECO-19a streaming, provenance and lossless UTC bar index acceptance tests.

Only synthetic fixtures. No internet, exchange, TradingView or real SOL PnL.
"""
import csv
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tracemalloc
import unittest

from quant.ecosystem.market_data import Provenance
from quant.ecosystem.market_data.streaming import (
    StreamImportError, get_manifest, ingest_ohlcv,
    iter_candles, read_window_as_dataset,
)

UTC = timezone.utc
FIRST = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
PROVENANCE = Provenance(
    source="synthetic test source",license="CC0 fixture",symbol="SOL-USDT-DEMO",
    market_type="linear_futures",contract_multiplier="1",
    interval_seconds=60,synthetic=True,
)
FIELDS=("timestamp","open","high","low","close","volume")


def row(i, **kwargs):
    data=dict(timestamp=(FIRST+timedelta(minutes=i)).isoformat(),
              open="100",high="101",low="99",close="100.5",volume="10")
    data.update(kwargs)
    return data


def make_file(path, indexes, *, compress=False, corrupt=None):
    if compress:
        with gzip.open(path,"wt",newline="",encoding="utf-8") as handle:
            w=csv.DictWriter(handle,fieldnames=FIELDS)
            w.writeheader()
            for i in indexes:
                w.writerow(row(i))
    else:
        with path.open("w",newline="",encoding="utf-8") as handle:
            w=csv.DictWriter(handle,fieldnames=FIELDS)
            w.writeheader()
            for i in indexes:
                w.writerow(row(i))
    if corrupt is not None:
        content=path.read_bytes()
        path.write_bytes(content[:corrupt])


class TestStreamingIndex(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.source=self.root/"ohlcv.csv.gz"
        self.dest=self.root/"bars.sqlite"

    def test_gzip_import_manifest_hash_ranges_and_small_window(self):
        make_file(self.source,[0,1,2,4,5,6],compress=True)
        r=ingest_ohlcv(self.source,self.dest,PROVENANCE,
                       expected_start=FIRST,
                       expected_end=FIRST+timedelta(minutes=8))
        self.assertEqual(r["rows"],6)
        self.assertEqual(r["missing_bars"],2)
        self.assertEqual(r["gap_runs"],2)
        self.assertEqual([g["missing_bars"] for g in r["gap_details"]],[1,1])
        self.assertEqual(r["coverage_claim"],"explicit_expected_window")
        self.assertTrue(r["source_file_sha256"])
        self.assertFalse(r["parity_with_TradingView_verified"])
        self.assertFalse(r["oos_verified"])
        self.assertEqual(r["live_orders_sent"],0)
        self.assertEqual(get_manifest(self.dest),r)
        rows=list(iter_candles(self.dest,FIRST.isoformat(),
                               (FIRST+timedelta(minutes=6)).isoformat()))
        self.assertEqual(len(rows),5)
        self.assertEqual(rows[2]["close"],"100.5")
        window=read_window_as_dataset(
            self.dest,FIRST,FIRST+timedelta(minutes=8))
        self.assertEqual(len(window.bars),6)
        self.assertEqual(window.report["missing_bars"],2)
        self.assertEqual(window.provenance.symbol,"SOL-USDT-DEMO")

    def test_gzip_stable_fingerprint_same_source_across_different_outputs(self):
        make_file(self.source,range(20),compress=True)
        r1=ingest_ohlcv(self.source,self.dest,PROVENANCE)
        r2=ingest_ohlcv(self.source,self.root/"another.sqlite",PROVENANCE)
        self.assertEqual(r1,r2)
        self.assertEqual(r1["canonical_candles_sha256"],
                         r2["canonical_candles_sha256"])
        self.assertEqual(r1["source_file_sha256"],r2["source_file_sha256"])
        with self.assertRaisesRegex(StreamImportError,"overwrite"):
            ingest_ohlcv(self.source,self.dest,PROVENANCE)

    def test_12_column_binance_layout_requires_matching_close_ms(self):
        source=self.root/"binance.csv.gz"
        with gzip.open(source,"wt",newline="") as f:
            w=csv.writer(f)
            for i in range(3):
                t=int((FIRST+timedelta(minutes=i)).timestamp()*1000)
                w.writerow([t,"100","101","99","100","1000",
                            t+59999,"0","0","0","0","0"])
        r=ingest_ohlcv(source,self.dest,PROVENANCE,layout="binance_klines")
        self.assertEqual(r["rows"],3)
        self.assertEqual(r["source_layout"],"binance_klines")
        self.assertTrue(r["provenance"]["synthetic"])
        self.assertEqual(len(list(iter_candles(
            self.dest,FIRST,FIRST+timedelta(minutes=3)))),3)

    def test_improper_binance_boundary_fails(self):
        source=self.root/"bad-binance.csv"
        t=int(FIRST.timestamp()*1000)
        with source.open("w",newline="") as f:
            csv.writer(f).writerow([t,"100","101","99","100","1000",
                                    t+60000,"0","0","0","0","0"])
        with self.assertRaisesRegex(StreamImportError,"close boundary"):
            ingest_ohlcv(source,self.dest,PROVENANCE,layout="binance_klines")
        self.assertFalse(self.dest.exists())

    def test_duplicate_and_unordered_bar_rejected_not_sorted(self):
        for seq in ([0,1,1,2],[0,2,1,3]):
            make_file(self.source,seq,compress=True)
            with self.assertRaisesRegex(StreamImportError,"duplicate or out-of-order"):
                ingest_ohlcv(self.source,self.dest,PROVENANCE)
            self.assertFalse(self.dest.exists())
            self.assertFalse(list(self.root.glob("*.partial")))

    def test_reject_gaps_fails_without_partial_or_placeholder_bars(self):
        make_file(self.source,[0,1,3],compress=True)
        with self.assertRaisesRegex(StreamImportError,"missing OHLCV"):
            ingest_ohlcv(self.source,self.dest,PROVENANCE,reject_gaps=True)
        self.assertFalse(self.dest.exists())
        result=ingest_ohlcv(self.source,self.dest,PROVENANCE,reject_gaps=False)
        self.assertEqual(result["missing_bars"],1)
        self.assertEqual(len(list(iter_candles(
            self.dest,FIRST,FIRST+timedelta(minutes=4)))),3)

    def test_bad_price_volume_bar_and_naive_timestamps_rejected(self):
        cases=[
            dict(high="99"),dict(low="101"),dict(open="0"),
            dict(close="NaN"),dict(volume="-1"),
            dict(timestamp="2026-01-01T00:00:00"),
            dict(timestamp="2026-01-01T00:00:00.500+00:00"),
        ]
        for i,malformed in enumerate(cases):
            f=self.root/f"bad-{i}.csv"
            with f.open("w",newline="") as out:
                w=csv.DictWriter(out,fieldnames=FIELDS)
                w.writeheader()
                w.writerow(row(0,**malformed))
            with self.assertRaises(StreamImportError):
                ingest_ohlcv(f,self.dest,PROVENANCE)
            self.assertFalse(self.dest.exists())

    def test_truncated_gzip_fails_atomically(self):
        make_file(self.source,range(300),compress=True,corrupt=60)
        with self.assertRaises(StreamImportError):
            ingest_ohlcv(self.source,self.dest,PROVENANCE)
        self.assertFalse(self.dest.exists())
        self.assertFalse(list(self.root.glob("*.partial")))

    def test_invalid_expected_window_and_bounded_view(self):
        make_file(self.source,range(20),compress=True)
        with self.assertRaisesRegex(StreamImportError,"outside declared coverage"):
            ingest_ohlcv(self.source,self.dest,PROVENANCE,
                        expected_end=FIRST+timedelta(minutes=10))
        self.assertFalse(self.dest.exists())
        r=ingest_ohlcv(self.source,self.dest,PROVENANCE)
        self.assertEqual(r["partial_days_total"],1)
        with self.assertRaisesRegex(StreamImportError,"window is too large"):
            read_window_as_dataset(self.dest,FIRST,FIRST+timedelta(days=365))
        with self.assertRaisesRegex(StreamImportError,"window must"):
            read_window_as_dataset(self.dest,FIRST,FIRST+timedelta(minutes=10),
                                   max_bars=100_001)

    def test_cli_refuses_implicit_real_data_without_explicit_synthetic_flag(self):
        source=self.root/"provenance-flag.csv"
        make_file(source,[0,1,2])
        details=asdict(PROVENANCE)
        del details["synthetic"]
        prov=self.root/"missing_flag.json"
        prov.write_text(json.dumps(details),encoding="utf-8")
        run=subprocess.run([
            sys.executable,"scripts/ecosystem_stream_ohlcv.py",
            "--source",str(source),"--provenance",str(prov),
            "--output",str(self.dest),
        ],capture_output=True,text=True)
        self.assertEqual(run.returncode,2)
        self.assertIn("explicitly specify",run.stderr)
        self.assertIn("synthetic",run.stderr)
        self.assertFalse(self.dest.exists())

    def test_large_fixture_does_not_materialize_entire_series_in_memory(self):
        large=self.root/"large.csv.gz"
        n=40_000
        with gzip.open(large,"wt",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=FIELDS)
            w.writeheader()
            for i in range(n):
                w.writerow(row(i))
        tracemalloc.start()
        try:
            result=ingest_ohlcv(large,self.dest,PROVENANCE,reject_gaps=True)
            _,peak=tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(result["rows"],n)
        self.assertEqual(result["gap_runs"],0)
        self.assertLess(peak,32*1024*1024)
        self.assertEqual(result["max_row_buffer"],4096)
        day=FIRST+timedelta(days=20)
        rows=list(iter_candles(self.dest,day,day+timedelta(minutes=10)))
        self.assertEqual(len(rows),10)

    def test_external_cli_needs_explicit_provenance_and_exports_report(self):
        source=self.root/"data.csv"
        make_file(source,[0,1,2])
        prov=self.root/"provenance.json"
        prov.write_text(json.dumps(asdict(PROVENANCE)))
        out=subprocess.run([
            sys.executable,"scripts/ecosystem_stream_ohlcv.py",
            "--source",str(source),"--layout","canonical",
            "--provenance",str(prov),"--output",str(self.dest),
        ],capture_output=True,text=True)
        self.assertEqual(out.returncode,0,out.stderr)
        report=json.loads(out.stdout)
        self.assertEqual(report["rows"],3)
        self.assertEqual(report["live_orders_sent"],0)
        self.assertTrue(self.dest.exists())
        second=subprocess.run([
            sys.executable,"scripts/ecosystem_stream_ohlcv.py",
            "--source",str(source),"--provenance",str(prov),
            "--output",str(self.dest),
        ],capture_output=True,text=True)
        self.assertNotEqual(second.returncode,0)
        self.assertIn("refusing overwrite",second.stderr)


if __name__ == "__main__":
    unittest.main()
