#!/usr/bin/env python3
"""Offline 1m -> completed 5m -> legacy Renko-0.1 -> IMBA research audit.

Never creates executable exchange orders or claims TradingView wick parity.
Optional --reference-alerts must contain confirmation time normalized to the
5-minute *source candle closing time*, not time of webhook receipt.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

from quant.ecosystem.market_data.streaming import get_manifest
from quant.ecosystem.market_data.renko_imba_audit import (
    ReferenceAlert, candidate_imba_signals, compare_confirmed_alerts,
)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--database",required=True)
    p.add_argument("--start",required=True,help="UTC start including warmup, 5m aligned")
    p.add_argument("--trading-start",required=True,help="first 5m-close to report")
    p.add_argument("--end",required=True,help="exclusive source 5m start")
    p.add_argument("--strategy-id",required=True)
    p.add_argument("--lookback",type=int,default=50)
    p.add_argument("--box",default="0.1")
    p.add_argument("--max-source-minutes",type=int,default=10080)
    p.add_argument("--reference-alerts",type=Path,
                   help="CSV reference_id,direction,confirmed_5m_close_at")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    batch=candidate_imba_signals(
        args.database,start=args.start,end=args.end,
        trading_start=args.trading_start,strategy_id=args.strategy_id,
        box=args.box,lookback=args.lookback,
        max_source_1min=args.max_source_minutes,
    )
    data=get_manifest(args.database)
    comparisons=None
    if args.reference_alerts:
        with args.reference_alerts.open("r",newline="",encoding="utf-8") as source:
            reader=csv.DictReader(source)
            required={"reference_id","direction","confirmed_5m_close_at"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                p.error("reference-alerts missing required source-close fields")
            records=[
                ReferenceAlert(row["reference_id"],int(row["direction"]),
                               row["confirmed_5m_close_at"])
                for row in reader
            ]
        comparisons=asdict(compare_confirmed_alerts(batch,records))
    report={
        "schema":"research-imba-5m-audit-1",
        "data_source":data["provenance"],
        "source_fingerprint":data["dataset_fingerprint"],
        "source_data_synthetic":data["provenance"]["synthetic"],
        "renko_source_timeframe_seconds":batch.renko_source_interval,
        "renko_close_only_no_tv_wicks":True,
        "candidate_imba_flip_count":len(batch.candidates),
        "five_minute_source_bars":batch.source_candles_5m,
        "candidate_bricks":batch.brick_count,
        "reference_comparison":comparisons,
        "full_tradingview_parity_verified":False,
        "independent_oos_verified":False,
        "candidate_only":True,
        "trade_target_signals_exported":0,
        "capital_risk_cleared":False,
        "live_orders_sent":0,
        "candidates":[{
            **asdict(c),
            "source_5m_open_at":c.source_5m_open_at.isoformat(),
            "available_at":c.available_at.isoformat(),
            "renko_brick_at":c.renko_brick_at.isoformat(),
            "box":str(c.box),
        } for c in batch.candidates],
    }
    if args.output.exists():
        p.error("output exists: overwrite refused")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,sort_keys=True,indent=2,default=str)+"\n",
                           encoding="utf-8")
    print(json.dumps({
        k:v for k,v in report.items() if k!="candidates"
    },sort_keys=True,default=str))


if __name__=="__main__":
    main()
