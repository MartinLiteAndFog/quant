#!/usr/bin/env python3
"""Offline market-to-ECO-03 replay. No exchange I/O or generated alpha.

Required: --bars UTC OHLCV CSV --signals timestamped target CSV
          --provenance JSON --output DIRECTORY
CSV bars: timestamp,open,high,low,close,volume
CSV signals: signal_id,strategy_id,target_quantity,signal_at,available_at,received_at
Optional funding: funding_id,at,known_at,rate,mark_price
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path

from quant.ecosystem.market_data import Provenance, normalize
from quant.ecosystem.integration.market_replay import (
    ReplayConfig, TargetSignal, FundingTick, replay_market,
)


def csv_rows(path: Path) -> list[dict]:
    with path.open("r",newline="",encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bars",type=Path,required=True)
    p.add_argument("--signals",type=Path,required=True)
    p.add_argument("--provenance",type=Path,required=True)
    p.add_argument("--funding",type=Path)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--max-target-quantity",default="100")
    p.add_argument("--allow-partial-days",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    provenance=Provenance(**json.loads(args.provenance.read_text(encoding="utf-8")))
    dataset=normalize(csv_rows(args.bars),provenance,start=args.start,end=args.end)
    targets=[TargetSignal(**r) for r in csv_rows(args.signals)]
    funding=[FundingTick(**r) for r in csv_rows(args.funding)] if args.funding else []
    config=ReplayConfig(max_abs_target_quantity=Decimal(args.max_target_quantity),
                        require_full_utc_days=not args.allow_partial_days)
    output=replay_market(dataset,targets,funding=funding,config=config)
    manifest={
        "schema_version":1,
        "mode":"PAPER ONLY: signals supplied externally; not a verified backtest",
        "market_source":dataset.report["provenance"],
        "source_data_synthetic":output.synthetic,
        "signals":len(targets),
        "paper_fills":len(output.fills),
        "days":len(output.days),
        "dataset_fingerprint":output.data_hash,
        "signal_and_funding_fingerprint":output.signal_hash,
        "net_pnl":str(sum((day.net_pnl for day in output.days),Decimal("0"))),
        "market_data_quality":dataset.report,
        "verified_oos":False,
        "capital_risk_cleared":False,
        "live_orders_sent":0,
    }
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"manifest.json").write_text(
        json.dumps(manifest,sort_keys=True,indent=2,default=str)+"\n",encoding="utf-8"
    )
    for name,records in (
        ("paper_fills.jsonl",(asdict(fill) for fill in output.fills)),
        ("paper_events.jsonl",(asdict(e) for e in output.events())),
        ("daily_positions.jsonl",(
            {"day":d.day.isoformat(),"positions":dict(d.end_positions),
             "net_pnl":str(d.net_pnl),"completed_trades":dict(d.completed_trades)}
            for d in output.days
        )),
    ):
        with (args.output/name).open("w",encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record,sort_keys=True,default=str)+"\n")
    print(json.dumps(manifest,sort_keys=True,default=str))


if __name__=="__main__":
    main()
