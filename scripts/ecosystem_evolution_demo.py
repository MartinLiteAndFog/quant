#!/usr/bin/env python3
"""ECO-17 synthetic paper evolution preview, not a backtest or live executor.

From repo root: PYTHONPATH=src python scripts/ecosystem_evolution_demo.py --days 30
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import json

from quant.ecosystem.ecology.physiology import Weather
from quant.ecosystem.integration.evolution_cycle import (
    CycleConfig, evolution_report, start_evolution, step_evolution,
)
from quant.ecosystem.lifecycle.predation import PredatorConfig
from quant.ecosystem.sim_clock.clock import IntradayEvent


def main() -> None:
    parser=argparse.ArgumentParser(description="Synthetic paper-only evolutionary ecology")
    parser.add_argument("--days",type=int,default=30)
    parser.add_argument("--seed",type=int,default=2026)
    parser.add_argument("--equity",default="100")
    args=parser.parse_args()
    if not 1<=args.days<=90:
        parser.error("--days must be in [1,90] for this short CPU-only preview")
    config=CycleConfig(
        reproduction_every_days=5,
        predators=PredatorConfig(attacks_per_day=0),
    )
    first=date(2026,1,1)
    state=start_evolution("synthetic-bee-evolution",first,args.equity,
                          seed=args.seed,config=config)
    for index in range(args.days):
        day=first+timedelta(days=index)
        timestamp=datetime(day.year,day.month,day.day,12,tzinfo=timezone.utc)
        candle=Weather(timestamp,timestamp,.65 if index%7==0 else .8,True)
        living=state.ecology["organisms"]
        counts={id_:1 for id_,v in living.items() if v["status"]!="closing"}
        events=(
            [IntradayEvent(f"synthetic-pnl-{index}",timestamp,timestamp,
                           "pnl_delta",Decimal(".02"),"parent-00")]
            if "parent-00" in living else []
        )
        state=step_evolution(
            state,day,events,candle,trades=counts,opportunities=counts,
            free_margin="20",config=config,
        ).state
    result=evolution_report(state)
    result["checkpoint_sha256"]=sha256(state.to_json().encode()).hexdigest()
    result["birth_events"]=[
        e for e in state.ecology["evolution_events"] if e["kind"]=="birth"
    ][:8]
    result["notice"]="All parent/offspring results are synthetic fixtures; no verified SOL history, no real orders"
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
