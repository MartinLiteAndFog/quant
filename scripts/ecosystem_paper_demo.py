#!/usr/bin/env python3
"""Paper-only reproducible synthetic daily trading-ecosystem demonstration.

Usage: PYTHONPATH=src python scripts/ecosystem_paper_demo.py --days 30
No exchange/TradingView/Railway access, credentials, requests or writes.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import json

from quant.ecosystem.ecology.physiology import Weather
from quant.ecosystem.integration.research_world import (
    simple_demo_population, start_world, step_world,
)
from quant.ecosystem.sim_clock.clock import IntradayEvent


def synthetic_day(day: date, index: int) -> list[IntradayEvent]:
    morning = datetime(day.year, day.month, day.day, 9, tzinfo=timezone.utc)
    afternoon = datetime(day.year, day.month, day.day, 13, tzinfo=timezone.utc)
    # Intentionally handcrafted PnL; NOT generated from a real trading signal.
    gross = Decimal("0.12") if index % 5 in (0, 1, 2) else Decimal("-0.09")
    return [
        IntradayEvent(f"synthetic:pnl:{index}", morning, morning,
                      "pnl_delta", gross, "parent-00"),
        IntradayEvent(f"synthetic:fee:{index}", afternoon, afternoon,
                      "fee", Decimal("-0.01"), "parent-00"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-only synthetic ecosystem, no live market orders")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--equity", default="100")
    args = parser.parse_args()
    if not 1 <= args.days <= 3650:
        parser.error("--days must be between 1 and 3650")

    start = date(2026, 1, 1)
    population = simple_demo_population()
    state = start_world("synthetic-paper-ecosystem", start, args.equity, population, seed=args.seed)
    initial_nav = state.equity
    for index in range(args.days):
        day = start + timedelta(days=index)
        seen = datetime(day.year, day.month, day.day, 15, tzinfo=timezone.utc)
        sunshine = 0.35 if index % 7 == 0 else 0.8
        weather = Weather(seen, seen, sunshine, synthetic=True)
        state = step_world(
            state, day, synthetic_day(day, index), population, weather,
            free_margin="20",
        ).state

    game = state.ecology
    claims = sum((Decimal(v) for v in game["entitlements"].values()), Decimal("0"))
    paper_intents = sum((Decimal(v["entry_margin"]) for v in game["intents"].values()), Decimal("0"))
    result = {
        "mode": "SYNTHETIC PAPER DEMO — NOT A BACKTEST",
        "simulation_id": state.simulation_id, "days": args.days, "seed": state.seed,
        "as_of": state.as_of.isoformat(), "plants": len(game["plants"]),
        "initial_equity": str(initial_nav), "equity": str(state.equity),
        "reserve_10pct": str(state.reserve),
        "distributable_90pct": str(state.distributable),
        "entitlements_total": str(claims),
        "observed_entry_margin_intent": str(paper_intents),
        "water_depth": game["groundwater"]["depth"],
        "water_eligible_plants": sum(
            1 for item in game["intents"].values() if item["water_eligible"]
        ),
        "audit_events": len(state.audit),
        "checkpoint_sha256": sha256(state.to_json().encode()).hexdigest(),
        "live_orders_sent": 0,
    }
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
