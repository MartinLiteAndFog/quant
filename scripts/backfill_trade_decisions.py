"""One-shot backfill for the trade_decisions table.

Run after deploying the trade-counter feature to populate ``trade_decisions``
from existing ``action_events`` history. Idempotent: re-running yields the same
rows (every ``decision_id`` is deterministic and the upsert is a no-op on
conflict).

Usage::

    POSTGRES_URL=... python scripts/backfill_trade_decisions.py --venue kucoin --symbol SOL-USDT

Omit ``--venue`` / ``--symbol`` to backfill every venue / symbol.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill trade_decisions from action_events")
    p.add_argument("--venue", type=str, default=None, help="Optional venue filter, e.g. kucoin")
    p.add_argument("--symbol", type=str, default=None, help="Optional symbol filter, e.g. SOL-USDT")
    p.add_argument("--since-ts", type=str, default=None, help="Optional lower bound on ts (ISO8601)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from quant.execution.trade_decisions_store import (
        backfill_trade_decisions_from_action_events,
    )

    info = backfill_trade_decisions_from_action_events(
        venue=args.venue,
        symbol=args.symbol,
        since_ts=args.since_ts,
    )
    print(json.dumps(info, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
