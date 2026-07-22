"""Autonomous equity snapshot writer for fleet performance curves.

Root-cause fix (fleet cockpit audit 2026-07-22): ``equity_snapshots`` rows were
only ever written as a side effect of ``/health`` polls, gated behind
``FLEET_PERSIST_LIVE_EQUITY`` (default off). Postgres therefore held stale
seeds, and the fleet API faked curves by forward-filling them and stitching
live equity on the end. This writer persists a snapshot on its own clock,
independent of who polls what.

Started by:
  - ``bot_webhook`` (Railway pilots): default ON, account = BOT_INSTANCE_ID
  - ``webhook_server`` (dashboard service): default OFF, opt-in via env

Env:
  FLEET_EQUITY_WRITER_ENABLED   "1"/"0" — overrides the caller's default
  FLEET_EQUITY_WRITER_SEC       write interval seconds (default 900, min 120)
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional

from quant.utils.log import get_logger

log = get_logger("quant.equity_snapshot_writer")

_STARTED = threading.Event()


def _truthy(v: Optional[str]) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "on"}


def _interval_sec() -> float:
    try:
        return max(120.0, float(os.getenv("FLEET_EQUITY_WRITER_SEC", "900")))
    except Exception:
        return 900.0


def _fetch_equity(venue: str) -> Dict[str, Any]:
    from quant.execution.live_account import fetch_kraken_account, fetch_kucoin_account

    if venue == "kraken":
        # persist=False: this writer owns persistence; avoid double-gating.
        return fetch_kraken_account(persist=False)
    return fetch_kucoin_account(persist_account=None)


def write_equity_snapshot_once(*, venue: str, account: str) -> bool:
    """Fetch live equity for this process's credentials and persist one row."""
    payload = _fetch_equity(venue)
    if not payload.get("ok"):
        log.warning(
            "equity snapshot skipped venue=%s account=%s error=%s",
            venue, account, payload.get("error"),
        )
        return False
    try:
        equity = float(payload.get("equity") or 0.0)
    except Exception:
        return False
    if equity <= 0:
        return False

    import pandas as pd

    from quant.execution.event_store import insert_equity_snapshot

    insert_equity_snapshot(
        {
            "ts": pd.Timestamp.now("UTC"),
            "venue": venue,
            "account": account,
            "symbol": None,
            "equity": equity,
            "currency": payload.get("currency"),
            "source": "equity_snapshot_writer",
            "payload_json": payload,
        }
    )
    return True


def start_equity_snapshot_writer(
    *,
    venue: str,
    account: str,
    default_enabled: bool = True,
) -> Optional[threading.Thread]:
    """Start the background writer thread (idempotent per process)."""
    env = os.getenv("FLEET_EQUITY_WRITER_ENABLED")
    enabled = _truthy(env) if env is not None and env.strip() != "" else default_enabled
    if not enabled:
        log.info("equity snapshot writer disabled (venue=%s account=%s)", venue, account)
        return None
    if not account:
        log.warning("equity snapshot writer: no account tag; not starting")
        return None
    if _STARTED.is_set():
        return None
    _STARTED.set()

    def _loop() -> None:
        # Let credentials / broker clients warm up before the first write.
        time.sleep(15.0)
        while True:
            try:
                ok = write_equity_snapshot_once(venue=venue, account=account)
                if ok:
                    log.info("equity snapshot written venue=%s account=%s", venue, account)
            except Exception as e:
                log.warning("equity snapshot write failed: %s", e)
            time.sleep(_interval_sec())

    t = threading.Thread(target=_loop, name="equity-snapshot-writer", daemon=True)
    t.start()
    log.info(
        "equity snapshot writer started venue=%s account=%s interval=%ss",
        venue, account, int(_interval_sec()),
    )
    return t
