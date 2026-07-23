"""Centralized fleet equity poller.

Writes one ``equity_snapshots`` row per fleet bot every
``FLEET_EQUITY_POLL_SEC`` (default 900s = 15 min) by fanning out to each bot's
public ``/health`` endpoint, which already reports that account's live equity
from its own venue key.

Why this exists
---------------
Fleet charts (``fleet_api.build_fleet_performance``) build history from
``equity_snapshots`` rows and only *live-stitch* health equity onto a series
that already has a **fresh** snapshot. Accounts whose own snapshot writer was
not persisting therefore rendered empty curves even though ``/health`` returned
live equity:

  - Kraken legacy: the Kraken service's writer defaulted OFF and was hardcoded
    to ``venue=kucoin, account=futures`` — so no ``kraken/main`` rows existed.
  - Funded KuCoin sub-account pilots: per-process ``bot_webhook`` writers had
    credential / thread gaps, so their snapshot backbone was missing too.

One poller on the fleet service gives every reachable account a snapshot
backbone with **no extra credentials distributed** — it consumes the same
per-account live equity the fleet API already fans out to for the live stitch.

Runs on
-------
``webhook_server`` (the dashboard/fleet host that serves ``/api/fleet/*`` and
already probes bot health). Enable on exactly one host; other ``webhook_server``
deployments can opt out with ``FLEET_EQUITY_POLL_ENABLED=0`` (writes are
idempotent via ``insert_equity_snapshot`` on-conflict-do-nothing regardless).

Env
---
  FLEET_EQUITY_POLL_ENABLED   "1"/"0" — overrides the caller's default
  FLEET_EQUITY_POLL_SEC       write interval seconds (default 900, min 120)
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional

from quant.utils.log import get_logger

log = get_logger("quant.fleet_equity_poller")

_STARTED = threading.Event()


def _truthy(v: Optional[str]) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "on"}


def _interval_sec() -> float:
    try:
        return max(120.0, float(os.getenv("FLEET_EQUITY_POLL_SEC", "900")))
    except Exception:
        return 900.0


def _account_for(bot: Dict[str, Any]) -> str:
    """Snapshot account key: explicit override else the strategy instance.

    Mirrors ``fleet_api._load_account_points_for_bot`` so the rows this writer
    persists are read back on the exact ``(venue, account)`` the chart queries.
    """
    return str(bot.get("equity_account") or bot.get("strategy_instance") or "").strip()


def poll_once() -> int:
    """Probe every fleet bot's health and persist one equity snapshot each.

    Returns the number of rows written. Bots whose ``/health`` does not report a
    positive equity (unreachable, dry, or missing venue credentials) are skipped
    — no equity is invented for them.
    """
    import pandas as pd

    from quant.execution.event_store import insert_equity_snapshot
    from quant.execution.fleet_api import list_fleet_bots

    written = 0
    payload = list_fleet_bots(probe_health=True)
    for bot in payload.get("bots") or []:
        raw_equity = bot.get("equity")
        if raw_equity is None:
            continue
        try:
            equity = float(raw_equity)
        except Exception:
            continue
        if not (equity > 0) or equity != equity:  # skip <=0, NaN
            continue

        venue = str(bot.get("venue") or "kucoin")
        account = _account_for(bot)
        if not account:
            continue

        try:
            insert_equity_snapshot(
                {
                    "ts": pd.Timestamp.now("UTC"),
                    "venue": venue,
                    "account": account,
                    "symbol": None,
                    "equity": equity,
                    "currency": bot.get("currency"),
                    "source": "fleet_equity_poller",
                    "payload_json": {
                        "id": bot.get("id"),
                        "status": bot.get("status"),
                        "equity_source": bot.get("equity_source"),
                        "available": bot.get("available"),
                    },
                }
            )
            written += 1
        except Exception as e:
            log.warning(
                "fleet equity poll write failed id=%s venue=%s account=%s: %s",
                bot.get("id"), venue, account, e,
            )
    return written


def start_fleet_equity_poller(
    *,
    default_enabled: bool = True,
) -> Optional[threading.Thread]:
    """Start the background poller thread (idempotent per process)."""
    env = os.getenv("FLEET_EQUITY_POLL_ENABLED")
    enabled = _truthy(env) if env is not None and env.strip() != "" else default_enabled
    if not enabled:
        log.info("fleet equity poller disabled")
        return None
    if _STARTED.is_set():
        return None
    _STARTED.set()

    def _loop() -> None:
        # Let broker clients / credentials warm up before the first fan-out.
        time.sleep(15.0)
        while True:
            try:
                n = poll_once()
                if n:
                    log.info("fleet equity poller wrote %d snapshot(s)", n)
            except Exception as e:
                log.warning("fleet equity poll cycle failed: %s", e)
            time.sleep(_interval_sec())

    t = threading.Thread(target=_loop, name="fleet-equity-poller", daemon=True)
    t.start()
    log.info(
        "fleet equity poller started interval=%ss", int(_interval_sec())
    )
    return t
