"""Durable, fast TradingView inbox for the Blaupunkt multi-bot hedge fleet.

TradingView drops HTTP requests that take longer than 3 seconds. No KuCoin
or Postgres roundtrip belongs in the webhook acknowledgment path.

Only one Railway replica may run against this SQLite file and exchange account.
Orders are serialized by the fleet's account-wide lock. An interrupted
processing event is never replayed blindly.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from quant.execution.blaupunkt_hedge import fleet
from quant.execution.tv_signal_executor import TVExecConfig, TVSignal
from quant.utils.log import get_logger

log = get_logger("quant.hedge_webhook_queue")
_LOCK = threading.RLock()
_WAKE = threading.Event()
_STARTED = False


def _db_path() -> Path:
    return Path(os.getenv("HEDGE_WEBHOOK_QUEUE_FILE", "/data/blaupunkt-webhooks.sqlite3"))


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path), timeout=2.0)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=2000")
    db.execute("""
        CREATE TABLE IF NOT EXISTS hedge_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            received_at REAL NOT NULL,
            bot TEXT NOT NULL,
            action TEXT NOT NULL,
            side TEXT NOT NULL,
            symbol TEXT NOT NULL,
            sl_price REAL,
            tp1_pct REAL NOT NULL,
            dry_run INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            result_json TEXT,
            updated_at REAL NOT NULL
        )
    """)
    db.commit()
    return db


def enqueue(bot: str, signal: TVSignal, config: TVExecConfig) -> Dict[str, Any]:
    """Quick local disk commit; no broker IO and no remote SQL calls."""
    if not bot or not bot.replace("-", "").replace("_", "").isalnum() or len(bot) > 40:
        raise ValueError("invalid bot ID")
    if signal.action not in {"entry", "flip", "exit", "tp1", "tp2", "sl"}:
        raise ValueError("invalid action")
    now = time.time()
    # TradingView may resend timed-out alerts, occasionally 3+ times.
    # Dedup only near-identical alerts in a 4-second window; persistence
    # keeps this protection across restarts.
    with _LOCK:
        with _connect() as db:
            last = db.execute(
                """SELECT id, received_at, status
                   FROM hedge_alerts
                   WHERE lower(bot)=lower(?) AND action=? AND side=? AND symbol=?
                     AND received_at>? ORDER BY id DESC LIMIT 1""",
                (bot, signal.action, signal.side, signal.symbol, now - 4.0)
            ).fetchone()
            if last:
                _WAKE.set()
                return {"ok": True, "queued": True, "duplicate": True,
                        "event_id": int(last["id"]), "bot": bot}
            row = db.execute(
                """INSERT INTO hedge_alerts
                   (received_at, bot, action, side, symbol, sl_price,
                    tp1_pct, dry_run, status, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?)""",
                (now, bot, signal.action, signal.side, signal.symbol,
                 signal.sl_price, config.tp1_close_pct, int(config.dry_run), now),
            )
            event_id = int(row.lastrowid)
    _WAKE.set()
    return {"ok": True, "queued": True, "event_id": event_id, "bot": bot}


def status(limit: int = 20) -> Dict[str, Any]:
    with _LOCK:
        with _connect() as db:
            rows = db.execute(
                """SELECT id, received_at, bot, action, side, symbol,
                          status, result_json, updated_at
                   FROM hedge_alerts ORDER BY id DESC LIMIT ?""",
                (max(1, min(int(limit), 100)),),
            ).fetchall()
            count = db.execute(
                "SELECT status, count(*) AS n FROM hedge_alerts GROUP BY status"
            ).fetchall()
            return {
                "counts": {r["status"]: int(r["n"]) for r in count},
                "recent": [
                    {
                        "id": int(r["id"]), "received_at": r["received_at"],
                        "bot": r["bot"], "action": r["action"],
                        "side": r["side"], "symbol": r["symbol"],
                        "status": r["status"],
                        "result": json.loads(r["result_json"]) if r["result_json"] else None,
                    } for r in rows
                ]
            }


def _process_next() -> bool:
    with _LOCK:
        with _connect() as db:
            # Never continue if a previous order attempt was interrupted.
            if db.execute(
                "SELECT 1 FROM hedge_alerts WHERE status='interrupted' LIMIT 1"
            ).fetchone():
                log.error("Hedge queue quarantined: interrupted order attempt requires reconciliation")
                return False
            r = db.execute(
                "SELECT * FROM hedge_alerts WHERE status='queued' ORDER BY id LIMIT 1"
            ).fetchone()
            if r is None:
                return False
            data = dict(r)
            db.execute(
                "UPDATE hedge_alerts SET status='processing',updated_at=? WHERE id=?",
                (time.time(), data["id"]),
            )

    # No queue lock while waiting for exchange.
    try:
        if time.time() - data["received_at"] > float(os.getenv("HEDGE_MAX_ALERT_AGE_SEC", "120")):
            result = {"ok": False, "reason": "stale_alert"}
            state = "rejected"
        else:
            result = fleet().execute(
                bot_id=data["bot"], action=data["action"], side=data["side"],
                symbol=data["symbol"], sl_price=data["sl_price"],
                tp1_pct=float(data["tp1_pct"]), dry_run=bool(data["dry_run"]),
            )
            state = "executed" if result.get("ok") else "rejected"
    except Exception as exc:
        log.exception("Blaupunkt hedge alert rejected event=%s bot=%s", data["id"], data["bot"])
        result = {"ok": False, "reason": type(exc).__name__, "detail": str(exc)}
        state = "failed"
    with _LOCK:
        with _connect() as db:
            db.execute(
                "UPDATE hedge_alerts SET status=?,result_json=?,updated_at=? WHERE id=?",
                (state, json.dumps(result, default=str), time.time(), data["id"]),
            )
    log.info("Blaupunkt hedge event=%s bot=%s action=%s status=%s result=%s",
             data["id"], data["bot"], data["action"], state, result)
    return True


def _work() -> None:
    # Any thread that was mid-execution when the service died must be
    # reconciled before another alert is allowed to trade. Never replay an
    # in-flight trade simply because its HTTP response was lost.
    with _LOCK:
        with _connect() as db:
            db.execute(
                "UPDATE hedge_alerts SET status='interrupted',updated_at=? WHERE status='processing'",
                (time.time(),),
            )
    while True:
        try:
            did_work = _process_next()
            if did_work:
                continue
        except Exception:
            log.exception("hedge queue worker failed; will check again")
        _WAKE.wait(timeout=2.0)
        _WAKE.clear()


def start_worker() -> None:
    global _STARTED
    with _LOCK:
        if _STARTED:
            return
        _STARTED = True
        thread = threading.Thread(
            target=_work, name="blaupunkt-hedge-alert-worker", daemon=True
        )
        thread.start()
