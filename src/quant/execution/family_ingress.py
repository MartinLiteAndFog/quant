"""Opt-in, NO-TRADE multi-strategy registration and TradingView ingress.

This service records routed alerts for the Blaupunkt family. It deliberately
does NOT call any exchange executor. Enable the separate family order writer
only after a durable hedge-mode ledger, venue reconciliation and operator signoff.
The existing /webhook/tv-execute is unchanged and keeps its legacy behavior.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from quant.execution.event_store import get_conn

router = APIRouter(tags=["family-router"])

ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")
ACTIONS = {"entry", "exit", "flip", "tp1", "tp2", "sl"}
SIDE_ACTIONS = {"entry", "flip"}


def _enabled() -> None:
    if str(os.getenv("FAMILY_ROUTER_ENABLED", "")).lower() not in ("1", "true", "yes"):
        raise HTTPException(status_code=404, detail="family router disabled")


def _family_id() -> str:
    family_id = str(os.getenv("FAMILY_ID", "blaupunkt")).strip().lower()
    if not ID_RE.fullmatch(family_id):
        raise HTTPException(status_code=503, detail="invalid family configuration")
    return family_id


def _admin(auth: Optional[str]) -> None:
    _enabled()
    expected = str(os.getenv("FAMILY_ADMIN_TOKEN", "")).strip()
    if not expected:
        raise HTTPException(status_code=503, detail="family admin credentials not configured")
    if not hmac.compare_digest(str(auth or ""), expected):
        raise HTTPException(status_code=401, detail="unauthorized")


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _symbol(symbol: str) -> str:
    sym = str(symbol).upper().replace("-", "").replace("/", "").replace("_", "")
    if sym not in ("SOLUSDT", "SOLUSDTM"):
        raise HTTPException(status_code=422, detail="only SOL-USDT family signals supported")
    return "SOLUSDTM"


def _schema() -> None:
    # Does not alter any existing trading table or legacy webhook state.
    ddl = """
    create table if not exists family_strategy_registry (
      family_id text not null,
      strategy_id text primary key,
      parent_strategy_id text,
      display_name text not null,
      pool text not null check (pool in ('evolution', 'angel')),
      token_sha256 text not null,
      config_json jsonb not null default '{}'::jsonb,
      -- Registration is not authorization for trading.
      execution_status text not null default 'shadow'
        check (execution_status in ('shadow', 'disabled')),
      created_at timestamptz not null default now()
    );
    create index if not exists idx_family_strategy_registry_family
      on family_strategy_registry (family_id, created_at desc);
    create table if not exists family_signal_inbox (
      family_id text not null,
      strategy_id text not null references family_strategy_registry(strategy_id),
      event_id text not null,
      action text not null,
      side text not null,
      symbol text not null,
      payload_sha256 text not null,
      status text not null default 'recorded_no_execution',
      created_at timestamptz not null default now(),
      primary key (strategy_id, event_id)
    );
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(ddl)


class NewStrategy(BaseModel):
    strategy_id: str = Field(min_length=3, max_length=64)
    display_name: str = Field(min_length=1, max_length=100)
    pool: str = "evolution"
    parent_strategy_id: Optional[str] = None
    configuration: Dict[str, Any] = Field(default_factory=dict)


def _validate_registration(data: NewStrategy) -> None:
    if not ID_RE.fullmatch(data.strategy_id):
        raise HTTPException(status_code=422, detail="invalid strategy_id format")
    if data.parent_strategy_id is not None and not ID_RE.fullmatch(data.parent_strategy_id):
        raise HTTPException(status_code=422, detail="invalid parent_strategy_id")
    if data.parent_strategy_id == data.strategy_id:
        raise HTTPException(status_code=422, detail="strategy cannot parent itself")
    if data.pool not in ("evolution", "angel"):
        raise HTTPException(status_code=422, detail="pool must be evolution or angel")
    if len(json.dumps(data.configuration)) > 16_384:
        raise HTTPException(status_code=422, detail="configuration exceeds 16 KB")


@router.post("/api/family/strategies", status_code=201)
def register_strategy(
    data: NewStrategy,
    x_family_admin_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Create a shadow-only strategy and issue its per-strategy secret ONCE."""
    _admin(x_family_admin_token)
    _validate_registration(data)
    family = _family_id()
    secret = secrets.token_urlsafe(32)
    _schema()
    with get_conn() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                if data.parent_strategy_id:
                    cur.execute(
                        "select pool from family_strategy_registry "
                        "where family_id=%s and strategy_id=%s",
                        (family, data.parent_strategy_id),
                    )
                    ancestor = cur.fetchone()
                    if ancestor is None:
                        raise HTTPException(status_code=422, detail="unknown parent in this family")
                    if ancestor[0] != data.pool:
                        raise HTTPException(status_code=422, detail="child must inherit parent's pool")
                cur.execute(
                    """insert into family_strategy_registry
                       (family_id, strategy_id, parent_strategy_id,
                        display_name, pool, token_sha256, config_json)
                       values (%s,%s,%s,%s,%s,%s,%s::jsonb)
                       on conflict(strategy_id) do nothing
                       returning strategy_id""",
                    (
                        family, data.strategy_id, data.parent_strategy_id,
                        data.display_name, data.pool, _token_hash(secret),
                        json.dumps(data.configuration, separators=(",", ":")),
                    ),
                )
                if cur.fetchone() is None:
                    raise HTTPException(status_code=409, detail="strategy_id already registered")
    return {
        "family_id": family,
        "strategy_id": data.strategy_id,
        "parent_strategy_id": data.parent_strategy_id,
        "pool": data.pool,
        "execution_status": "shadow",
        "webhook_path": "/webhook/family",
        "token_once": secret,
        "tradingview_message_example": {
            "strategy_id": data.strategy_id,
            "event_id": "REPLACE_WITH_UNIQUE_SIGNAL_ID",
            "token": secret,
            "action": "entry",
            "side": "buy",
            "symbol": "SOL-USDT",
        },
        "note": "This endpoint records signals but NEVER places orders.",
    }


@router.get("/api/family/strategies")
def list_strategies(
    x_family_admin_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _admin(x_family_admin_token)
    family = _family_id()
    _schema()
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """select strategy_id, parent_strategy_id, display_name, pool,
                      execution_status, created_at
               from family_strategy_registry where family_id=%s
               order by created_at, strategy_id""",
            (family,),
        )
        rows = cur.fetchall()
    return {"family_id": family, "strategies": [
        {
            "strategy_id": r[0], "parent_strategy_id": r[1],
            "display_name": r[2], "pool": r[3],
            "execution_status": r[4], "created_at": r[5].isoformat(),
        } for r in rows
    ]}


@router.post("/webhook/family")
async def ingest_family_signal(request: Request) -> Dict[str, Any]:
    """Authenticate and deduplicate alerts. Execution is NOT implemented."""
    _enabled()
    try:
        payload = await request.json()
    except (ValueError, UnicodeError):
        raise HTTPException(status_code=400, detail="invalid JSON")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="expected JSON object")
    strategy_id = str(payload.get("strategy_id") or "").strip().lower()
    token = str(payload.get("token") or "").strip()
    if not ID_RE.fullmatch(strategy_id) or not token:
        raise HTTPException(status_code=401, detail="invalid strategy credentials")
    family = _family_id()
    _schema()
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """select token_sha256, execution_status
               from family_strategy_registry
               where family_id=%s and strategy_id=%s""",
            (family, strategy_id),
        )
        row = cur.fetchone()
    if row is None or not hmac.compare_digest(_token_hash(token), row[0]):
        raise HTTPException(status_code=401, detail="invalid strategy credentials")
    if row[1] != "shadow":
        raise HTTPException(status_code=403, detail="strategy disabled")

    event_id = str(payload.get("event_id") or "").strip()
    action = str(payload.get("action") or "").strip().lower()
    side = str(payload.get("side") or "").strip().lower()
    if not 1 <= len(event_id) <= 128:
        raise HTTPException(status_code=422, detail="unique event_id required (max 128)")
    if action not in ACTIONS:
        raise HTTPException(status_code=422, detail="unsupported action")
    if action in SIDE_ACTIONS and side not in ("buy", "sell"):
        raise HTTPException(status_code=422, detail="buy/sell side required")
    if action not in SIDE_ACTIONS and side not in ("", "buy", "sell"):
        raise HTTPException(status_code=422, detail="invalid side")
    symbol = _symbol(str(payload.get("symbol") or "SOL-USDT"))
    fingerprint = _token_hash(json.dumps(
        [strategy_id, event_id, action, side, symbol], separators=(",", ":")
    ))
    with get_conn() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    """insert into family_signal_inbox
                         (family_id, strategy_id, event_id, action, side,
                          symbol, payload_sha256)
                       values (%s,%s,%s,%s,%s,%s,%s)
                       on conflict(strategy_id,event_id) do nothing
                       returning event_id""",
                    (family, strategy_id, event_id, action, side, symbol, fingerprint),
                )
                inserted = cur.fetchone() is not None
                if not inserted:
                    cur.execute(
                        """select payload_sha256 from family_signal_inbox
                           where strategy_id=%s and event_id=%s""",
                        (strategy_id, event_id),
                    )
                    existing = cur.fetchone()
                    if existing is None or not hmac.compare_digest(fingerprint, existing[0]):
                        raise HTTPException(status_code=409, detail="event_id reused for a different signal")
    return {
        "ok": True,
        "family_id": family,
        "strategy_id": strategy_id,
        "event_id": event_id,
        "duplicate": not inserted,
        "status": "recorded_no_execution",
        "order_sent": False,
    }
