"""Read-only family controls and one-time legacy-parent bootstrap.

Mount beside family_ingress. No broker methods are called here.
"""
from __future__ import annotations

import hashlib
import json
import secrets
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Query

from quant.execution.event_store import get_conn
from quant.execution.family_ingress import _admin, _family_id, _schema
from quant.execution.family_tree_allocation import (
    AllocationError,
    StrategyNode,
    allocate_tree,
)

router = APIRouter(tags=["family-control"])


@router.post("/api/family/bootstrap-legacy")
def bootstrap_legacy(
    x_family_admin_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Register the existing Blaupunkt live bot as a root WITHOUT touching it.

    Its old /webhook/tv-execute URL and BOT_WEBHOOK_TOKEN remain unchanged.
    No new family token is issued for the legacy bot.
    """
    _admin(x_family_admin_token)
    family = _family_id()
    # Refuse to bootstrap a different service/account accidentally.
    if family != "blaupunkt":
        raise HTTPException(status_code=422, detail="bootstrap only approved for Blaupunkt")
    legacy = "blaupunkt"
    _schema()
    unused_secret_hash = hashlib.sha256(secrets.token_bytes(32)).hexdigest()
    with get_conn() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    """insert into family_strategy_registry
                       (family_id, strategy_id, parent_strategy_id, display_name,
                        pool, token_sha256, config_json, execution_status)
                       values (%s,%s,null,%s,'evolution',%s,%s::jsonb,'shadow')
                       on conflict (strategy_id) do nothing
                       returning strategy_id""",
                    (
                        family, legacy, "Blaupunkt (existing TradingView parent)",
                        unused_secret_hash,
                        json.dumps({"source": "legacy_tv_execute", "legacy_live": True}),
                    ),
                )
                created = cur.fetchone() is not None
                if not created:
                    cur.execute(
                        """select family_id, parent_strategy_id, pool
                           from family_strategy_registry where strategy_id=%s""",
                        (legacy,),
                    )
                    prior = cur.fetchone()
                    if prior != (family, None, "evolution"):
                        raise HTTPException(status_code=409, detail="existing blaupunkt registry conflicts")
    return {
        "ok": True, "family_id": family, "parent_strategy_id": legacy,
        "created": created, "legacy_webhook_unchanged": True,
        "execution_status": "shadow_registry_only",
        "order_sent": False,
    }


def _read_nodes(family: str) -> list[StrategyNode]:
    _schema()
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """select strategy_id, parent_strategy_id, pool, execution_status
               from family_strategy_registry where family_id=%s
               order by created_at, strategy_id""",
            (family,),
        )
        return [
            StrategyNode(strategy_id=r[0], parent_strategy_id=r[1], pool=r[2], status=r[3])
            for r in cur.fetchall()
        ]


@router.get("/api/family/allocation-preview")
def allocation_preview(
    equity: str = Query(..., description="Account equity entered manually; preview only"),
    x_family_admin_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _admin(x_family_admin_token)
    family = _family_id()
    nodes = _read_nodes(family)
    try:
        result = allocate_tree(equity, nodes)
    except AllocationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {
        "ok": True, "family_id": family, "is_preview": True,
        "venue_order_enabled": False,
        "equity": str(result.equity),
        "tradable_budget": str(result.usable_budget),
        "reserve": str(result.reserve),
        "active_parents": len(result.family_budgets),
        "family_budgets": {k: str(v) for k, v in result.family_budgets.items()},
        "bot_budgets": {k: str(v) for k, v in result.bot_budgets.items()},
        "conservation_ok": result.audit(),
        "notes": [
            "Preview equity is supplied by caller, not fetched from KuCoin.",
            "Amounts are theoretical margin budgets, NOT order quantities.",
            "No real orders or capital transfers occur.",
            "Angel pool not funded from the evolution 90%.",
        ],
    }


@router.get("/api/family/inbox")
def inbox(
    limit: int = Query(50, ge=1, le=200),
    x_family_admin_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _admin(x_family_admin_token)
    family = _family_id()
    _schema()
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """select i.strategy_id, i.event_id, i.action, i.side,
                      i.symbol, i.status, i.created_at
               from family_signal_inbox i
               where i.family_id=%s
               order by i.created_at desc
               limit %s""",
            (family, limit),
        )
        rows = cur.fetchall()
    return {
        "family_id": family,
        "signals": [
            {
                "strategy_id": r[0], "event_id": r[1],
                "action": r[2], "side": r[3], "symbol": r[4],
                "status": r[5], "received_at": r[6].isoformat(),
                "order_sent": False,
            } for r in rows
        ],
    }


@router.get("/api/family/template/{strategy_id}")
def alert_template(
    strategy_id: str,
    x_family_admin_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Show a TradingView alert example WITHOUT disclosing stored secrets."""
    _admin(x_family_admin_token)
    family = _family_id()
    _schema()
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """select strategy_id from family_strategy_registry
               where family_id=%s and strategy_id=%s""",
            (family, strategy_id),
        )
        if cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="strategy not found")
    return {
        "family_id": family,
        "strategy_id": strategy_id,
        "webhook_path": "/webhook/family",
        "alert_message": {
            "strategy_id": strategy_id,
            "token": "INSERT_STRATEGY_TOKEN_FROM_REGISTRATION",
            "event_id": "{{ticker}}:{{timenow}}:{{strategy.order.id}}",
            "action": "entry",
            "side": "buy",
            "symbol": "SOL-USDT",
        },
        "status": "shadow_no_orders",
    }
