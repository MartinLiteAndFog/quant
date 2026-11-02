"""Explicit, one-time 3x3 Blaupunkt pilot registration.

This is an opt-in administrative action on a SHADOW-ONLY control plane.
Registers ONE existing legacy parent and ELEVEN new profiles. All use one
shared FAMILY_WEBHOOK_TOKEN (configured separately on Railway).
Nothing touches TradingView or KuCoin orders.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException

from quant.execution.event_store import get_conn
from quant.execution.family_ingress import _admin, _family_id, _schema
from quant.execution.family_pilot_roster import BLAUPUNKT_PILOT_ROSTER

router = APIRouter(tags=["family-roster"])


@router.get("/api/family/pilot-roster")
def pilot_roster(x_family_admin_token: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """Preview intended IDs, with NO token creation or database mutation."""
    _admin(x_family_admin_token)
    return {
        "family_id": _family_id(),
        "parents": 3,
        "daughters_per_parent": 3,
        "total": 12,
        "shared_webhook_token_required": True,
        "shared_webhook_path": "/webhook/family",
        "strategies": [
            {
                "strategy_id": p.strategy_id,
                "parent_strategy_id": p.parent_strategy_id,
                "display_name": p.display_name,
                "pool": p.pool,
                "existing_legacy": p.existing_legacy,
            } for p in BLAUPUNKT_PILOT_ROSTER
        ],
        "order_sent": False,
    }


@router.post("/api/family/bootstrap-pilot", status_code=201)
def bootstrap_pilot(x_family_admin_token: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """Atomically create the Blaupunkt 3x3 roster. Call only ONCE.

    All IDs authenticate with ONE shared FAMILY_WEBHOOK_TOKEN. This call
    never creates or returns secrets. Existing manually registered children
    cause a transaction rollback; legacy root can be reused only when its
    original metadata matches.
    """
    _admin(x_family_admin_token)
    family_id = _family_id()
    if family_id != "blaupunkt":
        raise HTTPException(status_code=422, detail="pilot registration is Blaupunkt-only")
    _schema()
    with get_conn() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    "select pg_advisory_xact_lock(hashtext(%s)::bigint)",
                    ("blaupunkt:pilot-roster",),
                )
                for p in BLAUPUNKT_PILOT_ROSTER:
                    cur.execute(
                        """select family_id,parent_strategy_id,pool,config_json
                           from family_strategy_registry where strategy_id=%s""",
                        (p.strategy_id,),
                    )
                    row = cur.fetchone()
                    if not row:
                        continue
                    if p.existing_legacy:
                        same = (
                            row[0] == family_id and row[1] is None and
                            row[2] == "evolution" and
                            isinstance(row[3], dict) and
                            row[3].get("source") == "legacy_tv_execute"
                        )
                        if not same:
                            raise HTTPException(
                                status_code=409,
                                detail="legacy Blaupunkt root is registered differently",
                            )
                    else:
                        raise HTTPException(
                            status_code=409,
                            detail=f"{p.strategy_id} already registered; no changes made",
                        )
                for p in BLAUPUNKT_PILOT_ROSTER:
                    payload = (
                        {"source": "legacy_tv_execute", "legacy_live": True}
                        if p.existing_legacy else
                        {"source": "manual_tradingview", "roster": "blaupunkt-3x3-v1"}
                    )
                    cur.execute(
                        """insert into family_strategy_registry
                           (family_id,strategy_id,parent_strategy_id,display_name,
                            pool,config_json,execution_status)
                           values (%s,%s,%s,%s,%s,%s::jsonb,'shadow')
                           on conflict(strategy_id) do nothing""",
                        (
                            family_id, p.strategy_id, p.parent_strategy_id,
                            p.display_name, p.pool,
                            json.dumps(payload, separators=(",", ":")),
                        ),
                    )
    return {
        "family_id": family_id,
        "created_strategies": 11,
        "legacy_parent_unchanged": True,
        "status": "shadow_no_orders",
        "new_strategies": [
            {
                "strategy_id": p.strategy_id,
                "parent_strategy_id": p.parent_strategy_id,
                "webhook_path": "/webhook/family",
            }
            for p in BLAUPUNKT_PILOT_ROSTER if not p.existing_legacy
        ],
        "shared_auth_env": "FAMILY_WEBHOOK_TOKEN",
        "important": "Configure ONE shared webhook token on Railway, reuse it for all strategy IDs.",
        "order_sent": False,
    }
