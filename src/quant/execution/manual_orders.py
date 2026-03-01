from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from quant.execution.execution_state import write_execution_state
from quant.execution.kucoin_futures import KucoinFuturesBroker
from quant.execution.live_monitor import ExpectedTrade, record_expected
from quant.utils.log import get_logger

log = get_logger("quant.manual_orders")

_ACTION_ALIASES: Dict[str, str] = {
    "cancel_short": "flatten_short",
    "close_short": "flatten_short",
    "flatten_short": "flatten_short",
    "cancel_long": "flatten_long",
    "close_long": "flatten_long",
    "flatten_long": "flatten_long",
    "flatten": "flatten",
    "close": "flatten",
    "cancel_position": "flatten",
    "enter_long": "enter_long",
    "buy": "enter_long",
    "enter_short": "enter_short",
    "sell": "enter_short",
    "cancel_all_orders": "cancel_all_orders",
    "cancel_orders": "cancel_all_orders",
}


def _normalize_symbol(sym: str) -> str:
    s = (sym or "").strip().upper().replace("/", "-").replace(":", "-").replace(" ", "")
    return s or "UNKNOWN"


def _now_iso() -> str:
    return pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _default_live_dir() -> Path:
    if Path("/data").exists():
        return Path("/data/live")
    return Path("data/live")


def _manual_orders_log_path() -> Path:
    p = (os.getenv("MANUAL_ORDERS_LOG_JSONL") or "").strip()
    if p:
        return Path(p)
    return _default_live_dir() / "manual_orders.jsonl"


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")


def _canonical_action(action: str) -> str:
    a = str(action or "").strip().lower()
    if a not in _ACTION_ALIASES:
        raise ValueError(f"unsupported manual action: {action!r}")
    return _ACTION_ALIASES[a]


def _position_snapshot(broker: KucoinFuturesBroker, symbol: str) -> Dict[str, Any]:
    pos = float(broker.get_position(symbol))
    if pos > 0:
        side = "long"
    elif pos < 0:
        side = "short"
    else:
        side = "flat"
    return {"signed_qty": pos, "abs_qty": abs(pos), "side": side}


def _round_contract_qty(qty: float) -> int:
    try:
        q = int(float(qty))
    except Exception:
        q = 0
    return int(max(0, q))


def _resolve_ref_price(bid: float, ask: float, side: str) -> float:
    b = float(bid or 0.0)
    a = float(ask or 0.0)
    if side == "buy":
        if a > 0:
            return a
        if b > 0:
            return b
    else:
        if b > 0:
            return b
        if a > 0:
            return a
    if b > 0 and a > 0:
        return (a + b) / 2.0
    return max(a, b, 1.0)


def _record_manual_expected(
    *,
    ts_iso: str,
    symbol: str,
    side: str,
    action: str,
    qty: int,
    expected_px: float,
    client_oid: str,
    event_tag: str,
    note_extra: str = "",
) -> None:
    note = f"event={event_tag} source=manual_orders"
    if note_extra:
        note = f"{note} {note_extra.strip()}"
    record_expected(
        ExpectedTrade(
            ts=ts_iso,
            symbol=symbol,
            side=side,
            action=action,
            qty=float(qty),
            expected_px=float(expected_px) if expected_px > 0 else None,
            client_oid=client_oid,
            note=note,
        )
    )


def _write_flat_state(symbol: str, ts_iso: str, manual_action: str) -> None:
    write_execution_state(
        {
            "symbol": symbol,
            "side": None,
            "mode": "manual",
            "sl": None,
            "ttp": None,
            "entry_px": None,
            "entry_bar_ts": None,
            "ts": ts_iso,
            "manual_action": manual_action,
        }
    )


def _write_entry_state(symbol: str, side: str, ts_iso: str, expected_px: float, manual_action: str) -> None:
    write_execution_state(
        {
            "symbol": symbol,
            "side": side,
            "mode": "manual",
            "sl": None,
            "ttp": None,
            "entry_px": float(expected_px) if expected_px > 0 else None,
            "entry_bar_ts": int(pd.Timestamp(ts_iso).timestamp()),
            "ts": ts_iso,
            "manual_action": manual_action,
        }
    )


def execute_manual_action(
    *,
    broker: KucoinFuturesBroker,
    symbol: str,
    action: str,
    qty: Optional[float] = None,
    wait_sec: float = 20.0,
    dry_run: bool = False,
) -> Dict[str, Any]:
    sym = _normalize_symbol(symbol)
    canonical = _canonical_action(action)
    now_iso = _now_iso()
    pos_before = _position_snapshot(broker, sym)

    if canonical == "flatten":
        if pos_before["side"] == "short":
            canonical = "flatten_short"
        elif pos_before["side"] == "long":
            canonical = "flatten_long"
        else:
            out = {
                "ok": True,
                "noop": True,
                "symbol": sym,
                "action": "flatten",
                "reason": "already_flat",
                "position_before": pos_before,
                "ts": now_iso,
            }
            _append_jsonl(_manual_orders_log_path(), out)
            return out

    if canonical == "cancel_all_orders":
        if not dry_run:
            broker.cancel_all(sym)
        out = {
            "ok": True,
            "symbol": sym,
            "action": canonical,
            "dry_run": bool(dry_run),
            "position_before": pos_before,
            "ts": now_iso,
        }
        _append_jsonl(_manual_orders_log_path(), out)
        return out

    bid, ask = broker.get_best_bid_ask(sym)
    off_pct = float(os.getenv("MANUAL_ORDER_MARKETABLE_OFF_PCT", "0.0008"))
    timeout_s = int(max(1, float(wait_sec)))

    if canonical in ("flatten_short", "flatten_long"):
        side_before = str(pos_before["side"])
        want_side = "short" if canonical == "flatten_short" else "long"
        if side_before != want_side:
            out = {
                "ok": True,
                "noop": True,
                "symbol": sym,
                "action": canonical,
                "reason": f"no_{want_side}_position",
                "position_before": pos_before,
                "ts": now_iso,
            }
            _append_jsonl(_manual_orders_log_path(), out)
            return out

        pos_abs = float(pos_before["abs_qty"])
        req_qty = pos_abs if qty is None else min(pos_abs, max(0.0, float(qty)))
        trade_qty = _round_contract_qty(req_qty)
        if trade_qty <= 0:
            out = {
                "ok": True,
                "noop": True,
                "symbol": sym,
                "action": canonical,
                "reason": "qty_zero_after_rounding",
                "requested_qty": qty,
                "position_before": pos_before,
                "ts": now_iso,
            }
            _append_jsonl(_manual_orders_log_path(), out)
            return out

        order_side = "buy" if canonical == "flatten_short" else "sell"
        ref_px = _resolve_ref_price(float(bid), float(ask), order_side)
        limit_px = ref_px * (1.0 + off_pct) if order_side == "buy" else ref_px * (1.0 - off_pct)
        client_oid = f"manual-{canonical}-{int(time.time() * 1000)}"
        event_tag = f"manual_{canonical}"
        _record_manual_expected(
            ts_iso=now_iso,
            symbol=sym,
            side=want_side,
            action="exit_sl",
            qty=trade_qty,
            expected_px=ref_px,
            client_oid=client_oid,
            event_tag=event_tag,
            note_extra=f"order_side={order_side}",
        )

        order_id: Optional[str] = None
        filled: Optional[bool] = None
        if not dry_run:
            order_id = broker.place_marketable_limit(
                symbol=sym,
                side=order_side,
                qty=float(trade_qty),
                limit_price=float(limit_px),
                reduce_only=True,
                client_id=client_oid,
            )
            filled = bool(broker.wait_filled(sym, order_id, timeout_s=timeout_s))
            if filled:
                _write_flat_state(sym, now_iso, canonical)

        out = {
            "ok": True,
            "symbol": sym,
            "action": canonical,
            "dry_run": bool(dry_run),
            "position_before": pos_before,
            "order_side": order_side,
            "qty": float(trade_qty),
            "ref_price": float(ref_px),
            "limit_price": float(limit_px),
            "client_oid": client_oid,
            "order_id": order_id,
            "filled": filled,
            "ts": now_iso,
        }
        _append_jsonl(_manual_orders_log_path(), out)
        return out

    if canonical in ("enter_long", "enter_short"):
        trade_qty = _round_contract_qty(float(qty) if qty is not None else 0.0)
        if trade_qty <= 0:
            raise ValueError("enter_long/enter_short requires --qty > 0")
        order_side = "buy" if canonical == "enter_long" else "sell"
        side_name = "long" if canonical == "enter_long" else "short"
        ref_px = _resolve_ref_price(float(bid), float(ask), order_side)
        limit_px = ref_px * (1.0 + off_pct) if order_side == "buy" else ref_px * (1.0 - off_pct)
        client_oid = f"manual-{canonical}-{int(time.time() * 1000)}"
        event_tag = f"manual_{canonical}"
        _record_manual_expected(
            ts_iso=now_iso,
            symbol=sym,
            side=side_name,
            action="entry",
            qty=trade_qty,
            expected_px=ref_px,
            client_oid=client_oid,
            event_tag=event_tag,
            note_extra=f"order_side={order_side}",
        )

        order_id: Optional[str] = None
        filled: Optional[bool] = None
        if not dry_run:
            order_id = broker.place_marketable_limit(
                symbol=sym,
                side=order_side,
                qty=float(trade_qty),
                limit_price=float(limit_px),
                reduce_only=False,
                client_id=client_oid,
            )
            filled = bool(broker.wait_filled(sym, order_id, timeout_s=timeout_s))
            if filled:
                _write_entry_state(sym, side_name, now_iso, ref_px, canonical)

        out = {
            "ok": True,
            "symbol": sym,
            "action": canonical,
            "dry_run": bool(dry_run),
            "position_before": pos_before,
            "order_side": order_side,
            "qty": float(trade_qty),
            "ref_price": float(ref_px),
            "limit_price": float(limit_px),
            "client_oid": client_oid,
            "order_id": order_id,
            "filled": filled,
            "ts": now_iso,
        }
        _append_jsonl(_manual_orders_log_path(), out)
        return out

    raise ValueError(f"unsupported manual action: {action!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manual KuCoin order helper (with expected-trade logging)")
    p.add_argument("--symbol", default=os.getenv("LIVE_SYMBOL", os.getenv("DASHBOARD_SYMBOL", "SOL-USDT")))
    p.add_argument("--action", required=True, choices=sorted(_ACTION_ALIASES.keys()))
    p.add_argument("--qty", type=float, default=None, help="Contracts (required for enter_long/enter_short)")
    p.add_argument("--wait-sec", type=float, default=float(os.getenv("MANUAL_ORDER_WAIT_SEC", "20")))
    p.add_argument("--dry-run", action="store_true", help="Do not send order; only log expected action")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    broker = KucoinFuturesBroker()
    try:
        res = execute_manual_action(
            broker=broker,
            symbol=str(args.symbol),
            action=str(args.action),
            qty=args.qty,
            wait_sec=float(args.wait_sec),
            dry_run=bool(args.dry_run),
        )
        print(json.dumps(res, ensure_ascii=False, sort_keys=True))
        log.info("manual order result: %s", res)
    except Exception as e:
        err = {"ok": False, "error": str(e), "action": args.action, "symbol": args.symbol, "ts": _now_iso()}
        print(json.dumps(err, ensure_ascii=False, sort_keys=True))
        log.warning("manual order failed: %s", e)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
