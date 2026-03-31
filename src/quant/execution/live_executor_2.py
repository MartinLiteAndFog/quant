from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quant.execution.execution_state import write_execution_state
from quant.execution.kraken_futures import KrakenFuturesClient
from quant.execution.CHOPgate import get_live_gate_state
from quant.execution.oms import MakerFirstOMS, OmsDefaults
from quant.execution.event_builders import build_action_event, build_execution_event
from quant.execution.event_log import append_event_jsonl
from quant.execution.event_store import (
    insert_action_event,
    insert_execution_event,
    insert_equity_snapshot,
    upsert_closed_trade,
)
from quant.strategies.flip_engine import FlipParams, run_flip_state_machine
from quant.strategies.follow_tp2_engine import TP2Params, run_follow_tp2_state_machine
from quant.strategies.signal_io import read_signals_jsonl
from quant.strategies.imba import ImbaParams, get_latest_imba_barriers
from quant.utils.log import get_logger, log_throttled

log = get_logger("quant.live_executor_2")

try:
    from quant.execution.live_monitor import ExpectedTrade, record_expected
except Exception:
    @dataclass
    class ExpectedTrade:
        ts: str
        symbol: str
        side: str
        action: str
        qty: float
        expected_px: Optional[float] = None
        note: Optional[str] = None

    def record_expected(_: ExpectedTrade) -> None:
        return None


def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _norm_symbol(sym: str) -> str:
    return sym.strip().upper().replace("/", "-").replace(":", "-").replace(" ", "")


def _canon_symbol(sym: str) -> str:
    s = (sym or "").upper()
    return "".join(ch for ch in s if ch.isalnum())


def _safe_ts(v: Any) -> Optional[pd.Timestamp]:
    ts = pd.to_datetime(v, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _now_utc() -> pd.Timestamp:
    return pd.Timestamp.now("UTC")


def _now_iso() -> str:
    return _now_utc().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _env_float(primary: str, fallback: str, default: float) -> float:
    raw = os.getenv(primary)
    if raw is None or str(raw).strip() == "":
        raw = os.getenv(fallback, str(default))
    return float(raw)


class KrakenOmsBroker:
    """
    Adapter that gives Kraken Futures the broker surface expected by MakerFirstOMS
    and by the copied live executor flow.
    """

    def __init__(self, client: KrakenFuturesClient) -> None:
        self.client = client
        self.venue_symbol = (os.getenv("KRAKEN_VENUE_SYMBOL") or os.getenv("KRAKEN_FUTURES_SYMBOL") or client.symbol).strip()

    def _symbol(self, _: str) -> str:
        return self.venue_symbol

    def get_best_bid_ask(self, symbol: str) -> tuple[float, float]:
        sym = self._symbol(symbol)
        data = self.client._req("GET", "/derivatives/api/v3/tickers")
        tickers = data.get("tickers", []) if isinstance(data, dict) else []
        for row in tickers:
            if str(row.get("symbol")) != sym:
                continue
            bid = float(row.get("bid") or row.get("bestBid") or row.get("markPrice") or row.get("last") or 0.0)
            ask = float(row.get("ask") or row.get("bestAsk") or row.get("markPrice") or row.get("last") or 0.0)
            if bid > 0 or ask > 0:
                return bid, ask
        px = float(self.client.get_mark_price(symbol=sym) or 0.0)
        return px, px

    def get_1m_range_pct_proxy(self, symbol: str) -> Optional[float]:
        raw = os.getenv("KRAKEN_1M_RANGE_PCT_PROXY")
        if raw is None or str(raw).strip() == "":
            raw = os.getenv("LIVE_EXECUTOR_2_1M_RANGE_PCT_PROXY")
        if raw is None or str(raw).strip() == "":
            return None
        try:
            v = float(raw)
        except Exception:
            return None
        return v if v > 0 else None

    def get_position(self, symbol: str) -> float:
        pos = self.client.get_position(symbol=self._symbol(symbol))
        return float(pos.get("size_signed", 0.0) or 0.0)

    def get_account_balance(self, currency: str = "USDT") -> Dict[str, float]:
        eq = self.client.get_account_equity()
        return {"equity": float(eq.get("equity_usd", 0.0) or 0.0)}

    def get_contract_multiplier(self, symbol: str) -> float:
        return float(os.getenv("LIVE_EXECUTOR_2_CONTRACT_MULTIPLIER", os.getenv("LIVE_EXECUTOR_CONTRACT_MULTIPLIER", "1.0")))

    def cancel_all(self, symbol: str) -> None:
        self.client.cancel_all_orders(symbol=self._symbol(symbol))

    def place_limit(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        post_only: bool,
        reduce_only: bool,
        client_id: str,
    ) -> str:
        sym = self._symbol(symbol)
        side_n = self.client._norm_side(side)
        size_s = self.client._norm_size_str(qty)
        params: Dict[str, Any] = {
            "symbol": sym,
            "side": side_n,
            "size": size_s,
            "orderType": "lmt",
            "limitPrice": f"{float(price):.8f}",
            "reduceOnly": "true" if reduce_only else "false",
            "cliOrdId": str(client_id),
        }
        if post_only:
            params["postOnly"] = "true"
        data = self.client._req("POST", "/derivatives/api/v3/sendorder", params=params, private=True)
        send = data.get("sendStatus", data) if isinstance(data, dict) else {}
        return str(send.get("order_id") or send.get("orderId") or send.get("order_id".upper()) or "")

    def place_marketable_limit(
        self,
        symbol: str,
        side: str,
        qty: float,
        limit_price: float,
        reduce_only: bool,
        client_id: str,
    ) -> str:
        return self.place_limit(
            symbol=symbol,
            side=side,
            qty=qty,
            price=limit_price,
            post_only=False,
            reduce_only=reduce_only,
            client_id=client_id,
        )

    def place_market(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool,
        client_id: str,
    ) -> str:
        res = self.client.place_market(
            side=side,
            size=qty,
            symbol=self._symbol(symbol),
            reduce_only=reduce_only,
            cli_ord_id=client_id,
        )
        data = res.get("data", {}) if isinstance(res, dict) else {}
        send = data.get("sendStatus", data) if isinstance(data, dict) else {}
        return str(send.get("order_id") or send.get("orderId") or "")

    def wait_filled(self, symbol: str, order_id: str, timeout_s: int) -> bool:
        sym = self._symbol(symbol)
        t0 = time.time()
        order_id = str(order_id or "")
        while (time.time() - t0) < max(1, int(timeout_s)):
            try:
                open_orders = self.client.get_open_orders(symbol=sym)
                if not any(str(o.get("order_id") or "") == order_id for o in open_orders):
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        return False

    def cancel_all_stop_orders(self, symbol: str) -> None:
        self.client.cancel_all_reduce_only_orders(symbol=self._symbol(symbol))

    def place_stop_market(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        reduce_only: bool,
        client_id: str,
    ) -> str:
        res = self.client.place_stop_market(
            side=side,
            size=qty,
            stop_price=stop_price,
            symbol=self._symbol(symbol),
            reduce_only=reduce_only,
            cli_ord_id=client_id,
        )
        data = res.get("data", {}) if isinstance(res, dict) else {}
        send = data.get("sendStatus", data) if isinstance(data, dict) else {}
        return str(send.get("order_id") or send.get("orderId") or "")


    def place_take_profit_market(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        reduce_only: bool,
        client_id: str,
    ) -> str:
        res = self.client.place_take_profit_market(
            side=side,
            size=qty,
            stop_price=stop_price,
            symbol=self._symbol(symbol),
            reduce_only=reduce_only,
            cli_ord_id=client_id,
        )
        data = res.get("data", {}) if isinstance(res, dict) else {}
        send = data.get("sendStatus", data) if isinstance(data, dict) else {}
        return str(send.get("order_id") or send.get("orderId") or "")

    def place_trigger_entry_market(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        reduce_only: bool,
        client_id: str,
    ) -> str:
        res = self.client.place_trigger_entry_market(
            side=side,
            size=qty,
            stop_price=stop_price,
            symbol=self._symbol(symbol),
            reduce_only=reduce_only,
            cli_ord_id=client_id,
        )
        data = res.get("data", {}) if isinstance(res, dict) else {}
        send = data.get("sendStatus", data) if isinstance(data, dict) else {}
        return str(send.get("order_id") or send.get("orderId") or "")


    def cancel_order(
        self,
        order_id: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> None:
        self.client.cancel_order(
            order_id=str(order_id) if order_id else None,
            cli_ord_id=str(client_id) if client_id else None,
        )

    def list_open_orders(self, symbol: str) -> list[dict]:
        rows = self.client.get_open_orders(symbol=self._symbol(symbol))
        return rows if isinstance(rows, list) else []

    def list_open_stop_orders(self, symbol: str) -> list[dict]:
        rows = self.client.get_open_orders(symbol=self._symbol(symbol))
        if not isinstance(rows, list):
            return []
        out = []
        for row in rows:
            order_type = str(row.get("order_type") or row.get("orderType") or "").strip().lower()
            stop_price = row.get("stop_price", row.get("stopPrice"))
            if order_type in ("stp", "stop", "take_profit") or stop_price not in (None, "", 0, 0.0):
                out.append(row)
        return out


def _events_root() -> Path:
    if Path("/data").exists():
        return Path("/data/events")
    return Path("data/events")


_EQUITY_SNAPSHOT_BASE_INTERVAL_SEC = float(
    os.getenv("LIVE_EXECUTOR_2_SNAPSHOT_BASE_SEC", "60")
)
_LAST_EQUITY_SNAPSHOT_TS: Optional[pd.Timestamp] = None


def _append_equity_snapshot(
    *,
    ts_iso: str,
    equity: Optional[float],
    position_qty: Optional[float] = None,
    position_side: Optional[int] = None,
    payload: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> None:
    global _LAST_EQUITY_SNAPSHOT_TS

    try:
        eq = float(equity) if equity is not None else None
        if eq is None or eq <= 0:
            return

        ts = pd.to_datetime(ts_iso, utc=True, errors="coerce")
        if pd.isna(ts):
            ts = pd.Timestamp.now("UTC")

        if not force:
            min_interval = max(1.0, float(_EQUITY_SNAPSHOT_BASE_INTERVAL_SEC))
            if _LAST_EQUITY_SNAPSHOT_TS is not None:
                delta_sec = (ts - _LAST_EQUITY_SNAPSHOT_TS).total_seconds()
                if delta_sec < min_interval:
                    return

        base_payload: Dict[str, Any] = dict(payload or {"equity_usd": eq})
        if position_qty is not None:
            base_payload["position_qty"] = float(position_qty)
        if position_side is not None:
            base_payload["position_side"] = int(position_side)
        base_payload["snapshot_kind"] = "event" if force else "base"

        insert_equity_snapshot(
            {
                "ts": ts,
                "venue": "kraken",
                "account": "main",
                "symbol": None,
                "equity": eq,
                "currency": "USD",
                "source": "live_executor_2",
                "payload_json": base_payload,
            }
        )
        _LAST_EQUITY_SNAPSHOT_TS = ts
    except Exception:
        pass


def _append_action_event(
    *,
    strategy: str,
    symbol: str,
    ts_iso: str,
    seq: int,
    engine_action: str,
    action_side: str,
    reason_code: str,
    position_before: int,
    position_after: int,
    engine_mode_before: str,
    engine_mode_after: str,
    blocked: bool = False,
    block_reason: Optional[str] = None,
    payload_json: Optional[Dict[str, Any]] = None,
) -> None:
    event = build_action_event(
        strategy=strategy,
        symbol=symbol.replace("-", ""),
        ts=ts_iso,
        seq=int(seq),
        engine_action=engine_action,
        action_side=action_side,
        reason_code=reason_code,
        venue="kraken",
        source_event_id=None,
        source_signal_event_id=None,
        position_before=int(position_before),
        position_after=int(position_after),
        engine_mode_before=engine_mode_before,
        engine_mode_after=engine_mode_after,
        blocked=bool(blocked),
        block_reason=block_reason,
    )
    event["strategy_instance"] = "live_executor_2"
    event["config_hash"] = "live_executor_2_v1"

    if payload_json:
        event["payload_json"] = dict(payload_json)

    out_path = _events_root() / "action_events" / f"{pd.Timestamp.now('UTC').strftime('%Y%m%d')}.jsonl"
    append_event_jsonl(out_path, event)

    try:
        insert_action_event(
            {
                "event_id": event["event_id"],
                "ts": event["ts"],
                "seq": event["seq"],
                "strategy": event["strategy"],
                "strategy_instance": event.get("strategy_instance"),
                "config_hash": event.get("config_hash"),
                "symbol": event["symbol"],
                "venue": event["venue"],
                "source_signal_event_id": None,
                "source_event_id": event.get("source_event_id"),
                "engine_action": event["engine_action"],
                "action_side": event.get("action_side"),
                "position_before": event.get("position_before"),
                "position_after": event.get("position_after"),
                "qty_before": event.get("qty_before"),
                "qty_after": event.get("qty_after"),
                "engine_mode_before": event.get("engine_mode_before"),
                "engine_mode_after": event.get("engine_mode_after"),
                "reason_code": event["reason_code"],
                "reason_detail": event.get("reason_detail"),
                "blocked": bool(event.get("blocked", False)),
                "block_reason": event.get("block_reason"),
                "regime_state": event.get("regime_state"),
                "gate_name": event.get("gate_name"),
                "payload_json": dict(event),
            }
        )
    except Exception as e:
        log.warning("kraken postgres action event failed: %s", e)


def _append_execution_event(
    *,
    strategy: str,
    symbol: str,
    ts_iso: str,
    seq: int,
    execution_kind: str,
    order_action: str,
    reason_code: str,
    position_before: int,
    position_after: int,
    order_id: Optional[str] = None,
    client_oid: Optional[str] = None,
    side: Optional[str] = None,
    qty: Optional[float] = None,
    price: Optional[float] = None,
    reduce_only: Optional[bool] = None,
    status: Optional[str] = None,
    reject_reason: Optional[str] = None,
    payload_json: Optional[Dict[str, Any]] = None,
) -> None:
    event = build_execution_event(
        strategy=strategy,
        symbol=symbol.replace("-", ""),
        ts=ts_iso,
        seq=int(seq),
        execution_kind=execution_kind,
        order_action=order_action,
        reason_code=reason_code,
        venue="kraken",
        source_event_id=None,
        source_signal_event_id=None,
        position_before=int(position_before),
        position_after=int(position_after),
        blocked=False,
        block_reason=None,
        client_oid=client_oid,
        order_id=order_id,
        side=side,
        qty=qty,
        price=price,
        reduce_only=reduce_only,
        status=status,
        reject_reason=reject_reason,
        strategy_instance="live_executor_2",
        config_hash="live_executor_2_v1",
        payload_json=payload_json or {},
    )

    out_path = _events_root() / "execution_events" / f"{pd.Timestamp.now('UTC').strftime('%Y%m%d')}.jsonl"
    append_event_jsonl(out_path, event)

    try:
        insert_execution_event(
            {
                "event_id": event["event_id"],
                "ts": event["ts"],
                "seq": event["seq"],
                "symbol": event["symbol"],
                "venue": event["venue"],
                "source_action_event_id": None,
                "execution_stage": str(event.get("execution_kind") or "fill"),
                "order_id": event.get("order_id"),
                "client_oid": event.get("client_oid"),
                "side": event.get("side") or event.get("order_action"),
                "qty": event.get("qty"),
                "price": event.get("price"),
                "reduce_only": event.get("reduce_only"),
                "status": event.get("status"),
                "reject_reason": event.get("reject_reason"),
                "payload_json": dict(event),
            }
        )
    except Exception as e:
        log.warning("kraken postgres execution event failed: %s", e)


def _resolve_trade_exit_price(details: Dict[str, Any], fallback_px: Optional[float]) -> Optional[float]:
    for key in ("price", "avg_fill_price", "fill_price", "avgPrice", "limit_px", "limitPrice"):
        px = _coerce_float(details.get(key))
        if px is not None and px > 0:
            return float(px)
    if fallback_px is not None and float(fallback_px) > 0:
        return float(fallback_px)
    return None


def _append_closed_trade(
    *,
    symbol: str,
    current_side: str,
    terminal: Optional[Dict[str, Any]],
    details: Dict[str, Any],
    event_name: str,
    action: str,
    position_before: int,
    position_after: int,
    seq: int,
    qty_default: float,
    exit_px_fallback: Optional[float],
) -> None:
    entry_px_realized = _coerce_float((terminal or {}).get("entry_px"))
    exit_px_realized = _resolve_trade_exit_price(details, exit_px_fallback)
    qty_realized = _coerce_float(details.get("qty", qty_default))

    entry_ts_raw = (terminal or {}).get("entry_bar_ts")
    entry_ts = pd.to_datetime(entry_ts_raw, utc=True, errors="coerce")
    exit_ts = pd.to_datetime(_now_iso(), utc=True)

    if entry_px_realized is None or exit_px_realized is None or qty_realized is None:
        log.warning(
            "kraken closed trade skipped: missing prices/qty action=%s event=%s entry_px=%s exit_px=%s qty=%s",
            action,
            event_name,
            entry_px_realized,
            exit_px_realized,
            qty_realized,
        )
        return

    if entry_px_realized <= 0 or qty_realized <= 0:
        return

    side_mult = 1.0 if current_side == "long" else -1.0
    pnl_pct_realized = (
        ((float(exit_px_realized) - float(entry_px_realized)) / float(entry_px_realized))
        * 100.0
        * side_mult
    )

    try:
        upsert_closed_trade(
            {
                "trade_id": f"{symbol}:{_now_iso()}:{action}:{seq}",
                "venue": "kraken",
                "symbol": symbol,
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "side": current_side,
                "qty": float(qty_realized),
                "entry_price": float(entry_px_realized),
                "exit_price": float(exit_px_realized),
                "pnl_pct": float(pnl_pct_realized),
                "exit_event": event_name,
                "strategy": "live_executor_2",
                "strategy_instance": "live_executor_2",
                "config_hash": "live_executor_2_v1",
                "source_action_event_id": None,
                "payload_json": {
                    "kind": "closed_trade",
                    "action": action,
                    "event_name": event_name,
                    "position_before": position_before,
                    "position_after": position_after,
                },
            }
        )
    except Exception as e:
        log.warning("kraken postgres closed trade failed: %s", e)


def _record_ttp_external_exit(
    state: ExecutorState,
    *,
    symbol: str,
    prior_side: str,
    terminal: Optional[Dict[str, Any]],
    ttp_px: Optional[float],
    event_name: str,
    action: str,
    execution_seq: int,
    qty: float,
    exit_details: Dict[str, Any],
) -> None:
    position_before = 1 if prior_side == "long" else (-1 if prior_side == "short" else 0)
    order_action = "sell" if prior_side == "long" else "buy"
    event_side = str(exit_details.get("side") or order_action).strip().lower() or order_action
    event_qty = _coerce_float(exit_details.get("qty"))
    event_price = _coerce_float(exit_details.get("price"))

    _append_execution_event(
        strategy="live_executor_2",
        symbol=symbol,
        ts_iso=_now_iso(),
        seq=int(execution_seq),
        execution_kind="fill",
        order_action=event_side,
        reason_code=event_name,
        position_before=position_before,
        position_after=0,
        order_id=str(exit_details.get("order_id") or "") or None,
        client_oid=str(exit_details.get("client_id") or "") or None,
        side=event_side,
        qty=(float(event_qty) if event_qty is not None else float(qty)),
        price=(float(event_price) if event_price is not None else ttp_px),
        reduce_only=True,
        status="fill",
        reject_reason=None,
        payload_json={
            "action": action,
            "source": "ttp_external_exit",
            "result": exit_details,
        },
    )

    _append_closed_trade(
        symbol=symbol,
        current_side=prior_side,
        terminal=terminal,
        details=exit_details,
        event_name=event_name,
        action=action,
        position_before=position_before,
        position_after=0,
        seq=int(execution_seq),
        qty_default=float(qty),
        exit_px_fallback=ttp_px,
    )

    state.open_leg_mode = None
    state.open_leg_id = None
    state.open_leg_side = None
    state.open_leg_entry_bar_ts = None
    state.ttp_reenter_exit_recorded = True


def _new_ttp_reenter_leg_id(side: str, source_ts: Optional[str] = None) -> str:
    seed = str(source_ts or _now_iso()).replace("-", "").replace(":", "").replace(".", "")
    return f"ttp_reenter:{side}:{seed}:{int(time.time() * 1000)}"


def _resolve_ttp_trail_pct() -> float:
    raw = os.getenv("LIVE_FLIP_TTP_TRAIL_PCT", os.getenv("LIVE_TTP_TRAIL_PCT", "0.012"))
    try:
        v = float(raw)
    except Exception:
        v = 0.012
    return float(max(1e-6, min(0.5, v)))


def _coerce_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if not (x == x):
        return None
    return x


def _opposite_imba_barrier(
    *,
    terminal_pos: int,
    imba_levels: Optional[Dict[str, Any]],
) -> Optional[float]:
    if not isinstance(imba_levels, dict):
        return None
    if terminal_pos > 0:
        return _coerce_float(imba_levels.get("short_barrier"))
    if terminal_pos < 0:
        return _coerce_float(imba_levels.get("long_barrier"))
    return None


def _opposite_imba_supersedes_stop(
    *,
    terminal_pos: int,
    base_stop: Optional[float],
    imba_levels: Optional[Dict[str, Any]],
) -> bool:
    if base_stop is None or terminal_pos == 0:
        return False
    opposite_barrier = _opposite_imba_barrier(terminal_pos=terminal_pos, imba_levels=imba_levels)
    if opposite_barrier is None:
        return False
    if terminal_pos > 0:
        return float(opposite_barrier) > float(base_stop)
    return float(opposite_barrier) < float(base_stop)


def _scale_delta_epsilon() -> float:
    step = _coerce_float(os.getenv("KRAKEN_SIZE_STEP", os.getenv("LIVE_EXECUTOR_2_SIZE_STEP", "0.1")))
    if step is None or step <= 0:
        step = 0.1
    env_abs = _coerce_float(os.getenv("LIVE_EXECUTOR_2_MIN_SCALE_DELTA_ABS"))
    if env_abs is not None and env_abs > 0:
        return float(env_abs)
    return float(step * 0.5)


def _apply_live_ttp_guard(
    terminal: Dict[str, Any],
    *,
    live_pos: float,
    live_mid: float,
    ttp_trail_pct: float,
) -> Dict[str, Any]:
    if not isinstance(terminal, dict):
        return {}
    out = dict(terminal)
    if abs(float(live_pos)) <= 1e-12:
        return out
    mid = _coerce_float(live_mid)
    if mid is None or mid <= 0:
        return out

    live_side = "long" if float(live_pos) > 0 else "short"
    side_raw = str(out.get("side") or "").strip().lower()
    side = side_raw if side_raw in ("long", "short") else live_side
    if side != live_side:
        side = live_side
    out["side"] = side

    mode = str(out.get("mode") or "").strip().upper()
    if mode != "TTP":
        return out

    trail = float(max(1e-6, ttp_trail_pct))
    cur_ttp = _coerce_float(out.get("ttp"))
    if side == "long":
        floor_ttp = float(mid * (1.0 - trail))
        out["ttp"] = floor_ttp if cur_ttp is None else float(max(cur_ttp, floor_ttp))
    else:
        cap_ttp = float(mid * (1.0 + trail))
        out["ttp"] = cap_ttp if cur_ttp is None else float(min(cur_ttp, cap_ttp))
    return out


@dataclass
class ExecutorState:
    last_signal_ts: Optional[str] = None
    last_signal_value: Optional[int] = None
    last_event_sig: Optional[str] = None
    last_action: Optional[str] = None
    n_actions: int = 0
    n_executions: int = 0
    last_terminal_sig: Optional[str] = None
    last_gate_on: Optional[int] = None
    latched_exit_engine: Optional[str] = None
    last_live_side: Optional[str] = None

    open_leg_mode: Optional[str] = None
    open_leg_id: Optional[str] = None
    open_leg_side: Optional[str] = None
    open_leg_entry_bar_ts: Optional[str] = None

    tp2_leg_id: Optional[str] = None
    tp2_leg_side: Optional[str] = None
    tp2_entry_bar_ts: Optional[str] = None
    tp2_tp1_done: bool = False
    tp2_tp1_pending: bool = False
    tp2_size_rem: float = 1.0
    tp2_remaining_qty_abs: Optional[float] = None
    tp2_tp1_hit_ts: Optional[str] = None
    tp2_tp1_hit_px: Optional[float] = None
    tp2_last_consumed_tp1_hit_ts: Optional[str] = None
    flat_until_new_signal_ts: Optional[str] = None
    flat_latch_reason: Optional[str] = None

    pending_follow_entry: bool = False
    pending_follow_entry_source_side: Optional[str] = None
    pending_follow_entry_side: Optional[str] = None
    pending_follow_entry_reason: Optional[str] = None
    pending_follow_entry_source_ts: Optional[str] = None
    pending_follow_entry_expires_at: Optional[str] = None

    ttp_reenter_pending: bool = False
    ttp_reenter_prior_side: Optional[str] = None
    ttp_reenter_target_side: Optional[str] = None
    ttp_reenter_source_ts: Optional[str] = None
    ttp_reenter_expires_at: Optional[str] = None
    ttp_reenter_exit_recorded: bool = False
    ttp_reenter_cooldown_until: Optional[str] = None
    ttp_reenter_last_attempt_key: Optional[str] = None


def _read_state(path: Path) -> ExecutorState:
    if not path.exists():
        return ExecutorState()
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return ExecutorState(
            last_signal_ts=d.get("last_signal_ts"),
            last_signal_value=d.get("last_signal_value"),
            last_event_sig=d.get("last_event_sig"),
            last_action=d.get("last_action"),
            n_actions=int(d.get("n_actions", 0)),
            n_executions=int(d.get("n_executions", 0)),
            last_terminal_sig=d.get("last_terminal_sig"),
            last_gate_on=(int(d.get("last_gate_on")) if d.get("last_gate_on") is not None else None),
            latched_exit_engine=d.get("latched_exit_engine"),
            last_live_side=d.get("last_live_side"),
            open_leg_mode=d.get("open_leg_mode"),
            open_leg_id=d.get("open_leg_id"),
            open_leg_side=d.get("open_leg_side"),
            open_leg_entry_bar_ts=d.get("open_leg_entry_bar_ts"),
            tp2_leg_id=d.get("tp2_leg_id"),
            tp2_leg_side=d.get("tp2_leg_side"),
            tp2_entry_bar_ts=d.get("tp2_entry_bar_ts"),
            tp2_tp1_done=bool(d.get("tp2_tp1_done", False)),
            tp2_tp1_pending=bool(d.get("tp2_tp1_pending", False)),
            tp2_size_rem=float(d.get("tp2_size_rem", 1.0) or 1.0),
            tp2_remaining_qty_abs=_coerce_float(d.get("tp2_remaining_qty_abs")),
            tp2_tp1_hit_ts=d.get("tp2_tp1_hit_ts"),
            tp2_tp1_hit_px=_coerce_float(d.get("tp2_tp1_hit_px")),
            tp2_last_consumed_tp1_hit_ts=d.get("tp2_last_consumed_tp1_hit_ts"),
            flat_until_new_signal_ts=d.get("flat_until_new_signal_ts"),
            flat_latch_reason=d.get("flat_latch_reason"),
            pending_follow_entry=bool(d.get("pending_follow_entry", False)),
            pending_follow_entry_source_side=d.get("pending_follow_entry_source_side"),
            pending_follow_entry_side=d.get("pending_follow_entry_side"),
            pending_follow_entry_reason=d.get("pending_follow_entry_reason"),
            pending_follow_entry_source_ts=d.get("pending_follow_entry_source_ts"),
            pending_follow_entry_expires_at=d.get("pending_follow_entry_expires_at"),
            ttp_reenter_pending=bool(d.get("ttp_reenter_pending", False)),
            ttp_reenter_prior_side=d.get("ttp_reenter_prior_side"),
            ttp_reenter_target_side=d.get("ttp_reenter_target_side"),
            ttp_reenter_source_ts=d.get("ttp_reenter_source_ts"),
            ttp_reenter_expires_at=d.get("ttp_reenter_expires_at"),
            ttp_reenter_exit_recorded=bool(d.get("ttp_reenter_exit_recorded", False)),
            ttp_reenter_cooldown_until=d.get("ttp_reenter_cooldown_until"),
            ttp_reenter_last_attempt_key=d.get("ttp_reenter_last_attempt_key"),
        )
    except Exception:
        return ExecutorState()


def _write_state(path: Path, st: ExecutorState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(st), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _clear_pending_follow_entry(state: ExecutorState) -> None:
    state.pending_follow_entry = False
    state.pending_follow_entry_source_side = None
    state.pending_follow_entry_side = None
    state.pending_follow_entry_reason = None
    state.pending_follow_entry_source_ts = None
    state.pending_follow_entry_expires_at = None


def _arm_pending_follow_entry(
    state: ExecutorState,
    *,
    source_side: Optional[str] = None,
    target_side: str,
    reason: str,
    source_ts: Optional[str] = None,
) -> None:
    ttl_sec = max(1.0, float(os.getenv("LIVE_EXECUTOR_2_FOLLOW_ENTRY_TTL_SEC", "20")))
    expires_at = _now_utc() + pd.Timedelta(seconds=ttl_sec)

    state.pending_follow_entry = True
    src = str(source_side or "").strip().lower()
    state.pending_follow_entry_source_side = src if src in ("long", "short") else None
    state.pending_follow_entry_side = str(target_side)
    state.pending_follow_entry_reason = str(reason)
    state.pending_follow_entry_source_ts = str(source_ts) if source_ts is not None else None
    state.pending_follow_entry_expires_at = expires_at.isoformat()


def _pending_follow_entry_is_active(state: ExecutorState) -> bool:
    if not bool(state.pending_follow_entry):
        return False

    target_side = str(state.pending_follow_entry_side or "").strip().lower()
    if target_side not in ("long", "short"):
        _clear_pending_follow_entry(state)
        return False

    expires_at = _safe_ts(state.pending_follow_entry_expires_at)
    if expires_at is not None and _now_utc() > expires_at:
        _clear_pending_follow_entry(state)
        return False

    return True


def _clear_ttp_reenter_handoff(state: ExecutorState) -> None:
    state.ttp_reenter_pending = False
    state.ttp_reenter_prior_side = None
    state.ttp_reenter_target_side = None
    state.ttp_reenter_source_ts = None
    state.ttp_reenter_expires_at = None
    state.ttp_reenter_exit_recorded = False
    state.ttp_reenter_cooldown_until = None
    state.ttp_reenter_last_attempt_key = None


def _arm_ttp_reenter_handoff(
    state: ExecutorState,
    *,
    prior_side: str,
    target_side: str,
    source_ts: Optional[str] = None,
) -> None:
    prior = str(prior_side or "").strip().lower()
    target = str(target_side or "").strip().lower()
    if prior not in ("long", "short") or target not in ("long", "short") or prior == target:
        _clear_ttp_reenter_handoff(state)
        return

    ttl_sec = max(1.0, float(os.getenv("LIVE_EXECUTOR_2_TTP_HANDOFF_TTL_SEC", "20")))
    expires_at = _now_utc() + pd.Timedelta(seconds=ttl_sec)

    state.ttp_reenter_pending = True
    state.ttp_reenter_prior_side = prior
    state.ttp_reenter_target_side = target
    state.ttp_reenter_source_ts = str(source_ts) if source_ts is not None else None
    state.ttp_reenter_expires_at = expires_at.isoformat()
    state.ttp_reenter_exit_recorded = False


def _ttp_reenter_handoff_context(state: ExecutorState) -> Optional[Dict[str, Optional[str]]]:
    if not bool(state.ttp_reenter_pending):
        return None

    prior = str(state.ttp_reenter_prior_side or "").strip().lower()
    target = str(state.ttp_reenter_target_side or "").strip().lower()
    if prior not in ("long", "short") or target not in ("long", "short") or prior == target:
        _clear_ttp_reenter_handoff(state)
        return None

    expires_at = _safe_ts(state.ttp_reenter_expires_at)
    if expires_at is not None and _now_utc() > expires_at:
        _clear_ttp_reenter_handoff(state)
        return None

    source_ts = str(state.ttp_reenter_source_ts or "").strip() or None
    handoff_key = f"{prior}:{target}:{source_ts or ''}"
    return {
        "prior_side": prior,
        "target_side": target,
        "source_ts": source_ts,
        "key": handoff_key,
    }


def _ttp_reenter_attempt_allowed(state: ExecutorState, handoff_key: str) -> bool:
    key = str(handoff_key or "").strip()
    if not key:
        return True
    if str(state.ttp_reenter_last_attempt_key or "").strip() != key:
        return True

    cooldown_until = _safe_ts(state.ttp_reenter_cooldown_until)
    if cooldown_until is not None and _now_utc() < cooldown_until:
        return False
    return True


def _mark_ttp_reenter_attempt(
    state: ExecutorState,
    handoff_key: str,
    *,
    cooldown_sec: Optional[float] = None,
) -> None:
    ttl = cooldown_sec
    if ttl is None:
        ttl = float(os.getenv("LIVE_EXECUTOR_2_TTP_REENTER_COOLDOWN_SEC", "1.0"))
    ttl = max(0.05, float(ttl))

    state.ttp_reenter_last_attempt_key = str(handoff_key or "")
    state.ttp_reenter_cooldown_until = (_now_utc() + pd.Timedelta(seconds=ttl)).isoformat()


def _ttp_reenter_handoff_action(
    state: ExecutorState,
    *,
    current_side: str,
    mid: float,
    terminal: Optional[Dict[str, Any]],
    source_ts: Optional[str] = None,
) -> Optional[Dict[str, Optional[str]]]:
    term = dict(terminal or {})
    if str(term.get("mode") or "").strip().upper() != "TTP":
        _clear_ttp_reenter_handoff(state)
        return None

    ttp_px = _coerce_float(term.get("ttp"))
    if current_side in ("long", "short") and ttp_px is not None and float(mid) > 0:
        crossed = (
            (current_side == "long" and float(mid) <= float(ttp_px))
            or (current_side == "short" and float(mid) >= float(ttp_px))
        )
        if crossed:
            _arm_ttp_reenter_handoff(
                state,
                prior_side=current_side,
                target_side=("short" if current_side == "long" else "long"),
                source_ts=source_ts,
            )

    ctx = _ttp_reenter_handoff_context(state)
    if ctx is None:
        return None

    prior_side = str(ctx["prior_side"] or "")
    target_side = str(ctx["target_side"] or "")

    if current_side in ("long", "short") and current_side == target_side:
        _clear_ttp_reenter_handoff(state)
        return None
    if current_side in ("long", "short") and current_side != prior_side:
        _clear_ttp_reenter_handoff(state)
        return None
    if current_side not in ("flat", prior_side):
        return None

    return {
        "action": f"ttp_confirm_reenter_{target_side}",
        "prior_side": prior_side,
        "target_side": target_side,
        "source_ts": ctx.get("source_ts"),
        "key": ctx.get("key"),
    }


def _derive_action_event_fields(
    *,
    action: str,
    current_side: str,
    want_side: Optional[str],
    terminal_pos: int,
    ttp_prior_side: Optional[str] = None,
) -> Tuple[str, int, int]:
    action_side = want_side if want_side is not None else current_side
    position_before = 1 if current_side == "long" else (-1 if current_side == "short" else 0)
    position_after = 1 if terminal_pos > 0 else (-1 if terminal_pos < 0 else 0)

    if action in ("ttp_confirm_reenter_short", "ttp_confirm_reenter_long"):
        prior_side = str(ttp_prior_side or current_side or "").strip().lower()
        action_side = "short" if action.endswith("short") else "long"
        position_before = 1 if prior_side == "long" else (-1 if prior_side == "short" else 0)
        position_after = 1 if action_side == "long" else -1
    elif action == "tp1_partial":
        action_side = current_side
        position_after = position_before
    elif action.startswith("exit_"):
        action_side = current_side
        position_after = 0
    elif action == "hold":
        position_after = position_before

    return action_side, int(position_before), int(position_after)


def _latest_signal(signals_root: Path, symbol: str) -> Optional[Dict[str, Any]]:
    wanted = _canon_symbol(symbol)
    candidate_dirs = []
    if signals_root.exists():
        for p in signals_root.iterdir():
            if p.is_dir() and _canon_symbol(p.name) == wanted:
                candidate_dirs.append(p)

    if not candidate_dirs:
        sym_dir = signals_root / _norm_symbol(symbol)
        if sym_dir.exists():
            candidate_dirs = [sym_dir]
    if not candidate_dirs:
        return None

    all_files = []
    for d in candidate_dirs:
        all_files.extend(d.glob("*.jsonl"))
    for fp in reversed(sorted(all_files)):
        try:
            with fp.open("r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            for ln in reversed(lines):
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                sig = obj.get("signal")
                ts = _safe_ts(obj.get("ts"))
                if ts is None:
                    continue
                try:
                    sig_i = int(sig)
                except Exception:
                    continue
                if sig_i == 0:
                    continue
                return {"ts": ts, "signal": 1 if sig_i > 0 else -1, "raw": obj}
        except Exception:
            continue
    return None


def _load_signals_df(signals_root: Path, symbol: str) -> pd.DataFrame:
    wanted = _canon_symbol(symbol)
    candidate_dirs = []
    if signals_root.exists():
        for p in signals_root.iterdir():
            if p.is_dir() and _canon_symbol(p.name) == wanted:
                candidate_dirs.append(p)
    if not candidate_dirs:
        sym_dir = signals_root / _norm_symbol(symbol)
        if sym_dir.exists():
            candidate_dirs = [sym_dir]
    if not candidate_dirs:
        return pd.DataFrame(columns=["ts", "signal"])

    parts: List[pd.DataFrame] = []
    all_files: List[Path] = []
    for d in candidate_dirs:
        all_files.extend(sorted(d.glob("*.jsonl")))
    for fp in all_files:
        try:
            parts.append(read_signals_jsonl(fp)[["ts", "signal"]].copy())
        except Exception:
            continue
    if not parts:
        return pd.DataFrame(columns=["ts", "signal"])
    out = pd.concat(parts, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    out["signal"] = pd.to_numeric(out["signal"], errors="coerce").fillna(0).astype(int).clip(-1, 1)
    out = out[out["signal"] != 0].copy()
    return out


def _load_renko_bars(path: Path, limit: int = 4000) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close"])
    df = pd.read_parquet(path)
    need = {"ts", "open", "high", "low", "close"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close"])
    df = df[["ts", "open", "high", "low", "close"]].copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ts", "close"]).sort_values("ts")
    if limit > 0:
        df = df.tail(int(limit))
    return df.reset_index(drop=True)


def _event_sig(row: pd.Series) -> str:
    ts = pd.Timestamp(row["ts"]).isoformat()
    seq = int(row.get("seq", 0))
    event = str(row.get("event", ""))
    side = int(row.get("side", 0))
    return f"{ts}|{seq}|{event}|{side}"


def _latest_backtest_event(
    renko_bars: pd.DataFrame,
    signals_df: pd.DataFrame,
) -> Tuple[Optional[pd.Series], Dict[str, Any]]:
    if renko_bars.empty or signals_df.empty:
        return None, {}
    params = FlipParams(
        fee_bps=float(os.getenv("LIVE_FLIP_FEE_BPS", "0")),
        ttp_trail_pct=float(os.getenv("LIVE_FLIP_TTP_TRAIL_PCT", "0.012")),
        min_sl_pct=float(os.getenv("LIVE_FLIP_MIN_SL_PCT", "0.015")),
        max_sl_pct=float(os.getenv("LIVE_FLIP_MAX_SL_PCT", "0.030")),
        swing_lookback=int(os.getenv("LIVE_FLIP_SWING_LOOKBACK", "50")),
        be_trigger_pct=float(os.getenv("LIVE_FLIP_BE_TRIGGER_PCT", "0")),
        be_offset_pct=float(os.getenv("LIVE_FLIP_BE_OFFSET_PCT", "0")),
    )
    _, events, terminal = run_flip_state_machine(
        bars=renko_bars,
        signals_df=signals_df,
        params=params,
        regime_on=None,
        regime_forces_flat=False,
    )
    if events is None or events.empty:
        return None, terminal
    events = events.sort_values(["ts", "seq"]).reset_index(drop=True)
    return events.iloc[-1], terminal


def _qty_from_equity_pct(
    *,
    equity: float,
    pos_pct: float,
    leverage: float,
    mid_price: float,
    contract_multiplier: float = 1.0,
) -> float:
    """
    Kraken sizing: size = floor((equity * pos_pct * leverage / mid_price) / step) * step
    contract_multiplier is ignored here and kept only to preserve the copied call shape.
    """
    equity = float(equity or 0.0)
    pos_pct = float(max(0.0, min(1.0, pos_pct)))
    leverage = float(leverage)
    mid_price = float(mid_price)
    if equity <= 0 or pos_pct <= 0 or leverage <= 0 or mid_price <= 0:
        return 0.0
    notional = equity * pos_pct * leverage
    raw_size = notional / mid_price
    step = float(os.getenv("KRAKEN_SIZE_STEP", os.getenv("LIVE_EXECUTOR_2_SIZE_STEP", "0.1")))
    if step <= 0:
        step = 0.1
    floored = math.floor(raw_size / step) * step
    return float(max(0.0, floored))


def _resolve_equity(broker: KrakenOmsBroker) -> float:
    try:
        bal = broker.get_account_balance(currency="USDT")
        return float(bal.get("equity", 0.0) or 0.0)
    except Exception:
        return 0.0


def _resolve_contract_multiplier(broker: KrakenOmsBroker, symbol: str) -> float:
    try:
        mult = float(broker.get_contract_multiplier(symbol))
        if mult > 0:
            return mult
    except Exception:
        pass
    return 1.0


def _kraken_strict_flatten_for_flip(
    *,
    broker: KrakenOmsBroker,
    symbol: str,
    current_side: str,
    qty: float,
) -> tuple[dict, float]:
    try:
        broker.client.cancel_all_reduce_only_orders(symbol=broker._symbol(symbol))
    except Exception as e:
        log.warning("executor flip cancel reduce-only failed: %s", e)

    close_side = "sell" if current_side == "long" else "buy"
    client_id = f"kraken-flatten-{pd.Timestamp.now('UTC').strftime('%Y%m%d%H%M%S%f')}"

    try:
        close_raw = broker.client.place_market(
            side=close_side,
            size=abs(float(qty)),
            symbol=broker._symbol(symbol),
            reduce_only=True,
            cli_ord_id=client_id,
        )
        close_data = close_raw.get("data") if isinstance(close_raw, dict) else None
        send = (close_data or {}).get("sendStatus", close_data or {}) if isinstance(close_data, dict) else {}

        flat_res = {
            "ok": bool(isinstance(close_raw, dict) and close_raw.get("ok", False)),
            "mode": "KRAKEN_FLATTEN_MKT",
            "details": {
                "symbol": symbol,
                "side": close_side,
                "qty": abs(float(qty)),
                "order_id": send.get("order_id") or send.get("orderId"),
                "client_id": send.get("cliOrdId") or client_id,
                "kind": "market",
                "raw": close_raw,
            },
        }
    except Exception as e:
        flat_res = {
            "ok": False,
            "mode": "KRAKEN_FLATTEN_MKT_ERR",
            "details": {
                "symbol": symbol,
                "side": close_side,
                "qty": abs(float(qty)),
                "error": str(e),
            },
        }
        try:
            return flat_res, float(broker.get_position(symbol))
        except Exception:
            return flat_res, 0.0

    pos_after_flat = float(broker.get_position(symbol))
    if abs(pos_after_flat) > 1e-12:
        t0 = time.time()
        while (time.time() - t0) < 5.0 and abs(pos_after_flat) > 1e-12:
            time.sleep(0.25)
            pos_after_flat = float(broker.get_position(symbol))

    return flat_res, pos_after_flat


def _verify_execution_fill_ratio(
    *,
    broker: KrakenOmsBroker,
    symbol: str,
    action: str,
    target_side: Optional[str],
    target_qty: float,
    min_ratio: float,
) -> None:
    try:
        pos_after = float(broker.get_position(symbol))
    except Exception as e:
        log_throttled(
            log,
            logging.WARNING,
            f"executor_verify_skipped:{symbol}",
            float(os.getenv("LIVE_EXECUTOR_LOG_THROTTLE_SEC", "60")),
            "executor verify skipped (position unavailable): %s",
            e,
        )
        return

    min_ratio = float(max(0.0, min(1.0, min_ratio)))
    if action.startswith("exit_"):
        ok = abs(pos_after) <= 1e-9
        if not ok:
            log_throttled(
                log,
                logging.WARNING,
                f"executor_verify_exit_fail:{symbol}:{action}",
                float(os.getenv("LIVE_EXECUTOR_LOG_THROTTLE_SEC", "60")),
                "executor verify FAIL action=%s expected_flat got_pos=%s",
                action,
                pos_after,
            )
        else:
            log_throttled(
                log,
                logging.INFO,
                f"executor_verify_exit_ok:{symbol}:{action}",
                float(os.getenv("LIVE_EXECUTOR_LOG_THROTTLE_SEC", "300")),
                "executor verify OK action=%s flat",
                action,
            )
        return

    if target_qty <= 0 or target_side not in ("long", "short"):
        return
    got_qty = abs(pos_after)
    ratio = (got_qty / float(target_qty)) if target_qty > 0 else 0.0
    got_side = "long" if pos_after > 0 else ("short" if pos_after < 0 else "flat")
    side_ok = got_side == target_side
    qty_ok = ratio >= min_ratio
    if side_ok and qty_ok:
        log_throttled(
            log,
            logging.INFO,
            f"executor_verify_ok:{symbol}:{action}:{target_side}",
            float(os.getenv("LIVE_EXECUTOR_LOG_THROTTLE_SEC", "300")),
            "executor verify OK action=%s side=%s qty=%s target=%s ratio=%.3f",
            action,
            got_side,
            got_qty,
            target_qty,
            ratio,
        )
    else:
        log_throttled(
            log,
            logging.WARNING,
            f"executor_verify_fail:{symbol}:{action}:{target_side}",
            float(os.getenv("LIVE_EXECUTOR_LOG_THROTTLE_SEC", "60")),
            "executor verify FAIL action=%s want_side=%s got_side=%s got_qty=%s target_qty=%s ratio=%.3f min_ratio=%.3f",
            action,
            target_side,
            got_side,
            got_qty,
            target_qty,
            ratio,
            min_ratio,
        )


def _sync_kraken_stop_loss(
    *,
    broker: KrakenOmsBroker,
    symbol: str,
    terminal: Optional[Dict[str, Any]],
    terminal_pos: int,
    dry_run: bool,
) -> None:
    """Sync a native reduce-only stop-market order on Kraken based on the flip engine terminal state."""
    native_stop_prefix = "quant-sl-"

    def _native_stop_orders() -> List[Dict[str, Any]]:
        try:
            rows = broker.list_open_stop_orders(symbol)
        except Exception:
            return []
        if not isinstance(rows, list):
            return []
        out: List[Dict[str, Any]] = []
        for row in rows:
            client_id = str(row.get("client_id") or row.get("cli_ord_id") or "").strip().lower()
            if client_id.startswith(native_stop_prefix):
                out.append(row)
        return out

    def _cancel_native_stop_orders() -> None:
        for row in _native_stop_orders():
            try:
                broker.cancel_order(
                    order_id=str(row.get("order_id")) if row.get("order_id") else None,
                    client_id=str(row.get("client_id") or row.get("cli_ord_id")) if (row.get("client_id") or row.get("cli_ord_id")) else None,
                )
            except Exception as e:
                log.warning("kraken cancel native stop failed: %s", e)

    def _resolved_native_stop_price() -> Optional[float]:
        if terminal is None:
            return None
        base_stop = _coerce_float(terminal.get("sl"))
        if base_stop is None:
            base_stop = _coerce_float(terminal.get("ttp"))

        imba_levels = terminal.get("imba_levels")
        if not isinstance(imba_levels, dict):
            return base_stop

        if base_stop is None:
            return _opposite_imba_barrier(terminal_pos=terminal_pos, imba_levels=imba_levels)
        if _opposite_imba_supersedes_stop(
            terminal_pos=terminal_pos,
            base_stop=base_stop,
            imba_levels=imba_levels,
        ):
            return None
        return base_stop

    if not _truthy(os.getenv("KRAKEN_NATIVE_SL_ENABLED", "1")):
        return

    if terminal_pos == 0 or terminal is None:
        try:
            _cancel_native_stop_orders()
        except Exception as e:
            log.warning("kraken cancel stop orders failed (flat): %s", e)
        return

    stop_price = _resolved_native_stop_price()
    if stop_price is None:
        _cancel_native_stop_orders()
        return

    stop_side = "sell" if terminal_pos > 0 else "buy"
    pos_qty = abs(float(broker.get_position(symbol)))
    if pos_qty <= 0:
        return

    if dry_run:
        log.info(
            "DRY_RUN kraken native SL: side=%s qty=%s stop=%.4f mode=%s",
            stop_side, pos_qty, stop_price, terminal.get("mode"),
        )
        return

    native_orders = _native_stop_orders()
    for row in native_orders:
        cur_px = _coerce_float(row.get("stop_price", row.get("stopPrice")))
        cur_side = str(row.get("side") or "").strip().lower()
        cur_qty = abs(_coerce_float(row.get("size")) or 0.0)
        if (
            cur_px is not None
            and abs(float(cur_px) - float(stop_price)) <= 1e-9
            and cur_side == stop_side
            and abs(cur_qty - float(pos_qty)) <= 1e-9
        ):
            return

    _cancel_native_stop_orders()

    try:
        order_id = broker.place_stop_market(
            symbol=symbol,
            side=stop_side,
            qty=pos_qty,
            stop_price=float(stop_price),
            reduce_only=True,
            client_id=f"quant-sl-{pd.Timestamp.now('UTC').strftime('%Y%m%d%H%M%S%f')}",
        )
        log.info(
            "kraken native SL placed: order_id=%s side=%s qty=%s stop=%.4f mode=%s",
            order_id, stop_side, pos_qty, stop_price, terminal.get("mode"),
        )
    except Exception as e:
        log.warning(
            "kraken native SL place failed: side=%s qty=%s stop=%.4f err=%s",
            stop_side, pos_qty, stop_price, e,
        )


def _write_dashboard_levels(
    symbol: str,
    terminal: Dict[str, Any],
    live_pos: Optional[float] = None,
    equity: Optional[float] = None,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    mid: Optional[float] = None,
) -> None:
    if not terminal:
        return

    def _norm_side(v: Any) -> Optional[str]:
        if isinstance(v, (int, float)):
            if float(v) > 0:
                return "long"
            if float(v) < 0:
                return "short"
            return None
        s = str(v or "").strip().lower()
        if s in ("1", "+1", "long", "buy"):
            return "long"
        if s in ("-1", "short", "sell"):
            return "short"
        return None

    side = _norm_side(terminal.get("side"))
    if side is None and live_pos is not None:
        side = _norm_side(live_pos)
    if side is None:
        return

    mode = str(terminal.get("mode", "")).upper() or "TTP"
    entry_px = _coerce_float(terminal.get("entry_px"))
    entry_bar_ts = terminal.get("entry_bar_ts")
    sl = _coerce_float(terminal.get("sl"))
    ttp = _coerce_float(terminal.get("ttp"))
    tp1 = _coerce_float(terminal.get("tp1"))
    tp2 = _coerce_float(terminal.get("tp2"))
    be_armed = bool(terminal.get("be_armed", False))
    tp1_done = bool(terminal.get("tp1_done", False))

    if live_pos is not None:
        lp = float(live_pos)
        if abs(lp) > 1e-12:
            live_side = "long" if lp > 0 else "short"
            if side != live_side:
                side = live_side
                entry_px = None
                entry_bar_ts = None
                sl = None
                ttp = None

    rows: List[Dict[str, Any]] = []
    if entry_px is not None:
        rows.append({"kind": "entry", "px": entry_px, "side": side, "mode": mode})
    if sl is not None:
        rows.append({"kind": "sl", "px": sl, "side": side, "mode": mode})
    if ttp is not None:
        rows.append({"kind": "ttp", "px": ttp, "side": side, "mode": mode})
    if tp1 is not None:
        rows.append({"kind": "tp1", "px": tp1, "side": side, "mode": mode})
    if tp2 is not None:
        rows.append({"kind": "tp2", "px": tp2, "side": side, "mode": mode})
    rows.append({"kind": "meta", "be_armed": be_armed, "tp1_done": tp1_done, "side": side, "mode": mode})

    base = Path("/data/live")
    try:
        base.mkdir(parents=True, exist_ok=True)
        out = base / f"active_levels_{symbol.replace('/', '-').replace(':', '-')}.json"
        out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        log_throttled(
            log,
            logging.WARNING,
            f"executor_write_levels:{symbol}",
            float(os.getenv("LIVE_EXECUTOR_LOG_THROTTLE_SEC", "60")),
            "executor failed to write active levels: %s",
            e,
        )

    write_execution_state({
        "symbol": symbol,
        "venue": "kraken",
        "strategy": terminal.get("strategy"),
        "exit_engine": terminal.get("exit_engine"),
        "latched_exit_engine": terminal.get("latched_exit_engine"),
        "ts": _now_iso(),
        "position": float(live_pos) if live_pos is not None else None,
        "side": side,
        "mode": mode,
        "sl": sl,
        "ttp": ttp,
        "tp1": tp1,
        "tp2": tp2,
        "entry_px": entry_px,
        "best_fav": _coerce_float(terminal.get("best_fav")),
        "ttp_trail_pct": _resolve_ttp_trail_pct(),
        "entry_bar_ts": int(pd.Timestamp(entry_bar_ts).timestamp()) if entry_bar_ts is not None else None,
        "live_pos": float(live_pos) if live_pos is not None else None,
        "equity": float(equity) if equity is not None else None,
        "market": {"bid": bid, "ask": ask, "mid": mid},
        "terminal": terminal,
    })


def _renko_path() -> Path:
    env = os.getenv("LIVE_RENKO_PATH")
    if env:
        return Path(env)
    p = Path("/data/live/renko_latest.parquet")
    if p.exists():
        return p
    return Path("data/live/renko_latest.parquet")


def _clear_tp2_leg_runtime(state: ExecutorState) -> None:
    state.tp2_leg_id = None
    state.tp2_leg_side = None
    state.tp2_entry_bar_ts = None
    state.tp2_tp1_done = False
    state.tp2_tp1_pending = False
    state.tp2_size_rem = 1.0
    state.tp2_remaining_qty_abs = None
    state.tp2_tp1_hit_ts = None
    state.tp2_tp1_hit_px = None
    state.tp2_last_consumed_tp1_hit_ts = None


def _sync_tp2_leg_runtime(
    state: ExecutorState,
    *,
    terminal: Dict[str, Any],
    exit_engine: str,
    live_pos: float,
    current_side: str,
) -> None:
    term = dict(terminal or {})
    term_mode = str(term.get("mode") or "").strip().upper()
    term_leg_id = str(term.get("leg_id") or "").strip()
    term_side = str(term.get("side") or "").strip().lower()
    term_entry_bar_ts = _safe_ts(term.get("entry_bar_ts"))
    term_tp1_done = bool(term.get("tp1_done", False))
    term_tp1_hit_ts = _safe_ts(term.get("tp1_hit_ts"))
    term_tp1_hit_ts_iso = term_tp1_hit_ts.isoformat() if term_tp1_hit_ts is not None else None
    term_tp1_hit_px = _coerce_float(term.get("tp1_hit_px"))
    term_size_rem = _coerce_float(term.get("size_rem"))

    live_qty_abs = abs(float(live_pos))
    persisted_leg_matches_live = (
        bool(state.tp2_leg_id)
        and str(state.tp2_leg_side or "").strip().lower() in ("long", "short")
        and current_side in ("long", "short")
        and current_side == str(state.tp2_leg_side or "").strip().lower()
        and live_qty_abs > 1e-12
    )

    if exit_engine != "tp2":
        _clear_tp2_leg_runtime(state)
        return

    term_has_valid_tp2_leg = (
        term_mode == "TP2"
        and bool(term_leg_id)
        and term_side in ("long", "short")
    )

    if not term_has_valid_tp2_leg:
        if persisted_leg_matches_live:
            # Keep the live leg authoritative when replay drifts or forgets the TP2 context.
            if state.tp2_tp1_done:
                prev = _coerce_float(state.tp2_remaining_qty_abs)
                if prev is None or prev <= 0:
                    state.tp2_remaining_qty_abs = live_qty_abs
                else:
                    state.tp2_remaining_qty_abs = min(float(prev), live_qty_abs)
                state.tp2_tp1_pending = False
                if state.tp2_size_rem > 0:
                    state.tp2_size_rem = min(float(state.tp2_size_rem), 1.0)
            else:
                state.tp2_remaining_qty_abs = live_qty_abs
            return

        _clear_tp2_leg_runtime(state)
        return

    is_new_leg = state.tp2_leg_id != term_leg_id
    if is_new_leg:
        _clear_tp2_leg_runtime(state)
        state.tp2_leg_id = term_leg_id
        state.tp2_leg_side = term_side
        state.tp2_entry_bar_ts = term_entry_bar_ts.isoformat() if term_entry_bar_ts is not None else None
        state.tp2_tp1_done = bool(term_tp1_done)
        state.tp2_tp1_pending = False
        if term_size_rem is not None:
            state.tp2_size_rem = max(0.0, min(1.0, float(term_size_rem)))
        else:
            state.tp2_size_rem = 0.5 if term_tp1_done else 1.0
        if current_side == term_side and live_qty_abs > 1e-12:
            state.tp2_remaining_qty_abs = live_qty_abs
    elif state.tp2_remaining_qty_abs is None and current_side == term_side and live_qty_abs > 1e-12:
        state.tp2_remaining_qty_abs = live_qty_abs

    if term_tp1_hit_ts_iso is not None:
        state.tp2_tp1_hit_ts = term_tp1_hit_ts_iso
    if term_tp1_hit_px is not None:
        state.tp2_tp1_hit_px = float(term_tp1_hit_px)

    if term_tp1_done:
        state.tp2_tp1_done = True
        state.tp2_tp1_pending = False
        if term_size_rem is not None:
            state.tp2_size_rem = max(0.0, min(1.0, float(term_size_rem)))
        if term_tp1_hit_ts_iso is not None:
            state.tp2_last_consumed_tp1_hit_ts = term_tp1_hit_ts_iso
        if current_side == term_side and live_qty_abs > 1e-12:
            prev = _coerce_float(state.tp2_remaining_qty_abs)
            if prev is None or prev <= 0:
                state.tp2_remaining_qty_abs = live_qty_abs
            else:
                state.tp2_remaining_qty_abs = min(float(prev), live_qty_abs)
        return

    if (
        term_tp1_hit_ts_iso is not None
        and term_tp1_hit_ts_iso != state.tp2_last_consumed_tp1_hit_ts
        and not state.tp2_tp1_done
        and not state.tp2_tp1_pending
    ):
        state.tp2_tp1_pending = True
        if term_size_rem is not None:
            state.tp2_size_rem = max(0.0, min(1.0, float(term_size_rem)))


def run_once(
    *,
    broker: KrakenOmsBroker,
    oms: MakerFirstOMS,
    symbol: str,
    signals_root: Path,
    state: ExecutorState,
    live_enabled: bool,
    dry_run: bool,
    leverage: float,
) -> ExecutorState:
    bid, ask = broker.get_best_bid_ask(symbol)
    mid = (bid + ask) / 2.0 if (bid and ask) else (ask or bid or 0.0)
    pos = float(broker.get_position(symbol))
    current_side = "long" if pos > 0 else ("short" if pos < 0 else "flat")

    gate = get_live_gate_state()
    gate_on = int(gate.get("gate_on", 0) or 0)
    gate_countertrend_on = int(gate.get("gate_countertrend_on", 0) or 0)
    gate_trend_on = int(gate.get("gate_trend_on", 0) or 0)

    if _pending_follow_entry_is_active(state):
        pending_side_now = str(state.pending_follow_entry_side or "").strip().lower()
        pending_source_side_now = str(state.pending_follow_entry_source_side or "").strip().lower()
        if current_side in ("long", "short"):
            if current_side == pending_side_now:
                log.info(
                    "executor clearing pending follow-entry: live position already on target side symbol=%s side=%s",
                    symbol,
                    current_side,
                )
                _clear_pending_follow_entry(state)
            elif pending_source_side_now in ("long", "short") and current_side == pending_source_side_now:
                log.info(
                    "executor keeping pending follow-entry armed until source side closes symbol=%s source_side=%s target_side=%s",
                    symbol,
                    pending_source_side_now,
                    pending_side_now,
                )
            else:
                log.warning(
                    "executor clearing pending follow-entry: live position on wrong side symbol=%s live_side=%s pending_side=%s",
                    symbol,
                    current_side,
                    pending_side_now,
                )
                _clear_pending_follow_entry(state)

    desired_exit_engine = "flip" if gate_countertrend_on == 1 else "tp2"
    if gate_countertrend_on != 1 and gate_trend_on != 1:
        desired_exit_engine = "flip" if gate_on == 1 else "tp2"

    persisted_tp2_live = (
        bool(state.tp2_leg_id)
        and str(state.tp2_leg_side or "").strip().lower() in ("long", "short")
        and current_side in ("long", "short")
        and current_side == str(state.tp2_leg_side or "").strip().lower()
        and abs(float(pos)) > 1e-12
    )

    if abs(float(pos)) <= 1e-12:
        state.latched_exit_engine = desired_exit_engine
    elif not state.latched_exit_engine:
        # Safety-first:
        # Never reclassify an already-open live trade into flip mode just because
        # the gate currently says "flip" and the persisted latch is missing.
        # This was the dangerous path that could turn an active TP2 leg into TTP/flip.
        state.latched_exit_engine = "tp2"

    if persisted_tp2_live:
        state.latched_exit_engine = "tp2"

    exit_engine = str(state.latched_exit_engine or desired_exit_engine)

    pos_pct = float(os.getenv("LIVE_EXECUTOR_2_POS_PCT", os.getenv("KRAKEN_EQUITY_PCT", os.getenv("LIVE_EXECUTOR_POS_PCT", "0.90"))))
    equity = _resolve_equity(broker)

    _append_equity_snapshot(
        ts_iso=_now_iso(),
        equity=equity,
        position_qty=abs(float(pos)),
        position_side=(1 if float(pos) > 0 else -1 if float(pos) < 0 else 0),
        payload={
            "equity_usd": float(equity) if equity is not None else None,
            "symbol": symbol,
            "position": float(pos),
            "side": current_side,
            "gate_on": gate_on,
            "source": "run_once",
        },
    )

    contract_multiplier = _resolve_contract_multiplier(broker, symbol)
    base_qty = _qty_from_equity_pct(
        equity=equity,
        pos_pct=pos_pct,
        leverage=leverage,
        mid_price=float(mid),
        contract_multiplier=contract_multiplier,
    )

    qty = int(base_qty) if isinstance(base_qty, int) else float(base_qty)
    log.info(
        "executor sizing: equity=%.2f pos_pct=%.2f leverage=%.1f mid=%.4f mult=%.4f base_qty=%s -> qty=%s",
        equity, pos_pct, leverage, mid, contract_multiplier, base_qty, qty,
    )

    renko_bars = _load_renko_bars(_renko_path(), limit=int(os.getenv("LIVE_RENKO_LIMIT", "4000")))
    signals_df = _load_signals_df(signals_root, symbol)
    imba_levels = get_latest_imba_barriers(
        renko_bars,
        ImbaParams(
            lookback=int(os.getenv("LIVE_IMBA_LOOKBACK", "250")),
            fixed_sl_abs=float(os.getenv("LIVE_IMBA_SL_ABS", "1.5")),
        ),
    )
    imba_long_barrier = _coerce_float(imba_levels.get("long_barrier"))
    imba_short_barrier = _coerce_float(imba_levels.get("short_barrier"))
    imba_barrier_ts = imba_levels.get("ts")

    if exit_engine == "flip":
        ev, terminal = _latest_backtest_event(renko_bars=renko_bars, signals_df=signals_df)
    else:
        tp2_params = TP2Params(
            fee_bps=float(os.getenv("LIVE_TP2_FEE_BPS", os.getenv("LIVE_FLIP_FEE_BPS", "0"))),
            tp1_pct=float(os.getenv("LIVE_TP1_PCT", "0.04")),
            tp2_pct=float(os.getenv("LIVE_TP2_PCT", "0.08")),
            tp1_frac=float(os.getenv("LIVE_TP1_FRAC", "0.5")),
            min_sl_pct=float(os.getenv("LIVE_TP2_MIN_SL_PCT", "0.03")),
            max_sl_pct=float(os.getenv("LIVE_TP2_MAX_SL_PCT", "0.08")),
            swing_lookback=int(os.getenv("LIVE_TP2_SWING_LOOKBACK", "180")),
            flip_on_opposite=_truthy(os.getenv("LIVE_TP2_FLIP_ON_OPPOSITE", "1")),
            be_after_tp1=_truthy(os.getenv("LIVE_TP2_BE_AFTER_TP1", "1")),
            be_offset_pct=float(os.getenv("LIVE_TP2_BE_OFFSET_PCT", "0.0")),
        )
        _, events_df, terminal = run_follow_tp2_state_machine(
            bars=renko_bars,
            signals_df=signals_df,
            params=tp2_params,
            regime_on=None,
            regime_forces_flat=False,
        )
        if events_df is not None and not events_df.empty:
            events_df = events_df.sort_values(["ts", "seq"]).reset_index(drop=True)
            ev = events_df.iloc[-1]
        else:
            ev = None

    fallback_used = False
    fallback_sig_ts_iso: Optional[str] = None
    fallback_sig_v: Optional[int] = None

    terminal_pos = int(terminal.get("pos", 0)) if terminal else 0
    terminal_sig = f"{terminal_pos}|{terminal.get('mode', '')}|{terminal.get('entry_px', '')}|{terminal.get('ttp', '')}|{terminal.get('sl', '')}"

    sig_now = _latest_signal(signals_root=signals_root, symbol=symbol)
    sig_now_v = int(sig_now["signal"]) if sig_now is not None else 0

    flat_latch_active = False
    flat_latch_ts = _safe_ts(state.flat_until_new_signal_ts)
    if flat_latch_ts is not None:
        if sig_now is not None and sig_now["ts"] > flat_latch_ts:
            state.flat_until_new_signal_ts = None
            state.flat_latch_reason = None
        else:
            flat_latch_active = True

    if terminal_pos == 0 and not flat_latch_active:
        if current_side == "long" and sig_now_v > 0:
            log.warning(
                "GUARD FIRED: terminal_pos=0 but live long + latest signal long -> suppress flat symbol=%s sig_ts=%s bars=%s signals=%s",
                symbol,
                sig_now["ts"].isoformat() if sig_now is not None else None,
                len(renko_bars),
                len(signals_df),
            )
            terminal_pos = 1
            terminal_sig = f"guard_long|{sig_now['ts'].isoformat() if sig_now is not None else ''}"
            terminal = dict(terminal or {})
            terminal["pos"] = 1
            terminal["side"] = "long"
        elif current_side == "short" and sig_now_v < 0:
            log.warning(
                "GUARD FIRED: terminal_pos=0 but live short + latest signal short -> suppress flat symbol=%s sig_ts=%s bars=%s signals=%s",
                symbol,
                sig_now["ts"].isoformat() if sig_now is not None else None,
                len(renko_bars),
                len(signals_df),
            )
            terminal_pos = -1
            terminal_sig = f"guard_short|{sig_now['ts'].isoformat() if sig_now is not None else ''}"
            terminal = dict(terminal or {})
            terminal["pos"] = -1
            terminal["side"] = "short"

    if ev is not None:
        try:
            state_payload = write_execution_state({
                "symbol": symbol,
                "venue": "kraken",
                "strategy": exit_engine,
                "exit_engine": exit_engine,
                "latched_exit_engine": state.latched_exit_engine,
                "ts": _now_iso(),
                "position": float(pos),
                "side": current_side,
                "equity": float(equity),
                "backtest_event": {
                    "ts": pd.Timestamp(ev["ts"]).isoformat() if "ts" in ev else None,
                    "event": str(ev.get("event", "")),
                    "side": int(ev.get("side", 0)),
                    "seq": int(ev.get("seq", 0)),
                },
                "terminal": terminal,
                "market": {"bid": bid, "ask": ask, "mid": mid},
                "sizing": {"equity": float(equity), "pos_pct": pos_pct, "leverage": leverage, "contract_multiplier": contract_multiplier, "qty": qty},
                "imba_levels": {
                    "ts": imba_barrier_ts,
                    "long_barrier": imba_long_barrier,
                    "short_barrier": imba_short_barrier,
                },
                "gate": gate,
            })
            log.debug("executor wrote state=%s", state_payload)
        except Exception as e:
            log_throttled(
                log,
                logging.WARNING,
                f"executor_state_write:{symbol}",
                float(os.getenv("LIVE_EXECUTOR_LOG_THROTTLE_SEC", "60")),
                "executor state write failed: %s",
                e,
            )

    terminal = _apply_live_ttp_guard(
        terminal,
        live_pos=pos,
        live_mid=float(mid),
        ttp_trail_pct=_resolve_ttp_trail_pct(),
    )
    terminal = dict(terminal or {})
    terminal["strategy"] = exit_engine
    terminal["exit_engine"] = exit_engine
    terminal["latched_exit_engine"] = state.latched_exit_engine
    terminal["imba_levels"] = {
        "ts": imba_barrier_ts,
        "long_barrier": imba_long_barrier,
        "short_barrier": imba_short_barrier,
    }

    _sync_tp2_leg_runtime(
        state,
        terminal=terminal,
        exit_engine=exit_engine,
        live_pos=pos,
        current_side=current_side,
    )

    terminal["tp2_leg_state"] = {
        "active": bool(state.tp2_leg_id),
        "leg_id": state.tp2_leg_id,
        "side": state.tp2_leg_side,
        "entry_bar_ts": state.tp2_entry_bar_ts,
        "tp1_done": bool(state.tp2_tp1_done),
        "tp1_pending": bool(state.tp2_tp1_pending),
        "size_rem": float(state.tp2_size_rem),
        "remaining_qty_abs": state.tp2_remaining_qty_abs,
        "tp1_hit_ts": state.tp2_tp1_hit_ts,
        "tp1_hit_px": state.tp2_tp1_hit_px,
        "last_consumed_tp1_hit_ts": state.tp2_last_consumed_tp1_hit_ts,
        "flat_until_new_signal_ts": state.flat_until_new_signal_ts,
        "flat_latch_reason": state.flat_latch_reason,
        "flat_latch_active": flat_latch_active,
    }

    if (
        bool(state.tp2_leg_id)
        and current_side in ("long", "short")
        and current_side == str(state.tp2_leg_side or "").strip().lower()
        and abs(float(pos)) > 1e-12
    ):
        state.open_leg_mode = "tp2"
        state.open_leg_id = state.tp2_leg_id
        state.open_leg_side = state.tp2_leg_side
        state.open_leg_entry_bar_ts = state.tp2_entry_bar_ts

    _write_dashboard_levels(
        symbol=symbol,
        terminal=terminal,
        live_pos=pos,
        equity=equity,
        bid=bid,
        ask=ask,
        mid=mid,
    )

    want_side = "long" if terminal_pos > 0 else ("short" if terminal_pos < 0 else None)
    if flat_latch_active and current_side == "flat":
        want_side = None

    tp2_leg_active = (
        exit_engine == "tp2"
        and bool(state.tp2_leg_id)
        and current_side in ("long", "short")
        and current_side == str(state.tp2_leg_side or "").strip().lower()
    )
    terminal_mode = str(terminal.get("mode") or "").strip().upper()
    qty_effective = float(qty)
    if tp2_leg_active and state.tp2_tp1_done:
        rem_abs = _coerce_float(state.tp2_remaining_qty_abs)
        if rem_abs is not None and rem_abs > 0:
            qty_effective = float(rem_abs)
        elif abs(float(pos)) > 1e-12:
            qty_effective = abs(float(pos))

    tp1_partial_qty = 0.0
    if terminal_mode != "WAIT" and tp2_leg_active and state.tp2_tp1_pending and not state.tp2_tp1_done and abs(float(pos)) > 1e-12:
        tp1_frac = _coerce_float(terminal.get("tp1_frac"))
        if tp1_frac is None or tp1_frac <= 0.0 or tp1_frac >= 1.0:
            tp1_frac = 0.5
        tp1_partial_qty = max(0.0, min(abs(float(pos)), abs(float(pos)) * float(tp1_frac)))

    event_name = str(ev.get("event", "")) if ev is not None else "none"
    sig_side_now = "long" if sig_now_v > 0 else ("short" if sig_now_v < 0 else None)
    ttp_source_ts = (
        pd.Timestamp(ev["ts"]).isoformat()
        if ev is not None and "ts" in ev
        else (str(terminal.get("entry_bar_ts") or "").strip() or _now_iso())
    )

    def _ok(res: Any) -> bool:
        if isinstance(res, dict):
            return bool(res.get("ok", False))
        return bool(getattr(res, "ok", False))

    def _details(res: Any) -> Dict[str, Any]:
        if isinstance(res, dict):
            d = res.get("details", {})
            return d if isinstance(d, dict) else {}
        d = getattr(res, "details", {})
        return d if isinstance(d, dict) else {}

    def _mode(res: Any) -> str:
        if isinstance(res, dict):
            return str(res.get("mode", ""))
        return str(getattr(res, "mode", ""))

    def _execution_side(default_side: str, details: Dict[str, Any]) -> str:
        return str(details.get("side") or default_side)

    def _execution_qty(default_qty: float, details: Dict[str, Any]) -> Optional[float]:
        return _coerce_float(details.get("qty", default_qty))

    def _execution_price(details: Dict[str, Any]) -> Optional[float]:
        return _coerce_float(details.get("price"))
    effective_terminal_mode = str(terminal_mode)
    wait_same_side_imba_confirmed = (
        terminal_mode == "WAIT"
        and current_side in ("long", "short")
        and sig_side_now == current_side
        and _coerce_float(terminal.get("ttp")) is not None
    )
    if wait_same_side_imba_confirmed:
        effective_terminal_mode = "TTP"
    terminal = dict(terminal or {})
    terminal["mode"] = effective_terminal_mode

    if (
        current_side == "long"
        and imba_short_barrier is not None
        and float(mid) <= float(imba_short_barrier)
        and not _pending_follow_entry_is_active(state)
    ):
        _arm_pending_follow_entry(
            state,
            source_side="long",
            target_side="short",
            reason="opposite_imba_close",
            source_ts=ttp_source_ts,
        )
    elif (
        current_side == "short"
        and imba_long_barrier is not None
        and float(mid) >= float(imba_long_barrier)
        and not _pending_follow_entry_is_active(state)
    ):
        _arm_pending_follow_entry(
            state,
            source_side="short",
            target_side="long",
            reason="opposite_imba_close",
            source_ts=ttp_source_ts,
        )
    ttp_handoff = _ttp_reenter_handoff_action(
        state,
        current_side=current_side,
        mid=float(mid),
        terminal=terminal,
        source_ts=ttp_source_ts,
    )

    def _retag_stop_order(
        *,
        old_kind: str,
        new_kind: str,
        side: str,
        qty: float,
        stop_price: float,
    ) -> bool:
        row = oms.find_stop_order_by_kind(symbol, old_kind)
        if not row:
            return False

        cur_px = _coerce_float(row.get("stop_price", row.get("stopPrice")))
        if cur_px is None:
            return False

        if abs(float(cur_px) - float(stop_price)) > 1e-9:
            return False

        oms.cancel_orders_by_kind(symbol, old_kind)
        oms.arm_stop_entry(
            symbol=symbol,
            side=side,
            qty=float(qty),
            stop_price=float(stop_price),
            kind=new_kind,
        )
        return True

    if current_side == "flat":
        if (
            not _pending_follow_entry_is_active(state)
            and state.last_live_side in ("long", "short")
        ):
            opposite_imba_filled = False
            target_flip_side: Optional[str] = None
            if state.last_live_side == "long":
                opp_order = oms.find_stop_order_by_kind(symbol, "opposite_imba_short")
                if opp_order is None:
                    opposite_imba_filled = True
                    target_flip_side = "short"
            elif state.last_live_side == "short":
                opp_order = oms.find_stop_order_by_kind(symbol, "opposite_imba_long")
                if opp_order is None:
                    opposite_imba_filled = True
                    target_flip_side = "long"

            if opposite_imba_filled and target_flip_side is not None:
                target_pos = 1 if target_flip_side == "long" else -1
                if terminal_pos == target_pos:
                    log.info(
                        "executor detected opposite_imba fill (order gone, terminal agrees) was=%s target=%s terminal_pos=%s -> arming follow entry",
                        state.last_live_side, target_flip_side, terminal_pos,
                    )
                    _arm_pending_follow_entry(
                        state,
                        source_side=str(state.last_live_side),
                        target_side=target_flip_side,
                        reason="opposite_imba_stop_filled",
                        source_ts=ttp_source_ts,
                    )
                else:
                    log.info(
                        "executor detected opposite_imba fill but terminal disagrees was=%s target=%s terminal_pos=%s -> skipping follow entry",
                        state.last_live_side, target_flip_side, terminal_pos,
                    )

        if _pending_follow_entry_is_active(state):
            pending_side = str(state.pending_follow_entry_side or "").strip().lower()
            if float(qty) > 0:
                res = oms.enter_market(symbol=symbol, side=pending_side, qty=float(qty))
                log.info(
                    "executor pending follow-entry result symbol=%s side=%s qty=%s reason=%s result=%s",
                    symbol,
                    pending_side,
                    qty,
                    state.pending_follow_entry_reason,
                    res,
                )
                if _ok(res):
                    details = _details(res)
                    _clear_pending_follow_entry(state)
                    state.open_leg_mode = str(desired_exit_engine)
                    state.open_leg_id = str(terminal.get("leg_id") or "") or None
                    state.open_leg_side = str(pending_side)
                    entry_bar_ts = terminal.get("entry_bar_ts")
                    state.open_leg_entry_bar_ts = str(entry_bar_ts) if entry_bar_ts is not None else None

                    exec_qty = _execution_qty(float(qty), details)

                    state.n_executions += 1
                    _append_execution_event(
                        strategy="live_executor_2",
                        symbol=symbol,
                        ts_iso=_now_iso(),
                        seq=int(state.n_executions),
                        execution_kind="fill",
                        order_action=_execution_side("buy" if pending_side == "long" else "sell", details),
                        reason_code=event_name,
                        position_before=0,
                        position_after=(1 if pending_side == "long" else -1),
                        order_id=str(details.get("order_id") or "") or None,
                        client_oid=str(details.get("client_id") or "") or None,
                        side=_execution_side("buy" if pending_side == "long" else "sell", details),
                        qty=exec_qty,
                        price=_execution_price(details),
                        reduce_only=False,
                        status=_mode(res) or "fill",
                        reject_reason=None,
                        payload_json={"action": "follow_entry_after_close", "result": details, "event_name": event_name},
                    )

                    fresh_equity = _resolve_equity(broker)
                    _append_equity_snapshot(
                        ts_iso=_now_iso(),
                        equity=fresh_equity,
                        position_qty=float(exec_qty if exec_qty is not None else qty),
                        position_side=(1 if pending_side == "long" else -1),
                        payload={
                            "equity_usd": float(fresh_equity) if fresh_equity is not None else None,
                            "symbol": symbol,
                            "position": float(qty),
                            "side": pending_side,
                            "gate_on": gate_on,
                            "source": "post_follow_entry_fill",
                            "event_name": event_name,
                        },
                        force=True,
                    )

                    state.last_terminal_sig = terminal_sig
                    state.last_action = "follow_entry_after_close"
                    state.last_gate_on = int(gate_on)
                    state.last_live_side = pending_side
                    return state
                else:
                    state.last_terminal_sig = terminal_sig
                    state.last_action = "follow_entry_pending_retry"
                    state.last_gate_on = int(gate_on)
                    state.last_live_side = current_side
                    return state
            else:
                log.warning(
                    "executor pending follow-entry skipped qty=0 symbol=%s side=%s reason=%s",
                    symbol,
                    pending_side,
                    state.pending_follow_entry_reason,
                )

        if ttp_handoff is None:
            flat_entry_results = []

            def _stop_px(row: Optional[Dict[str, Any]]) -> Optional[float]:
                if not row:
                    return None
                return _coerce_float(row.get("stop_price", row.get("stopPrice")))

            long_existing = oms.find_stop_order_by_kind(symbol, "flat_entry_long")
            short_existing = oms.find_stop_order_by_kind(symbol, "flat_entry_short")

            if imba_long_barrier is not None and float(qty) > 0:
                cur_px = _stop_px(long_existing)
                if cur_px is None or abs(float(cur_px) - float(imba_long_barrier)) > 1e-9:
                    if long_existing:
                        oms.cancel_orders_by_kind(symbol, "flat_entry_long")
                    flat_entry_results.append(
                        oms.arm_stop_entry(
                            symbol=symbol,
                            side="long",
                            qty=float(qty),
                            stop_price=float(imba_long_barrier),
                            kind="flat_entry_long",
                        )
                    )

            if imba_short_barrier is not None and float(qty) > 0:
                cur_px = _stop_px(short_existing)
                if cur_px is None or abs(float(cur_px) - float(imba_short_barrier)) > 1e-9:
                    if short_existing:
                        oms.cancel_orders_by_kind(symbol, "flat_entry_short")
                    flat_entry_results.append(
                        oms.arm_stop_entry(
                            symbol=symbol,
                            side="short",
                            qty=float(qty),
                            stop_price=float(imba_short_barrier),
                            kind="flat_entry_short",
                        )
                    )

            log.info(
                "executor flat-state entries synced symbol=%s qty=%s long_barrier=%s short_barrier=%s changed=%s",
                symbol,
                qty,
                imba_long_barrier,
                imba_short_barrier,
                len(flat_entry_results),
            )

            state.last_terminal_sig = terminal_sig
            state.last_action = "sync_flat_entries"
            state.last_gate_on = int(gate_on)
            state.last_live_side = current_side
            return state

        log.info(
            "executor preserving flat TTP handoff symbol=%s prior_side=%s target_side=%s",
            symbol,
            ttp_handoff.get("prior_side"),
            ttp_handoff.get("target_side"),
        )


    if state.last_live_side == "flat" and current_side == "long" and imba_short_barrier is not None:
        _retag_stop_order(
            old_kind="flat_entry_short",
            new_kind="opposite_imba_short",
            side="short",
            qty=float(abs(pos)),
            stop_price=float(imba_short_barrier),
        )
    elif state.last_live_side == "flat" and current_side == "short" and imba_long_barrier is not None:
        _retag_stop_order(
            old_kind="flat_entry_long",
            new_kind="opposite_imba_long",
            side="long",
            qty=float(abs(pos)),
            stop_price=float(imba_long_barrier),
        )

    def _sync_stop_order(
        *,
        kind: str,
        side: str,
        qty: float,
        stop_price: Optional[float],
        reduce_only: bool,
    ) -> bool:
        if stop_price is None or qty <= 0:
            oms.cancel_orders_by_kind(symbol, kind)
            return False

        row = oms.find_stop_order_by_kind(symbol, kind)
        cur_px = _coerce_float(row.get("stop_price", row.get("stopPrice"))) if row else None

        if cur_px is not None and abs(float(cur_px) - float(stop_price)) <= 1e-9:
            return False

        if row:
            oms.cancel_orders_by_kind(symbol, kind)

        oms.arm_stop_exit(
            symbol=symbol,
            side=side,
            qty=float(qty),
            stop_price=float(stop_price),
            kind=kind,
            reduce_only=reduce_only,
        )
        return True

    def _sync_take_profit_order(
        *,
        kind: str,
        side: str,
        qty: float,
        stop_price: Optional[float],
        reduce_only: bool,
    ) -> bool:
        if stop_price is None or qty <= 0:
            oms.cancel_orders_by_kind(symbol, kind)
            return False

        row = oms.find_stop_order_by_kind(symbol, kind)
        cur_px = _coerce_float(row.get("stop_price", row.get("stopPrice"))) if row else None

        if cur_px is not None and abs(float(cur_px) - float(stop_price)) <= 1e-9:
            return False

        if row:
            oms.cancel_orders_by_kind(symbol, kind)

        oms.arm_take_profit_exit(
            symbol=symbol,
            side=side,
            qty=float(qty),
            stop_price=float(stop_price),
            kind=kind,
            reduce_only=reduce_only,
        )
        return True

    leg_mode = str(state.open_leg_mode or "").strip().lower()

    if leg_mode == "tp2" and current_side in ("long", "short") and abs(float(pos)) > 1e-12:
        live_qty = float(abs(pos))
        tp1_px = _coerce_float(terminal.get("tp1"))
        tp2_px = _coerce_float(terminal.get("tp2"))
        sl_px = _coerce_float(terminal.get("sl"))
        imba_levels = terminal.get("imba_levels") if isinstance(terminal, dict) else None
        opposite_supersedes_sl = _opposite_imba_supersedes_stop(
            terminal_pos=(1 if current_side == "long" else -1),
            base_stop=sl_px,
            imba_levels=imba_levels if isinstance(imba_levels, dict) else None,
        )

        changed = 0
        if opposite_supersedes_sl:
            oms.cancel_orders_by_kind(symbol, "tp2_sl")
        else:
            changed += int(_sync_stop_order(
                kind="tp2_sl",
                side=current_side,
                qty=live_qty,
                stop_price=sl_px,
                reduce_only=True,
            ))
        changed += int(_sync_take_profit_order(
            kind="tp2_tp1",
            side=current_side,
            qty=live_qty * 0.5,
            stop_price=tp1_px,
            reduce_only=True,
        ))
        changed += int(_sync_take_profit_order(
            kind="tp2_tp2",
            side=current_side,
            qty=live_qty,
            stop_price=tp2_px,
            reduce_only=True,
        ))

        if current_side == "long":
            changed += int(_sync_stop_order(
                kind="opposite_imba_short",
                side="long",
                qty=live_qty,
                stop_price=imba_short_barrier,
                reduce_only=True,
            ))
            oms.cancel_orders_by_kind(symbol, "opposite_imba_long")
        else:
            changed += int(_sync_stop_order(
                kind="opposite_imba_long",
                side="short",
                qty=live_qty,
                stop_price=imba_long_barrier,
                reduce_only=True,
            ))
            oms.cancel_orders_by_kind(symbol, "opposite_imba_short")

        log.info(
            "executor tp2-state synced symbol=%s side=%s qty=%s sl=%s tp1=%s tp2=%s changed=%s",
            symbol,
            current_side,
            live_qty,
            sl_px,
            tp1_px,
            tp2_px,
            changed,
        )

    elif str(effective_terminal_mode or "").strip().upper() == "WAIT" and current_side in ("long", "short") and abs(float(pos)) > 1e-12:
        live_qty = float(abs(pos))
        sl_px = _coerce_float(terminal.get("sl"))
        imba_levels = terminal.get("imba_levels") if isinstance(terminal, dict) else None
        opposite_supersedes_sl = _opposite_imba_supersedes_stop(
            terminal_pos=(1 if current_side == "long" else -1),
            base_stop=sl_px,
            imba_levels=imba_levels if isinstance(imba_levels, dict) else None,
        )

        changed = 0
        if opposite_supersedes_sl:
            oms.cancel_orders_by_kind(symbol, "wait_sl")
        else:
            changed += int(_sync_stop_order(
                kind="wait_sl",
                side=current_side,
                qty=live_qty,
                stop_price=sl_px,
                reduce_only=True,
            ))

        oms.cancel_orders_by_kind(symbol, "tp2_sl")
        oms.cancel_orders_by_kind(symbol, "tp2_tp1")
        oms.cancel_orders_by_kind(symbol, "tp2_tp2")
        oms.cancel_orders_by_kind(symbol, "ttp_exit")

        if current_side == "long":
            changed += int(_sync_stop_order(
                kind="opposite_imba_short",
                side="long",
                qty=live_qty,
                stop_price=imba_short_barrier,
                reduce_only=True,
            ))
            oms.cancel_orders_by_kind(symbol, "opposite_imba_long")
        else:
            changed += int(_sync_stop_order(
                kind="opposite_imba_long",
                side="short",
                qty=live_qty,
                stop_price=imba_long_barrier,
                reduce_only=True,
            ))
            oms.cancel_orders_by_kind(symbol, "opposite_imba_short")

        log.info(
            "executor wait-state synced symbol=%s side=%s qty=%s sl=%s changed=%s",
            symbol,
            current_side,
            live_qty,
            sl_px,
            changed,
        )

    elif str(effective_terminal_mode or "").strip().upper() == "TTP" and current_side in ("long", "short") and abs(float(pos)) > 1e-12:
        live_qty = float(abs(pos))
        ttp_px = _coerce_float(terminal.get("ttp"))

        changed = 0
        changed += int(_sync_stop_order(
            kind="ttp_exit",
            side=current_side,
            qty=live_qty,
            stop_price=ttp_px,
            reduce_only=True,
        ))

        oms.cancel_orders_by_kind(symbol, "tp2_sl")
        oms.cancel_orders_by_kind(symbol, "tp2_tp1")
        oms.cancel_orders_by_kind(symbol, "tp2_tp2")
        oms.cancel_orders_by_kind(symbol, "wait_sl")

        if current_side == "long":
            changed += int(_sync_stop_order(
                kind="opposite_imba_short",
                side="long",
                qty=live_qty,
                stop_price=imba_short_barrier,
                reduce_only=True,
            ))
            oms.cancel_orders_by_kind(symbol, "opposite_imba_long")
        else:
            changed += int(_sync_stop_order(
                kind="opposite_imba_long",
                side="short",
                qty=live_qty,
                stop_price=imba_long_barrier,
                reduce_only=True,
            ))
            oms.cancel_orders_by_kind(symbol, "opposite_imba_short")

        log.info(
            "executor ttp-state synced symbol=%s side=%s qty=%s ttp=%s changed=%s",
            symbol,
            current_side,
            live_qty,
            ttp_px,
            changed,
        )

    if wait_same_side_imba_confirmed and ttp_handoff is None and current_side in ("long", "short"):
        state.last_terminal_sig = terminal_sig
        state.last_action = "sync_wait_to_ttp_orders"
        state.last_gate_on = int(gate_on)
        state.last_live_side = current_side
        return state

    ttp_prior_side: Optional[str] = None
    ttp_target_side: Optional[str] = None
    ttp_handoff_key: Optional[str] = None

    action = "hold"
    if ttp_handoff is not None:
        ttp_prior_side = str(ttp_handoff.get("prior_side") or "").strip().lower() or None
        ttp_target_side = str(ttp_handoff.get("target_side") or "").strip().lower() or None
        ttp_handoff_key = str(ttp_handoff.get("key") or "").strip() or None
        if ttp_handoff_key is None or _ttp_reenter_attempt_allowed(state, ttp_handoff_key):
            action = str(ttp_handoff["action"])
        else:
            log.info(
                "executor ttp reenter cooldown active symbol=%s key=%s until=%s",
                symbol,
                ttp_handoff_key,
                state.ttp_reenter_cooldown_until,
            )
    elif leg_mode == "tp2" and current_side in ("long", "short"):
        action = "hold"
    elif terminal_mode == "WAIT" and current_side in ("long", "short"):
        action = "hold"
    elif tp2_leg_active and state.tp2_tp1_pending and not state.tp2_tp1_done and tp1_partial_qty > 0:
        action = "tp1_partial"
    elif flat_latch_active and current_side == "flat":
        action = "hold"

    native_stop_synced = False
    if live_enabled and (not dry_run) and action == "hold":
        _sync_kraken_stop_loss(
            broker=broker,
            symbol=symbol,
            terminal=terminal,
            terminal_pos=terminal_pos,
            dry_run=dry_run,
        )
        native_stop_synced = True


    if terminal_sig == state.last_terminal_sig and action == "hold":
        log_throttled(
            log,
            logging.INFO,
            f"executor_nochange:{symbol}",
            float(os.getenv("LIVE_EXECUTOR_LOG_THROTTLE_SEC", "30")),
            "executor no change: terminal unchanged and already satisfied pos=%s terminal_pos=%s",
            pos,
            terminal_pos,
        )
        state.last_gate_on = int(gate_on)
        state.last_live_side = current_side
        return state

    ts_iso = _now_iso()

    event_sig = _event_sig(ev) if ev is not None else f"none|{terminal_sig}"
    if event_sig == state.last_event_sig and action == "hold":
        log_throttled(
            log,
            logging.INFO,
            f"executor_hold_dedup:{symbol}",
            float(os.getenv("LIVE_EXECUTOR_LOG_THROTTLE_SEC", "30")),
            "executor hold dedup symbol=%s event=%s",
            symbol,
            event_name,
        )
        state.last_terminal_sig = terminal_sig
        state.last_gate_on = int(gate_on)
        state.last_live_side = current_side
        return state

    engine_mode = str(terminal.get("mode", ""))
    action_side, position_before, position_after = _derive_action_event_fields(
        action=action,
        current_side=current_side,
        want_side=want_side,
        terminal_pos=terminal_pos,
        ttp_prior_side=ttp_prior_side,
    )

    state.n_actions += 1
    action_seq = int(state.n_actions)

    _append_action_event(
        strategy="live_executor_2",
        symbol=symbol,
        ts_iso=ts_iso,
        seq=action_seq,
        engine_action=action,
        action_side=action_side,
        reason_code=event_name,
        position_before=position_before,
        position_after=position_after,
        engine_mode_before=engine_mode,
        engine_mode_after=engine_mode,
        blocked=False,
        block_reason=None,
        payload_json={
            "terminal_pos": terminal_pos,
            "current_side": current_side,
            "want_side": want_side,
            "qty": float(qty),
            "mid": float(mid),
            "event_name": event_name,
            "fallback_used": fallback_used,
            "terminal_sig": terminal_sig,
            "last_terminal_sig": state.last_terminal_sig,
            "event_sig": event_sig,
            "last_event_sig": state.last_event_sig,
            "live_pos": float(pos),
            "gate": gate,
            "ev": ev if isinstance(ev, dict) else None,
        },
    )

    exp_side: Optional[str] = None
    exp_action: Optional[str] = None
    exp_qty: Optional[float] = None
    exp_note: Optional[str] = None

    if action in ("enter_long", "enter_short") and want_side is not None:
        exp_side = want_side
        exp_action = "entry"
        exp_qty = float(qty)
        exp_note = f"executor action={action} event={event_name} current={current_side}"
    elif action in ("ttp_confirm_reenter_short", "ttp_confirm_reenter_long") and ttp_target_side in ("long", "short"):
        exp_side = str(ttp_target_side)
        exp_action = "entry"
        exp_qty = float(qty_effective)
        exp_note = f"executor action={action} event={event_name} current={ttp_prior_side or current_side}"
    elif action == "tp1_partial" and current_side in ("long", "short") and tp1_partial_qty > 0:
        exp_side = current_side
        exp_action = "exit_tp"
        exp_qty = float(tp1_partial_qty)
        exp_note = f"executor action={action} event={event_name} current={current_side}"
    elif action in ("flip_to_long", "flip_to_short") and want_side is not None:
        exp_side = want_side
        exp_action = "exit_flip"
        exp_qty = float(qty)
        exp_note = f"executor action={action} event={event_name} current={current_side}"
    elif action.startswith("exit_") and abs(pos) > 1e-12:
        exp_side = current_side
        if event_name in ("sl_exit", "be_exit"):
            exp_action = "exit_sl"
        elif event_name in ("tp_exit",):
            exp_action = "exit_tp"
        else:
            exp_action = "exit_flip"
        exp_qty = abs(float(pos))
        exp_note = f"executor action={action} event={event_name} current={current_side}"
    elif action.startswith("scale_") and want_side is not None:
        add_qty = max(0.0, float(qty_effective) - abs(float(pos)))
        if add_qty > 0:
            exp_side = want_side
            exp_action = "entry"
            exp_qty = float(add_qty)
            exp_note = f"executor action={action} event=scale current={current_side}"

    if exp_side is not None and exp_action is not None and exp_qty is not None and exp_qty > 0:
        record_expected(
            ExpectedTrade(
                ts=ts_iso,
                symbol=symbol,
                side=exp_side,
                action=exp_action,
                qty=float(exp_qty),
                expected_px=float(mid) if mid > 0 else None,
                note=exp_note,
            )
        )

    if not live_enabled:
        log.warning("executor LIVE_TRADING_ENABLED=0 -> simulated action=%s", action)
    elif dry_run:
        log.warning("executor DRY_RUN=1 -> simulated action=%s", action)
    else:
        def _ok(res: Any) -> bool:
            if isinstance(res, dict):
                return bool(res.get("ok", False))
            return bool(getattr(res, "ok", False))

        def _details(res: Any) -> Dict[str, Any]:
            if isinstance(res, dict):
                d = res.get("details", {})
                return d if isinstance(d, dict) else {}
            d = getattr(res, "details", {})
            return d if isinstance(d, dict) else {}

        def _mode(res: Any) -> str:
            if isinstance(res, dict):
                return str(res.get("mode", ""))
            return str(getattr(res, "mode", ""))

        def _execution_side(default_side: str, details: Dict[str, Any]) -> str:
            return str(details.get("side") or default_side)

        def _execution_qty(default_qty: float, details: Dict[str, Any]) -> Optional[float]:
            return _coerce_float(details.get("qty", default_qty))

        def _execution_price(details: Dict[str, Any]) -> Optional[float]:
            return _coerce_float(details.get("price"))

        target_qty_for_verify = float(qty_effective)
        target_side_for_verify: Optional[str] = want_side if action.startswith(("enter_", "flip_to_", "scale_")) else None

        if action == "tp1_partial" and current_side in ("long", "short") and tp1_partial_qty > 0:
            res = oms.partial_tp1_market(symbol=symbol, side=current_side, qty=float(tp1_partial_qty))
            log.info("executor tp1 partial result=%s qty=%s", res, tp1_partial_qty)
            target_side_for_verify = current_side
            target_qty_for_verify = max(0.0, abs(float(pos)) - float(tp1_partial_qty))
            if _ok(res):
                state.open_leg_mode = "tp2"
                state.open_leg_id = str(terminal.get("leg_id") or state.open_leg_id or "") or None
                state.open_leg_side = str(current_side)
                entry_bar_ts = terminal.get("entry_bar_ts")
                state.open_leg_entry_bar_ts = str(entry_bar_ts) if entry_bar_ts is not None else state.open_leg_entry_bar_ts
                details = _details(res)
                state.n_executions += 1
                _append_execution_event(
                    strategy="live_executor_2",
                    symbol=symbol,
                    ts_iso=_now_iso(),
                    seq=int(state.n_executions),
                    execution_kind="fill",
                    order_action=_execution_side("sell" if current_side == "long" else "buy", details),
                    reason_code=event_name,
                    position_before=position_before,
                    position_after=position_after,
                    order_id=str(details.get("order_id") or "") or None,
                    client_oid=str(details.get("client_id") or "") or None,
                    side=_execution_side("sell" if current_side == "long" else "buy", details),
                    qty=_execution_qty(float(tp1_partial_qty), details),
                    price=_execution_price(details),
                    reduce_only=True,
                    status=_mode(res) or "fill",
                    reject_reason=None,
                    payload_json={"action": action, "result": details, "event_name": event_name},
                )

                pos_after_tp1 = abs(float(broker.get_position(symbol)))
                term_size_rem = _coerce_float(terminal.get("size_rem"))
                if term_size_rem is not None:
                    state.tp2_size_rem = max(0.0, min(1.0, float(term_size_rem)))
                state.tp2_tp1_done = True
                state.tp2_tp1_pending = False
                state.tp2_remaining_qty_abs = pos_after_tp1
                state.tp2_last_consumed_tp1_hit_ts = (
                    state.tp2_tp1_hit_ts
                    or (str(terminal.get("tp1_hit_ts")) if terminal.get("tp1_hit_ts") is not None else None)
                )

                fresh_equity = _resolve_equity(broker)
                _append_equity_snapshot(
                    ts_iso=_now_iso(),
                    equity=fresh_equity,
                    position_qty=pos_after_tp1,
                    position_side=(1 if current_side == "long" else -1),
                    payload={
                        "equity_usd": float(fresh_equity) if fresh_equity is not None else None,
                        "symbol": symbol,
                        "position": pos_after_tp1,
                        "side": current_side,
                        "gate_on": gate_on,
                        "source": "post_tp1_partial_fill",
                        "event_name": event_name,
                    },
                    force=True,
                )

        elif action.startswith("enter_") and want_side is not None:
            res = oms.enter_market(symbol=symbol, side=want_side, qty=float(qty))
            log.info("executor enter result=%s", res)
            if _ok(res):
                if _pending_follow_entry_is_active(state):
                    log.info(
                        "executor clearing pending follow-entry after normal entry success symbol=%s pending_side=%s entered_side=%s",
                        symbol,
                        state.pending_follow_entry_side,
                        want_side,
                    )
                    _clear_pending_follow_entry(state)
                state.open_leg_mode = str(exit_engine)
                state.open_leg_id = str(terminal.get("leg_id") or "") or None
                state.open_leg_side = str(want_side)
                entry_bar_ts = terminal.get("entry_bar_ts")
                state.open_leg_entry_bar_ts = str(entry_bar_ts) if entry_bar_ts is not None else None
                details = _details(res)
                state.n_executions += 1
                _append_execution_event(
                    strategy="live_executor_2",
                    symbol=symbol,
                    ts_iso=_now_iso(),
                    seq=int(state.n_executions),
                    execution_kind="fill",
                    order_action=_execution_side("buy" if want_side == "long" else "sell", details),
                    reason_code=event_name,
                    position_before=position_before,
                    position_after=position_after,
                    order_id=str(details.get("order_id") or "") or None,
                    client_oid=str(details.get("client_id") or "") or None,
                    side=_execution_side("buy" if want_side == "long" else "sell", details),
                    qty=_execution_qty(float(qty), details),
                    price=_execution_price(details),
                    reduce_only=False,
                    status=_mode(res) or "fill",
                    reject_reason=None,
                    payload_json={"action": action, "result": details, "event_name": event_name},
                )
                
                fresh_equity = _resolve_equity(broker)
                _append_equity_snapshot(
                    ts_iso=_now_iso(),
                    equity=fresh_equity,
                    position_qty=abs(float(qty)),
                    position_side=(1 if want_side == "long" else -1),
                    payload={
                        "equity_usd": float(fresh_equity) if fresh_equity is not None else None,
                        "symbol": symbol,
                        "position": float(qty),
                        "side": want_side,
                        "gate_on": gate_on,
                        "source": "post_entry_fill",
                        "event_name": event_name,
                    },
                    force=True,
                )

                if want_side == "long" and imba_short_barrier is not None:
                    _retag_stop_order(
                        old_kind="flat_entry_short",
                        new_kind="opposite_imba_short",
                        side="short",
                        qty=float(qty),
                        stop_price=float(imba_short_barrier),
                    )
                elif want_side == "short" and imba_long_barrier is not None:
                    _retag_stop_order(
                        old_kind="flat_entry_long",
                        new_kind="opposite_imba_long",
                        side="long",
                        qty=float(qty),
                        stop_price=float(imba_long_barrier),
                    )

        elif action.startswith("flip_to_") and want_side is not None:
            flat_res, pos_after_flat = _kraken_strict_flatten_for_flip(
                broker=broker,
                symbol=symbol,
                current_side=current_side,
                qty=abs(float(pos)),
            )
            log.info("executor flip flatten result=%s", flat_res)

            if _ok(flat_res):
                flat_details = _details(flat_res)
                state.n_executions += 1
                _append_execution_event(
                    strategy="live_executor_2",
                    symbol=symbol,
                    ts_iso=_now_iso(),
                    seq=int(state.n_executions),
                    execution_kind="fill",
                    order_action=_execution_side("sell" if current_side == "long" else "buy", flat_details),
                    reason_code=event_name,
                    position_before=position_before,
                    position_after=0,
                    order_id=str(flat_details.get("order_id") or "") or None,
                    client_oid=str(flat_details.get("client_id") or "") or None,
                    side=_execution_side("sell" if current_side == "long" else "buy", flat_details),
                    qty=_execution_qty(abs(float(pos)), flat_details),
                    price=_execution_price(flat_details),
                    reduce_only=True,
                    status=_mode(flat_res) or "fill",
                    reject_reason=None,
                    payload_json={"action": "flip_flatten", "result": flat_details, "event_name": event_name},
                )
                
                closed_trade_event = "signal_flip_exit" if action.startswith("flip_to_") else event_name

                _append_closed_trade(
                    symbol=symbol,
                    current_side=current_side,
                    terminal=terminal,
                    details=flat_details,
                    event_name=closed_trade_event,
                    action="flip_flatten",
                    position_before=position_before,
                    position_after=0,
                    seq=int(state.n_executions),
                    qty_default=abs(float(pos)),
                    exit_px_fallback=float(mid) if mid and mid > 0 else None,
                )

                if abs(pos_after_flat) > 1e-12:
                    log.warning("executor flip aborted: not flat after flatten pos_after=%s", pos_after_flat)
                    _clear_pending_follow_entry(state)
                else:
                    state.latched_exit_engine = desired_exit_engine
                    fresh_bid, fresh_ask = broker.get_best_bid_ask(symbol)
                    fresh_mid = (fresh_bid + fresh_ask) / 2.0 if (fresh_bid and fresh_ask) else (fresh_ask or fresh_bid or mid or 0.0)
                    fresh_equity = _resolve_equity(broker)
                    _append_equity_snapshot(
                        ts_iso=_now_iso(),
                        equity=fresh_equity,
                        position_qty=abs(float(pos_after_flat)),
                        position_side=(1 if float(pos_after_flat) > 0 else -1 if float(pos_after_flat) < 0 else 0),
                        payload={
                            "equity_usd": float(fresh_equity) if fresh_equity is not None else None,
                            "symbol": symbol,
                            "position": float(pos_after_flat),
                            "side": "flat" if abs(float(pos_after_flat)) <= 1e-12 else current_side,
                            "gate_on": gate_on,
                            "source": "post_flip_flatten",
                        },
                        force=True,
                    )

                    flip_qty = _qty_from_equity_pct(
                        equity=fresh_equity,
                        pos_pct=pos_pct,
                        leverage=leverage,
                        mid_price=float(fresh_mid),
                        contract_multiplier=contract_multiplier,
                    )
                    if flip_qty <= 0:
                        log.warning(
                            "executor flip aborted: qty=0 after flatten equity=%s pos_pct=%s leverage=%s mid=%s contract_multiplier=%s",
                            fresh_equity,
                            pos_pct,
                            leverage,
                            fresh_mid,
                            contract_multiplier,
                        )
                        _clear_pending_follow_entry(state)
                    else:
                        follow_source_ts = (
                            sig_now["ts"].isoformat() if sig_now is not None else (
                                pd.Timestamp(ev["ts"]).isoformat() if ev is not None and "ts" in ev else ts_iso
                            )
                        )
                        log.info(
                            "executor armed pending follow-entry symbol=%s side=%s qty=%s event=%s source_ts=%s",
                            symbol,
                            want_side,
                            flip_qty,
                            event_name,
                            follow_source_ts,
                        )
                        target_qty_for_verify = float(flip_qty)
                        _arm_pending_follow_entry(
                            state,
                            target_side=str(want_side),
                            reason=str(action),
                            source_ts=follow_source_ts,
                        )
            else:
                log.warning("executor flip aborted: flatten failed")

        elif action.startswith("exit_"):
            try:
                broker.client.cancel_all_reduce_only_orders(symbol=broker._symbol(symbol))
            except Exception as e:
                log.warning("executor exit cancel reduce-only failed: %s", e)

            res = oms.flatten_market(symbol=symbol, side=current_side, qty=abs(float(pos)))

            log.info("executor exit result=%s", res)
            target_side_for_verify = None
            target_qty_for_verify = abs(float(pos))
            if _ok(res):
                details = _details(res)
                state.n_executions += 1
                _append_execution_event(
                    strategy="live_executor_2",
                    symbol=symbol,
                    ts_iso=_now_iso(),
                    seq=int(state.n_executions),
                    execution_kind="fill",
                    order_action=_execution_side("sell" if current_side == "long" else "buy", details),
                    reason_code=event_name,
                    position_before=position_before,
                    position_after=0,
                    order_id=str(details.get("order_id") or "") or None,
                    client_oid=str(details.get("client_id") or "") or None,
                    side=_execution_side("sell" if current_side == "long" else "buy", details),
                    qty=_execution_qty(abs(float(pos)), details),
                    price=_execution_price(details),
                    reduce_only=True,
                    status=_mode(res) or "fill",
                    reject_reason=None,
                    payload_json={"action": action, "result": details, "event_name": event_name},
                )

                _append_closed_trade(
                    symbol=symbol,
                    current_side=current_side,
                    terminal=terminal,
                    details=details,
                    event_name=event_name,
                    action=action,
                    position_before=position_before,
                    position_after=0,
                    seq=int(state.n_executions),
                    qty_default=abs(float(pos)),
                    exit_px_fallback=float(mid) if mid and mid > 0 else None,
                )

                state.open_leg_mode = None
                state.open_leg_id = None
                state.open_leg_side = None
                state.open_leg_entry_bar_ts = None

                if exit_engine == "tp2" and event_name in ("tp2_exit", "be_exit", "sl_exit"):
                    state.flat_until_new_signal_ts = (
                        pd.Timestamp(ev["ts"]).isoformat()
                        if ev is not None and "ts" in ev
                        else _now_iso()
                    )
                    state.flat_latch_reason = event_name
                    _clear_tp2_leg_runtime(state)

                fresh_equity = _resolve_equity(broker)
                _append_equity_snapshot(
                    ts_iso=_now_iso(),
                    equity=fresh_equity,
                    position_qty=0.0,
                    position_side=0,
                    payload={
                        "equity_usd": float(fresh_equity) if fresh_equity is not None else None,
                        "symbol": symbol,
                        "position": 0.0,
                        "side": "flat",
                        "gate_on": gate_on,
                        "source": "post_exit_fill",
                        "event_name": event_name,
                    },
                    force=True,
                )

        elif action in ("ttp_confirm_reenter_short", "ttp_confirm_reenter_long"):
            prior_side = str(ttp_prior_side or current_side or "").strip().lower()
            reenter_side = str(ttp_target_side or ("short" if action.endswith("short") else "long")).strip().lower()
            ttp_px = _coerce_float(terminal.get("ttp"))
            confirm_checks = max(1, int(os.getenv("LIVE_EXECUTOR_2_TTP_CONFIRM_CHECKS", "3")))
            confirm_sleep = max(0.05, float(os.getenv("LIVE_EXECUTOR_2_TTP_CONFIRM_SLEEP_SEC", "0.20")))
            if ttp_handoff_key is not None:
                _mark_ttp_reenter_attempt(state, ttp_handoff_key)

            pos_after_ttp = float(broker.get_position(symbol))
            checks_used = 1
            while abs(pos_after_ttp) > 1e-12 and checks_used < confirm_checks:
                time.sleep(confirm_sleep)
                pos_after_ttp = float(broker.get_position(symbol))
                checks_used += 1

            if abs(pos_after_ttp) > 1e-12:
                log.error(
                    "executor ttp confirm failed symbol=%s current_side=%s reenter_side=%s ttp=%s pos_after=%s checks=%s",
                    symbol,
                    prior_side,
                    reenter_side,
                    ttp_px,
                    pos_after_ttp,
                    checks_used,
                )
                target_side_for_verify = None
                target_qty_for_verify = 0.0
            else:
                if not bool(state.ttp_reenter_exit_recorded):
                    ttp_exit_qty = abs(float(pos)) if abs(float(pos)) > 1e-12 else max(0.0, float(qty_effective))
                    exit_details = {
                        "order_id": f"ttp-exit:{symbol}:{int(time.time() * 1000)}",
                        "price": ttp_px,
                        "qty": ttp_exit_qty,
                    }
                    state.n_executions += 1
                    _record_ttp_external_exit(
                        state,
                        symbol=symbol,
                        prior_side=prior_side,
                        terminal=terminal,
                        ttp_px=ttp_px,
                        event_name="ttp_external_exit",
                        action=action,
                        execution_seq=int(state.n_executions),
                        qty=ttp_exit_qty,
                        exit_details=exit_details,
                    )

                fresh_bid, fresh_ask = broker.get_best_bid_ask(symbol)
                fresh_mid = (fresh_bid + fresh_ask) / 2.0 if (fresh_bid and fresh_ask) else (fresh_ask or fresh_bid or mid or 0.0)
                fresh_equity = _resolve_equity(broker)
                reenter_qty = _qty_from_equity_pct(
                    equity=fresh_equity,
                    pos_pct=pos_pct,
                    leverage=leverage,
                    mid_price=float(fresh_mid),
                    contract_multiplier=contract_multiplier,
                )

                if reenter_qty <= 0:
                    log.error(
                        "executor ttp reenter aborted qty=0 symbol=%s side=%s equity=%s mid=%s",
                        symbol,
                        reenter_side,
                        fresh_equity,
                        fresh_mid,
                    )
                    target_side_for_verify = None
                    target_qty_for_verify = 0.0
                else:
                    res = oms.enter_market(symbol=symbol, side=reenter_side, qty=float(reenter_qty))
                    log.info(
                        "executor ttp reenter result=%s side=%s qty=%s checks=%s",
                        res,
                        reenter_side,
                        reenter_qty,
                        checks_used,
                    )
                    target_side_for_verify = reenter_side
                    target_qty_for_verify = float(reenter_qty)
                    if _ok(res):
                        details = _details(res)
                        _clear_ttp_reenter_handoff(state)
                        state.open_leg_mode = str(exit_engine)
                        state.open_leg_id = _new_ttp_reenter_leg_id(reenter_side, source_ts=ttp_source_ts)
                        state.open_leg_side = str(reenter_side)
                        state.open_leg_entry_bar_ts = _now_iso()

                        state.n_executions += 1
                        _append_execution_event(
                            strategy="live_executor_2",
                            symbol=symbol,
                            ts_iso=_now_iso(),
                            seq=int(state.n_executions),
                            execution_kind="fill",
                            order_action=_execution_side("buy" if reenter_side == "long" else "sell", details),
                            reason_code="ttp_reenter",
                            position_before=(1 if prior_side == "long" else -1 if prior_side == "short" else 0),
                            position_after=(1 if reenter_side == "long" else -1),
                            order_id=str(details.get("order_id") or "") or None,
                            client_oid=str(details.get("client_id") or "") or None,
                            side=_execution_side("buy" if reenter_side == "long" else "sell", details),
                            qty=_execution_qty(float(reenter_qty), details),
                            price=_execution_price(details),
                            reduce_only=False,
                            status=_mode(res) or "fill",
                            reject_reason=None,
                            payload_json={"action": action, "result": details, "event_name": event_name},
                        )

                        fresh_equity = _resolve_equity(broker)
                        _append_equity_snapshot(
                            ts_iso=_now_iso(),
                            equity=fresh_equity,
                            position_qty=float(reenter_qty),
                            position_side=(1 if reenter_side == "long" else -1),
                            payload={
                                "equity_usd": float(fresh_equity) if fresh_equity is not None else None,
                                "symbol": symbol,
                                "position": float(reenter_qty),
                                "side": reenter_side,
                                "gate_on": gate_on,
                                "source": "post_ttp_reenter_fill",
                                "event_name": event_name,
                            },
                            force=True,
                        )

        elif action.startswith("scale_") and want_side is not None:
            add_qty = max(0.0, float(qty_effective) - abs(float(pos)))
            if add_qty <= _scale_delta_epsilon():
                add_qty = 0.0
            if add_qty > 0:
                target_qty_for_verify = abs(float(pos)) + float(add_qty)
                res = oms.enter_market(symbol=symbol, side=want_side, qty=add_qty)
                log.info("executor scale result=%s add_qty=%s target_qty=%s pos_before=%s", res, add_qty, qty_effective, pos)
                if _ok(res):
                    details = _details(res)
                    state.n_executions += 1
                    _append_execution_event(
                        strategy="live_executor_2",
                        symbol=symbol,
                        ts_iso=_now_iso(),
                        seq=int(state.n_executions),
                        execution_kind="fill",
                        order_action=_execution_side("buy" if want_side == "long" else "sell", details),
                        reason_code=event_name,
                        position_before=position_before,
                        position_after=position_after,
                        order_id=str(details.get("order_id") or "") or None,
                        client_oid=str(details.get("client_id") or "") or None,
                        side=_execution_side("buy" if want_side == "long" else "sell", details),
                        qty=_execution_qty(float(add_qty), details),
                        price=_execution_price(details),
                        reduce_only=False,
                        status=_mode(res) or "fill",
                        reject_reason=None,
                        payload_json={"action": action, "result": details, "event_name": event_name},
                    )
                    fresh_equity = _resolve_equity(broker)
                    new_pos_qty = abs(float(pos)) + float(add_qty)
                    _append_equity_snapshot(
                        ts_iso=_now_iso(),
                        equity=fresh_equity,
                        position_qty=new_pos_qty,
                        position_side=(1 if want_side == "long" else -1),
                        payload={
                            "equity_usd": float(fresh_equity) if fresh_equity is not None else None,
                            "symbol": symbol,
                            "position": new_pos_qty,
                            "side": want_side,
                            "gate_on": gate_on,
                            "source": "post_scale_fill",
                            "event_name": event_name,
                        },
                        force=True,
                    )
            else:
                log.info("executor scale skipped add_qty=0 target_qty=%s pos_before=%s", qty_effective, pos)

        else:
            log.info("executor hold symbol=%s pos=%s terminal_pos=%s event=%s", symbol, pos, terminal_pos, event_name)
            target_side_for_verify = None

        _verify_execution_fill_ratio(
            broker=broker,
            symbol=symbol,
            action=action,
            target_side=target_side_for_verify,
            target_qty=float(target_qty_for_verify),
            min_ratio=float(os.getenv("LIVE_EXECUTOR_MIN_FILL_RATIO", "0.95")),
        )

        log.info(
            "pre-sl-sync symbol=%s action=%s terminal_pos=%s terminal=%s dry_run=%s",
            symbol,
            action,
            terminal_pos,
            terminal,
            dry_run,
        )

        if not native_stop_synced:
            _sync_kraken_stop_loss(
                broker=broker,
                symbol=symbol,
                terminal=terminal,
                terminal_pos=terminal_pos,
                dry_run=dry_run,
            )

    if fallback_used and fallback_sig_ts_iso is not None and fallback_sig_v is not None:
        state.last_signal_ts = fallback_sig_ts_iso
        state.last_signal_value = int(fallback_sig_v)
    elif ev is not None and "ts" in ev:
        ev_ts = pd.Timestamp(ev["ts"]).isoformat()
        state.last_signal_ts = ev_ts
        state.last_signal_value = int(terminal_pos) if terminal_pos != 0 else 0

    if ev is not None:
        state.last_event_sig = _event_sig(ev)
    state.last_terminal_sig = terminal_sig
    state.last_action = action
    state.last_gate_on = int(gate_on)
    return state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live execution worker 2 (signals -> OMS -> Kraken)")
    p.add_argument("--symbol", default=os.getenv("LIVE_SYMBOL", "SOL-USDT"))
    p.add_argument("--signals-dir", default=os.getenv("SIGNALS_DIR", "data/signals"))
    p.add_argument("--state-file", default=os.getenv("LIVE_EXECUTOR_2_STATE", os.getenv("LIVE_EXECUTOR_STATE", "data/live/live_executor_2_state.json")))
    p.add_argument("--poll-sec", type=float, default=float(os.getenv("LIVE_EXECUTOR_2_POLL_SEC", os.getenv("LIVE_EXECUTOR_POLL_SEC", "5"))))
    p.add_argument("--once", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    symbol = str(args.symbol).upper()
    signals_root = Path(args.signals_dir)
    state_path = Path(args.state_file)

    live_enabled = _truthy(os.getenv("LIVE_TRADING_ENABLED", "0"))
    dry_run = _truthy(os.getenv("LIVE_EXECUTOR_2_DRY_RUN", os.getenv("LIVE_EXECUTOR_DRY_RUN", "1")))
    leverage = float(os.getenv("LIVE_EXECUTOR_2_LEVERAGE", os.getenv("KRAKEN_LEVERAGE", os.getenv("LIVE_EXECUTOR_LEVERAGE", "1"))))

    allowlist_raw = os.getenv("LIVE_EXECUTOR_2_SYMBOL_ALLOWLIST", os.getenv("LIVE_EXECUTOR_SYMBOL_ALLOWLIST", "SOL-USDT"))
    allowlist = {s.strip().upper() for s in allowlist_raw.split(",") if s.strip()}
    if symbol not in allowlist:
        raise RuntimeError(f"symbol '{symbol}' not allowed. Set LIVE_EXECUTOR_2_SYMBOL_ALLOWLIST.")

    client = KrakenFuturesClient()
    broker = KrakenOmsBroker(client=client)
    oms = MakerFirstOMS(broker=broker, cfg=OmsDefaults())
    st = _read_state(state_path)

    log.info(
        "executor_2 start symbol=%s live_enabled=%s dry_run=%s leverage=%s pos_pct=%s signals=%s",
        symbol,
        live_enabled,
        dry_run,
        leverage,
        os.getenv("LIVE_EXECUTOR_2_POS_PCT", os.getenv("KRAKEN_EQUITY_PCT", os.getenv("LIVE_EXECUTOR_POS_PCT", "0.90"))),
        str(signals_root),
    )

    while True:
        try:
            st = run_once(
                broker=broker,
                oms=oms,
                symbol=symbol,
                signals_root=signals_root,
                state=st,
                live_enabled=live_enabled,
                dry_run=dry_run,
                leverage=leverage,
            )
            _write_state(state_path, st)
        except Exception as e:
            log_throttled(
                log,
                logging.WARNING,
                f"executor_loop_error:{symbol}",
                float(os.getenv("LIVE_EXECUTOR_LOG_THROTTLE_SEC", "30")),
                "executor loop error: %s",
                e,
            )
            _write_state(state_path, st)

        if args.once:
            break
        time.sleep(max(1.0, float(args.poll_sec)))


if __name__ == "__main__":
    main()