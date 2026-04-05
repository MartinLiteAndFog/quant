# src/quant/execution/tv_signal_executor.py
"""
TradingView webhook -> KuCoin direct execution (hot-path).

Background thread keeps a pre-computed cache of position, equity, sizing,
and gate state. When a webhook arrives the only broker call is place_market().
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from quant.execution.CHOPgate import get_live_gate_state
from quant.execution.event_builders import build_action_event, build_execution_event
from quant.execution.event_log import append_event_jsonl
from quant.execution.event_store import insert_action_event, insert_execution_event
from quant.execution.kucoin_futures import KucoinFuturesBroker
from quant.execution.live_executor import (
    _qty_from_equity_pct,
    _resolve_contract_multiplier,
    _resolve_equity,
)
from quant.utils.log import get_logger, log_throttled

import logging

log = get_logger("quant.tv_executor")

VALID_ACTIONS = {"entry", "exit", "flip", "tp1", "tp2", "sl"}
VALID_SIDES = {"buy", "sell"}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class TVExecConfig:
    symbol: str
    pos_pct: float
    leverage: float
    tp1_close_pct: float
    dry_run: bool
    gate_mode: str
    cache_sec: float
    cache_max_age_sec: float
    emergency_sl_pct: float

    @classmethod
    def from_env(cls) -> TVExecConfig:
        return cls(
            symbol=os.getenv("LIVE_SYMBOL", "SOL-USDT"),
            pos_pct=float(os.getenv("TV_EXEC_POS_PCT", "0.50")),
            leverage=float(os.getenv("TV_EXEC_LEVERAGE", "10.0")),
            tp1_close_pct=float(os.getenv("TV_EXEC_TP1_PCT", "0.50")),
            dry_run=_truthy(os.getenv("TV_EXEC_DRY_RUN", "1")),
            gate_mode=os.getenv("TV_EXEC_GATE_MODE", "countertrend").strip().lower(),
            cache_sec=float(os.getenv("TV_EXEC_CACHE_SEC", "10")),
            cache_max_age_sec=float(os.getenv("TV_EXEC_CACHE_MAX_AGE_SEC", "60")),
            emergency_sl_pct=float(os.getenv("TV_EXEC_EMERGENCY_SL_PCT", "0.023")),
        )


# ---------------------------------------------------------------------------
# Signal parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TVSignal:
    action: str   # entry | exit | flip | tp1 | tp2 | sl
    side: str     # buy | sell | "" (not needed for exit/tp/sl)
    symbol: str


def parse_tv_signal(payload: Dict[str, Any], default_symbol: str = "") -> TVSignal:
    action = str(payload.get("action", "")).strip().lower()
    if action not in VALID_ACTIONS:
        raise ValueError(f"invalid action '{action}', must be one of {VALID_ACTIONS}")

    side = str(payload.get("side", "")).strip().lower()
    if action in ("entry", "flip") and side not in VALID_SIDES:
        raise ValueError(f"action '{action}' requires side (buy|sell), got '{side}'")

    sym = ""
    for k in ("symbol", "ticker", "pair"):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            sym = v.strip()
            break
    if not sym:
        sym = default_symbol or "SOL-USDT"

    return TVSignal(action=action, side=side, symbol=sym)


def _close_side_for_position(position_side: str) -> str:
    return "sell" if position_side == "long" else "buy"


# ---------------------------------------------------------------------------
# Pre-computed cache
# ---------------------------------------------------------------------------

@dataclass
class TVCache:
    position: float
    current_side: str          # "long" | "short" | "flat"
    equity: float
    mid_price: float
    bid: float
    ask: float
    contract_multiplier: float
    qty: int
    gate_on: int
    gate_allows_entry: bool
    gate_source: str
    updated_at: float


_cache: Optional[TVCache] = None
_cache_lock = threading.Lock()
_exec_locks: Dict[str, threading.Lock] = {}
_exec_locks_guard = threading.Lock()
_broker: Optional[KucoinFuturesBroker] = None
_config: Optional[TVExecConfig] = None
_seq_lock = threading.Lock()
_seq_counter: int = 0
_ready = threading.Event()


def _next_seq() -> int:
    global _seq_counter
    with _seq_lock:
        _seq_counter += 1
        return _seq_counter


def _exec_lock_for(symbol: str) -> threading.Lock:
    with _exec_locks_guard:
        if symbol not in _exec_locks:
            _exec_locks[symbol] = threading.Lock()
        return _exec_locks[symbol]


def _build_cache(broker: KucoinFuturesBroker, config: TVExecConfig) -> TVCache:
    pos = float(broker.get_position(config.symbol))
    current_side = "long" if pos > 0 else ("short" if pos < 0 else "flat")

    equity = _resolve_equity(broker)
    bid, ask = broker.get_best_bid_ask(config.symbol)
    mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else (ask or bid or 0.0)
    mult = _resolve_contract_multiplier(broker, config.symbol)

    qty = _qty_from_equity_pct(
        equity=equity,
        pos_pct=config.pos_pct,
        leverage=config.leverage,
        mid_price=mid,
        contract_multiplier=mult,
    )

    gate = get_live_gate_state()
    gate_on = int(gate.get("gate_on", 0) or 0)
    gate_allows = True
    if config.gate_mode == "countertrend":
        gate_allows = True
    elif config.gate_mode == "block_all" and gate_on == 1:
        gate_allows = False

    return TVCache(
        position=pos,
        current_side=current_side,
        equity=equity,
        mid_price=mid,
        bid=bid,
        ask=ask,
        contract_multiplier=mult,
        qty=int(qty),
        gate_on=gate_on,
        gate_allows_entry=gate_allows,
        gate_source=str(gate.get("source", "")),
        updated_at=time.time(),
    )


def _tv_cache_refresh_loop(broker: KucoinFuturesBroker, config: TVExecConfig) -> None:
    global _cache
    while True:
        try:
            new_cache = _build_cache(broker, config)
            with _cache_lock:
                _cache = new_cache
            if not _ready.is_set():
                _ready.set()
                log.info(
                    "tv_executor cache ready: pos=%.1f side=%s equity=%.2f qty=%d gate_on=%d",
                    new_cache.position, new_cache.current_side, new_cache.equity,
                    new_cache.qty, new_cache.gate_on,
                )
        except Exception as e:
            log_throttled(
                log, logging.WARNING,
                "tv_cache_refresh_fail",
                30.0,
                "tv_executor cache refresh failed: %s", e,
            )
        time.sleep(max(2.0, config.cache_sec))


def _get_cache(max_age: float) -> TVCache:
    with _cache_lock:
        c = _cache
    if c is None:
        raise RuntimeError("tv_executor cache not ready")
    age = time.time() - c.updated_at
    if age > max_age:
        raise RuntimeError(f"tv_executor cache stale ({age:.0f}s > {max_age:.0f}s)")
    return c


def _refresh_position_in_cache(broker: KucoinFuturesBroker, config: TVExecConfig) -> None:
    """Post-execution: update position+sizing in cache immediately."""
    global _cache
    try:
        pos = float(broker.get_position(config.symbol))
        current_side = "long" if pos > 0 else ("short" if pos < 0 else "flat")
        equity = _resolve_equity(broker)
        bid, ask = broker.get_best_bid_ask(config.symbol)
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else (ask or bid or 0.0)
        mult = _resolve_contract_multiplier(broker, config.symbol)
        qty = _qty_from_equity_pct(
            equity=equity, pos_pct=config.pos_pct,
            leverage=config.leverage, mid_price=mid,
            contract_multiplier=mult,
        )
        with _cache_lock:
            if _cache is not None:
                _cache = TVCache(
                    position=pos, current_side=current_side,
                    equity=equity, mid_price=mid, bid=bid, ask=ask,
                    contract_multiplier=mult, qty=int(qty),
                    gate_on=_cache.gate_on,
                    gate_allows_entry=_cache.gate_allows_entry,
                    gate_source=_cache.gate_source,
                    updated_at=time.time(),
                )
    except Exception as e:
        log.warning("tv_executor post-exec cache refresh failed: %s", e)


# ---------------------------------------------------------------------------
# Event logging (fire-and-forget)
# ---------------------------------------------------------------------------

def _events_root() -> Path:
    if Path("/data").exists():
        return Path("/data/events")
    return Path("data/events")


def _now_iso() -> str:
    return pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _log_action(
    *,
    symbol: str,
    seq: int,
    action: str,
    action_side: str,
    reason: str,
    pos_before: int,
    pos_after: int,
    blocked: bool = False,
    block_reason: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    ts = _now_iso()
    event = build_action_event(
        strategy="tv_executor",
        symbol=symbol.replace("-", ""),
        ts=ts,
        seq=seq,
        engine_action=action,
        action_side=action_side,
        reason_code=reason,
        venue="kucoin",
        position_before=pos_before,
        position_after=pos_after,
        blocked=blocked,
        block_reason=block_reason,
    )
    event["strategy_instance"] = "tv_executor"
    event["config_hash"] = "tv_executor_v1"
    if payload:
        event["payload_json"] = payload

    day = pd.Timestamp.now("UTC").strftime("%Y%m%d")
    out_path = _events_root() / "action_events" / f"{day}.jsonl"
    try:
        append_event_jsonl(out_path, event)
    except Exception:
        pass
    try:
        insert_action_event({
            "event_id": event["event_id"],
            "ts": event["ts"],
            "seq": event["seq"],
            "strategy": event["strategy"],
            "strategy_instance": "tv_executor",
            "config_hash": "tv_executor_v1",
            "symbol": event["symbol"],
            "venue": event["venue"],
            "source_signal_event_id": None,
            "source_event_id": None,
            "engine_action": event["engine_action"],
            "action_side": event.get("action_side"),
            "position_before": event.get("position_before"),
            "position_after": event.get("position_after"),
            "engine_mode_before": event.get("engine_mode_before"),
            "engine_mode_after": event.get("engine_mode_after"),
            "reason_code": event["reason_code"],
            "blocked": bool(event.get("blocked", False)),
            "block_reason": event.get("block_reason"),
            "payload_json": dict(event),
        })
    except Exception as e:
        log.warning("tv_executor postgres action event failed: %s", e)


def _log_execution(
    *,
    symbol: str,
    seq: int,
    kind: str,
    order_action: str,
    reason: str,
    pos_before: int,
    pos_after: int,
    order_id: Optional[str] = None,
    client_oid: Optional[str] = None,
    side: Optional[str] = None,
    qty: Optional[float] = None,
    reduce_only: Optional[bool] = None,
    status: Optional[str] = None,
) -> None:
    ts = _now_iso()
    event = build_execution_event(
        strategy="tv_executor",
        symbol=symbol.replace("-", ""),
        ts=ts,
        seq=seq,
        execution_kind=kind,
        order_action=order_action,
        reason_code=reason,
        venue="kucoin",
        position_before=pos_before,
        position_after=pos_after,
        client_oid=client_oid,
        order_id=order_id,
        side=side,
        qty=qty,
        reduce_only=reduce_only,
        status=status,
        strategy_instance="tv_executor",
        config_hash="tv_executor_v1",
    )
    day = pd.Timestamp.now("UTC").strftime("%Y%m%d")
    out_path = _events_root() / "execution_events" / f"{day}.jsonl"
    try:
        append_event_jsonl(out_path, event)
    except Exception:
        pass
    try:
        insert_execution_event({
            "event_id": event["event_id"],
            "ts": event["ts"],
            "seq": event["seq"],
            "symbol": event["symbol"],
            "venue": event["venue"],
            "source_action_event_id": None,
            "execution_stage": kind,
            "order_id": order_id,
            "client_oid": client_oid,
            "side": side,
            "qty": qty,
            "reduce_only": reduce_only,
            "status": status,
            "payload_json": dict(event),
        })
    except Exception as e:
        log.warning("tv_executor postgres execution event failed: %s", e)


def _log_bg(fn, **kwargs) -> None:
    """Fire-and-forget logging in background thread."""
    threading.Thread(target=fn, kwargs=kwargs, daemon=True).start()


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def _client_oid(action: str) -> str:
    return f"quant:tv:{action}:{int(time.time() * 1000)}"


def _place_market(
    broker: KucoinFuturesBroker,
    symbol: str,
    side: str,
    qty: int,
    reduce_only: bool,
    action_label: str,
) -> str:
    cid = _client_oid(action_label)
    order_id = broker.place_market(
        symbol=symbol,
        side=side,
        qty=qty,
        reduce_only=reduce_only,
        client_id=cid,
    )
    log.info(
        "tv_executor order: action=%s side=%s qty=%d reduce_only=%s order_id=%s",
        action_label, side, qty, reduce_only, order_id,
    )
    return order_id


def _cancel_emergency_sl(broker: KucoinFuturesBroker, symbol: str) -> None:
    try:
        broker.cancel_all_stop_orders(symbol)
    except Exception as e:
        log.warning("tv_executor cancel emergency SL failed: %s", e)


def _place_emergency_sl(
    broker: KucoinFuturesBroker,
    symbol: str,
    position_side: str,
    qty: int,
    mid_price: float,
    sl_pct: float,
) -> Optional[str]:
    if sl_pct <= 0 or qty <= 0 or mid_price <= 0:
        return None

    if position_side == "long":
        sl_price = mid_price * (1.0 - sl_pct)
        sl_side = "sell"
    elif position_side == "short":
        sl_price = mid_price * (1.0 + sl_pct)
        sl_side = "buy"
    else:
        return None

    cid = _client_oid("emergency_sl")
    try:
        oid = broker.place_stop_market(
            symbol=symbol,
            side=sl_side,
            qty=qty,
            stop_price=round(sl_price, 4),
            reduce_only=True,
            client_id=cid,
        )
        log.info(
            "tv_executor emergency SL placed: side=%s qty=%d sl_price=%.4f (%.1f%% from %.4f) order_id=%s",
            sl_side, qty, sl_price, sl_pct * 100, mid_price, oid,
        )
        return oid
    except Exception as e:
        log.error("tv_executor emergency SL placement failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Core dispatch
# ---------------------------------------------------------------------------

def execute_tv_signal(signal: TVSignal, config: TVExecConfig) -> Dict[str, Any]:
    """
    Hot-path: reads pre-computed cache, places market order(s), returns result.
    Called from webhook handler in a thread pool.
    """
    lock = _exec_lock_for(signal.symbol)
    with lock:
        return _execute_locked(signal, config)


def _execute_locked(signal: TVSignal, config: TVExecConfig) -> Dict[str, Any]:
    broker = _broker
    if broker is None:
        return {"ok": False, "action": signal.action, "reason": "broker_not_initialized"}

    try:
        cache = _get_cache(config.cache_max_age_sec)
    except RuntimeError as e:
        return {"ok": False, "action": signal.action, "reason": str(e)}

    pos = cache.position
    side = cache.current_side  # long | short | flat
    qty = cache.qty
    seq = _next_seq()

    pos_before_i = 1 if side == "long" else (-1 if side == "short" else 0)

    # ----- entry -----
    if signal.action == "entry":
        want_side = "long" if signal.side == "buy" else "short"

        if side == want_side:
            log.info("tv_executor skip duplicate entry: already %s", side)
            return {"ok": True, "action": "entry", "reason": "duplicate_skip"}

        if side != "flat":
            # Opposite position: treat as flip
            return _do_flip(broker, config, signal.symbol, signal.side, cache, seq)

        if not cache.gate_allows_entry:
            _log_bg(_log_action, symbol=signal.symbol, seq=seq, action="entry",
                    action_side=want_side, reason="tv_entry",
                    pos_before=pos_before_i, pos_after=0,
                    blocked=True, block_reason=f"gate_on={cache.gate_on}")
            log.info("tv_executor entry blocked by gate (gate_on=%d)", cache.gate_on)
            return {"ok": False, "action": "entry", "reason": f"gate_blocked:gate_on={cache.gate_on}"}

        if qty <= 0:
            return {"ok": False, "action": "entry", "reason": "qty_zero_check_equity"}

        if config.dry_run:
            log.warning("tv_executor DRY_RUN entry %s qty=%d", want_side, qty)
            return {"ok": True, "action": "entry", "reason": "dry_run", "qty": qty, "side": want_side}

        order_side = signal.side  # buy or sell
        oid = _place_market(broker, signal.symbol, order_side, qty, reduce_only=False, action_label="entry")
        pos_after_i = 1 if want_side == "long" else -1

        _place_emergency_sl(broker, signal.symbol, want_side, qty, cache.mid_price, config.emergency_sl_pct)
        _refresh_position_in_cache(broker, config)

        _log_bg(_log_action, symbol=signal.symbol, seq=seq, action="entry",
                action_side=want_side, reason="tv_entry",
                pos_before=pos_before_i, pos_after=pos_after_i)
        _log_bg(_log_execution, symbol=signal.symbol, seq=seq, kind="market_fill",
                order_action=order_side, reason="tv_entry",
                pos_before=pos_before_i, pos_after=pos_after_i,
                order_id=oid, side=order_side, qty=float(qty), reduce_only=False, status="sent")

        return {"ok": True, "action": "entry", "side": want_side, "qty": qty, "order_id": oid}

    # ----- exit -----
    if signal.action == "exit":
        if side == "flat":
            return {"ok": True, "action": "exit", "reason": "already_flat"}

        close_qty = abs(int(pos))
        if close_qty <= 0:
            return {"ok": True, "action": "exit", "reason": "already_flat"}

        if config.dry_run:
            log.warning("tv_executor DRY_RUN exit %s qty=%d", side, close_qty)
            return {"ok": True, "action": "exit", "reason": "dry_run", "qty": close_qty}

        close_side = _close_side_for_position(side)
        oid = _place_market(broker, signal.symbol, close_side, close_qty, reduce_only=True, action_label="exit")

        _refresh_position_in_cache(broker, config)

        _log_bg(_log_action, symbol=signal.symbol, seq=seq, action="exit",
                action_side=side, reason="tv_exit",
                pos_before=pos_before_i, pos_after=0)
        _log_bg(_log_execution, symbol=signal.symbol, seq=seq, kind="market_fill",
                order_action=close_side, reason="tv_exit",
                pos_before=pos_before_i, pos_after=0,
                order_id=oid, side=close_side, qty=float(close_qty), reduce_only=True, status="sent")

        return {"ok": True, "action": "exit", "qty": close_qty, "order_id": oid}

    # ----- flip -----
    if signal.action == "flip":
        return _do_flip(broker, config, signal.symbol, signal.side, cache, seq)

    # ----- tp1 -----
    if signal.action == "tp1":
        if side == "flat":
            return {"ok": True, "action": "tp1", "reason": "already_flat"}

        close_qty = max(1, int(abs(pos) * config.tp1_close_pct))
        if close_qty > abs(int(pos)):
            close_qty = abs(int(pos))

        if config.dry_run:
            log.warning("tv_executor DRY_RUN tp1 %s qty=%d", side, close_qty)
            return {"ok": True, "action": "tp1", "reason": "dry_run", "qty": close_qty}

        close_side = _close_side_for_position(side)
        oid = _place_market(broker, signal.symbol, close_side, close_qty, reduce_only=True, action_label="tp1")

        _refresh_position_in_cache(broker, config)

        _log_bg(_log_action, symbol=signal.symbol, seq=seq, action="tp1_partial",
                action_side=side, reason="tv_tp1",
                pos_before=pos_before_i, pos_after=pos_before_i)
        _log_bg(_log_execution, symbol=signal.symbol, seq=seq, kind="market_fill",
                order_action=close_side, reason="tv_tp1",
                pos_before=pos_before_i, pos_after=pos_before_i,
                order_id=oid, side=close_side, qty=float(close_qty), reduce_only=True, status="sent")

        return {"ok": True, "action": "tp1", "qty": close_qty, "order_id": oid}

    # ----- tp2 -----
    if signal.action == "tp2":
        if side == "flat":
            return {"ok": True, "action": "tp2", "reason": "already_flat"}

        close_qty = abs(int(pos))
        if close_qty <= 0:
            return {"ok": True, "action": "tp2", "reason": "already_flat"}

        if config.dry_run:
            log.warning("tv_executor DRY_RUN tp2 %s qty=%d", side, close_qty)
            return {"ok": True, "action": "tp2", "reason": "dry_run", "qty": close_qty}

        close_side = _close_side_for_position(side)
        oid = _place_market(broker, signal.symbol, close_side, close_qty, reduce_only=True, action_label="tp2")

        _refresh_position_in_cache(broker, config)

        _log_bg(_log_action, symbol=signal.symbol, seq=seq, action="tp2_close",
                action_side=side, reason="tv_tp2",
                pos_before=pos_before_i, pos_after=0)
        _log_bg(_log_execution, symbol=signal.symbol, seq=seq, kind="market_fill",
                order_action=close_side, reason="tv_tp2",
                pos_before=pos_before_i, pos_after=0,
                order_id=oid, side=close_side, qty=float(close_qty), reduce_only=True, status="sent")

        return {"ok": True, "action": "tp2", "qty": close_qty, "order_id": oid}

    # ----- sl -----
    if signal.action == "sl":
        if side == "flat":
            return {"ok": True, "action": "sl", "reason": "already_flat"}

        close_qty = abs(int(pos))
        if close_qty <= 0:
            return {"ok": True, "action": "sl", "reason": "already_flat"}

        if config.dry_run:
            log.warning("tv_executor DRY_RUN sl %s qty=%d", side, close_qty)
            return {"ok": True, "action": "sl", "reason": "dry_run", "qty": close_qty}

        # Cancel all pending orders first
        try:
            broker.cancel_all(signal.symbol)
        except Exception:
            pass
        try:
            broker.cancel_all_stop_orders(signal.symbol)
        except Exception:
            pass

        close_side = _close_side_for_position(side)
        oid = _place_market(broker, signal.symbol, close_side, close_qty, reduce_only=True, action_label="sl")

        _refresh_position_in_cache(broker, config)

        _log_bg(_log_action, symbol=signal.symbol, seq=seq, action="sl_exit",
                action_side=side, reason="tv_sl",
                pos_before=pos_before_i, pos_after=0)
        _log_bg(_log_execution, symbol=signal.symbol, seq=seq, kind="market_fill",
                order_action=close_side, reason="tv_sl",
                pos_before=pos_before_i, pos_after=0,
                order_id=oid, side=close_side, qty=float(close_qty), reduce_only=True, status="sent")

        return {"ok": True, "action": "sl", "qty": close_qty, "order_id": oid}

    return {"ok": False, "action": signal.action, "reason": "unknown_action"}


def _do_flip(
    broker: KucoinFuturesBroker,
    config: TVExecConfig,
    symbol: str,
    order_side: str,
    cache: TVCache,
    seq: int,
) -> Dict[str, Any]:
    """Flip: single net-off market order (close + open opposite in one shot)."""
    want_side = "long" if order_side == "buy" else "short"
    current_side = cache.current_side
    pos = cache.position
    qty = cache.qty
    pos_before_i = 1 if current_side == "long" else (-1 if current_side == "short" else 0)

    if current_side == want_side:
        log.info("tv_executor skip duplicate flip: already %s", current_side)
        return {"ok": True, "action": "flip", "reason": "duplicate_skip"}

    if not cache.gate_allows_entry:
        # Gate blocks: only flatten, no re-entry
        if current_side == "flat":
            _log_bg(_log_action, symbol=symbol, seq=seq, action="flip",
                    action_side=want_side, reason="tv_flip",
                    pos_before=pos_before_i, pos_after=0,
                    blocked=True, block_reason=f"gate_on={cache.gate_on}")
            log.info("tv_executor flip blocked by gate (flat, gate_on=%d)", cache.gate_on)
            return {"ok": False, "action": "flip", "reason": f"gate_blocked:gate_on={cache.gate_on}"}

        # Flatten only
        if config.dry_run:
            log.warning("tv_executor DRY_RUN flip->flatten_only (gate blocks re-entry) %s qty=%d", current_side, abs(int(pos)))
            return {"ok": True, "action": "flip", "reason": "dry_run_flatten_only_gate"}

        _cancel_emergency_sl(broker, symbol)
        close_side = "sell" if current_side == "long" else "buy"
        close_qty = abs(int(pos))
        oid = _place_market(broker, symbol, close_side, close_qty, reduce_only=True, action_label="flip_flatten")

        _refresh_position_in_cache(broker, config)

        _log_bg(_log_action, symbol=symbol, seq=seq, action="flip_flatten_only",
                action_side=want_side, reason="tv_flip_gate_block",
                pos_before=pos_before_i, pos_after=0)
        _log_bg(_log_execution, symbol=symbol, seq=seq, kind="market_fill",
                order_action=close_side, reason="tv_flip_flatten",
                pos_before=pos_before_i, pos_after=0,
                order_id=oid, side=close_side, qty=float(close_qty), reduce_only=True, status="sent")

        return {"ok": True, "action": "flip", "reason": "flatten_only_gate_blocked", "qty_closed": close_qty, "order_id": oid}

    if qty <= 0:
        return {"ok": False, "action": "flip", "reason": "qty_zero_check_equity"}

    if config.dry_run:
        total = qty + abs(int(pos)) if current_side != "flat" else qty
        log.warning("tv_executor DRY_RUN flip %s->%s qty=%d", current_side, want_side, total)
        return {"ok": True, "action": "flip", "reason": "dry_run", "qty": total, "side": want_side}

    if current_side == "flat":
        oid = _place_market(broker, symbol, order_side, qty, reduce_only=False, action_label="flip_entry")
        pos_after_i = 1 if want_side == "long" else -1

        _place_emergency_sl(broker, symbol, want_side, qty, cache.mid_price, config.emergency_sl_pct)
        _refresh_position_in_cache(broker, config)

        _log_bg(_log_action, symbol=symbol, seq=seq, action="flip_entry",
                action_side=want_side, reason="tv_flip",
                pos_before=0, pos_after=pos_after_i)
        _log_bg(_log_execution, symbol=symbol, seq=seq, kind="market_fill",
                order_action=order_side, reason="tv_flip_entry",
                pos_before=0, pos_after=pos_after_i,
                order_id=oid, side=order_side, qty=float(qty), reduce_only=False, status="sent")

        return {"ok": True, "action": "flip", "side": want_side, "qty": qty, "order_id": oid}

    # Net-off flip: cancel old SL, single order, place new SL
    _cancel_emergency_sl(broker, symbol)
    total_qty = qty + abs(int(pos))
    oid = _place_market(broker, symbol, order_side, total_qty, reduce_only=False, action_label="flip")
    pos_after_i = 1 if want_side == "long" else -1

    _place_emergency_sl(broker, symbol, want_side, qty, cache.mid_price, config.emergency_sl_pct)
    _refresh_position_in_cache(broker, config)

    _log_bg(_log_action, symbol=symbol, seq=seq, action="flip",
            action_side=want_side, reason="tv_flip",
            pos_before=pos_before_i, pos_after=pos_after_i)
    _log_bg(_log_execution, symbol=symbol, seq=seq, kind="market_fill",
            order_action=order_side, reason="tv_flip",
            pos_before=pos_before_i, pos_after=pos_after_i,
            order_id=oid, side=order_side, qty=float(total_qty), reduce_only=False, status="sent")

    return {"ok": True, "action": "flip", "side": want_side, "qty": total_qty, "order_id": oid}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def start_tv_executor() -> None:
    """Call from webhook_server lifespan to start the cache refresh thread."""
    global _broker, _config

    if not _truthy(os.getenv("ENABLE_TV_EXECUTOR", "0")):
        log.info("tv_executor disabled (ENABLE_TV_EXECUTOR != 1)")
        return

    _config = TVExecConfig.from_env()
    _broker = KucoinFuturesBroker(
        api_key=os.getenv("TV_EXEC_KUCOIN_API_KEY", "") or None,
        api_secret=os.getenv("TV_EXEC_KUCOIN_API_SECRET", "") or None,
        passphrase=os.getenv("TV_EXEC_KUCOIN_PASSPHRASE", "") or None,
    )

    t = threading.Thread(
        target=_tv_cache_refresh_loop,
        args=(_broker, _config),
        daemon=True,
        name="tv-cache-refresh",
    )
    t.start()

    # Block until first cache is populated
    if not _ready.wait(timeout=30):
        log.warning("tv_executor cache did not become ready within 30s")
    else:
        log.info("tv_executor started (dry_run=%s, gate_mode=%s, cache_sec=%.0f)",
                 _config.dry_run, _config.gate_mode, _config.cache_sec)
