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

from quant.execution.bot_profiles import (
    active_profile,
    strategy_config_hash,
    strategy_instance_id,
)
from quant.execution.CHOPgate import get_live_gate_state
from quant.execution.event_builders import build_action_event, build_execution_event
from quant.execution.event_log import append_event_jsonl
from quant.execution.event_store import (
    insert_action_event,
    insert_execution_event,
    load_open_leg_from_execution_events,
    upsert_closed_trade,
)
from quant.execution.kucoin_futures import KucoinFuturesBroker
from quant.execution.live_executor import (
    _live_order_qty,
    _resolve_contract_multiplier,
    _resolve_equity,
)
from quant.utils.log import get_logger, log_throttled

import logging

log = get_logger("quant.tv_executor")

VALID_ACTIONS = {"entry", "exit", "flip", "tp1", "tp2", "sl"}
VALID_SIDES = {"buy", "sell"}

# Per-symbol open-leg snapshot so exits can write closed_trades (fleet % curves).
# TV webhook mode does not run live_executor, which previously owned this write.
_open_legs: Dict[str, Dict[str, Any]] = {}
_open_legs_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _order_leverage() -> float:
    """The leverage actually sent to KuCoin on the order — what the exchange applies."""
    return float(
        os.getenv("KUCOIN_FUTURES_ORDER_LEVERAGE", os.getenv("LIVE_EXECUTOR_LEVERAGE", "1"))
    )


def _effective_leverage() -> float:
    """Single source of truth for leverage.

    Sizing and the order must agree. They used to be independent:
    `TV_EXEC_LEVERAGE` defaulted to 10 and drove position sizing, while
    `KUCOIN_FUTURES_ORDER_LEVERAGE` drove the leverage KuCoin actually applied.
    Setting one to 10 and leaving the other at 3 sized the position for 10x but
    opened it at 3x, requiring 3.3x more margin than budgeted.

    The order leverage wins, because that is what the exchange enforces.
    """
    order_lev = _order_leverage()
    raw = os.getenv("TV_EXEC_LEVERAGE")
    if raw is None or not str(raw).strip():
        return order_lev

    sizing_lev = float(raw)
    if abs(sizing_lev - order_lev) > 1e-9:
        log.error(
            "TV_EXEC_LEVERAGE=%s disagrees with the leverage sent to KuCoin (%s). "
            "Using %s so sizing matches the exchange. Set KUCOIN_FUTURES_ORDER_LEVERAGE "
            "and LIVE_EXECUTOR_LEVERAGE to change actual leverage.",
            sizing_lev, order_lev, order_lev,
        )
        return order_lev
    return sizing_lev


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
    flip_delay_sec: float
    # A flip already closes the old position and opens the new one. When
    # TradingView also sends the matching `exit`, the two land milliseconds
    # apart and the exit flattens the position the flip just opened — a
    # round-trip that pays two lots of fees and holds nothing. Ignore an exit
    # this soon after an open; 0 disables the guard.
    exit_after_open_guard_sec: float
    # A percentage stop is blind to leverage. At 25x the liquidation price sits
    # ~3% from entry, so a 2.5% backstop leaves almost no room, and any stop
    # wider than that (a strategy's own level, say) would sit *past* liquidation
    # — the exchange closes the position first and the stop never fires. Keep
    # the stop at least this fraction of the entry->liquidation distance on the
    # safe side of liquidation. 0 disables the clamp.
    sl_liq_buffer_frac: float

    @classmethod
    def from_env(cls) -> TVExecConfig:
        return cls(
            symbol=os.getenv("LIVE_SYMBOL", "SOL-USDT"),
            pos_pct=float(os.getenv("TV_EXEC_POS_PCT", "0.50")),
            leverage=_effective_leverage(),
            tp1_close_pct=float(os.getenv("TV_EXEC_TP1_PCT", "0.50")),
            dry_run=_truthy(os.getenv("TV_EXEC_DRY_RUN", "1")),
            gate_mode=os.getenv("TV_EXEC_GATE_MODE", "countertrend").strip().lower(),
            cache_sec=float(os.getenv("TV_EXEC_CACHE_SEC", "10")),
            cache_max_age_sec=float(os.getenv("TV_EXEC_CACHE_MAX_AGE_SEC", "60")),
            emergency_sl_pct=float(os.getenv("TV_EXEC_EMERGENCY_SL_PCT", "0.025")),
            flip_delay_sec=float(os.getenv("TV_EXEC_FLIP_DELAY_SEC", "2.0")),
            exit_after_open_guard_sec=float(
                os.getenv("TV_EXEC_EXIT_AFTER_OPEN_GUARD_SEC", "5.0")
            ),
            sl_liq_buffer_frac=float(
                os.getenv("TV_EXEC_SL_LIQ_BUFFER_FRAC", "0.25")
            ),
        )


# ---------------------------------------------------------------------------
# Signal parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TVSignal:
    action: str   # entry | exit | flip | tp1 | tp2 | sl
    side: str     # buy | sell | "" (not needed for exit/tp/sl)
    symbol: str
    # Absolute stop price from the strategy, when the alert carries one. The
    # emergency stop is only a backstop for a flash crash that TradingView is
    # too slow to react to; if the strategy states where its own stop sits, that
    # is the more accurate level — and it may legitimately be wider than the
    # percentage fallback. Optional, so alerts without it keep working.
    sl_price: Optional[float] = None


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

    sl_price: Optional[float] = None
    for k in ("sl_price", "sl", "stop_price", "stop", "stoploss", "stop_loss"):
        v = payload.get(k)
        if v is None or isinstance(v, bool) or (isinstance(v, str) and not v.strip()):
            continue
        try:
            parsed = float(v)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            sl_price = parsed
            break

    return TVSignal(action=action, side=side, symbol=sym, sl_price=sl_price)


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

    qty = _live_order_qty(
        equity=equity,
        pos_pct=config.pos_pct,
        leverage=config.leverage,
        mid_price=mid,
        contract_multiplier=mult,
    )

    # Gate decoupled from quant. Only the "block_all" mode actually uses the
    # gate decision; in every other mode (countertrend / default) the result is
    # ignored, so we skip get_live_gate_state() entirely. That call reaches into
    # quant's shared Postgres daily_gate_history (+ Redis) on every refresh, and
    # the pilots are meant to simply execute the TradingView orders with no
    # quant-computed gate. Skipping it removes that coupling from the hot loop.
    gate_on = 0
    gate_allows = True
    gate_source = "disabled"
    if config.gate_mode == "block_all":
        gate = get_live_gate_state()
        gate_on = int(gate.get("gate_on", 0) or 0)
        gate_source = str(gate.get("source", ""))
        if gate_on == 1:
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
        gate_source=gate_source,
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
        qty = _live_order_qty(
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


def _record_open_leg(
    *,
    symbol: str,
    side: str,
    qty: float,
    entry_price: Optional[float],
) -> None:
    if side not in ("long", "short") or qty <= 0:
        return
    px = float(entry_price) if entry_price and entry_price > 0 else None
    with _open_legs_lock:
        _open_legs[str(symbol)] = {
            "side": side,
            "qty": float(qty),
            "entry_ts": _now_iso(),
            "entry_price": px,
            # Monotonic stamp for the exit-after-flip guard. Wall-clock is not
            # safe here: an NTP step could make a fresh position look old and
            # let the guard through.
            "opened_monotonic": time.monotonic(),
        }


def _seconds_since_open(symbol: str) -> Optional[float]:
    """Age of the currently tracked open leg, or None if there isn't one."""
    with _open_legs_lock:
        leg = _open_legs.get(str(symbol)) or {}
    opened = leg.get("opened_monotonic")
    if opened is None:
        return None
    return max(0.0, time.monotonic() - float(opened))


def _append_tv_closed_trade(
    *,
    symbol: str,
    exit_event: str,
    exit_price: Optional[float],
    qty: Optional[float] = None,
) -> None:
    """Write a closed_trades row tagged with this bot's strategy_instance."""
    with _open_legs_lock:
        leg = dict(_open_legs.get(str(symbol)) or {})
        if exit_event in ("exit", "sl", "tp2", "flip_close", "flip_flatten"):
            _open_legs.pop(str(symbol), None)

    if not leg:
        # The in-memory leg is wiped on every redeploy, so a close that lands
        # after a restart would otherwise never write closed_trades. Recover the
        # entry from the durable execution_events (Plan A). The caller only
        # reaches here with a live position, so the latest opening fill is it.
        try:
            leg = load_open_leg_from_execution_events(
                strategy_instance=strategy_instance_id(),
                symbol=str(symbol),
            ) or {}
        except Exception as e:
            log.warning("tv_executor open-leg reconstruct failed: %s", e)
            leg = {}
        if leg:
            log.info(
                "tv_executor reconstructed open leg from execution_events "
                "event=%s side=%s qty=%s entry_px=%s",
                exit_event, leg.get("side"), leg.get("qty"), leg.get("entry_price"),
            )

    if not leg:
        return
    entry_px = leg.get("entry_price")
    exit_px = float(exit_price) if exit_price and float(exit_price) > 0 else None
    if not entry_px or not exit_px:
        log.info("tv_executor skip closed_trade (missing prices) event=%s", exit_event)
        return
    side = str(leg.get("side") or "")
    qty_realized = float(qty if qty is not None else leg.get("qty") or 0.0)
    if qty_realized <= 0:
        return
    side_mult = 1.0 if side == "long" else -1.0
    pnl_pct = ((exit_px - float(entry_px)) / float(entry_px)) * 100.0 * side_mult
    try:
        upsert_closed_trade(
            {
                "trade_id": f"{symbol}:{_now_iso()}:{exit_event}:{strategy_instance_id()}",
                "venue": "kucoin",
                "symbol": symbol,
                "entry_ts": leg.get("entry_ts") or _now_iso(),
                "exit_ts": _now_iso(),
                "side": side,
                "qty": qty_realized,
                "entry_price": float(entry_px),
                "exit_price": float(exit_px),
                "pnl_pct": float(pnl_pct),
                "exit_event": exit_event,
                "strategy": "tv_executor",
                "strategy_instance": strategy_instance_id(),
                "config_hash": strategy_config_hash(),
                "source_action_event_id": None,
                "payload_json": {
                    "kind": "closed_trade",
                    "bot_profile": active_profile(),
                    "exit_event": exit_event,
                    "leg_reconstructed": bool(leg.get("reconstructed")),
                },
            }
        )
    except Exception as e:
        log.warning("tv_executor postgres closed trade failed: %s", e)


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
    # Tag with this bot's own instance so the four pilot sub-accounts stay
    # separable in Postgres for fill analysis. A shared "tv_executor" tag
    # would merge all of them into one indistinguishable stream.
    event["strategy_instance"] = strategy_instance_id()
    event["config_hash"] = strategy_config_hash()
    event["bot_profile"] = active_profile()
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
            "strategy_instance": strategy_instance_id(),
            "config_hash": strategy_config_hash(),
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
    price: Optional[float] = None,
    mid_price: Optional[float] = None,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
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
        strategy_instance=strategy_instance_id(),
        config_hash=strategy_config_hash(),
    )
    # Market context at decision time — required to compute realised slippage
    # against the eventual fill. Fall back to the live cache so every call site
    # records context without having to thread it through by hand.
    if mid_price is None or bid is None or ask is None:
        with _cache_lock:
            snap = _cache
        if snap is not None:
            mid_price = snap.mid_price if mid_price is None else mid_price
            bid = snap.bid if bid is None else bid
            ask = snap.ask if ask is None else ask

    ref_price = price if price is not None else mid_price
    for key, value in (
        ("price", ref_price), ("mid_price", mid_price), ("bid", bid), ("ask", ask),
        ("bot_profile", active_profile()),
    ):
        if value is not None:
            event[key] = value

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
            "price": ref_price,
            "reject_reason": None,
            "strategy_instance": strategy_instance_id(),
            "config_hash": strategy_config_hash(),
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


def _verify_stop_registered(
    broker: KucoinFuturesBroker,
    order_id: Optional[str],
    expected_stop_price: float,
) -> bool:
    """Confirm the exchange actually accepted the order as a resting stop.

    The failure this catches is silent and expensive: sent to the wrong
    endpoint, KuCoin drops the trigger and keeps the rest, leaving a plain
    reduce-only market order that fills at once and flattens the position it
    was meant to protect. A registered stop comes back with `stop` set to
    'down'/'up'; an empty `stop`, or a status showing it already filled, means
    there is no protection on this position.

    Best-effort and never raises — a failed check must not undo a good entry.
    """
    if not order_id:
        return False
    try:
        order = broker.get_order(str(order_id))
    except Exception as e:
        log.warning("tv_executor could not verify emergency SL %s: %s", order_id, e)
        return False
    stop = str(order.get("stop") or "").strip().lower()
    filled = str(order.get("status") or "").strip().lower() == "done"
    if stop in ("down", "up") and not filled:
        log.info(
            "tv_executor emergency SL verified resting: order_id=%s stop=%s stopPrice=%s",
            order_id, stop, order.get("stopPrice"),
        )
        return True
    log.error(
        "tv_executor EMERGENCY SL DID NOT REGISTER AS A STOP: order_id=%s stop=%r status=%r "
        "size=%s expected_stop=%.4f — the position is UNPROTECTED and this order may have "
        "filled immediately at market",
        order_id, order.get("stop"), order.get("status"), order.get("size"), expected_stop_price,
    )
    return False


def _liquidation_price(
    broker: KucoinFuturesBroker,
    symbol: str,
    attempts: int = 3,
    delay_sec: float = 0.2,
) -> Optional[float]:
    """Read the exchange's liquidation price for the open position.

    Called right after the entry fills, so the position may not be visible on
    the very first read — hence the short retry. Best-effort and never raises:
    without a liquidation price the caller keeps the plain percentage stop,
    which is what it did before this existed.
    """
    for attempt in range(max(1, attempts)):
        try:
            info = broker.get_position_info(symbol)
        except Exception as e:
            log.warning("tv_executor could not read position for liq price: %s", e)
            return None
        raw = (info or {}).get("raw") or {}
        try:
            liq = float(raw.get("liquidationPrice") or 0)
        except (TypeError, ValueError):
            liq = 0.0
        if liq > 0:
            return liq
        if attempt < attempts - 1:
            time.sleep(delay_sec)
    log.warning(
        "tv_executor no liquidation price for %s after %d attempts — "
        "emergency SL will not be liquidation-clamped",
        symbol, attempts,
    )
    return None


def _clamp_stop_inside_liquidation(
    sl_price: float,
    position_side: str,
    mid_price: float,
    liq_price: Optional[float],
    buffer_frac: float,
) -> float:
    """Pull a stop back inside the liquidation price.

    A stop at or beyond liquidation is not protection: the exchange closes the
    position at the liquidation price first, so the stop never triggers and the
    loss is the whole margin instead of the intended slice. This only ever
    tightens the stop — a stop already comfortably inside is returned untouched.

    The buffer is a fraction of the entry->liquidation distance rather than a
    flat percentage so it scales with leverage on its own: wide at 5x, tight at
    25x, always proportional to the room actually available.
    """
    if buffer_frac <= 0 or liq_price is None or liq_price <= 0 or mid_price <= 0:
        return sl_price

    if position_side == "long":
        gap = mid_price - liq_price
        if gap <= 0:
            log.error(
                "tv_executor liquidation price %.4f is not below a long entry at %.4f — "
                "cannot clamp; leaving stop at %.4f",
                liq_price, mid_price, sl_price,
            )
            return sl_price
        floor = liq_price + gap * buffer_frac
        if sl_price >= floor:
            return sl_price
        log.warning(
            "tv_executor emergency SL %.4f sits too close to liquidation %.4f "
            "(long, entry %.4f) — tightening to %.4f (%.0f%% of the entry->liq gap)",
            sl_price, liq_price, mid_price, floor, buffer_frac * 100,
        )
        return floor

    if position_side == "short":
        gap = liq_price - mid_price
        if gap <= 0:
            log.error(
                "tv_executor liquidation price %.4f is not above a short entry at %.4f — "
                "cannot clamp; leaving stop at %.4f",
                liq_price, mid_price, sl_price,
            )
            return sl_price
        ceiling = liq_price - gap * buffer_frac
        if sl_price <= ceiling:
            return sl_price
        log.warning(
            "tv_executor emergency SL %.4f sits too close to liquidation %.4f "
            "(short, entry %.4f) — tightening to %.4f (%.0f%% of the entry->liq gap)",
            sl_price, liq_price, mid_price, ceiling, buffer_frac * 100,
        )
        return ceiling

    return sl_price


def _place_emergency_sl(
    broker: KucoinFuturesBroker,
    symbol: str,
    position_side: str,
    qty: int,
    mid_price: float,
    sl_pct: float,
    strategy_sl_price: Optional[float] = None,
    liq_buffer_frac: float = 0.0,
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

    # Prefer the strategy's own stop when the alert states one — it is the real
    # level, and may sit wider than the percentage backstop. Only accept it if
    # it is on the protective side of the current price: a stop on the wrong
    # side triggers the instant it is placed, which is precisely the failure
    # this whole path just had to be fixed for.
    if strategy_sl_price is not None and strategy_sl_price > 0:
        protective = (
            strategy_sl_price < mid_price if position_side == "long"
            else strategy_sl_price > mid_price
        )
        if protective:
            log.info(
                "tv_executor emergency SL using strategy stop %.4f (backstop would be %.4f)",
                strategy_sl_price, sl_price,
            )
            sl_price = float(strategy_sl_price)
        else:
            log.warning(
                "tv_executor ignoring strategy stop %.4f for %s at %.4f — wrong side of price, "
                "would trigger immediately; falling back to %.2f%% backstop at %.4f",
                strategy_sl_price, position_side, mid_price, sl_pct * 100, sl_price,
            )

    # Whatever level we arrived at — percentage backstop or the strategy's own
    # stop — it is only protection if it triggers before the exchange
    # liquidates. Checked last so it constrains both paths.
    if liq_buffer_frac > 0:
        liq_price = _liquidation_price(broker, symbol)
        clamped = _clamp_stop_inside_liquidation(
            sl_price, position_side, mid_price, liq_price, liq_buffer_frac
        )
        still_protective = (
            clamped < mid_price if position_side == "long" else clamped > mid_price
        )
        if not still_protective:
            log.error(
                "tv_executor CANNOT PLACE A SAFE STOP for %s at %.4f: liquidation %.4f is so "
                "close that any stop inside it would trigger immediately. Leverage is too high "
                "for a meaningful stop — placing the %.2f%% backstop at %.4f anyway, but this "
                "position is effectively protected by liquidation only",
                position_side, mid_price, liq_price or 0.0, sl_pct * 100, sl_price,
            )
        else:
            sl_price = clamped

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
        effective_pct = abs(mid_price - sl_price) / mid_price * 100 if mid_price > 0 else 0.0
        log.info(
            "tv_executor emergency SL placed: side=%s qty=%d sl_price=%.4f (%.2f%% from %.4f) order_id=%s",
            sl_side, qty, sl_price, effective_pct, mid_price, oid,
        )
        _verify_stop_registered(broker, oid, sl_price)
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


def _ensure_margin_mode_while_flat(broker: KucoinFuturesBroker, symbol: str) -> None:
    """Correct the margin mode at the one moment it is safe to: while flat.

    An account left in CROSS silently ignores the configured leverage. Startup
    can't always fix it — if a position is open when the bot boots, the switch
    has to wait until that position closes, which may be days later. Checking
    here means the correction lands on the next entry instead.
    """
    want = str(os.getenv("KUCOIN_FUTURES_MARGIN_MODE", "")).strip()
    if not want:
        return
    try:
        broker.ensure_margin_mode(symbol, want)
    except Exception as e:
        log.warning("margin mode check before entry failed: %s", e)


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
            return _do_flip(broker, config, signal.symbol, signal.side, cache, seq, signal.sl_price)

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

        # We are flat and about to open — the only safe moment to correct the
        # margin mode. CROSS makes KuCoin ignore the configured leverage, and
        # switching with a position open would move its liquidation price.
        _ensure_margin_mode_while_flat(broker, signal.symbol)

        order_side = signal.side  # buy or sell
        oid = _place_market(broker, signal.symbol, order_side, qty, reduce_only=False, action_label="entry")
        pos_after_i = 1 if want_side == "long" else -1

        _place_emergency_sl(broker, signal.symbol, want_side, qty, cache.mid_price,
                            config.emergency_sl_pct, signal.sl_price,
                            config.sl_liq_buffer_frac)
        _refresh_position_in_cache(broker, config)

        _log_bg(_log_action, symbol=signal.symbol, seq=seq, action="entry",
                action_side=want_side, reason="tv_entry",
                pos_before=pos_before_i, pos_after=pos_after_i)
        _log_bg(_log_execution, symbol=signal.symbol, seq=seq, kind="market_fill",
                order_action=order_side, reason="tv_entry",
                pos_before=pos_before_i, pos_after=pos_after_i,
                order_id=oid, side=order_side, qty=float(qty), reduce_only=False, status="sent")
        _record_open_leg(
            symbol=signal.symbol,
            side=want_side,
            qty=float(qty),
            entry_price=cache.mid_price,
        )

        return {"ok": True, "action": "entry", "side": want_side, "qty": qty, "order_id": oid}

    # ----- exit -----
    if signal.action == "exit":
        if side == "flat":
            return {"ok": True, "action": "exit", "reason": "already_flat"}

        close_qty = abs(int(pos))
        if close_qty <= 0:
            return {"ok": True, "action": "exit", "reason": "already_flat"}

        # A flip closes the old position and opens the new one by itself. If
        # TradingView also sends the matching `exit` for that same reversal, it
        # arrives milliseconds later and flattens the position the flip just
        # opened — the bot ends up flat having paid entry+exit fees for nothing.
        # An exit belongs to the position it was computed against, so one that
        # lands this soon after an open cannot have been meant for it.
        age = _seconds_since_open(signal.symbol)
        guard = float(config.exit_after_open_guard_sec or 0.0)
        if guard > 0 and age is not None and age < guard:
            _log_bg(_log_action, symbol=signal.symbol, seq=seq, action="exit",
                    action_side=side, reason="tv_exit",
                    pos_before=pos_before_i, pos_after=pos_before_i,
                    blocked=True,
                    block_reason=f"exit_after_open_guard:{age:.2f}s<{guard:.2f}s")
            log.warning(
                "tv_executor exit ignored: position opened %.2fs ago (< %.2fs guard) "
                "— treating as the redundant exit of a flip, not a close of this position",
                age, guard,
            )
            return {
                "ok": True,
                "action": "exit",
                "reason": "exit_after_open_guard",
                "position_age_sec": age,
                "guard_sec": guard,
            }

        if config.dry_run:
            log.warning("tv_executor DRY_RUN exit %s qty=%d", side, close_qty)
            return {"ok": True, "action": "exit", "reason": "dry_run", "qty": close_qty}

        close_side = "sell" if side == "long" else "buy"
        oid = _place_market(broker, signal.symbol, close_side, close_qty, reduce_only=True, action_label="exit")

        _refresh_position_in_cache(broker, config)

        _log_bg(_log_action, symbol=signal.symbol, seq=seq, action="exit",
                action_side=side, reason="tv_exit",
                pos_before=pos_before_i, pos_after=0)
        _log_bg(_log_execution, symbol=signal.symbol, seq=seq, kind="market_fill",
                order_action=close_side, reason="tv_exit",
                pos_before=pos_before_i, pos_after=0,
                order_id=oid, side=close_side, qty=float(close_qty), reduce_only=True, status="sent")
        _append_tv_closed_trade(
            symbol=signal.symbol,
            exit_event="exit",
            exit_price=cache.mid_price,
            qty=float(close_qty),
        )

        return {"ok": True, "action": "exit", "qty": close_qty, "order_id": oid}

    # ----- flip -----
    if signal.action == "flip":
        return _do_flip(broker, config, signal.symbol, signal.side, cache, seq, signal.sl_price)

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

        close_side = "sell" if side == "long" else "buy"
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

        close_side = "sell" if side == "long" else "buy"
        oid = _place_market(broker, signal.symbol, close_side, close_qty, reduce_only=True, action_label="tp2")

        _refresh_position_in_cache(broker, config)

        _log_bg(_log_action, symbol=signal.symbol, seq=seq, action="tp2_close",
                action_side=side, reason="tv_tp2",
                pos_before=pos_before_i, pos_after=0)
        _log_bg(_log_execution, symbol=signal.symbol, seq=seq, kind="market_fill",
                order_action=close_side, reason="tv_tp2",
                pos_before=pos_before_i, pos_after=0,
                order_id=oid, side=close_side, qty=float(close_qty), reduce_only=True, status="sent")
        _append_tv_closed_trade(
            symbol=signal.symbol,
            exit_event="tp2",
            exit_price=cache.mid_price,
            qty=float(close_qty),
        )

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

        close_side = "sell" if side == "long" else "buy"
        oid = _place_market(broker, signal.symbol, close_side, close_qty, reduce_only=True, action_label="sl")

        _refresh_position_in_cache(broker, config)

        _log_bg(_log_action, symbol=signal.symbol, seq=seq, action="sl_exit",
                action_side=side, reason="tv_sl",
                pos_before=pos_before_i, pos_after=0)
        _log_bg(_log_execution, symbol=signal.symbol, seq=seq, kind="market_fill",
                order_action=close_side, reason="tv_sl",
                pos_before=pos_before_i, pos_after=0,
                order_id=oid, side=close_side, qty=float(close_qty), reduce_only=True, status="sent")
        _append_tv_closed_trade(
            symbol=signal.symbol,
            exit_event="sl",
            exit_price=cache.mid_price,
            qty=float(close_qty),
        )

        return {"ok": True, "action": "sl", "qty": close_qty, "order_id": oid}

    return {"ok": False, "action": signal.action, "reason": "unknown_action"}


def _do_flip(
    broker: KucoinFuturesBroker,
    config: TVExecConfig,
    symbol: str,
    order_side: str,
    cache: TVCache,
    seq: int,
    strategy_sl_price: Optional[float] = None,
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
        _append_tv_closed_trade(
            symbol=symbol,
            exit_event="flip_flatten",
            exit_price=cache.mid_price,
            qty=float(close_qty),
        )

        return {"ok": True, "action": "flip", "reason": "flatten_only_gate_blocked", "qty_closed": close_qty, "order_id": oid}

    if qty <= 0:
        return {"ok": False, "action": "flip", "reason": "qty_zero_check_equity"}

    if config.dry_run:
        log.warning(
            "tv_executor DRY_RUN flip %s->%s close_qty=%d wait=%.1fs open_qty=%d",
            current_side, want_side, abs(int(pos)), config.flip_delay_sec, qty,
        )
        return {
            "ok": True, "action": "flip", "reason": "dry_run", "side": want_side,
            "close_qty": abs(int(pos)), "qty": qty,
            "flip_delay_sec": config.flip_delay_sec,
        }

    if current_side == "flat":
        oid = _place_market(broker, symbol, order_side, qty, reduce_only=False, action_label="flip_entry")
        pos_after_i = 1 if want_side == "long" else -1

        _place_emergency_sl(broker, symbol, want_side, qty, cache.mid_price,
                            config.emergency_sl_pct, strategy_sl_price,
                            config.sl_liq_buffer_frac)
        _refresh_position_in_cache(broker, config)

        _log_bg(_log_action, symbol=symbol, seq=seq, action="flip_entry",
                action_side=want_side, reason="tv_flip",
                pos_before=0, pos_after=pos_after_i)
        _log_bg(_log_execution, symbol=symbol, seq=seq, kind="market_fill",
                order_action=order_side, reason="tv_flip_entry",
                pos_before=0, pos_after=pos_after_i,
                order_id=oid, side=order_side, qty=float(qty), reduce_only=False, status="sent")
        _record_open_leg(
            symbol=symbol,
            side=want_side,
            qty=float(qty),
            entry_price=cache.mid_price,
        )

        return {"ok": True, "action": "flip", "side": want_side, "qty": qty, "order_id": oid}

    # Two-legged flip: close the existing position, settle, then open the
    # opposite side. A single net-off order is faster but leaves no clean
    # close fill to analyse, and KuCoin can partially fill the combined size
    # and strand the position mid-flip.
    _cancel_emergency_sl(broker, symbol)
    pos_after_i = 1 if want_side == "long" else -1

    close_side = "sell" if current_side == "long" else "buy"
    close_qty = abs(int(pos))
    close_oid = _place_market(
        broker, symbol, close_side, close_qty, reduce_only=True, action_label="flip_close"
    )
    _log_bg(_log_execution, symbol=symbol, seq=seq, kind="market_fill",
            order_action=close_side, reason="tv_flip_close",
            pos_before=pos_before_i, pos_after=0,
            order_id=close_oid, side=close_side, qty=float(close_qty),
            reduce_only=True, status="sent")
    _append_tv_closed_trade(
        symbol=symbol,
        exit_event="flip_close",
        exit_price=cache.mid_price,
        qty=float(close_qty),
    )

    # Let the close settle so the opposite entry sizes against a flat book.
    if config.flip_delay_sec > 0:
        time.sleep(config.flip_delay_sec)

    # Confirm we are actually flat before reversing; opening the opposite side
    # on top of an unclosed position would double the intended exposure.
    try:
        pos_now = float(broker.get_position(symbol))
    except Exception as e:
        log.warning("tv_executor flip: position check failed after close: %s", e)
        pos_now = 0.0

    if abs(pos_now) > 0:
        log.error(
            "tv_executor flip aborted: still holding %.1f contracts after close leg; not reversing",
            pos_now,
        )
        _refresh_position_in_cache(broker, config)
        _log_bg(_log_action, symbol=symbol, seq=seq, action="flip_close_only",
                action_side=want_side, reason="tv_flip_close_incomplete",
                pos_before=pos_before_i, pos_after=pos_before_i,
                blocked=True, block_reason=f"residual_position={pos_now}")
        return {
            "ok": False, "action": "flip", "reason": "close_leg_incomplete",
            "residual_position": pos_now, "order_id": close_oid,
        }

    open_oid = _place_market(
        broker, symbol, order_side, qty, reduce_only=False, action_label="flip_open"
    )

    _place_emergency_sl(broker, symbol, want_side, qty, cache.mid_price,
                        config.emergency_sl_pct, strategy_sl_price,
                        config.sl_liq_buffer_frac)
    _refresh_position_in_cache(broker, config)

    _log_bg(_log_action, symbol=symbol, seq=seq, action="flip",
            action_side=want_side, reason="tv_flip",
            pos_before=pos_before_i, pos_after=pos_after_i)
    _log_bg(_log_execution, symbol=symbol, seq=seq, kind="market_fill",
            order_action=order_side, reason="tv_flip_open",
            pos_before=0, pos_after=pos_after_i,
            order_id=open_oid, side=order_side, qty=float(qty),
            reduce_only=False, status="sent")
    _record_open_leg(
        symbol=symbol,
        side=want_side,
        qty=float(qty),
        entry_price=cache.mid_price,
    )

    return {
        "ok": True, "action": "flip", "side": want_side, "qty": qty,
        "close_order_id": close_oid, "order_id": open_oid,
        "flip_delay_sec": config.flip_delay_sec,
    }


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
