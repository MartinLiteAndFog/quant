from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
from typing import Any, Dict, Optional, Tuple

from quant.execution.kraken_futures import KrakenFuturesClient
from quant.utils.log import get_logger

log = get_logger("quant.kraken_tv_executor")

TARGET_ACTIONS = {"entry", "flip"}
UNSUPPORTED_ACTIONS = {"exit", "sl", "tp2"}
VALID_SIDES = {"buy", "sell"}
ORDER_SUCCESS_STATUSES = {"placed", "received", "success", "accepted", "filled", "partiallyfilled"}

_DEDUP_LOCK = threading.Lock()
_SEEN_FINGERPRINTS: Dict[str, float] = {}
_EXEC_LOCK = threading.Lock()


def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def _floor_to_step(value: float, step: float) -> float:
    value_dec = Decimal(str(max(0.0, float(value))))
    step_dec = Decimal(str(max(float(step), 0.00000001)))
    units = (value_dec / step_dec).to_integral_value(rounding=ROUND_FLOOR)
    floored = units * step_dec
    decimals = max(0, -step_dec.as_tuple().exponent)
    return float(round(floored, decimals))


def _signed_side(size_signed: float) -> str:
    if size_signed > 0:
        return "long"
    if size_signed < 0:
        return "short"
    return "flat"


def _is_sol_symbol(symbol: str) -> bool:
    cleaned = "".join(ch for ch in str(symbol or "").upper() if ch.isalnum())
    return cleaned in {"SOLUSDT", "SOLUSD", "PFSOLUSD"}


def _position_tolerance(step: float) -> float:
    return max(float(step) / 2.0, 0.00000001)


def _same_direction(left: float, right: float) -> bool:
    return left != 0 and right != 0 and math.copysign(1.0, left) == math.copysign(1.0, right)


def _opposite_direction(left: float, right: float) -> bool:
    return left != 0 and right != 0 and math.copysign(1.0, left) != math.copysign(1.0, right)


def _at_target(current_signed: float, desired_signed: float, step: float) -> bool:
    return _same_direction(current_signed, desired_signed) and abs(current_signed - desired_signed) <= _position_tolerance(step)


@dataclass(frozen=True)
class KrakenTVConfig:
    venue_symbol: str
    display_symbol: str
    pos_pct: float
    leverage: float
    tp1_frac: float
    dry_run: bool
    size_step: float
    dedup_ttl_sec: float
    cancel_reduce_only_on_flip: bool
    verify_after_order: bool
    refill_partial: bool

    @classmethod
    def from_env(cls) -> "KrakenTVConfig":
        return cls(
            venue_symbol=_env_first("KRAKEN_FUTURES_SYMBOL", "KRAKEN_VENUE_SYMBOL", default="PF_SOLUSD"),
            display_symbol=_env_first("LIVE_SYMBOL", "KRAKEN_SYMBOL", default="SOL-USDT"),
            pos_pct=float(_env_first("KRAKEN_TV_POS_PCT", "LIVE_EXECUTOR_2_POS_PCT", default="0.90")),
            leverage=float(
                _env_first(
                    "KRAKEN_TV_LEVERAGE",
                    "KRAKEN_FUTURES_ORDER_LEVERAGE",
                    "LIVE_EXECUTOR_2_LEVERAGE",
                    default="10",
                )
            ),
            tp1_frac=float(_env_first("KRAKEN_TV_TP1_FRAC", "LIVE_TP1_FRAC", default="0.50")),
            dry_run=_truthy(_env_first("KRAKEN_TV_DRY_RUN", "LIVE_EXECUTOR_2_DRY_RUN", default="1")),
            size_step=float(_env_first("KRAKEN_SIZE_STEP", default="0.1")),
            dedup_ttl_sec=float(_env_first("KRAKEN_TV_DEDUP_TTL_SEC", default="300")),
            cancel_reduce_only_on_flip=_truthy(
                _env_first("KRAKEN_TV_CANCEL_REDUCE_ONLY_ON_FLIP", default="1")
            ),
            verify_after_order=_truthy(_env_first("KRAKEN_TV_VERIFY_AFTER_ORDER", default="1")),
            refill_partial=_truthy(_env_first("KRAKEN_TV_REFILL_PARTIAL", default="0")),
        )


@dataclass(frozen=True)
class KrakenTVSignal:
    action: str
    side: str
    symbol: str
    reason: str
    fingerprint: str


def _fingerprint(payload: Dict[str, Any]) -> str:
    parts = [
        str(payload.get("source", "")),
        str(payload.get("action", "")),
        str(payload.get("side", "")),
        str(payload.get("reason", "")),
        str(payload.get("bar_time", "")),
        str(payload.get("bar_index", "")),
    ]
    return "|".join(parts).lower()


def parse_kraken_tv_signal(payload: Dict[str, Any], default_symbol: str = "SOL-USDT") -> KrakenTVSignal:
    action = str(payload.get("action", "")).strip().lower()
    if action in UNSUPPORTED_ACTIONS:
        raise ValueError(f"unsupported TradingView action for Kraken executor: {action}")
    if action not in TARGET_ACTIONS and action != "tp1":
        raise ValueError("action must be one of entry, flip, tp1")

    side = str(payload.get("side", "")).strip().lower()
    if action in TARGET_ACTIONS and side not in VALID_SIDES:
        raise ValueError(f"action {action!r} requires side buy|sell")
    if action == "tp1" and side and side not in VALID_SIDES:
        raise ValueError("tp1 side must be empty, buy, or sell")

    symbol = ""
    for key in ("symbol", "ticker", "pair"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            symbol = value.strip()
            break
    symbol = symbol or default_symbol
    if not _is_sol_symbol(symbol):
        raise ValueError(f"unsupported symbol for Kraken SOL executor: {symbol}")

    return KrakenTVSignal(
        action=action,
        side=side,
        symbol=symbol,
        reason=str(payload.get("reason", "") or ""),
        fingerprint=_fingerprint(payload),
    )


def _claim_fingerprint(fingerprint: str, ttl_sec: float) -> bool:
    now = time.time()
    ttl = max(0.0, float(ttl_sec))
    if ttl <= 0:
        return True
    with _DEDUP_LOCK:
        expired = [key for key, seen_at in _SEEN_FINGERPRINTS.items() if now - seen_at > ttl]
        for key in expired:
            _SEEN_FINGERPRINTS.pop(key, None)
        seen_at = _SEEN_FINGERPRINTS.get(fingerprint)
        if seen_at is not None and now - seen_at <= ttl:
            return False
        _SEEN_FINGERPRINTS[fingerprint] = now
        return True


def _release_fingerprint(fingerprint: str) -> None:
    with _DEDUP_LOCK:
        _SEEN_FINGERPRINTS.pop(fingerprint, None)


def compute_target_size(equity_usd: float, mark_price: float, leverage: float, pos_pct: float, step: float) -> float:
    if equity_usd <= 0 or mark_price <= 0 or leverage <= 0 or pos_pct <= 0:
        return 0.0
    raw = float(equity_usd) * float(pos_pct) * float(leverage) / float(mark_price)
    return _floor_to_step(raw, step)


def _position_snapshot(client: KrakenFuturesClient, venue_symbol: str) -> Tuple[Dict[str, Any], float]:
    pos = client.get_position(symbol=venue_symbol)
    return pos, float(pos.get("size_signed", 0.0) or 0.0)


def _order_status(result: Optional[Dict[str, Any]]) -> str:
    if not isinstance(result, dict):
        return ""
    data = result.get("data")
    if not isinstance(data, dict):
        data = result
    send_status = data.get("sendStatus")
    if isinstance(send_status, dict):
        status = send_status.get("status") or send_status.get("result")
    else:
        status = data.get("status") or data.get("result")
    return str(status or "").strip().lower()


def _order_reject_reason(result: Optional[Dict[str, Any]]) -> str:
    if not isinstance(result, dict):
        return "missing order response"
    data = result.get("data")
    if not isinstance(data, dict):
        data = result
    send_status = data.get("sendStatus")
    if isinstance(send_status, dict):
        for key in ("rejectReason", "reason", "error", "message"):
            value = send_status.get(key)
            if value:
                return str(value)
    for key in ("error", "message", "rejectReason", "reason"):
        value = data.get(key)
        if value:
            return str(value)
    return f"unexpected order status: {_order_status(result) or 'unknown'}"


def _assert_order_accepted(result: Optional[Dict[str, Any]], label: str) -> Dict[str, Any]:
    if isinstance(result, dict) and result.get("ok") is False:
        raise RuntimeError(f"{label} rejected by Kraken: {_order_reject_reason(result)}")
    status = _order_status(result)
    if status and status not in ORDER_SUCCESS_STATUSES:
        raise RuntimeError(f"{label} rejected by Kraken: {_order_reject_reason(result)}")
    if not isinstance(result, dict):
        raise RuntimeError(f"{label} returned no Kraken response")
    return result


def _place_market_checked(
    client: KrakenFuturesClient,
    side: str,
    *,
    size: float,
    symbol: str,
    reduce_only: bool,
    label: str,
) -> Dict[str, Any]:
    result = client.place_market(side, size=size, symbol=symbol, reduce_only=reduce_only)
    return _assert_order_accepted(result, label)


def _wait_for_flat(
    client: KrakenFuturesClient,
    venue_symbol: str,
    step: float,
    attempts: int = 3,
    delay_sec: float = 0.25,
) -> Tuple[Dict[str, Any], float]:
    tolerance = _position_tolerance(step)
    last_pos, last_signed = _position_snapshot(client, venue_symbol)
    for attempt in range(max(1, attempts)):
        if abs(last_signed) <= tolerance:
            return last_pos, last_signed
        if attempt < attempts - 1:
            time.sleep(delay_sec)
            last_pos, last_signed = _position_snapshot(client, venue_symbol)
    return last_pos, last_signed


def execute_kraken_tv_signal(
    signal: KrakenTVSignal,
    config: KrakenTVConfig,
    client: Optional[KrakenFuturesClient] = None,
) -> Dict[str, Any]:
    if not _claim_fingerprint(signal.fingerprint, config.dedup_ttl_sec):
        return {
            "ok": True,
            "deduped": True,
            "action": signal.action,
            "side": signal.side,
            "fingerprint": signal.fingerprint,
        }

    client = client or KrakenFuturesClient()
    try:
        with _EXEC_LOCK:
            if signal.action == "tp1":
                return _execute_tp1(client, signal, config)
            return _execute_target_side(client, signal, config)
    except Exception:
        _release_fingerprint(signal.fingerprint)
        raise


def _execute_tp1(
    client: KrakenFuturesClient,
    signal: KrakenTVSignal,
    config: KrakenTVConfig,
) -> Dict[str, Any]:
    pos_raw, current_signed = _position_snapshot(client, config.venue_symbol)
    current_abs = abs(current_signed)
    close_size = _floor_to_step(current_abs * config.tp1_frac, config.size_step)
    if current_abs <= 0 or close_size <= 0:
        return {
            "ok": True,
            "action": "tp1",
            "reason": "already_flat_or_below_step",
            "position_before": pos_raw,
            "close_size": close_size,
            "dry_run": config.dry_run,
        }

    order_side = "sell" if current_signed > 0 else "buy"
    result: Optional[Dict[str, Any]] = None
    if not config.dry_run:
        result = _place_market_checked(
            client,
            order_side,
            size=close_size,
            symbol=config.venue_symbol,
            reduce_only=True,
            label="kraken tv tp1",
        )

    position_after = None
    if config.verify_after_order and not config.dry_run:
        position_after = client.get_position(symbol=config.venue_symbol)

    return {
        "ok": True,
        "action": "tp1",
        "order_side": order_side,
        "order_size": close_size,
        "tp1_frac": config.tp1_frac,
        "position_before": pos_raw,
        "position_after": position_after,
        "dry_run": config.dry_run,
        "order_result": result,
        "fingerprint": signal.fingerprint,
    }


def _execute_target_side(
    client: KrakenFuturesClient,
    signal: KrakenTVSignal,
    config: KrakenTVConfig,
) -> Dict[str, Any]:
    mark = float(client.get_mark_price(symbol=config.venue_symbol) or 0.0)
    equity = client.get_account_equity()
    equity_usd = float(equity.get("equity_usd", 0.0) or 0.0)
    pos_raw, current_signed = _position_snapshot(client, config.venue_symbol)

    target_abs = compute_target_size(
        equity_usd=equity_usd,
        mark_price=mark,
        leverage=config.leverage,
        pos_pct=config.pos_pct,
        step=config.size_step,
    )
    desired_signed = target_abs if signal.side == "buy" else -target_abs
    delta = desired_signed - current_signed
    order_size = _floor_to_step(abs(delta), config.size_step)
    order_side = "buy" if delta > 0 else "sell"

    if target_abs <= 0 or order_size <= 0:
        return {
            "ok": True,
            "action": signal.action,
            "side": signal.side,
            "reason": "already_at_target_or_below_step",
            "mark": mark,
            "equity": equity,
            "target_size": target_abs,
            "current_signed": current_signed,
            "desired_signed": desired_signed,
            "order_size": order_size,
            "position_before": pos_raw,
            "dry_run": config.dry_run,
        }

    direction_change = _opposite_direction(current_signed, desired_signed)
    cancel_result: Optional[Dict[str, Any]] = None
    close_result: Optional[Dict[str, Any]] = None
    close_position_after = None
    close_signed_after: Optional[float] = None
    order_result: Optional[Dict[str, Any]] = None
    net_position_after = None
    net_signed_after: Optional[float] = None
    net_order_error: Optional[str] = None
    fallback_used = False
    fallback_reason: Optional[str] = None
    refill_result: Optional[Dict[str, Any]] = None
    position_after = None
    order_plan = [
        {
            "side": order_side,
            "size": order_size,
            "reduce_only": False,
            "role": "net_target",
        }
    ]

    if direction_change:
        order_plan[0]["role"] = "optimistic_net_flip"
        order_plan.append(
            {
                "side": "sell" if current_signed > 0 else "buy",
                "size": _floor_to_step(abs(current_signed), config.size_step),
                "reduce_only": True,
                "role": "fallback_close_current",
            }
        )
        order_plan.append(
            {
                "side": signal.side,
                "size": target_abs,
                "reduce_only": False,
                "role": "fallback_open_target",
            }
        )

    if not config.dry_run:
        if direction_change and config.cancel_reduce_only_on_flip:
            cancel_result = client.cancel_all_reduce_only_orders(symbol=config.venue_symbol)

        if direction_change:
            try:
                order_result = _place_market_checked(
                    client,
                    order_side,
                    size=order_size,
                    symbol=config.venue_symbol,
                    reduce_only=False,
                    label="kraken tv optimistic flip",
                )
            except RuntimeError as exc:
                net_order_error = str(exc)

            net_position_after, net_signed_after = _position_snapshot(client, config.venue_symbol)
            if net_order_error:
                fallback_reason = "net_order_rejected"
            elif _at_target(float(net_signed_after or 0.0), desired_signed, config.size_step):
                fallback_reason = None
            elif _opposite_direction(float(net_signed_after or 0.0), desired_signed):
                fallback_reason = "net_order_left_old_side"
            elif abs(float(net_signed_after or 0.0)) <= _position_tolerance(config.size_step):
                fallback_reason = "net_order_left_flat"
            elif config.refill_partial and _same_direction(float(net_signed_after or 0.0), desired_signed):
                fallback_reason = "net_order_partial_refill"

            if fallback_reason in {"net_order_rejected", "net_order_left_old_side", "net_order_left_flat"}:
                fallback_used = True
                after_signed = float(net_signed_after or 0.0)
                if abs(after_signed) > _position_tolerance(config.size_step):
                    close_side = "sell" if after_signed > 0 else "buy"
                    close_size = _floor_to_step(abs(after_signed), config.size_step)
                    if close_size > 0:
                        close_result = _place_market_checked(
                            client,
                            close_side,
                            size=close_size,
                            symbol=config.venue_symbol,
                            reduce_only=True,
                            label="kraken tv fallback close",
                        )
                    close_position_after, close_signed_after = _wait_for_flat(
                        client,
                        config.venue_symbol,
                        config.size_step,
                    )
                    if abs(float(close_signed_after or 0.0)) > _position_tolerance(config.size_step):
                        raise RuntimeError(
                            f"kraken tv fallback close did not flatten position: signed={close_signed_after}"
                        )
                else:
                    close_position_after = net_position_after
                    close_signed_after = after_signed

                mark = float(client.get_mark_price(symbol=config.venue_symbol) or 0.0)
                equity = client.get_account_equity()
                equity_usd = float(equity.get("equity_usd", 0.0) or 0.0)
                target_abs = compute_target_size(
                    equity_usd=equity_usd,
                    mark_price=mark,
                    leverage=config.leverage,
                    pos_pct=config.pos_pct,
                    step=config.size_step,
                )
                desired_signed = target_abs if signal.side == "buy" else -target_abs
                order_size = target_abs
                order_side = signal.side
                order_plan[-1]["size"] = target_abs
                if order_size > 0:
                    refill_result = _place_market_checked(
                        client,
                        order_side,
                        size=order_size,
                        symbol=config.venue_symbol,
                        reduce_only=False,
                        label="kraken tv fallback open",
                    )
            elif fallback_reason == "net_order_partial_refill":
                fallback_used = True
                after_signed = float(net_signed_after or 0.0)
                refill_delta = desired_signed - after_signed
                refill_size = _floor_to_step(abs(refill_delta), config.size_step)
                if refill_size > 0:
                    refill_side = "buy" if refill_delta > 0 else "sell"
                    refill_result = _place_market_checked(
                        client,
                        refill_side,
                        size=refill_size,
                        symbol=config.venue_symbol,
                        reduce_only=False,
                        label="kraken tv optimistic flip refill",
                    )
        else:
            reducing_same_side = (
                current_signed != 0
                and desired_signed != 0
                and _same_direction(current_signed, desired_signed)
                and abs(desired_signed) < abs(current_signed)
            )
            order_plan[0]["reduce_only"] = reducing_same_side

        if not direction_change and order_size > 0:
            order_result = _place_market_checked(
                client,
                order_side,
                size=order_size,
                symbol=config.venue_symbol,
                reduce_only=bool(order_plan[-1]["reduce_only"]),
                label="kraken tv target entry",
            )

        if config.verify_after_order:
            position_after = client.get_position(symbol=config.venue_symbol)
            if config.refill_partial:
                after_signed = float(position_after.get("size_signed", 0.0) or 0.0)
                refill_delta = desired_signed - after_signed
                refill_size = _floor_to_step(abs(refill_delta), config.size_step)
                if refill_size > 0:
                    refill_side = "buy" if refill_delta > 0 else "sell"
                    refill_result = _place_market_checked(
                        client,
                        refill_side,
                        size=refill_size,
                        symbol=config.venue_symbol,
                        reduce_only=False,
                        label="kraken tv target refill",
                    )
                    position_after = client.get_position(symbol=config.venue_symbol)

    log.info(
        "kraken tv %s %s: current=%s desired=%s order=%s %s direction_change=%s fallback=%s equity=%.2f mark=%.4f dry_run=%s order_result=%s close_result=%s refill_result=%s",
        signal.action,
        signal.side,
        current_signed,
        desired_signed,
        order_side,
        order_size,
        direction_change,
        fallback_reason,
        equity_usd,
        mark,
        config.dry_run,
        order_result,
        close_result,
        refill_result,
    )

    return {
        "ok": True,
        "action": signal.action,
        "side": signal.side,
        "reason": signal.reason,
        "venue_symbol": config.venue_symbol,
        "display_symbol": config.display_symbol,
        "mark": mark,
        "equity": equity,
        "pos_pct": config.pos_pct,
        "leverage": config.leverage,
        "target_size": target_abs,
        "current_signed": current_signed,
        "current_side": _signed_side(current_signed),
        "desired_signed": desired_signed,
        "desired_side": _signed_side(desired_signed),
        "order_side": order_side,
        "order_size": order_size,
        "order_plan": order_plan,
        "direction_change": direction_change,
        "cancel_reduce_only_result": cancel_result,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "net_order_error": net_order_error,
        "net_position_after": net_position_after,
        "net_signed_after": net_signed_after,
        "close_result": close_result,
        "close_position_after": close_position_after,
        "close_signed_after": close_signed_after,
        "order_result": order_result,
        "refill_result": refill_result,
        "position_before": pos_raw,
        "position_after": position_after,
        "dry_run": config.dry_run,
        "fingerprint": signal.fingerprint,
    }
