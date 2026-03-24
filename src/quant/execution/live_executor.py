from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quant.execution.execution_state import write_execution_state
from quant.execution.CHOPgate import get_live_gate_state
from quant.execution.kucoin_futures import KucoinFuturesBroker
from quant.execution.oms import MakerFirstOMS, OmsDefaults
from quant.execution.event_builders import build_action_event, build_execution_event
from quant.execution.event_log import append_event_jsonl
from quant.execution.event_store import (
    insert_action_event,
    insert_execution_event,
    upsert_closed_trade,
)
from quant.strategies.flip_engine import FlipParams, run_flip_state_machine
from quant.strategies.follow_tp2_engine import TP2Params, run_follow_tp2_state_machine
from quant.strategies.signal_io import read_signals_jsonl
from quant.utils.log import get_logger, log_throttled

log = get_logger("quant.live_executor")

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


def _read_live_gate_from_redis(symbol: str) -> Optional[Dict[str, Any]]:
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        return None
    try:
        import redis as redis_lib

        canon = _canon_symbol(symbol)
        key = f"gate:{canon}:latest"
        r = redis_lib.from_url(redis_url, decode_responses=True)
        raw = r.get(key)
        if not raw:
            return None
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _events_root() -> Path:
    if Path("/data").exists():
        return Path("/data/events")
    return Path("data/events")


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
        venue="kucoin",
        source_event_id=None,
        source_signal_event_id=None,
        position_before=int(position_before),
        position_after=int(position_after),
        engine_mode_before=engine_mode_before,
        engine_mode_after=engine_mode_after,
        blocked=bool(blocked),
        block_reason=block_reason,
    )
    event["strategy_instance"] = "live_executor"
    event["config_hash"] = "live_executor_v1"

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
        log.warning("kucoin postgres action event failed: %s", e)


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
        venue="kucoin",
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
        strategy_instance="live_executor",
        config_hash="live_executor_v1",
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
        log.warning("kucoin postgres execution event failed: %s", e)


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
            "kucoin closed trade skipped: missing prices/qty action=%s event=%s entry_px=%s exit_px=%s qty=%s",
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
                "venue": "kucoin",
                "symbol": symbol,
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "side": current_side,
                "qty": float(qty_realized),
                "entry_price": float(entry_px_realized),
                "exit_price": float(exit_px_realized),
                "pnl_pct": float(pnl_pct_realized),
                "exit_event": event_name,
                "strategy": "live_executor",
                "strategy_instance": "live_executor",
                "config_hash": "live_executor_v1",
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
        log.warning("kucoin postgres closed trade failed: %s", e)


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
        )
    except Exception:
        return ExecutorState()


def _write_state(path: Path, st: ExecutorState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(st), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


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
) -> int:
    """
    qty = floor(equity * pos_pct * leverage / (mid_price * contract_multiplier))
    """
    equity = float(equity or 0.0)
    pos_pct = float(max(0.0, min(1.0, pos_pct)))
    leverage = float(leverage)
    mid_price = float(mid_price)
    contract_multiplier = float(contract_multiplier)
    if equity <= 0 or pos_pct <= 0 or leverage <= 0 or mid_price <= 0 or contract_multiplier <= 0:
        return 0
    notional = equity * pos_pct * leverage
    per_contract = mid_price * contract_multiplier
    return int(notional // per_contract)


def _resolve_equity(broker: KucoinFuturesBroker) -> float:
    try:
        bal = broker.get_account_balance(currency="USDT")
        return float(bal.get("equity", 0.0) or 0.0)
    except Exception:
        return 0.0


def _resolve_contract_multiplier(broker: KucoinFuturesBroker, symbol: str) -> float:
    try:
        mult = float(broker.get_contract_multiplier(symbol))
        if mult > 0:
            return mult
    except Exception:
        pass
    return float(os.getenv("LIVE_EXECUTOR_CONTRACT_MULTIPLIER", "0.1"))


def _verify_execution_fill_ratio(
    *,
    broker: KucoinFuturesBroker,
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


def _sync_kucoin_stop_loss(
    *,
    broker: KucoinFuturesBroker,
    symbol: str,
    terminal: Optional[Dict[str, Any]],
    terminal_pos: int,
    dry_run: bool,
) -> None:
    """Sync a native stop-market order on KuCoin based on the flip engine terminal state."""
    if not _truthy(os.getenv("KUCOIN_NATIVE_SL_ENABLED", "0")):
        return

    if terminal_pos == 0 or terminal is None:
        try:
            broker.cancel_all_stop_orders(symbol)
        except Exception as e:
            log.warning("kucoin cancel stop orders failed (flat): %s", e)
        return

    stop_price = terminal.get("sl") or terminal.get("ttp")
    if stop_price is None:
        return

    stop_side = "sell" if terminal_pos > 0 else "buy"
    pos_qty = abs(float(broker.get_position(symbol)))
    if pos_qty <= 0:
        return

    if dry_run:
        log.info(
            "DRY_RUN kucoin native SL: side=%s qty=%s stop=%.4f mode=%s",
            stop_side, pos_qty, stop_price, terminal.get("mode"),
        )
        return

    try:
        broker.cancel_all_stop_orders(symbol)
    except Exception as e:
        log.warning("kucoin cancel stop orders pre-place failed: %s", e)

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
            "kucoin native SL placed: order_id=%s side=%s qty=%s stop=%.4f mode=%s",
            order_id, stop_side, pos_qty, stop_price, terminal.get("mode"),
        )
    except Exception as e:
        log.warning(
            "kucoin native SL place failed: side=%s qty=%s stop=%.4f err=%s",
            stop_side, pos_qty, stop_price, e,
        )


def _write_dashboard_levels(symbol: str, terminal: Dict[str, Any], live_pos: Optional[float] = None) -> None:
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
        "side": side,
        "mode": mode,
        "sl": sl,
        "ttp": ttp,
        "entry_px": entry_px,
        "best_fav": _coerce_float(terminal.get("best_fav")),
        "ttp_trail_pct": _resolve_ttp_trail_pct(),
        "entry_bar_ts": int(pd.Timestamp(entry_bar_ts).timestamp()) if entry_bar_ts is not None else None,
        "live_pos": float(live_pos) if live_pos is not None else None,
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


def run_once(
    *,
    broker: KucoinFuturesBroker,
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

    pos_pct = float(os.getenv("LIVE_EXECUTOR_POS_PCT", "0.90"))
    equity = _resolve_equity(broker)
    contract_multiplier = _resolve_contract_multiplier(broker, symbol)
    qty = _qty_from_equity_pct(
        equity=equity,
        pos_pct=pos_pct,
        leverage=leverage,
        mid_price=float(mid),
        contract_multiplier=contract_multiplier,
    )
    log.info(
        "executor sizing: equity=%.2f pos_pct=%.2f leverage=%.1f mid=%.4f mult=%.4f -> qty=%d",
        equity, pos_pct, leverage, mid, contract_multiplier, qty,
    )

    renko_bars = _load_renko_bars(_renko_path(), limit=int(os.getenv("LIVE_RENKO_LIMIT", "4000")))
    signals_df = _load_signals_df(signals_root, symbol)

    gate = _read_live_gate_from_redis(symbol)
    if not gate:
        gate = get_live_gate_state()
    gate_on = int(gate.get("gate_on", 0) or 0)
    exit_engine = "flip" if gate_on == 1 else "tp2"

    if gate_on == 1:
        ev, terminal = _latest_backtest_event(renko_bars=renko_bars, signals_df=signals_df)
    else:
        tp2_params = TP2Params(
            fee_bps=float(os.getenv("LIVE_TP2_FEE_BPS", os.getenv("LIVE_FLIP_FEE_BPS", "0"))),
            tp1_pct=float(os.getenv("LIVE_TP1_PCT", "0.07")),
            tp2_pct=float(os.getenv("LIVE_TP2_PCT", "0.11")),
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

    if terminal_pos == 0:
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
                "venue": "kucoin",
                "strategy": exit_engine,
                "exit_engine": exit_engine,
                "gate_on": gate_on,
                "gate_state": gate,
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
    _write_dashboard_levels(symbol=symbol, terminal=terminal, live_pos=pos)

    want_side = "long" if terminal_pos > 0 else ("short" if terminal_pos < 0 else None)

    action = "hold"
    if current_side == "flat" and want_side == "long":
        action = "enter_long"
    elif current_side == "flat" and want_side == "short":
        action = "enter_short"
    elif current_side == "long" and want_side is None:
        action = "exit_long"
    elif current_side == "short" and want_side is None:
        action = "exit_short"
    elif current_side == "long" and want_side == "short":
        action = "flip_to_short"
    elif current_side == "short" and want_side == "long":
        action = "flip_to_long"
    elif current_side == "long" and want_side == "long":
        if abs(pos) < float(qty):
            action = "scale_long"
    elif current_side == "short" and want_side == "short":
        if abs(pos) < float(qty):
            action = "scale_short"

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
        return state


    event_name = str(ev.get("event", "")) if ev is not None else "none"
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
        return state

    engine_mode = str(terminal.get("mode", ""))
    action_side = want_side if want_side is not None else current_side
    position_before = 1 if current_side == "long" else (-1 if current_side == "short" else 0)
    position_after = 1 if terminal_pos > 0 else (-1 if terminal_pos < 0 else 0)
    if action == "hold":
        position_after = position_before

    state.n_actions += 1
    action_seq = int(state.n_actions)

    _append_action_event(
        strategy="live_executor",
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
        add_qty = max(0.0, float(qty) - abs(float(pos)))
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

        target_qty_for_verify = float(qty)
        target_side_for_verify: Optional[str] = want_side if action.startswith(("enter_", "flip_to_", "scale_")) else None

        if action.startswith("enter_") and want_side is not None:
            res = oms.enter_market(symbol=symbol, side=want_side, qty=float(qty))
            log.info("executor enter result=%s", res)
            if _ok(res):
                details = _details(res)
                state.n_executions += 1
                _append_execution_event(
                    strategy="live_executor",
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

        elif action.startswith("flip_to_") and want_side is not None:
            flat_res = oms.flatten_market(symbol=symbol, side=current_side, qty=abs(float(pos)))
            log.info("executor flip flatten result=%s", flat_res)

            if _ok(flat_res):
                flat_details = _details(flat_res)
                state.n_executions += 1
                _append_execution_event(
                    strategy="live_executor",
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

                pos_after_flat = float(broker.get_position(symbol))
                if abs(pos_after_flat) > 1e-12:
                    log.warning("executor flip aborted: not flat after flatten pos_after=%s", pos_after_flat)
                else:
                    fresh_bid, fresh_ask = broker.get_best_bid_ask(symbol)
                    fresh_mid = (fresh_bid + fresh_ask) / 2.0 if (fresh_bid and fresh_ask) else (fresh_ask or fresh_bid or mid or 0.0)
                    fresh_equity = _resolve_equity(broker)
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
                    else:
                        log.info("executor flip re-size: pre=%s post=%s equity=%s contract_mult=%s", qty, flip_qty, fresh_equity, contract_multiplier)
                        target_qty_for_verify = float(flip_qty)
                        res = oms.enter_market(symbol=symbol, side=want_side, qty=float(flip_qty))
                        log.info("executor flip re-enter result=%s", res)
                        if _ok(res):
                            details = _details(res)
                            state.n_executions += 1
                            _append_execution_event(
                                strategy="live_executor",
                                symbol=symbol,
                                ts_iso=_now_iso(),
                                seq=int(state.n_executions),
                                execution_kind="fill",
                                order_action=_execution_side("buy" if want_side == "long" else "sell", details),
                                reason_code=event_name,
                                position_before=0,
                                position_after=position_after,
                                order_id=str(details.get("order_id") or "") or None,
                                client_oid=str(details.get("client_id") or "") or None,
                                side=_execution_side("buy" if want_side == "long" else "sell", details),
                                qty=_execution_qty(float(flip_qty), details),
                                price=_execution_price(details),
                                reduce_only=False,
                                status=_mode(res) or "fill",
                                reject_reason=None,
                                payload_json={"action": action, "result": details, "event_name": event_name},
                            )
            else:
                log.warning("executor flip aborted: flatten failed")

        elif action.startswith("exit_"):
            res = oms.flatten_market(symbol=symbol, side=current_side, qty=abs(float(pos)))
            log.info("executor exit result=%s", res)
            target_side_for_verify = None
            target_qty_for_verify = abs(float(pos))
            if _ok(res):
                details = _details(res)
                state.n_executions += 1
                _append_execution_event(
                    strategy="live_executor",
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

        elif action.startswith("scale_") and want_side is not None:
            add_qty = max(0.0, float(qty) - abs(float(pos)))
            if add_qty > 0:
                target_qty_for_verify = abs(float(pos)) + float(add_qty)
                res = oms.enter_market(symbol=symbol, side=want_side, qty=add_qty)
                log.info("executor scale result=%s add_qty=%s target_qty=%s pos_before=%s", res, add_qty, qty, pos)
                if _ok(res):
                    details = _details(res)
                    state.n_executions += 1
                    _append_execution_event(
                        strategy="live_executor",
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
            else:
                log.info("executor scale skipped add_qty=0 target_qty=%s pos_before=%s", qty, pos)

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

        _sync_kucoin_stop_loss(
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
    return state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live execution worker (signals -> OMS -> KuCoin)")
    p.add_argument("--symbol", default=os.getenv("LIVE_SYMBOL", "SOL-USDT"))
    p.add_argument("--signals-dir", default=os.getenv("SIGNALS_DIR", "data/signals"))
    p.add_argument("--state-file", default=os.getenv("LIVE_EXECUTOR_STATE", "data/live/live_executor_state.json"))
    p.add_argument("--poll-sec", type=float, default=float(os.getenv("LIVE_EXECUTOR_POLL_SEC", "5")))
    p.add_argument("--once", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    symbol = str(args.symbol).upper()
    signals_root = Path(args.signals_dir)
    state_path = Path(args.state_file)

    live_enabled = _truthy(os.getenv("LIVE_TRADING_ENABLED", "0"))
    dry_run = _truthy(os.getenv("LIVE_EXECUTOR_DRY_RUN", "1"))
    leverage = float(os.getenv("LIVE_EXECUTOR_LEVERAGE", "1"))

    allowlist_raw = os.getenv("LIVE_EXECUTOR_SYMBOL_ALLOWLIST", "SOL-USDT")
    allowlist = {s.strip().upper() for s in allowlist_raw.split(",") if s.strip()}
    if symbol not in allowlist:
        raise RuntimeError(f"symbol '{symbol}' not allowed. Set LIVE_EXECUTOR_SYMBOL_ALLOWLIST.")

    broker = KucoinFuturesBroker()
    oms = MakerFirstOMS(broker=broker, cfg=OmsDefaults())
    st = _read_state(state_path)

    log.info(
        "executor start symbol=%s live_enabled=%s dry_run=%s leverage=%s pos_pct=%s signals=%s",
        symbol,
        live_enabled,
        dry_run,
        leverage,
        os.getenv("LIVE_EXECUTOR_POS_PCT", "0.90"),
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