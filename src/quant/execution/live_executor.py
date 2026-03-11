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
from quant.execution.kucoin_futures import KucoinFuturesBroker
from quant.execution.oms import MakerFirstOMS, OmsDefaults
from quant.execution.event_builders import build_action_event
from quant.execution.event_log import append_event_jsonl
from quant.execution.event_store import insert_action_event
from quant.strategies.flip_engine import FlipParams, run_flip_state_machine
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


def _snap_signals_to_bars(
    signals_df: pd.DataFrame,
    bars: pd.DataFrame,
    tolerance: pd.Timedelta = pd.Timedelta(minutes=5),
) -> pd.DataFrame:
    if signals_df.empty or bars.empty:
        return signals_df

    sig = signals_df.copy()
    sig["ts"] = pd.to_datetime(sig["ts"], utc=True, errors="coerce")
    sig = sig.dropna(subset=["ts"])
    if sig.empty:
        return sig

    bar_times = pd.DatetimeIndex(pd.to_datetime(bars["ts"], utc=True, errors="coerce")).dropna()
    if len(bar_times) == 0:
        return sig

    bar_times_sorted = bar_times.sort_values()
    snapped: list = []
    n_snapped = 0
    for t in sig["ts"]:
        if t in bar_times_sorted:
            snapped.append(t)
            continue
        idx = bar_times_sorted.searchsorted(t)
        candidates = []
        if idx > 0:
            candidates.append(bar_times_sorted[idx - 1])
        if idx < len(bar_times_sorted):
            candidates.append(bar_times_sorted[idx])
        if candidates:
            nearest = min(candidates, key=lambda bt: abs(bt - t))
            if abs(nearest - t) <= tolerance:
                snapped.append(nearest)
                n_snapped += 1
                continue
        snapped.append(t)

    if n_snapped > 0:
        log.info("executor snapped %d/%d signal timestamps to nearest renko bar", n_snapped, len(sig))

    sig["ts"] = snapped
    return sig.drop_duplicates("ts", keep="last").reset_index(drop=True)


def _latest_backtest_event(
    renko_bars: pd.DataFrame,
    signals_df: pd.DataFrame,
) -> Tuple[Optional[pd.Series], Dict[str, Any]]:
    if renko_bars.empty or signals_df.empty:
        return None, {}
    signals_df = _snap_signals_to_bars(signals_df, renko_bars)
    params = FlipParams(
        fee_bps=float(os.getenv("LIVE_FLIP_FEE_BPS", "0")),
        ttp_trail_pct=float(os.getenv("LIVE_FLIP_TTP_TRAIL_PCT", "0.012")),
        min_sl_pct=float(os.getenv("LIVE_FLIP_MIN_SL_PCT", "0.015")),
        max_sl_pct=float(os.getenv("LIVE_FLIP_MAX_SL_PCT", "0.030")),
        swing_lookback=int(os.getenv("LIVE_FLIP_SWING_LOOKBACK", "50")),
        be_trigger_pct=float(os.getenv("LIVE_FLIP_BE_TRIGGER_PCT", "0")),
        be_offset_pct=float(os.getenv("LIVE_FLIP_BE_OFFSET_PCT", "0")),
    )
    _, events, terminal = run_flip_state_machine(bars=renko_bars, signals_df=signals_df, params=params, regime_on=None)
    if events is None or events.empty:
        return None, terminal
    events = events.sort_values(["ts", "seq"]).reset_index(drop=True)
    return events.iloc[-1], terminal


def _qty_from_available_balance(
    *,
    available_usdt: float,
    leverage: float,
    mid_price: float,
    contract_multiplier: float = 1.0,
    available_fraction: float = 0.95,
) -> int:
    if available_usdt <= 0 or leverage <= 0 or mid_price <= 0 or contract_multiplier <= 0:
        return 0
    frac = float(max(0.0, min(1.0, available_fraction)))
    usable_margin = float(available_usdt) * frac
    notional = usable_margin * float(leverage)
    per_contract_notional = float(mid_price) * float(contract_multiplier)
    return int(notional // per_contract_notional)


def _resolve_available_usdt(
    broker: KucoinFuturesBroker,
    *,
    available_fraction: float,
) -> float:
    try:
        bal = broker.get_account_balance(currency="USDT")
        available = float(bal.get("available", 0.0) or 0.0)
        if available > 0:
            frac = float(max(0.0, min(1.0, available_fraction)))
            return float(available * frac)
    except Exception:
        pass
    return 0.0


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
    entry_px = terminal.get("entry_px")
    entry_bar_ts = terminal.get("entry_bar_ts")
    sl = terminal.get("sl")
    ttp = terminal.get("ttp")

    if live_pos is not None:
        lp = float(live_pos)
        if abs(lp) <= 1e-12:
            side = None
            entry_px = None
            entry_bar_ts = None
            sl = None
            ttp = None
        else:
            live_side = "long" if lp > 0 else "short"
            if side != live_side:
                side = live_side
                entry_px = None
                entry_bar_ts = None
                sl = None
                ttp = None
            elif entry_px is None:
                log_throttled(
                    log,
                    logging.WARNING,
                    f"executor_skip_state_overwrite:{symbol}",
                    float(os.getenv("LIVE_EXECUTOR_LOG_THROTTLE_SEC", "120")),
                    "executor skip dashboard-state overwrite: live_pos=%s terminal_side=%s terminal_entry_px=%s",
                    live_pos,
                    side,
                    entry_px,
                )
                return

    write_execution_state({
        "symbol": symbol,
        "side": side,
        "mode": terminal.get("mode"),
        "sl": sl,
        "ttp": ttp,
        "entry_px": entry_px,
        "best_fav": terminal.get("best_fav"),
        "ttp_trail_pct": _resolve_ttp_trail_pct(),
        "entry_bar_ts": int(pd.Timestamp(entry_bar_ts).timestamp()) if entry_bar_ts is not None else None,
    })


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
    now_ts = _now_utc()

    renko_path = Path(os.getenv("LIVE_EXECUTOR_RENKO_PARQUET", os.getenv("DASHBOARD_RENKO_PARQUET", "data/live/renko_latest.parquet")))
    renko_bars = _load_renko_bars(renko_path, limit=int(os.getenv("LIVE_EXECUTOR_RENKO_LIMIT", "4000")))
    signals_df = _load_signals_df(signals_root=signals_root, symbol=symbol)

    ev, terminal_state = _latest_backtest_event(
        renko_bars=renko_bars,
        signals_df=signals_df,
    )

    terminal_pos = int(terminal_state.get("pos", 0)) if terminal_state else 0
    terminal_entry_ts = terminal_state.get("entry_bar_ts") if terminal_state else None
    terminal_sig = f"{terminal_pos}|{terminal_entry_ts}"

    fallback_used = False
    fallback_sig_ts_iso: Optional[str] = None
    fallback_sig_v: Optional[int] = None

    if terminal_pos == 0 and (ev is None or (terminal_state and terminal_state.get("pos", 0) == 0)):
        max_age = float(os.getenv("LIVE_EXECUTOR_FALLBACK_MAX_SIGNAL_AGE_SEC", "30"))
        require_monotonic = _truthy(os.getenv("LIVE_EXECUTOR_FALLBACK_REQUIRE_MONOTONIC", "1"))
        sig = _latest_signal(signals_root=signals_root, symbol=symbol)
        if sig is not None:
            sig_v = int(sig["signal"])
            sig_ts = pd.Timestamp(sig["ts"])
            sig_ts_iso = sig_ts.isoformat()
            age_sec = float((now_ts - sig_ts).total_seconds())
            if age_sec < 0:
                sig = None
            else:
                is_newer = True
                if require_monotonic and state.last_signal_ts:
                    prev = _safe_ts(state.last_signal_ts)
                    if prev is not None:
                        is_newer = sig_ts > prev
                is_dup = (state.last_signal_ts == sig_ts_iso and state.last_signal_value == sig_v)
                if (age_sec <= max_age) and is_newer and (not is_dup):
                    terminal_pos = 1 if sig_v > 0 else -1
                    terminal_sig = f"{terminal_pos}|fallback|{sig_ts_iso}"
                    fallback_used = True
                    fallback_sig_ts_iso = sig_ts_iso
                    fallback_sig_v = terminal_pos
                    log.info(
                        "executor fallback: using direct signal ts=%s sig=%s age_sec=%.3f symbol=%s",
                        sig_ts_iso, terminal_pos, age_sec, symbol,
                    )

    if terminal_pos == 0 and ev is None:
        sig_check = _latest_signal(signals_root=signals_root, symbol=symbol)
        if sig_check is None:
            log_throttled(
                log,
                logging.INFO,
                f"executor_no_signal:{symbol}",
                float(os.getenv("LIVE_EXECUTOR_NO_SIGNAL_LOG_SEC", "60")),
                "executor no signal yet symbol=%s",
                symbol,
            )
            return state

    bid, ask = broker.get_best_bid_ask(symbol)
    mid = (bid + ask) / 2.0 if (bid and ask) else (ask or bid or 0.0)

    available_fraction = float(os.getenv("LIVE_EXECUTOR_AVAILABLE_FRACTION", "0.95"))
    available_usdt = _resolve_available_usdt(
        broker=broker,
        available_fraction=available_fraction,
    )

    contract_multiplier = 1.0
    try:
        multiplier_getter = getattr(broker, "get_contract_multiplier", None)
        if callable(multiplier_getter):
            contract_multiplier = float(multiplier_getter(symbol))
    except Exception as e:
        log.warning("executor failed to fetch contract multiplier symbol=%s err=%s", symbol, e)
        contract_multiplier = 1.0

    qty = _qty_from_available_balance(
        available_usdt=available_usdt,
        leverage=leverage,
        mid_price=float(mid),
        contract_multiplier=float(contract_multiplier),
        available_fraction=1.0,
    )
    if qty <= 0:
        log_throttled(
            log,
            logging.WARNING,
            f"executor_qty_zero:{symbol}",
            float(os.getenv("LIVE_EXECUTOR_LOG_THROTTLE_SEC", "60")),
            "executor qty=0 (available_usdt=%s leverage=%s mid=%s contract_multiplier=%s available_fraction=%s) -> skip",
            available_usdt,
            leverage,
            mid,
            contract_multiplier,
            available_fraction,
        )
        state.last_terminal_sig = terminal_sig
        state.last_action = "skip_qty_0"
        return state

    pos = float(broker.get_position(symbol))

    terminal_state = _apply_live_ttp_guard(
        terminal_state,
        live_pos=pos,
        live_mid=float(mid),
        ttp_trail_pct=_resolve_ttp_trail_pct(),
    )
    _write_dashboard_levels(symbol, terminal_state, live_pos=pos)

    current_side = "long" if pos > 0 else ("short" if pos < 0 else "flat")
    want_side: Optional[str] = None
    if terminal_pos > 0:
        want_side = "long"
    elif terminal_pos < 0:
        want_side = "short"

    action: str
    if terminal_pos > 0:
        if abs(pos) < 1e-12:
            action = "enter_long"
        elif pos < 0:
            action = "flip_to_long"
        else:
            action = "hold"
    elif terminal_pos < 0:
        if abs(pos) < 1e-12:
            action = "enter_short"
        elif pos > 0:
            action = "flip_to_short"
        else:
            action = "hold"
    else:
        if abs(pos) > 1e-12:
            action = f"exit_{current_side}"
        else:
            action = "hold"

    if action == "hold" and want_side is not None and current_side == want_side and abs(pos) + 1e-12 < float(qty):
        action = f"scale_{want_side}"

        state.n_actions += 1
    event_seq = int(state.n_actions)

    if action in ("enter_long", "flip_to_long", "scale_long"):
        action_side = "long"
        position_after = 1
    elif action in ("enter_short", "flip_to_short", "scale_short"):
        action_side = "short"
        position_after = -1
    elif action.startswith("exit_"):
        action_side = "flat"
        position_after = 0
    else:
        action_side = current_side
        position_after = terminal_pos if terminal_pos != 0 else (1 if pos > 0 else -1 if pos < 0 else 0)

    _append_action_event(
        strategy="live_executor",
        symbol=symbol,
        ts_iso=ts_iso,
        seq=event_seq,
        engine_action=action,
        action_side=action_side,
        reason_code=event_name,
        position_before=(1 if pos > 0 else -1 if pos < 0 else 0),
        position_after=position_after,
        engine_mode_before=str(terminal_state.get("mode", "UNKNOWN") if terminal_state else "UNKNOWN"),
        engine_mode_after=str(terminal_state.get("mode", "UNKNOWN") if terminal_state else "UNKNOWN"),
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
        },
    )

    event_sig = _event_sig(ev) if ev is not None else None
    if event_sig is not None and event_sig == state.last_event_sig and action == "hold":
        return state

    event_name = str(ev.get("event")) if isinstance(ev, (dict, pd.Series)) and ev is not None and "event" in ev else "none"

    ts_iso = now_ts.isoformat()

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

        target_qty_for_verify = float(qty)
        target_side_for_verify: Optional[str] = want_side if action.startswith(("enter_", "flip_to_", "scale_")) else None

        if action.startswith("enter_") and want_side is not None:
            res = oms.enter(symbol=symbol, side=want_side, qty=float(qty))
            log.info("executor enter result=%s", res)

        elif action.startswith("flip_to_") and want_side is not None:
            flat_res = oms.exit_tp_or_flip(symbol=symbol, side=current_side, qty=abs(float(pos)), flip_to=None)
            log.info("executor flip flatten result=%s", flat_res)

            if _ok(flat_res):
                pos_after_flat = float(broker.get_position(symbol))
                if abs(pos_after_flat) > 1e-12:
                    log.warning("executor flip aborted: not flat after flatten pos_after=%s", pos_after_flat)
                else:
                    fresh_bid, fresh_ask = broker.get_best_bid_ask(symbol)
                    fresh_mid = (fresh_bid + fresh_ask) / 2.0 if (fresh_bid and fresh_ask) else (fresh_ask or fresh_bid or mid or 0.0)
                    fresh_available_usdt = _resolve_available_usdt(
                        broker=broker,
                        available_fraction=available_fraction,
                    )
                    flip_qty = _qty_from_available_balance(
                        available_usdt=fresh_available_usdt,
                        leverage=leverage,
                        mid_price=float(fresh_mid),
                        contract_multiplier=float(contract_multiplier),
                        available_fraction=1.0,
                    )
                    if flip_qty <= 0:
                        log.warning(
                            "executor flip aborted: qty=0 after flatten available_usdt=%s leverage=%s mid=%s contract_multiplier=%s",
                            fresh_available_usdt,
                            leverage,
                            fresh_mid,
                            contract_multiplier,
                        )
                    else:
                        log.info("executor flip re-size: pre=%s post=%s available_usdt=%s", qty, flip_qty, fresh_available_usdt)
                        target_qty_for_verify = float(flip_qty)
                        res = oms.enter(symbol=symbol, side=want_side, qty=float(flip_qty))
                        log.info("executor flip re-enter result=%s", res)
            else:
                log.warning("executor flip aborted: flatten failed")

        elif action.startswith("exit_"):
            res = oms.exit_sl(symbol=symbol, side=current_side, qty=abs(float(pos)))
            log.info("executor exit result=%s", res)
            target_side_for_verify = None
            target_qty_for_verify = abs(float(pos))

        elif action.startswith("scale_") and want_side is not None:
            add_qty = max(0.0, float(qty) - abs(float(pos)))
            if add_qty > 0:
                target_qty_for_verify = abs(float(pos)) + float(add_qty)
                res = oms.enter(symbol=symbol, side=want_side, qty=add_qty)
                log.info("executor scale result=%s add_qty=%s target_qty=%s pos_before=%s", res, add_qty, qty, pos)
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
    state.n_actions += 1
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
        "executor start symbol=%s live_enabled=%s dry_run=%s leverage=%s signals=%s available_fraction=%s",
        symbol,
        live_enabled,
        dry_run,
        leverage,
        str(signals_root),
        os.getenv("LIVE_EXECUTOR_AVAILABLE_FRACTION", "0.95"),
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