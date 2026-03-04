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
from quant.strategies.flip_engine import FlipParams, run_flip_state_machine
from quant.strategies.signal_io import read_signals_jsonl
from quant.utils.log import get_logger, log_throttled

log = get_logger("quant.live_executor")

try:
    from quant.execution.live_monitor import ExpectedTrade, record_expected
except Exception:
    # Keep executor runnable even if monitoring module is not packaged in target runtime.
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


def _now_iso() -> str:
    return pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


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
    if not (x == x):  # NaN check
        return None
    return x


def _apply_live_ttp_guard(
    terminal: Dict[str, Any],
    *,
    live_pos: float,
    live_mid: float,
    ttp_trail_pct: float,
) -> Dict[str, Any]:
    """
    Keep TTP consistent with configured trail distance relative to latest live quote.
    This prevents stale-renko drift where dashboard TTP can lag far from market.
    """
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
        # Do not allow stale TTP below the live-trail floor.
        floor_ttp = float(mid * (1.0 - trail))
        out["ttp"] = floor_ttp if cur_ttp is None else float(max(cur_ttp, floor_ttp))
    else:
        # Do not allow stale TTP above the live-trail cap.
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
        )
    except Exception:
        return ExecutorState()


def _write_state(path: Path, st: ExecutorState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(st), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _latest_signal(signals_root: Path, symbol: str) -> Optional[Dict[str, Any]]:
    # Accept both SOLUSDT and SOL-USDT style directories.
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

    # Read newest files first, newest line wins.
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
    """Snap signal timestamps to the nearest renko bar within *tolerance*.

    The live signal worker and the renko cache updater build renko
    independently, so a signal's ``ts`` may not exactly match any bar in
    ``renko_latest.parquet``.  Without snapping, ``align_impulses_exact``
    silently drops the signal and no trade is activated.
    """
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
    renko_bars: pd.DataFrame, signals_df: pd.DataFrame,
) -> Tuple[Optional[pd.Series], Dict[str, Any]]:
    """Returns (latest_event_row, terminal_state_dict)."""
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


def _qty_from_max_eur(max_eur: float, leverage: float, mid_price: float, contract_multiplier: float = 1.0) -> int:
    if max_eur <= 0 or leverage <= 0 or mid_price <= 0 or contract_multiplier <= 0:
        return 0
    notional = float(max_eur) * float(leverage)
    per_contract_notional = float(mid_price) * float(contract_multiplier)
    return int(notional // per_contract_notional)


def _resolve_max_eur(
    broker: KucoinFuturesBroker,
    configured_max_eur: float,
    *,
    use_full_equity: bool,
    equity_fraction: float,
) -> float:
    try:
        bal = broker.get_account_balance(currency="USDT")
        eq = float(bal.get("equity", 0.0) or 0.0)
        if eq > 0:
            frac = float(max(0.0, min(1.0, equity_fraction)))
            return float(eq * frac)
    except Exception:
        pass
    return float(configured_max_eur)


def _verify_execution_fill_ratio(
    *,
    broker: KucoinFuturesBroker,
    symbol: str,
    action: str,
    target_side: Optional[str],
    target_qty: float,
    min_ratio: float,
) -> None:
    """
    Lightweight runtime checker:
    warn if resulting position deviates materially from intended target.
    """
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
    """Write current flip-engine state to execution_state.json for the dashboard."""
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
            # After manual flatten (or any external close), clear stale in-position levels.
            side = None
            entry_px = None
            entry_bar_ts = None
            sl = None
            ttp = None
        else:
            live_side = "long" if lp > 0 else "short"
            if side != live_side:
                # Live position is source-of-truth for side; avoid showing stale opposite state.
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
    max_eur: float,
    leverage: float,
) -> ExecutorState:
    renko_path = Path(os.getenv("LIVE_EXECUTOR_RENKO_PARQUET", os.getenv("DASHBOARD_RENKO_PARQUET", "data/live/renko_latest.parquet")))
    renko_bars = _load_renko_bars(renko_path, limit=int(os.getenv("LIVE_EXECUTOR_RENKO_LIMIT", "4000")))
    signals_df = _load_signals_df(signals_root=signals_root, symbol=symbol)
    ev, terminal_state = _latest_backtest_event(renko_bars=renko_bars, signals_df=signals_df)
    _used_fallback = False
    use_fallback_signal = (ev is None)
    if ev is not None:
        ev_sig = _event_sig(ev)
        if state.last_event_sig == ev_sig:
            use_fallback_signal = True

    if use_fallback_signal:
        sig = _latest_signal(signals_root=signals_root, symbol=symbol)
        if sig is None:
            log_throttled(
                log,
                logging.INFO,
                f"executor_no_signal:{symbol}",
                float(os.getenv("LIVE_EXECUTOR_NO_SIGNAL_LOG_SEC", "60")),
                "executor no signal yet symbol=%s",
                symbol,
            )
            return state
        ts = sig["ts"]
        sig_v = int(sig["signal"])
        ts_iso = ts.isoformat()
        if state.last_signal_ts == ts_iso and state.last_signal_value == sig_v:
            return state
        event = "fallback_signal"
        event_side = sig_v
        ev_sig = f"{ts_iso}|0|fallback_signal|{sig_v}"
        _used_fallback = True
        if ev is not None:
            log.info(
                "executor fallback: flip-engine event stale, using direct signal ts=%s sig=%s symbol=%s",
                ts_iso, sig_v, symbol,
            )
    else:
        ts = pd.Timestamp(ev["ts"])
        ts_iso = ts.isoformat()
        event = str(ev["event"])
        event_side = int(ev["side"])
        sig_v = 1 if event_side > 0 else -1

    bid, ask = broker.get_best_bid_ask(symbol)
    mid = (bid + ask) / 2.0 if (bid and ask) else (ask or bid or 0.0)
    use_full_equity = _truthy(os.getenv("LIVE_EXECUTOR_USE_FULL_EQUITY", "1"))
    equity_fraction = float(os.getenv("LIVE_EXECUTOR_EQUITY_FRACTION", "1.0"))
    sizing_max_eur = _resolve_max_eur(
        broker=broker,
        configured_max_eur=max_eur,
        use_full_equity=use_full_equity,
        equity_fraction=equity_fraction,
    )
    contract_multiplier = 1.0
    try:
        multiplier_getter = getattr(broker, "get_contract_multiplier", None)
        if callable(multiplier_getter):
            contract_multiplier = float(multiplier_getter(symbol))
    except Exception as e:
        log.warning("executor failed to fetch contract multiplier symbol=%s err=%s", symbol, e)
        contract_multiplier = 1.0
    qty = _qty_from_max_eur(
        max_eur=sizing_max_eur,
        leverage=leverage,
        mid_price=float(mid),
        contract_multiplier=float(contract_multiplier),
    )
    if qty <= 0:
        log_throttled(
            log,
            logging.WARNING,
            f"executor_qty_zero:{symbol}",
            float(os.getenv("LIVE_EXECUTOR_LOG_THROTTLE_SEC", "60")),
            "executor qty=0 (configured_max_eur=%s sizing_max_eur=%s leverage=%s mid=%s contract_multiplier=%s use_full_equity=%s) -> skip",
            max_eur,
            sizing_max_eur,
            leverage,
            mid,
            contract_multiplier,
            use_full_equity,
        )
        state.last_signal_ts = ts_iso
        state.last_signal_value = sig_v
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
    target_side = (-event_side if event in ("signal_flip_exit", "tp_exit") else event_side)
    want_side = "long" if target_side > 0 else "short"
    current_side = "long" if pos > 0 else ("short" if pos < 0 else "flat")

    action = None
    if event == "entry":
        action = f"enter_{want_side}" if abs(pos) < 1e-12 else "hold"
    elif event in ("signal_flip_exit", "tp_exit"):
        if abs(pos) < 1e-12:
            action = f"enter_{want_side}"
        else:
            action = f"flip_to_{want_side}" if ((pos > 0 and target_side < 0) or (pos < 0 and target_side > 0)) else "hold"
    elif event in ("sl_exit", "be_exit"):
        action = f"exit_{current_side}" if abs(pos) > 1e-12 else "hold"
    else:
        action = f"enter_{want_side}" if abs(pos) < 1e-12 else (f"flip_to_{want_side}" if ((pos > 0 and sig_v < 0) or (pos < 0 and sig_v > 0)) else "hold")
    # Keep exposure near target size while in-position.
    if action == "hold" and current_side == want_side and abs(pos) + 1e-12 < float(qty):
        action = f"scale_{want_side}"

    # Record expected trade intents for fills-reason mapping in dashboard.
    exp_side: Optional[str] = None
    exp_action: Optional[str] = None
    exp_qty: Optional[float] = None
    exp_note: Optional[str] = None
    if action in ("enter_long", "enter_short"):
        exp_side = want_side
        exp_action = "entry"
        exp_qty = float(qty)
        exp_note = f"executor action={action} event={event} current={current_side}"
    elif action in ("flip_to_long", "flip_to_short"):
        exp_side = want_side
        exp_action = "exit_flip"
        exp_qty = float(qty)
        exp_note = f"executor action={action} event={event} current={current_side}"
    elif action.startswith("exit_") and abs(pos) > 1e-12:
        exp_side = current_side
        if event in ("sl_exit", "be_exit"):
            exp_action = "exit_sl"
        elif event in ("tp_exit",):
            exp_action = "exit_tp"
        else:
            exp_action = "exit_flip"
        exp_qty = abs(float(pos))
        exp_note = f"executor action={action} event={event} current={current_side}"
    elif action.startswith("scale_"):
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

        if action.startswith("enter_"):
            res = oms.enter(symbol=symbol, side=want_side, qty=float(qty))
            log.info("executor enter result=%s", res)
        elif action.startswith("flip_to_"):
            flat_res = oms.exit_tp_or_flip(symbol=symbol, side=current_side, qty=abs(float(pos)), flip_to=None)
            log.info("executor flip flatten result=%s", flat_res)
            if bool(getattr(flat_res, "ok", False)):
                flip_sizing = _resolve_max_eur(
                    broker=broker, configured_max_eur=max_eur,
                    use_full_equity=True, equity_fraction=equity_fraction,
                )
                flip_qty = _qty_from_max_eur(
                    max_eur=flip_sizing, leverage=leverage,
                    mid_price=float(mid), contract_multiplier=float(contract_multiplier),
                )
                if flip_qty <= 0:
                    flip_qty = qty
                log.info("executor flip re-size: pre=%s post=%s", qty, flip_qty)
                res = oms.enter(symbol=symbol, side=want_side, qty=float(flip_qty))
                log.info("executor flip re-enter result=%s", res)
            else:
                log.warning("executor flip aborted: flatten failed")
        elif action.startswith("exit_"):
            res = oms.exit_sl(symbol=symbol, side=current_side, qty=abs(float(pos)))
            log.info("executor exit result=%s", res)
        elif action.startswith("scale_"):
            add_qty = max(0.0, float(qty) - abs(float(pos)))
            if add_qty > 0:
                res = oms.enter(symbol=symbol, side=want_side, qty=add_qty)
                log.info("executor scale result=%s add_qty=%s target_qty=%s pos_before=%s", res, add_qty, qty, pos)
            else:
                log.info("executor scale skipped add_qty=0 target_qty=%s pos_before=%s", qty, pos)
        else:
            log.info("executor hold symbol=%s pos=%s sig=%s event=%s", symbol, pos, sig_v, event)

        _verify_execution_fill_ratio(
            broker=broker,
            symbol=symbol,
            action=action or "hold",
            target_side=want_side if action.startswith(("enter_", "flip_to_", "scale_")) else None,
            target_qty=float(qty),
            min_ratio=float(os.getenv("LIVE_EXECUTOR_MIN_FILL_RATIO", "0.95")),
        )

    state.last_signal_ts = ts_iso
    state.last_signal_value = sig_v
    if not _used_fallback:
        state.last_event_sig = ev_sig
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
    max_eur = float(os.getenv("LIVE_EXECUTOR_MAX_EUR", "20"))
    leverage = float(os.getenv("LIVE_EXECUTOR_LEVERAGE", "1"))

    allowlist_raw = os.getenv("LIVE_EXECUTOR_SYMBOL_ALLOWLIST", "SOL-USDT")
    allowlist = {s.strip().upper() for s in allowlist_raw.split(",") if s.strip()}
    if symbol not in allowlist:
        raise RuntimeError(f"symbol '{symbol}' not allowed. Set LIVE_EXECUTOR_SYMBOL_ALLOWLIST.")

    broker = KucoinFuturesBroker()
    oms = MakerFirstOMS(broker=broker, cfg=OmsDefaults())
    st = _read_state(state_path)

    log.info(
        "executor start symbol=%s live_enabled=%s dry_run=%s max_eur=%s leverage=%s signals=%s",
        symbol,
        live_enabled,
        dry_run,
        max_eur,
        leverage,
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
                max_eur=max_eur,
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
