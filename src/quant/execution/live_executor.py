from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from quant.execution.execution_state import write_execution_state
from quant.execution.kucoin_futures import KucoinFuturesBroker
from quant.execution.oms import MakerFirstOMS, OmsDefaults
from quant.utils.log import get_logger

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


def _now_iso() -> str:
    return pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class ExecutorState:
    last_signal_ts: Optional[str] = None
    last_signal_value: Optional[int] = None
    last_action: Optional[str] = None
    n_actions: int = 0


@dataclass
class TrailingState:
    side: Optional[str] = None
    mode: Optional[str] = None
    entry_ref: Optional[float] = None
    best_fav: Optional[float] = None
    last_updated: Optional[str] = None


def _read_state(path: Path) -> ExecutorState:
    if not path.exists():
        return ExecutorState()
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return ExecutorState(
            last_signal_ts=d.get("last_signal_ts"),
            last_signal_value=d.get("last_signal_value"),
            last_action=d.get("last_action"),
            n_actions=int(d.get("n_actions", 0)),
        )
    except Exception:
        return ExecutorState()


def _write_state(path: Path, st: ExecutorState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(st), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _read_trailing_state(path: Path) -> TrailingState:
    if not path.exists():
        return TrailingState()
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return TrailingState(
            side=d.get("side"),
            mode=d.get("mode"),
            entry_ref=float(d["entry_ref"]) if d.get("entry_ref") is not None else None,
            best_fav=float(d["best_fav"]) if d.get("best_fav") is not None else None,
            last_updated=d.get("last_updated"),
        )
    except Exception:
        return TrailingState()


def _write_trailing_state(path: Path, st: TrailingState) -> None:
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
        all_files.extend((d / "countertrend").glob("*.jsonl"))
        all_files.extend((d / "trendfollower").glob("*.jsonl"))

    for fp in reversed(sorted(set(all_files))):
        try:
            with fp.open("r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            for ln in reversed(lines):
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                ts = _safe_ts(obj.get("ts"))
                if ts is None:
                    continue
                try:
                    sig_i = int(obj.get("signal"))
                except Exception:
                    continue
                if sig_i == 0:
                    continue
                return {
                    "ts": ts,
                    "signal": 1 if sig_i > 0 else -1,
                    "raw": obj,
                }
        except Exception:
            continue
    return None


def _qty_from_max_eur(max_eur: float, leverage: float, mid_price: float) -> int:
    if max_eur <= 0 or leverage <= 0 or mid_price <= 0:
        return 0
    notional = float(max_eur) * float(leverage)
    return int(notional // float(mid_price))


def _update_trailing_levels(
    *,
    broker: KucoinFuturesBroker,
    symbol: str,
    trailing: TrailingState,
    pos: float,
    ttp_trail_pct: float,
    sl_wait_pct: float,
) -> TrailingState:
    if abs(pos) < 1e-12:
        return TrailingState()

    bid, ask = broker.get_best_bid_ask(symbol)
    px = (bid + ask) / 2.0 if (bid and ask) else (ask or bid or 0.0)
    if px <= 0:
        return trailing

    side = "long" if pos > 0 else "short"
    if trailing.side != side or trailing.entry_ref is None or trailing.best_fav is None:
        trailing = TrailingState(
            side=side,
            mode="TTP",
            entry_ref=float(px),
            best_fav=float(px),
            last_updated=_now_iso(),
        )

    best = float(trailing.best_fav or px)
    if side == "long":
        best = max(best, float(px))
    else:
        best = min(best, float(px))
    trailing.best_fav = best
    trailing.last_updated = _now_iso()

    if side == "long":
        ttp = best * (1.0 - float(ttp_trail_pct))
        sl = float(trailing.entry_ref) * (1.0 - float(sl_wait_pct))
        tp1 = float(trailing.entry_ref) * (1.0 + float(ttp_trail_pct))
        tp2 = float(trailing.entry_ref) * (1.0 + float(2.0 * ttp_trail_pct))
    else:
        ttp = best * (1.0 + float(ttp_trail_pct))
        sl = float(trailing.entry_ref) * (1.0 + float(sl_wait_pct))
        tp1 = float(trailing.entry_ref) * (1.0 - float(ttp_trail_pct))
        tp2 = float(trailing.entry_ref) * (1.0 - float(2.0 * ttp_trail_pct))

    write_execution_state(
        {
            "symbol": symbol,
            "mode": trailing.mode or "TTP",
            "side": side,
            "signal": 1 if side == "long" else -1,
            "sl": float(sl),
            "ttp": float(ttp),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "entry_ref": float(trailing.entry_ref),
            "best_fav": float(best),
            "ts": trailing.last_updated,
        }
    )
    return trailing


def run_once(
    *,
    broker: KucoinFuturesBroker,
    oms: MakerFirstOMS,
    symbol: str,
    signals_root: Path,
    state: ExecutorState,
    trailing: TrailingState,
    live_enabled: bool,
    dry_run: bool,
    max_eur: float,
    leverage: float,
    ttp_trail_pct: float,
    sl_wait_pct: float,
) -> tuple[ExecutorState, TrailingState]:
    pos = float(broker.get_position(symbol))
    trailing = _update_trailing_levels(
        broker=broker,
        symbol=symbol,
        trailing=trailing,
        pos=pos,
        ttp_trail_pct=ttp_trail_pct,
        sl_wait_pct=sl_wait_pct,
    )

    sig = _latest_signal(signals_root=signals_root, symbol=symbol)
    if sig is None:
        log.info("executor no signal yet symbol=%s", symbol)
        return state, trailing

    ts = sig["ts"]
    sig_v = int(sig["signal"])
    ts_iso = ts.isoformat()
    if state.last_signal_ts == ts_iso and state.last_signal_value == sig_v:
        return state, trailing

    bid, ask = broker.get_best_bid_ask(symbol)
    mid = (bid + ask) / 2.0 if (bid and ask) else (ask or bid or 0.0)
    qty = _qty_from_max_eur(max_eur=max_eur, leverage=leverage, mid_price=float(mid))
    if qty <= 0:
        log.warning("executor qty=0 (max_eur=%s leverage=%s mid=%s) -> skip", max_eur, leverage, mid)
        state.last_signal_ts = ts_iso
        state.last_signal_value = sig_v
        state.last_action = "skip_qty_0"
        return state, trailing

    want_side = "long" if sig_v > 0 else "short"
    current_side = "long" if pos > 0 else ("short" if pos < 0 else "flat")
    if abs(pos) < 1e-12:
        action = f"enter_{want_side}"
    elif (pos > 0 and sig_v < 0) or (pos < 0 and sig_v > 0):
        action = f"flip_to_{want_side}"
    else:
        action = "hold"

    if action in ("enter_long", "enter_short", "flip_to_long", "flip_to_short"):
        record_expected(
            ExpectedTrade(
                ts=ts_iso,
                symbol=symbol,
                side=want_side,
                action=("entry" if action.startswith("enter_") else "exit_flip"),
                qty=float(qty),
                expected_px=float(mid) if mid > 0 else None,
                note=f"executor action={action} current={current_side}",
            )
        )

    if not live_enabled:
        log.warning("executor LIVE_TRADING_ENABLED=0 -> simulated action=%s", action)
    elif dry_run:
        log.warning("executor DRY_RUN=1 -> simulated action=%s", action)
    else:
        if action.startswith("enter_"):
            res = oms.enter(symbol=symbol, side=want_side, qty=float(qty))
            log.info("executor enter result=%s", res)
        elif action.startswith("flip_to_"):
            res = oms.exit_tp_or_flip(symbol=symbol, side=current_side, qty=abs(float(pos)), flip_to=want_side)
            log.info("executor flip result=%s", res)
        else:
            log.info("executor hold symbol=%s pos=%s sig=%s", symbol, pos, sig_v)

    state.last_signal_ts = ts_iso
    state.last_signal_value = sig_v
    state.last_action = action
    state.n_actions += 1
    return state, trailing


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live execution worker (signals -> OMS -> KuCoin)")
    p.add_argument("--symbol", default=os.getenv("LIVE_SYMBOL", "SOL-USDT"))
    p.add_argument("--signals-dir", default=os.getenv("SIGNALS_DIR", "data/signals"))
    p.add_argument("--state-file", default=os.getenv("LIVE_EXECUTOR_STATE", "data/live/live_executor_state.json"))
    p.add_argument("--trailing-state-file", default=os.getenv("LIVE_TRAILING_STATE", "data/live/live_trailing_state.json"))
    p.add_argument("--poll-sec", type=float, default=float(os.getenv("LIVE_EXECUTOR_POLL_SEC", "5")))
    p.add_argument("--once", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    symbol = str(args.symbol).upper()
    signals_root = Path(args.signals_dir)
    state_path = Path(args.state_file)
    trailing_state_path = Path(args.trailing_state_file)

    live_enabled = _truthy(os.getenv("LIVE_TRADING_ENABLED", "0"))
    dry_run = _truthy(os.getenv("LIVE_EXECUTOR_DRY_RUN", "1"))
    max_eur = float(os.getenv("LIVE_EXECUTOR_MAX_EUR", "20"))
    leverage = float(os.getenv("LIVE_EXECUTOR_LEVERAGE", "1"))
    ttp_trail_pct = float(os.getenv("LIVE_TTP_TRAIL_PCT", "0.012"))
    sl_wait_pct = float(os.getenv("LIVE_WAIT_SL_PCT", "0.02"))

    allowlist_raw = os.getenv("LIVE_EXECUTOR_SYMBOL_ALLOWLIST", "SOLUSDT,SOL-USDT")
    allowlist = {s.strip().upper() for s in allowlist_raw.split(",") if s.strip()}
    if symbol not in allowlist:
        raise RuntimeError(f"symbol '{symbol}' not allowed. Set LIVE_EXECUTOR_SYMBOL_ALLOWLIST.")

    broker = KucoinFuturesBroker()
    oms = MakerFirstOMS(broker=broker, cfg=OmsDefaults())
    st = _read_state(state_path)
    trailing = _read_trailing_state(trailing_state_path)

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
            st, trailing = run_once(
                broker=broker,
                oms=oms,
                symbol=symbol,
                signals_root=signals_root,
                state=st,
                trailing=trailing,
                live_enabled=live_enabled,
                dry_run=dry_run,
                max_eur=max_eur,
                leverage=leverage,
                ttp_trail_pct=ttp_trail_pct,
                sl_wait_pct=sl_wait_pct,
            )
            _write_state(state_path, st)
            _write_trailing_state(trailing_state_path, trailing)
        except Exception as e:
            log.warning("executor loop error: %s", e)
            _write_state(state_path, st)
            _write_trailing_state(trailing_state_path, trailing)

        if args.once:
            break
        time.sleep(max(1.0, float(args.poll_sec)))


if __name__ == "__main__":
    main()
