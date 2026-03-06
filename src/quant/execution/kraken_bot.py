from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from quant.execution.kraken_futures import KrakenFuturesClient
from quant.utils.log import get_logger

log = get_logger("quant.kraken_bot")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _live_default(rel_path: str) -> str:
    if Path("/data").exists():
        return str(Path("/data/live/kraken") / rel_path)
    return str(Path("data/live/kraken") / rel_path)


def _now_iso() -> str:
    return pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _fetch_json(url: str, timeout_s: int = 4) -> Dict[str, Any]:
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "quant-kraken-bot/1"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode("utf-8"))


def _publish_metrics(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _append_equity(path: Path, ts: str, equity_usd: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts_i = int(pd.Timestamp(ts).timestamp())
    snap = pd.DataFrame([{"ts": ts_i, "equity_usd": float(equity_usd)}])
    if path.exists():
        try:
            old = pd.read_csv(path)
        except Exception:
            old = pd.DataFrame(columns=["ts", "equity_usd"])
        df = pd.concat([old, snap], ignore_index=True)
    else:
        df = snap
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df["equity_usd"] = pd.to_numeric(df["equity_usd"], errors="coerce")
    df = df.dropna(subset=["ts", "equity_usd"]).sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    max_rows = int(max(200, int(os.getenv("KRAKEN_EQUITY_MAX_ROWS", "5000"))))
    if len(df) > max_rows:
        df = df.tail(max_rows)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Bot state
# ---------------------------------------------------------------------------

@dataclass
class BotState:
    pos_side: int = 0           # +1 long, -1 short, 0 flat
    entry_px: float = 0.0
    best_fav: float = 0.0
    size_full: float = 0.0
    size_rem: float = 0.0
    mode: str = "FLAT"          # FLAT, FLIP_TTP, FLIP_WAIT, TP2_OPEN, TP2_BE
    engine: str = "none"        # "flip" or "tp2"
    gate_on: int = 0
    last_signal_ts: str = ""
    tp1_done: bool = False


def save_state(state: BotState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), ensure_ascii=False), encoding="utf-8")


def load_state(path: Path) -> Optional[BotState]:
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return BotState(**{k: v for k, v in d.items() if k in BotState.__dataclass_fields__})
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class FlipParams:
    ttp_trail_pct: float = 0.012
    min_sl_pct: float = 0.015
    max_sl_pct: float = 0.030
    swing_lookback: int = 50  # backtest caps to 50 (flip_engine.py:183)


@dataclass
class TP2Params:
    tp1_pct: float = 0.015
    tp2_pct: float = 0.030
    tp1_frac: float = 0.5
    min_sl_pct: float = 0.030
    max_sl_pct: float = 0.080
    swing_lookback: int = 50  # backtest caps to 50 (renko_runner_tp2.py:259)
    flip_on_opposite: bool = True


def load_flip_params() -> FlipParams:
    return FlipParams(
        ttp_trail_pct=float(os.getenv("KRAKEN_TTP_TRAIL_PCT", "0.012")),
        min_sl_pct=float(os.getenv("KRAKEN_FLIP_MIN_SL_PCT", "0.015")),
        max_sl_pct=float(os.getenv("KRAKEN_FLIP_MAX_SL_PCT", "0.030")),
        swing_lookback=min(int(os.getenv("KRAKEN_FLIP_SWING_LOOKBACK", "50")), 50),
    )


def load_tp2_params() -> TP2Params:
    return TP2Params(
        tp1_pct=float(os.getenv("KRAKEN_TP1_PCT", "0.015")),
        tp2_pct=float(os.getenv("KRAKEN_TP2_PCT", "0.030")),
        tp1_frac=float(os.getenv("KRAKEN_TP1_FRAC", "0.5")),
        min_sl_pct=float(os.getenv("KRAKEN_TP2_MIN_SL_PCT", "0.030")),
        max_sl_pct=float(os.getenv("KRAKEN_TP2_MAX_SL_PCT", "0.080")),
        swing_lookback=min(int(os.getenv("KRAKEN_TP2_SWING_LOOKBACK", "50")), 50),
        flip_on_opposite=_truthy(os.getenv("KRAKEN_FLIP_ON_OPPOSITE", "1")),
    )


# ---------------------------------------------------------------------------
# Swing SL computation (mirrors backtest)
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def compute_swing_sl(
    pos_side: int,
    entry_px: float,
    swing_low: float,
    swing_high: float,
    min_sl_pct: float,
    max_sl_pct: float,
) -> float:
    if pos_side > 0:
        raw = (entry_px - swing_low) / entry_px if entry_px > 0 else max_sl_pct
        sl_pct = _clamp(raw, min_sl_pct, max_sl_pct)
        return entry_px * (1.0 - sl_pct)
    else:
        raw = (swing_high - entry_px) / entry_px if entry_px > 0 else max_sl_pct
        sl_pct = _clamp(raw, min_sl_pct, max_sl_pct)
        return entry_px * (1.0 + sl_pct)


# ---------------------------------------------------------------------------
# Pure state machine logic
# ---------------------------------------------------------------------------

def run_once_logic(
    state: BotState,
    gate_on: int,
    signal: int,
    signal_ts: str,
    mark: float,
    swing_low: float,
    swing_high: float,
    target_size: float,
    flip_p: FlipParams,
    tp2_p: TP2Params,
) -> Tuple[BotState, List[Dict[str, Any]]]:
    """
    Pure state machine tick. Returns (new_state, actions).
    Actions: [{"action": "close_all"}, {"action": "enter_long", "size": ...}, ...]
    """
    s = BotState(**asdict(state))
    actions: List[Dict[str, Any]] = []

    new_signal = (signal != 0 and signal_ts and signal_ts != s.last_signal_ts)

    # --- 1. Gate transition: regime_exit ---
    if gate_on != s.gate_on and s.mode != "FLAT":
        actions.append({"action": "close_all", "reason": "regime_exit"})
        s.pos_side = 0
        s.entry_px = 0.0
        s.best_fav = 0.0
        s.size_full = 0.0
        s.size_rem = 0.0
        s.mode = "FLAT"
        s.tp1_done = False

    s.gate_on = gate_on
    s.engine = "flip" if gate_on == 1 else "tp2"

    # --- 2. FLAT: new signal enters ---
    if s.mode == "FLAT" and new_signal:
        s.last_signal_ts = signal_ts
        side = 1 if signal > 0 else -1
        order_side = "buy" if side > 0 else "sell"
        actions.append({"action": f"enter_{('long' if side > 0 else 'short')}", "side": order_side, "size": target_size})
        s.pos_side = side
        s.entry_px = mark
        s.best_fav = mark
        s.size_full = target_size
        s.size_rem = target_size
        s.tp1_done = False
        s.mode = "FLIP_TTP" if s.engine == "flip" else "TP2_OPEN"
        return s, actions

    # Update last_signal_ts for dedup even if we didn't act on it while flat
    if new_signal:
        s.last_signal_ts = signal_ts

    if s.mode == "FLAT":
        return s, actions

    # --- 3. FLIP_TTP: trailing take-profit, opposite signal flip ---
    if s.mode == "FLIP_TTP":
        if s.pos_side > 0:
            s.best_fav = max(s.best_fav, mark)
        else:
            s.best_fav = min(s.best_fav, mark)

        # Opposite IMBA signal → immediate flip (before TTP check, matching backtest)
        if new_signal:
            imp = 1 if signal > 0 else -1
            if imp == -s.pos_side:
                old_side = "sell" if s.pos_side > 0 else "buy"
                new_side_str = "long" if imp > 0 else "short"
                actions.append({"action": "close_all", "reason": "signal_flip"})
                actions.append({"action": f"enter_{new_side_str}", "side": ("buy" if imp > 0 else "sell"), "size": target_size})
                s.pos_side = imp
                s.entry_px = mark
                s.best_fav = mark
                s.size_full = target_size
                s.size_rem = target_size
                return s, actions
            elif imp == s.pos_side:
                pass  # same-dir signal: TTP already armed, do nothing

        # TTP trail check
        if s.pos_side > 0:
            ttp_stop = s.best_fav * (1.0 - flip_p.ttp_trail_pct)
            triggered = mark <= ttp_stop
        else:
            ttp_stop = s.best_fav * (1.0 + flip_p.ttp_trail_pct)
            triggered = mark >= ttp_stop

        if triggered:
            new_side = -s.pos_side
            new_side_str = "long" if new_side > 0 else "short"
            actions.append({"action": "close_all", "reason": "ttp_flip"})
            actions.append({"action": f"enter_{new_side_str}", "side": ("buy" if new_side > 0 else "sell"), "size": target_size})
            s.pos_side = new_side
            s.entry_px = mark
            s.best_fav = mark
            s.size_full = target_size
            s.size_rem = target_size
            s.mode = "FLIP_WAIT"
            return s, actions

        return s, actions

    # --- 4. FLIP_WAIT: swing SL, signal re-arms TTP ---
    if s.mode == "FLIP_WAIT":
        sl_px = compute_swing_sl(s.pos_side, s.entry_px, swing_low, swing_high, flip_p.min_sl_pct, flip_p.max_sl_pct)

        if (s.pos_side > 0 and mark <= sl_px) or (s.pos_side < 0 and mark >= sl_px):
            actions.append({"action": "close_all", "reason": "swing_sl"})
            s.pos_side = 0
            s.entry_px = 0.0
            s.best_fav = 0.0
            s.size_full = 0.0
            s.size_rem = 0.0
            s.mode = "FLAT"
            s.tp1_done = False
            return s, actions

        if new_signal:
            s.mode = "FLIP_TTP"
            s.best_fav = mark  # re-arm TTP from current price

        return s, actions

    # --- 5. TP2_OPEN: TP1 partial, TP2 full, swing SL, opposite signal ---
    if s.mode == "TP2_OPEN":
        tp1_px = s.entry_px * (1.0 + tp2_p.tp1_pct) if s.pos_side > 0 else s.entry_px * (1.0 - tp2_p.tp1_pct)
        tp2_px = s.entry_px * (1.0 + tp2_p.tp2_pct) if s.pos_side > 0 else s.entry_px * (1.0 - tp2_p.tp2_pct)
        sl_px = compute_swing_sl(s.pos_side, s.entry_px, swing_low, swing_high, tp2_p.min_sl_pct, tp2_p.max_sl_pct)

        tp2_hit = (s.pos_side > 0 and mark >= tp2_px) or (s.pos_side < 0 and mark <= tp2_px)
        tp1_hit = (s.pos_side > 0 and mark >= tp1_px) or (s.pos_side < 0 and mark <= tp1_px)
        sl_hit = (s.pos_side > 0 and mark <= sl_px) or (s.pos_side < 0 and mark >= sl_px)

        if tp2_hit:
            actions.append({"action": "close_all", "reason": "tp2_exit"})
            s.pos_side = 0
            s.entry_px = 0.0
            s.best_fav = 0.0
            s.size_full = 0.0
            s.size_rem = 0.0
            s.mode = "FLAT"
            s.tp1_done = False
            return s, actions

        if tp1_hit and not s.tp1_done and s.size_rem > 0:
            partial = _clamp(tp2_p.tp1_frac, 0.0, 1.0) * s.size_full
            close_side = "sell" if s.pos_side > 0 else "buy"
            actions.append({"action": "close_partial", "close_side": close_side, "size": partial, "reason": "tp1_exit"})
            s.size_rem = s.size_full - partial
            s.tp1_done = True
            s.mode = "TP2_BE"
            return s, actions

        if sl_hit:
            actions.append({"action": "close_all", "reason": "swing_sl"})
            s.pos_side = 0
            s.entry_px = 0.0
            s.best_fav = 0.0
            s.size_full = 0.0
            s.size_rem = 0.0
            s.mode = "FLAT"
            s.tp1_done = False
            return s, actions

        if new_signal:
            imp = 1 if signal > 0 else -1
            if imp == -s.pos_side:
                actions.append({"action": "close_all", "reason": "signal_exit"})
                s.pos_side = 0
                s.entry_px = 0.0
                s.best_fav = 0.0
                s.size_full = 0.0
                s.size_rem = 0.0
                s.mode = "FLAT"
                s.tp1_done = False
                if tp2_p.flip_on_opposite:
                    new_side_str = "long" if imp > 0 else "short"
                    actions.append({"action": f"enter_{new_side_str}", "side": ("buy" if imp > 0 else "sell"), "size": target_size})
                    s.pos_side = imp
                    s.entry_px = mark
                    s.best_fav = mark
                    s.size_full = target_size
                    s.size_rem = target_size
                    s.mode = "TP2_OPEN"
                return s, actions

        return s, actions

    # --- 6. TP2_BE: breakeven + TP2 remainder ---
    if s.mode == "TP2_BE":
        be_px = s.entry_px
        tp2_px = s.entry_px * (1.0 + tp2_p.tp2_pct) if s.pos_side > 0 else s.entry_px * (1.0 - tp2_p.tp2_pct)

        be_hit = (s.pos_side > 0 and mark <= be_px) or (s.pos_side < 0 and mark >= be_px)
        tp2_hit = (s.pos_side > 0 and mark >= tp2_px) or (s.pos_side < 0 and mark <= tp2_px)

        if be_hit:
            actions.append({"action": "close_all", "reason": "be_exit"})
            s.pos_side = 0
            s.entry_px = 0.0
            s.best_fav = 0.0
            s.size_full = 0.0
            s.size_rem = 0.0
            s.mode = "FLAT"
            s.tp1_done = False
            return s, actions

        if tp2_hit:
            actions.append({"action": "close_all", "reason": "tp2_exit"})
            s.pos_side = 0
            s.entry_px = 0.0
            s.best_fav = 0.0
            s.size_full = 0.0
            s.size_rem = 0.0
            s.mode = "FLAT"
            s.tp1_done = False
            return s, actions

        if new_signal:
            imp = 1 if signal > 0 else -1
            if imp == -s.pos_side:
                actions.append({"action": "close_all", "reason": "signal_exit"})
                s.pos_side = 0
                s.entry_px = 0.0
                s.best_fav = 0.0
                s.size_full = 0.0
                s.size_rem = 0.0
                s.mode = "FLAT"
                s.tp1_done = False
                return s, actions

        return s, actions

    return s, actions


# ---------------------------------------------------------------------------
# Action executor
# ---------------------------------------------------------------------------

def execute_actions(
    client: KrakenFuturesClient,
    actions: List[Dict[str, Any]],
    dry_run: bool,
    equity_pct: float = 0.9,
    leverage: float = 1.0,
) -> List[Dict[str, Any]]:
    results = []
    just_closed = False
    for a in actions:
        if dry_run:
            log.info("DRY_RUN: %s", a)
            results.append({**a, "executed": False, "dry_run": True})
            if a.get("action") == "close_all":
                just_closed = True
            continue
        try:
            act = a["action"]
            if act == "close_all":
                res = client.close_position()
                results.append({**a, "executed": True, "result": res})
                just_closed = True
            elif act == "close_partial":
                res = client.place_market(a["close_side"], size=float(a["size"]), reduce_only=True)
                results.append({**a, "executed": True, "result": res})
            elif act.startswith("enter_"):
                size = float(a["size"])
                if just_closed:
                    try:
                        mark = client.get_mark_price()
                        eq = client.get_account_equity()
                        fresh_eq = float(eq.get("equity_usd", 0.0) or 0.0)
                        fresh_size = compute_target_size(fresh_eq, mark, leverage, equity_pct)
                        if fresh_size > 0:
                            log.info("flip re-size: pre=%.1f post=%.1f equity=%.2f", size, fresh_size, fresh_eq)
                            size = fresh_size
                    except Exception as e:
                        log.warning("flip re-size failed, using original: %s", e)
                res = client.place_market(a["side"], size=size)
                results.append({**a, "size": size, "executed": True, "result": res})
                just_closed = False
            else:
                results.append({**a, "executed": False, "error": "unknown_action"})
        except Exception as e:
            log.warning("action failed: %s error=%s", a, e)
            results.append({**a, "executed": False, "error": str(e)})
    return results


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_gate(url: str) -> Dict[str, Any]:
    try:
        obj = _fetch_json(url)
        return {"gate_on": int(obj.get("gate_on", 0) or 0), "ts": str(obj.get("ts", "")), "source": obj.get("source", "api")}
    except Exception as e:
        log.warning("gate fetch failed: %s", e)
        return {"gate_on": 0, "ts": "", "source": "error", "error": str(e)}


def fetch_signal(url: str) -> Dict[str, Any]:
    try:
        obj = _fetch_json(url)
        return {"signal": int(obj.get("signal", 0) or 0), "ts": str(obj.get("ts", ""))}
    except Exception as e:
        log.warning("signal fetch failed: %s", e)
        return {"signal": 0, "ts": ""}


def fetch_renko(url: str, lookback: int) -> Dict[str, Any]:
    try:
        redis_url = os.getenv("REDIS_URL", "").strip()
        if redis_url:
            import json
            import redis as redis_lib

            r = redis_lib.from_url(redis_url, decode_responses=True)
            raw = r.get("renko:SOLUSDT:latest")
            if raw:
                obj = json.loads(raw)
                bars = obj.get("bars", []) or []
                lb = min(max(int(lookback), 1), len(bars) if bars else 1)
                tail = bars[-lb:] if bars else []

                if tail:
                    swing_low = min(float(x.get("low", 0) or 0) for x in tail)
                    swing_high = max(float(x.get("high", 0) or 0) for x in tail)
                else:
                    swing_low = float(obj.get("swing_low_50", 0) or 0)
                    swing_high = float(obj.get("swing_high_50", 0) or 0)

                return {
                    "swing_low": swing_low,
                    "swing_high": swing_high,
                    "last_close": float(obj.get("close", 0) or 0),
                    "source": "redis",
                    "ts": str(obj.get("ts", "")),
                }
    except Exception as e:
        log.warning("renko redis fetch failed: %s", e)

    try:
        obj = _fetch_json(f"{url}?lookback={lookback}")
        return {
            "swing_low": float(obj.get("swing_low", 0) or 0),
            "swing_high": float(obj.get("swing_high", 0) or 0),
            "last_close": float(obj.get("last_close", 0) or 0),
            "source": "api",
            "ts": str(obj.get("ts", "")),
        }
    except Exception as e:
        log.warning("renko fetch failed: %s", e)
        return {"swing_low": 0.0, "swing_high": 0.0, "last_close": 0.0, "source": "none", "ts": ""}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def compute_target_size(
    equity_usd: float,
    mark_price: float,
    leverage: float,
    equity_pct: float,
) -> float:
    """
    Compute position size in SOL from equity percentage.
    
    equity_pct=0.90 means use 90% of equity * leverage / mark_price.
    Result is floored to 1 decimal to meet Kraken's minimum tick.
    """
    if mark_price <= 0 or equity_usd <= 0:
        return 0.0
    notional = equity_usd * equity_pct * leverage
    size = notional / mark_price
    return float(int(size * 10) / 10)  # floor to 0.1 SOL


def run_once(
    client: KrakenFuturesClient,
    state: BotState,
    gate_url: str,
    signal_url: str,
    renko_url: str,
    equity_pct: float,
    leverage: float,
    dry_run: bool,
    flip_p: FlipParams,
    tp2_p: TP2Params,
    metrics_path: Path,
    equity_path: Path,
    state_path: Path,
) -> BotState:
    gate = fetch_gate(gate_url)
    sig = fetch_signal(signal_url)
    mark = client.get_mark_price()
    eq = client.get_account_equity()
    equity_usd = float(eq.get("equity_usd", 0.0) or 0.0)

    target_size = compute_target_size(equity_usd, mark, leverage, equity_pct)

    active_lookback = flip_p.swing_lookback if state.engine == "flip" else tp2_p.swing_lookback
    renko = fetch_renko(renko_url, active_lookback)

    new_state, actions = run_once_logic(
        state=state,
        gate_on=gate["gate_on"],
        signal=sig["signal"],
        signal_ts=sig["ts"],
        mark=mark,
        swing_low=renko["swing_low"],
        swing_high=renko["swing_high"],
        target_size=target_size,
        flip_p=flip_p,
        tp2_p=tp2_p,
    )

    action_results = execute_actions(client, actions, dry_run=dry_run, equity_pct=equity_pct, leverage=leverage)

    save_state(new_state, state_path)

    ts = _now_iso()
    row = {
        "ts": ts,
        "equity_usd": equity_usd,
        "wallet_usd": float(eq.get("wallet_usd", 0.0) or 0.0),
        "upnl_usd": float(eq.get("upnl_usd", 0.0) or 0.0),
        "mark_price": round(mark, 4),
        "target_size": target_size,
        "gate_on": gate["gate_on"],
        "gate_source": gate.get("source", "?"),
        "engine": new_state.engine,
        "mode": new_state.mode,
        "pos_side": new_state.pos_side,
        "entry_px": round(new_state.entry_px, 4),
        "best_fav": round(new_state.best_fav, 4),
        "size_rem": round(new_state.size_rem, 6),
        "tp1_done": new_state.tp1_done,
        "signal": sig["signal"],
        "signal_ts": sig["ts"],
        "actions": [a.get("action", "") + ":" + a.get("reason", "") for a in actions],
        "dry_run": dry_run,
    }
    _publish_metrics(metrics_path, row)
    _append_equity(equity_path, ts=ts, equity_usd=float(row["equity_usd"]))

    side_str = {1: "long", -1: "short", 0: "flat"}.get(new_state.pos_side, "?")
    action_str = ", ".join(a.get("action", "") + ":" + a.get("reason", "") for a in actions) or "hold"
    log.info(
        "bot ts=%s gate=%s engine=%s mode=%s pos=%s/%s entry=%.2f mark=%.2f sig=%s action=[%s]",
        ts, gate["gate_on"], new_state.engine, new_state.mode,
        side_str, round(new_state.size_rem, 4),
        new_state.entry_px, mark,
        sig["signal"], action_str,
    )
    return new_state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kraken Futures SOL-USD execution bot (backtest-parity)")
    p.add_argument("--once", action="store_true")
    p.add_argument("--poll-sec", type=float, default=float(os.getenv("KRAKEN_POLL_SEC", "10")))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    client = KrakenFuturesClient()

    gate_url = os.getenv("KRAKEN_GATE_URL", "http://127.0.0.1:8000/api/gate/solusd")
    signal_url = os.getenv("KRAKEN_SIGNAL_URL", "http://127.0.0.1:8000/api/signals/latest/solusd")
    renko_url = os.getenv("KRAKEN_RENKO_URL", "http://127.0.0.1:8000/api/renko/latest/solusd")

    equity_pct = float(os.getenv("KRAKEN_EQUITY_PCT", "0.90"))
    leverage = float(os.getenv("KRAKEN_LEVERAGE", "5"))
    dry_run = _truthy(os.getenv("KRAKEN_DRY_RUN", "1"))

    state_path = Path(os.getenv("KRAKEN_STATE_JSON", _live_default("bot_state.json")))
    metrics_path = Path(os.getenv("KRAKEN_METRICS_JSON", _live_default("metrics.json")))
    equity_path = Path(os.getenv("KRAKEN_EQUITY_CSV", _live_default("equity.csv")))

    flip_p = load_flip_params()
    tp2_p = load_tp2_params()

    state = load_state(state_path) or BotState()
    log.info("bot starting state=%s dry_run=%s equity_pct=%s leverage=%s", state.mode, dry_run, equity_pct, leverage)

    while True:
        try:
            state = run_once(
                client=client, state=state,
                gate_url=gate_url, signal_url=signal_url, renko_url=renko_url,
                equity_pct=equity_pct, leverage=leverage, dry_run=dry_run,
                flip_p=flip_p, tp2_p=tp2_p,
                metrics_path=metrics_path, equity_path=equity_path, state_path=state_path,
            )
        except Exception as e:
            log.warning("bot loop error: %s", e)
        if args.once:
            break
        time.sleep(max(1.0, float(args.poll_sec)))


if __name__ == "__main__":
    main()
