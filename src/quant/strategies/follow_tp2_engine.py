# src/quant/strategies/follow_tp2_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TP2Params:
    """
    Trend / Follow TP2 strategy:

    - FLAT + IMBA impulse -> ENTER in impulse direction.
    - In position:
        * TP1 can scale out a fraction.
        * After TP1, BE can arm at entry (0% BE).
        * TP2 closes the remainder.
        * SL closes the remainder.
        * Opposite IMBA can either close+flip or just close, depending on flip_on_opposite.

    Gate handling:
    - regime_on may be aligned and provided externally.
    - regime_forces_flat controls whether regime_off forces a flat exit.
      For the new live design, this should usually stay False.
    """

    fee_bps: float = 0.0  # ROUNDTRIP bps
    tp1_pct: float = 0.04
    tp2_pct: float = 0.08
    tp1_frac: float = 0.5
    min_sl_pct: float = 0.03
    max_sl_pct: float = 0.08
    swing_lookback: int = 180  # effective capped to 50
    flip_on_opposite: bool = True
    be_after_tp1: bool = True
    be_offset_pct: float = 0.0  # 0.0 => BE at entry


def _fee_roundtrip(fee_bps_roundtrip: float) -> float:
    return float(fee_bps_roundtrip) / 10_000.0


def _ensure_cols(df: pd.DataFrame, need: List[str], name: str) -> pd.DataFrame:
    missing = set(need) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")

    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    out = out.drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)

    for c in ["open", "high", "low", "close"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["close"]).reset_index(drop=True)
    return out


def _coerce_signals_df_to_series(signals_df: Optional[pd.DataFrame]) -> pd.Series:
    """
    Returns sparse impulses indexed by UTC ts with int in {-1,0,+1}.
    Accepts columns: signal or position or action.
    """
    if signals_df is None or len(signals_df) == 0:
        return pd.Series(dtype="int64")

    df = signals_df.copy()
    if "ts" not in df.columns:
        raise ValueError("signals_df must have column 'ts'")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    df = df.drop_duplicates(subset=["ts"], keep="last")

    cols = set(df.columns)
    if "signal" in cols:
        s = pd.to_numeric(df["signal"], errors="coerce").fillna(0).astype(int)
    elif "position" in cols:
        s = pd.to_numeric(df["position"], errors="coerce").fillna(0).astype(int)
    elif "action" in cols:

        def _map(a: Any) -> int:
            if a is None:
                return 0
            a = str(a).strip().lower()
            if a in ("long", "buy", "1", "+1"):
                return 1
            if a in ("short", "sell", "-1"):
                return -1
            return 0

        s = df["action"].map(_map).astype(int)
    else:
        raise ValueError("signals_df must contain one of: signal, position, action")

    s = np.sign(s).astype(int)
    out = pd.Series(s.values, index=pd.DatetimeIndex(df["ts"]), dtype="int64")
    out = out[out != 0]
    return out


def align_impulses_exact(times: pd.DatetimeIndex, signals_df: Optional[pd.DataFrame]) -> pd.Series:
    """
    Align impulses using exact timestamp matches only, same style as flip_engine.
    """
    out = pd.Series(0, index=times, dtype="int64")
    sig = _coerce_signals_df_to_series(signals_df)
    if len(sig) == 0:
        return out
    common = out.index.intersection(sig.index)
    if len(common) > 0:
        out.loc[common] = sig.loc[common].astype(int)
    return out


def _align_regime_ffill(times: pd.DatetimeIndex, regime_on: Optional[pd.Series]) -> Optional[pd.Series]:
    """
    Align regime to bar times using forward-fill.
    Default True before the first regime timestamp.
    """
    if regime_on is None or len(regime_on) == 0:
        return None

    r = regime_on.copy()
    r.index = pd.to_datetime(r.index, utc=True, errors="coerce")
    r = r[~r.index.isna()]
    r = r[~r.index.duplicated(keep="last")]
    r = r.sort_index()
    r = r.astype(bool)

    out = pd.Series(True, index=times, dtype="bool")
    out = out.to_frame("x")
    out["x"] = np.nan

    tmp = pd.DataFrame({"x": r.astype(int).values}, index=r.index)
    out = out.combine_first(tmp).sort_index()
    out["x"] = out["x"].ffill()
    out["x"] = out["x"].fillna(1).astype(int)
    return out["x"].astype(bool).reindex(times)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _compute_sl_price(
    *,
    pos: int,
    entry_px: float,
    swing_low_prev: float,
    swing_high_prev: float,
    min_sl_pct: float,
    max_sl_pct: float,
) -> Tuple[float, float]:
    if pos > 0:
        raw = (entry_px - swing_low_prev) / entry_px if entry_px > 0 else max_sl_pct
        sl_pct = _clamp(raw, min_sl_pct, max_sl_pct)
        sl_px = entry_px * (1.0 - sl_pct)
    else:
        raw = (swing_high_prev - entry_px) / entry_px if entry_px > 0 else max_sl_pct
        sl_pct = _clamp(raw, min_sl_pct, max_sl_pct)
        sl_px = entry_px * (1.0 + sl_pct)
    return float(sl_px), float(sl_pct)


def run_follow_tp2_state_machine(
    bars: pd.DataFrame,
    signals_df: Optional[pd.DataFrame],
    params: TP2Params,
    regime_on: Optional[pd.Series] = None,
    regime_forces_flat: bool = False,
) -> Tuple[pd.Series, pd.DataFrame, Dict[str, Any]]:
    """
    Returns:
      pos_series indexed by ts,
      events_df with columns: ts,event,side,price,pnl_pct,note,seq,size,
      terminal_state dict with keys like:
        pos, side, mode, entry_px, sl, tp1, tp2, be_px, be_armed, tp1_done, size_rem, entry_bar_ts
    """
    bars = _ensure_cols(bars, ["ts", "close"], name="bars")
    has_hl = ("high" in bars.columns) and ("low" in bars.columns)

    times = pd.DatetimeIndex(bars["ts"])
    close = pd.to_numeric(bars["close"], errors="coerce").astype(float).values
    high = pd.to_numeric(bars["high"], errors="coerce").astype(float).values if has_hl else close.copy()
    low = pd.to_numeric(bars["low"], errors="coerce").astype(float).values if has_hl else close.copy()

    impulses = align_impulses_exact(times, signals_df)
    regime = _align_regime_ffill(times, regime_on)

    fee_rt = _fee_roundtrip(params.fee_bps)
    eff_lb = int(min(max(int(params.swing_lookback or 1), 1), 50))

    swing_low = pd.Series(low, index=times).rolling(eff_lb, min_periods=1).min().shift(1)
    swing_high = pd.Series(high, index=times).rolling(eff_lb, min_periods=1).max().shift(1)
    swing_low = swing_low.fillna(pd.Series(low, index=times))
    swing_high = swing_high.fillna(pd.Series(high, index=times))

    pos = 0
    entry_px: Optional[float] = None
    entry_bar_ts: Optional[pd.Timestamp] = None
    size_rem = 0.0
    tp1_done = False
    be_armed = False

    events: List[Dict[str, Any]] = []
    seq = 0
    out_pos = pd.Series(0, index=times, dtype="int8")

    def emit(ts: pd.Timestamp, event: str, side: int, price: float, pnl_pct: float, note: str, size: Optional[float] = None) -> None:
        nonlocal seq
        events.append(
            {
                "ts": pd.Timestamp(ts),
                "event": str(event),
                "side": int(side),
                "price": float(price),
                "pnl_pct": float(pnl_pct),
                "note": str(note),
                "seq": int(seq),
                "size": (float(size) if size is not None else np.nan),
            }
        )
        seq += 1

    def realized_pnl_pct(exit_px: float, frac: float = 1.0) -> float:
        assert entry_px is not None and pos != 0
        if pos > 0:
            gross = (float(exit_px) - float(entry_px)) / float(entry_px)
        else:
            gross = (float(entry_px) - float(exit_px)) / float(entry_px)
        return float(frac * gross - frac * fee_rt)

    def current_tp1_px() -> Optional[float]:
        if entry_px is None or pos == 0 or tp1_done:
            return None
        return float(entry_px * (1.0 + params.tp1_pct)) if pos > 0 else float(entry_px * (1.0 - params.tp1_pct))

    def current_tp2_px() -> Optional[float]:
        if entry_px is None or pos == 0:
            return None
        return float(entry_px * (1.0 + params.tp2_pct)) if pos > 0 else float(entry_px * (1.0 - params.tp2_pct))

    def current_be_px() -> Optional[float]:
        if entry_px is None or not be_armed:
            return None
        if pos > 0:
            return float(entry_px * (1.0 + params.be_offset_pct))
        return float(entry_px * (1.0 - params.be_offset_pct))

    for i, ts in enumerate(times):
        px = float(close[i])
        h = float(high[i])
        l = float(low[i])

        gate = True if regime is None else bool(regime.iloc[i])
        out_pos.iloc[i] = pos

        if not gate and pos != 0 and regime_forces_flat:
            pnl = realized_pnl_pct(px, frac=size_rem)
            emit(ts, "regime_exit", pos, px, pnl, "Regime off -> flat", size=size_rem)
            pos = 0
            entry_px = None
            entry_bar_ts = None
            size_rem = 0.0
            tp1_done = False
            be_armed = False
            out_pos.iloc[i] = pos
            continue

        impulse = int(impulses.iloc[i]) if len(impulses) else 0

        if pos == 0:
            if gate and impulse != 0:
                pos = int(np.sign(impulse))
                entry_px = px
                entry_bar_ts = pd.Timestamp(ts)
                size_rem = 1.0
                tp1_done = False
                be_armed = False
                emit(ts, "entry", pos, px, 0.0, "ENTER on signal", size=1.0)
                out_pos.iloc[i] = pos
            continue

        assert entry_px is not None

        sl_px, sl_pct = _compute_sl_price(
            pos=pos,
            entry_px=float(entry_px),
            swing_low_prev=float(swing_low.iloc[i]),
            swing_high_prev=float(swing_high.iloc[i]),
            min_sl_pct=float(params.min_sl_pct),
            max_sl_pct=float(params.max_sl_pct),
        )

        tp1_px = current_tp1_px()
        tp2_px = current_tp2_px()
        be_px = current_be_px()

        if be_px is not None:
            if (pos > 0 and l <= be_px) or (pos < 0 and h >= be_px):
                pnl = realized_pnl_pct(be_px, frac=size_rem)
                emit(ts, "be_exit", pos, be_px, pnl, "BE hit -> flat", size=size_rem)
                pos = 0
                entry_px = None
                entry_bar_ts = None
                size_rem = 0.0
                tp1_done = False
                be_armed = False
                out_pos.iloc[i] = pos
                continue

        if (pos > 0 and l <= sl_px) or (pos < 0 and h >= sl_px):
            pnl = realized_pnl_pct(sl_px, frac=size_rem)
            emit(ts, "sl_exit", pos, sl_px, pnl, f"SL hit -> flat (sl_pct={sl_pct:.5f})", size=size_rem)
            pos = 0
            entry_px = None
            entry_bar_ts = None
            size_rem = 0.0
            tp1_done = False
            be_armed = False
            out_pos.iloc[i] = pos
            continue

        tp2_hit = tp2_px is not None and ((pos > 0 and h >= tp2_px) or (pos < 0 and l <= tp2_px))
        if tp2_hit:
            pnl = realized_pnl_pct(tp2_px, frac=size_rem)
            emit(ts, "tp2_exit", pos, tp2_px, pnl, "TP2 hit -> flat", size=size_rem)
            pos = 0
            entry_px = None
            entry_bar_ts = None
            size_rem = 0.0
            tp1_done = False
            be_armed = False
            out_pos.iloc[i] = pos
            continue

        tp1_hit = tp1_px is not None and ((pos > 0 and h >= tp1_px) or (pos < 0 and l <= tp1_px))
        if tp1_hit and (not tp1_done):
            frac = _clamp(float(params.tp1_frac), 0.0, 1.0)
            frac = min(frac, size_rem)
            pnl = realized_pnl_pct(tp1_px, frac=frac)
            emit(ts, "tp1_exit", pos, tp1_px, pnl, f"TP1 hit -> scale out frac={frac:.2f}", size=frac)
            size_rem = float(max(0.0, size_rem - frac))
            tp1_done = True
            if params.be_after_tp1 and size_rem > 1e-12:
                be_armed = True
                emit(ts, "be_armed", pos, float(entry_px), 0.0, "BE armed at entry after TP1", size=size_rem)
            if size_rem <= 1e-12:
                pos = 0
                entry_px = None
                entry_bar_ts = None
                size_rem = 0.0
                tp1_done = False
                be_armed = False
                out_pos.iloc[i] = pos
                continue

        if gate and impulse != 0 and int(np.sign(impulse)) == -pos:
            pnl = realized_pnl_pct(px, frac=size_rem)
            emit(ts, "signal_exit", pos, px, pnl, "Opposite signal -> close", size=size_rem)

            if params.flip_on_opposite:
                pos = int(np.sign(impulse))
                entry_px = px
                entry_bar_ts = pd.Timestamp(ts)
                size_rem = 1.0
                tp1_done = False
                be_armed = False
                emit(ts, "entry", pos, px, 0.0, "Flip: open opposite on same bar", size=1.0)
                out_pos.iloc[i] = pos
                continue

            pos = 0
            entry_px = None
            entry_bar_ts = None
            size_rem = 0.0
            tp1_done = False
            be_armed = False
            out_pos.iloc[i] = pos
            continue

        out_pos.iloc[i] = pos

    events_df = pd.DataFrame(events)

    terminal_state: Dict[str, Any] = {
        "pos": int(pos),
        "side": ("long" if pos > 0 else "short") if pos != 0 else None,
        "mode": "TP2" if pos != 0 else None,
        "entry_px": float(entry_px) if entry_px is not None else None,
        "size_rem": float(size_rem),
        "tp1_done": bool(tp1_done),
        "be_armed": bool(be_armed),
        "entry_bar_ts": pd.Timestamp(entry_bar_ts) if entry_bar_ts is not None else None,
    }

    if pos != 0 and entry_px is not None:
        n = len(bars) - 1
        sl_px, _ = _compute_sl_price(
            pos=pos,
            entry_px=float(entry_px),
            swing_low_prev=float(swing_low.iloc[n]),
            swing_high_prev=float(swing_high.iloc[n]),
            min_sl_pct=float(params.min_sl_pct),
            max_sl_pct=float(params.max_sl_pct),
        )
        terminal_state["sl"] = float(sl_px)
        terminal_state["tp1"] = current_tp1_px()
        terminal_state["tp2"] = current_tp2_px()
        terminal_state["be_px"] = current_be_px()
        terminal_state["ttp"] = None
        terminal_state["best_fav"] = None
    else:
        terminal_state["sl"] = None
        terminal_state["tp1"] = None
        terminal_state["tp2"] = None
        terminal_state["be_px"] = None
        terminal_state["ttp"] = None
        terminal_state["best_fav"] = None

    return out_pos, events_df, terminal_state