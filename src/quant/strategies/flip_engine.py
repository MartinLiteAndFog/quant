# src/quant/strategies/flip_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FlipParams:
    """
    Countertrend strategy (remembered as "countertrend"):

    - IMBA impulse defines direction.
    - When FLAT and impulse arrives -> ENTER in impulse direction, mode=TTP (TTP armed).
    - When IN POSITION and opposite impulse arrives -> signal_flip_exit + flip immediately, mode=TTP (because IMBA signal).
    - When IN POSITION and same-dir impulse arrives:
        * If mode != TTP -> ttp_on + set mode=TTP (re-arm TTP)
        * Else ignore.

    - TTP (trailing take profit) is a trailing stop that, when hit, causes tp_exit and FLIPS.
      After a TTP-based flip, we go into WAIT (SL-only) until an IMBA same-dir impulse re-arms TTP.

    - WAIT SL distance is:
        dist_pct = clamp( lookback_extreme_dist_pct , min_sl_pct, max_sl_pct )
      LONG: stop = entry * (1 - dist_pct)
      SHORT: stop = entry * (1 + dist_pct)
      swing_lookback is capped at 50 effective.

    Fees:
      fee_bps is ROUNDTRIP bps (e.g., 10 means total 10 bps for entry+exit).
    """

    fee_bps: float = 0.0  # ROUNDTRIP bps

    # Phase A (TTP)
    ttp_trail_pct: float = 0.012

    # Phase B (WAIT SL)
    min_sl_pct: float = 0.015
    max_sl_pct: float = 0.030
    swing_lookback: int = 50

    # Optional BE (disabled by default; kept for future)
    be_trigger_pct: float = 0.0
    be_offset_pct: float = 0.0


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
    Returns Series indexed by UTC ts with int in {-1,0,+1}, impulses only.
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
    """Align impulses using EXACT timestamp matches only."""
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
    Default True before the first regime timestamp (allow trading).
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


def run_flip_state_machine(
    bars: pd.DataFrame,
    signals_df: Optional[pd.DataFrame],
    params: FlipParams,
    regime_on: Optional[pd.Series] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    bars: DataFrame with at least ['ts','close'] and optionally ['high','low'] for swing stops.

    Returns:
      pos_series indexed by ts,
      events_df with columns: ts,event,side,price,pnl_pct,note,seq
    """
    bars = _ensure_cols(bars, ["ts", "close"], name="bars")
    has_hl = ("high" in bars.columns) and ("low" in bars.columns)

    times = pd.DatetimeIndex(bars["ts"])
    close = pd.to_numeric(bars["close"], errors="coerce").astype(float).values

    impulses = align_impulses_exact(times, signals_df)
    regime = _align_regime_ffill(times, regime_on)

    fee_rt = _fee_roundtrip(params.fee_bps)
    trail = float(params.ttp_trail_pct)

    min_sl = float(params.min_sl_pct)
    max_sl = float(params.max_sl_pct)
    lb_arg = int(params.swing_lookback) if params.swing_lookback else 0
    lb_eff = int(min(max(lb_arg, 0), 50))  # cap to 50 effective

    be_trig = float(getattr(params, "be_trigger_pct", 0.0))
    be_off = float(getattr(params, "be_offset_pct", 0.0))

    pos = 0  # -1/0/+1
    entry_px: Optional[float] = None
    best_fav: Optional[float] = None  # best favorable price since entry (for TTP)
    mode: Optional[str] = None  # None | "TTP" | "WAIT"

    events: List[Dict[str, Any]] = []
    seq = 0
    out_pos = pd.Series(0, index=times, dtype="int8")

    def emit(ts: pd.Timestamp, event: str, side: int, price: float, pnl_pct: float, note: str) -> None:
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
            }
        )
        seq += 1

    def realized_pnl_pct(exit_px: float) -> float:
        assert entry_px is not None and pos != 0
        if pos > 0:
            gross = (exit_px - entry_px) / entry_px
        else:
            gross = (entry_px - exit_px) / entry_px
        return float(gross - fee_rt)

    def _clamp(x: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, x)))

    def swing_sl_price(i: int) -> float:
        """
        WAIT SL: dist_pct = clamp(lookback_dist_pct, min_sl, max_sl)
        LONG: stop = entry*(1 - dist_pct)
        SHORT: stop = entry*(1 + dist_pct)
        """
        assert entry_px is not None and pos != 0
        e = float(entry_px)

        # Fallback if no HL data or no lookback
        if (not has_hl) or (lb_eff <= 1):
            dist = _clamp(min_sl, min_sl, max_sl)
            return float(e * (1.0 - dist) if pos > 0 else e * (1.0 + dist))

        j0 = max(0, i - lb_eff + 1)

        if pos > 0:
            swing_low = float(pd.to_numeric(bars["low"].iloc[j0 : i + 1], errors="coerce").min())
            if not np.isfinite(swing_low):
                dist = _clamp(min_sl, min_sl, max_sl)
                stop = e * (1.0 - dist)
            else:
                # distance as pct
                dist_pct = (e - swing_low) / e if e != 0 else max_sl
                dist = _clamp(float(dist_pct), min_sl, max_sl)
                stop = e * (1.0 - dist)
            # ensure strictly below entry
            eps = max(1e-9, abs(e) * 1e-9)
            stop = min(stop, e - eps)
            return float(stop)

        else:
            swing_high = float(pd.to_numeric(bars["high"].iloc[j0 : i + 1], errors="coerce").max())
            if not np.isfinite(swing_high):
                dist = _clamp(min_sl, min_sl, max_sl)
                stop = e * (1.0 + dist)
            else:
                dist_pct = (swing_high - e) / e if e != 0 else max_sl
                dist = _clamp(float(dist_pct), min_sl, max_sl)
                stop = e * (1.0 + dist)
            eps = max(1e-9, abs(e) * 1e-9)
            stop = max(stop, e + eps)
            return float(stop)

    def ttp_stop(best_fav_px: float) -> float:
        assert entry_px is not None and pos != 0
        if pos > 0:
            return float(best_fav_px * (1.0 - trail))
        else:
            return float(best_fav_px * (1.0 + trail))

    for i, ts in enumerate(times):
        px = float(close[i])

        gate = True if regime is None else bool(regime.iloc[i])
        out_pos.iloc[i] = pos

        impulse = int(impulses.iloc[i]) if len(impulses) else 0

        # =========================
        # FLAT: impulse opens a position, TTP armed
        # =========================
        if pos == 0:
            if gate and impulse != 0:
                pos = int(np.sign(impulse))
                entry_px = px
                best_fav = px
                mode = "TTP"
                emit(ts, "entry", pos, px, 0.0, "ENTER (flat) -> TTP")
            continue

        assert entry_px is not None and best_fav is not None and mode is not None

        # Update best favorable for TTP
        if pos > 0:
            best_fav = max(best_fav, px)
        else:
            best_fav = min(best_fav, px)

        # Optional BE (kept, default disabled)
        if be_trig > 0.0:
            if pos > 0:
                best_move = (best_fav - entry_px) / entry_px
                if best_move >= be_trig:
                    be_stop = entry_px * (1.0 + be_off)
                    if px <= be_stop:
                        pnl = realized_pnl_pct(be_stop)
                        emit(ts, "be_exit", pos, be_stop, pnl, "BE hit -> flat")
                        pos = 0
                        entry_px = None
                        best_fav = None
                        mode = None
                        continue
            else:
                best_move = (entry_px - best_fav) / entry_px
                if best_move >= be_trig:
                    be_stop = entry_px * (1.0 - be_off)
                    if px >= be_stop:
                        pnl = realized_pnl_pct(be_stop)
                        emit(ts, "be_exit", pos, be_stop, pnl, "BE hit -> flat")
                        pos = 0
                        entry_px = None
                        best_fav = None
                        mode = None
                        continue

        # =========================
        # IMBA handling (core countertrend):
        # - opposite impulse flips immediately and sets mode=TTP (IMBA signal!)
        # - same-dir impulse arms TTP (mode=TTP)
        # =========================
        if gate and impulse != 0:
            imp = int(np.sign(impulse))
            if imp == -pos:
                pnl = realized_pnl_pct(px)
                emit(ts, "signal_flip_exit", pos, px, pnl, "Opposite IMBA -> flip -> TTP")
                pos = -pos
                entry_px = px
                best_fav = px
                mode = "TTP"
                out_pos.iloc[i] = pos
                continue
            elif imp == pos:
                if mode != "TTP":
                    mode = "TTP"
                    emit(ts, "ttp_on", pos, px, 0.0, "IMBA same-dir -> TTP(on)")
                # if already TTP, do nothing

        # =========================
        # PRICE-based logic
        # =========================
        if mode == "TTP":
            stop = ttp_stop(best_fav)
            if (pos > 0 and px <= stop) or (pos < 0 and px >= stop):
                pnl = realized_pnl_pct(stop)
                emit(ts, "tp_exit", pos, stop, pnl, f"TTP hit -> flip (stop={stop:.4f})")
                pos = -pos
                entry_px = stop
                best_fav = stop
                mode = "WAIT"  # after TTP flip, we are in WAIT until IMBA re-arms TTP
                out_pos.iloc[i] = pos
                continue

        elif mode == "WAIT":
            stop = swing_sl_price(i)
            if (pos > 0 and px <= stop) or (pos < 0 and px >= stop):
                pnl = realized_pnl_pct(stop)
                emit(ts, "sl_exit", pos, stop, pnl, f"SL hit -> flat (sl={stop:.4f})")
                pos = 0
                entry_px = None
                best_fav = None
                mode = None
                out_pos.iloc[i] = pos
                continue

        out_pos.iloc[i] = pos

    events_df = pd.DataFrame(events)
    return out_pos, events_df