# src/quant/strategies/flip_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FlipParams:
    """
    Two-phase stop logic:

    Phase A: "TTP active"
      - Trigger: IMBA impulse while flat OR IMBA impulse while in WAIT.
      - Behavior: trailing stop (TTP) active immediately.
      - Exit:
          * If TTP is hit -> tp_exit and FLIP position.
          * If opposite IMBA arrives -> signal_flip_exit and FLIP position.
      - After any flip (tp_exit or opposite IMBA flip): switch to Phase B (WAIT with SL).

    Phase B: "WAIT with SL"
      - Trigger: after a flip (from Phase A).
      - Behavior: SL only:
          * Stop is at least sl_cap_pct away from entry (min distance),
            and can be farther using swing extreme over swing_lookback.
          * LONG: stop = min(entry*(1-sl_cap_pct), swing_low_lookback)
          * SHORT: stop = max(entry*(1+sl_cap_pct), swing_high_lookback)
      - Exit:
          * If SL hit -> sl_exit and go FLAT.
      - If an IMBA impulse arrives:
          * If same direction as current pos -> activate Phase A (TTP) (no pos change).
          * If opposite direction -> reverse into that direction and activate Phase A (TTP).

    Break-even (BE) (optional):
      - If be_trigger_pct > 0:
          * Once trade's best favorable move reaches be_trigger_pct,
            enforce a BE stop at entry*(1+be_offset_pct) for LONG and entry*(1-be_offset_pct) for SHORT.
      - BE is evaluated in both TTP and WAIT modes and will exit to FLAT with event 'be_exit'.
        (No flip on BE; it’s a protection mechanic.)

    Regime gating (optional):
      - If regime_on is provided (bool series):
          * When regime_on is False, we DO NOT:
              - open new entries from flat
              - activate TTP / reverse on IMBA impulses
              - flip on IMBA impulses
            We DO still:
              - manage open positions with TTP/SL/BE exits
    """

    fee_bps: float = 0.0

    # Phase A
    ttp_trail_pct: float = 0.012

    # Phase B
    sl_cap_pct: float = 0.015
    swing_lookback: int = 250

    # Break-even
    be_trigger_pct: float = 0.0
    be_offset_pct: float = 0.0


def _fee_roundtrip(fee_bps: float) -> float:
    return 2.0 * (float(fee_bps) / 10_000.0)


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
    Align regime to bar times using forward-fill:
      - regime_on can be daily/weekly timestamps
      - we map each bar ts to the latest regime value at or before it
    Default is True before the first regime timestamp (conservative = allow trading),
    but you can change that later if you prefer default False.
    """
    if regime_on is None or len(regime_on) == 0:
        return None

    r = regime_on.copy()
    r.index = pd.to_datetime(r.index, utc=True, errors="coerce")
    r = r[~r.index.isna()]
    r = r[~r.index.duplicated(keep="last")]
    r = r.sort_index()
    r = r.astype(bool)

    # build full index and ffill
    out = pd.Series(True, index=times, dtype="bool")
    out = out.to_frame("x")
    out["x"] = np.nan

    tmp = pd.DataFrame({"x": r.astype(int).values}, index=r.index)
    out = out.combine_first(tmp).sort_index()
    out["x"] = out["x"].ffill()

    # default True if still NaN (before first regime point)
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
    slcap = float(params.sl_cap_pct)
    lb = int(params.swing_lookback) if params.swing_lookback else 0
    be_trig = float(params.be_trigger_pct)
    be_off = float(params.be_offset_pct)

    pos = 0  # -1/0/+1
    entry_px: Optional[float] = None
    best_fav: Optional[float] = None  # best favorable price since entry
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

    def swing_sl_price(i: int) -> float:
        assert entry_px is not None and pos != 0
        e = float(entry_px)

        if pos > 0:
            cap_stop = e * (1.0 - slcap)
            stop = cap_stop
            if has_hl and lb > 1:
                j0 = max(0, i - lb + 1)
                swing_low = float(pd.to_numeric(bars["low"].iloc[j0 : i + 1], errors="coerce").min())
                if np.isfinite(swing_low):
                    stop = min(stop, swing_low)
            eps = max(1e-9, abs(e) * 1e-9)
            stop = min(stop, e - eps)
            return float(stop)
        else:
            cap_stop = e * (1.0 + slcap)
            stop = cap_stop
            if has_hl and lb > 1:
                j0 = max(0, i - lb + 1)
                swing_high = float(pd.to_numeric(bars["high"].iloc[j0 : i + 1], errors="coerce").max())
                if np.isfinite(swing_high):
                    stop = max(stop, swing_high)
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

        if pos == 0:
            if gate and impulse != 0:
                pos = int(np.sign(impulse))
                entry_px = px
                best_fav = px
                mode = "TTP"
                emit(ts, "entry", pos, px, 0.0, "ENTER (flat) -> TTP")
            continue

        assert entry_px is not None and best_fav is not None and mode is not None

        # update best favorable
        if pos > 0:
            best_fav = max(best_fav, px)
        else:
            best_fav = min(best_fav, px)

        # Break-even (optional)
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

        if mode == "TTP":
            stop = ttp_stop(best_fav)
            if (pos > 0 and px <= stop) or (pos < 0 and px >= stop):
                pnl = realized_pnl_pct(stop)
                emit(ts, "tp_exit", pos, stop, pnl, f"TTP hit -> flip (stop={stop:.4f})")
                pos = -pos
                entry_px = stop
                best_fav = stop
                mode = "WAIT"
                continue

            if gate and impulse != 0 and int(np.sign(impulse)) == -pos:
                pnl = realized_pnl_pct(px)
                emit(ts, "signal_flip_exit", pos, px, pnl, "Opposite IMBA -> flip -> TTP")
                pos = -pos
                entry_px = px
                best_fav = px
                mode = "TTP"
                continue

        elif mode == "WAIT":
            stop = swing_sl_price(i)
            if (pos > 0 and px <= stop) or (pos < 0 and px >= stop):
                pnl = realized_pnl_pct(stop)
                emit(ts, "sl_exit", pos, stop, pnl, f"WAIT SL hit -> flat (sl={stop:.4f})")
                pos = 0
                entry_px = None
                best_fav = None
                mode = None
                continue

            if gate and impulse != 0:
                imp = int(np.sign(impulse))
                if imp == pos:
                    mode = "TTP"
                    emit(ts, "ttp_on", pos, px, 0.0, "WAIT + same IMBA -> TTP")
                    continue
                else:
                    pnl = realized_pnl_pct(px)
                    emit(ts, "signal_flip_exit", pos, px, pnl, "WAIT + opposite IMBA -> flip -> TTP")
                    pos = -pos
                    entry_px = px
                    best_fav = px
                    mode = "TTP"
                    continue

        out_pos.iloc[i] = pos

    events_df = pd.DataFrame(events)
    return out_pos, events_df
