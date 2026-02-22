from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ImbaParams:
    """
    IMBA signal logic (intended to match the Pine "can_long/can_short" behavior):

      high_line = highest(high, lookback)
      low_line  = lowest(low, lookback)
      fib_236 = high_line - (high_line-low_line)*0.236
      fib_5   = high_line - (high_line-low_line)*0.5
      fib_786 = high_line - (high_line-low_line)*0.786

    Long zone:
      close >= fib_5 and close >= fib_236

    Short zone:
      close <= fib_5 and close <= fib_786

    IMPORTANT (TV-style):
      - Trend state is STICKY (no neutral/0 state).
      - A signal is emitted ONLY on true flips: +1 -> -1 or -1 -> +1.
      - "Same-direction re-arms" must never emit a signal.

    Notes about trend_state_mode:
      - We keep this param for backward compatibility, but "reset" is rejected because it
        creates a neutral state (0) that does not exist in the TV behavior you described.
    """

    lookback: int = 240
    start_ts: Optional[pd.Timestamp] = None  # optional filter, utc
    fixed_sl_abs: float = 1.5  # absolute stop distance in quote currency

    fib_236: float = 0.236
    fib_5: float = 0.5
    fib_786: float = 0.786

    trend_state_mode: str = "sticky"  # must be "sticky" for TV-style


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    need = {"ts", "open", "high", "low", "close"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"df_ohlcv missing columns: {sorted(missing)}")
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    out = out.drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    # Coerce numeric
    for c in ["open", "high", "low", "close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def compute_imba_signals(df_ohlcv: pd.DataFrame, params: ImbaParams) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      ts, signal, position, source, sl

    signal: +1 for long signal bar, -1 for short signal bar, 0 otherwise
    position: same as signal for those bars (for convenience)
    sl: absolute SL price based on entry close and fixed_sl_abs
    """
    df = _ensure_ohlcv(df_ohlcv)

    if params.start_ts is not None:
        start_ts = pd.to_datetime(params.start_ts, utc=True, errors="coerce")
        df = df[df["ts"] >= start_ts].copy().reset_index(drop=True)

    mode = str(params.trend_state_mode).strip().lower()
    if mode != "sticky":
        raise ValueError("trend_state_mode must be 'sticky' (TV-style: no neutral state)")

    lookback = int(params.lookback)
    if lookback <= 1:
        raise ValueError("lookback must be > 1")

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    # Rolling extrema
    high_line = high.rolling(lookback, min_periods=lookback).max()
    low_line = low.rolling(lookback, min_periods=lookback).min()

    rng = (high_line - low_line).astype(float)

    fib236 = high_line - rng * float(params.fib_236)
    fib5 = high_line - rng * float(params.fib_5)
    fib786 = high_line - rng * float(params.fib_786)

    long_zone = (close >= fib5) & (close >= fib236)
    short_zone = (close <= fib5) & (close <= fib786)

    signals = np.zeros(len(df), dtype=np.int8)

    # state: -1 short, +1 long. (No 0 in TV-style.)
    cur: int = 0  # internal init only; NOT a real trend-state
    initialized = False

    for i in range(len(df)):
        # Need fully formed fibs
        if not np.isfinite(fib5.iat[i]) or not np.isfinite(fib236.iat[i]) or not np.isfinite(fib786.iat[i]):
            continue

        in_long = bool(long_zone.iat[i])
        in_short = bool(short_zone.iat[i])

        # If neither zone holds, we do nothing (sticky trend).
        if not in_long and not in_short:
            continue

        # Determine desired trend state on this bar
        # If both zones happen to be true (rare edge), prefer keeping current if initialized,
        # otherwise choose long by default.
        if in_long and not in_short:
            new = 1
        elif in_short and not in_long:
            new = -1
        else:
            new = cur if initialized else 1

        if not initialized:
            # Establish initial trend WITHOUT emitting a signal
            cur = new
            initialized = True
            continue

        # Emit only on true flip: +1 <-> -1
        if new != cur:
            signals[i] = new
            cur = new

    out = pd.DataFrame({"ts": df["ts"].values, "signal": signals.astype(int)})
    out = out[out["signal"] != 0].copy()
    out["position"] = out["signal"]
    out["source"] = "imba"

    # fixed absolute stop (kept for compatibility; TV uses fib-based SL by default)
    sl_abs = float(params.fixed_sl_abs)
    entry = df.set_index("ts")["close"].astype(float)

    out = out.set_index("ts")
    out["entry"] = entry.reindex(out.index).astype(float)
    out["sl"] = np.where(out["signal"] > 0, out["entry"] - sl_abs, out["entry"] + sl_abs)
    out = out.reset_index().drop(columns=["entry"])

    out = out.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    return out


def write_signals_jsonl(signals: pd.DataFrame, out_path: Path) -> int:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sig = signals.copy()
    sig["ts"] = pd.to_datetime(sig["ts"], utc=True, errors="coerce")
    sig = sig.dropna(subset=["ts"])
    sig = sig.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for _, r in sig.iterrows():
            rec = {
                "ts": pd.Timestamp(r["ts"]).isoformat(),
                "signal": int(r["signal"]),
                "position": int(r.get("position", r["signal"])),
                "source": str(r.get("source", "imba")),
                "sl": float(r.get("sl")) if not pd.isna(r.get("sl")) else None,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def make_signals_from_ohlcv(
    df_ohlcv: pd.DataFrame,
    params: ImbaParams,
    out_jsonl: Path,
) -> Tuple[int, pd.DataFrame]:
    sig = compute_imba_signals(df_ohlcv, params=params)
    n = write_signals_jsonl(sig, out_jsonl)
    return n, sig