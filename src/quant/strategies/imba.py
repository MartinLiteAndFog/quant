# src/quant/strategies/imba.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import json
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ImbaParams:
    """
    IMBA signal logic (matching the Pine code you pasted):
      high_line = highest(high, lookback)
      low_line  = lowest(low, lookback)
      fib_236 = high_line - (high_line-low_line)*0.236
      fib_5   = high_line - (high_line-low_line)*0.5
      fib_786 = high_line - (high_line-low_line)*0.786

    Long signal when:
      close >= fib_5 and close >= fib_236 and NOT already in long trend
    Short signal when:
      close <= fib_5 and close <= fib_786 and NOT already in short trend

    Sticky trend: if neither long nor short condition, trend stays as-is.
    """

    lookback: int = 240
    start_ts: Optional[pd.Timestamp] = None  # optional filter, utc
    fixed_sl_abs: float = 1.5  # absolute stop distance in quote currency

    fib_236: float = 0.236
    fib_5: float = 0.5
    fib_786: float = 0.786


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    need = {"ts", "open", "high", "low", "close"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV missing columns: {sorted(missing)}")
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out = out.sort_values("ts")
    # Drop exact duplicate timestamps (keep last) to avoid downstream reindex dup issues
    out = out.drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    return out


def compute_imba_signals(df_ohlcv: pd.DataFrame, params: ImbaParams) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      ts, signal, position, source, sl

    Where:
      signal: +1 for long signal bar, -1 for short signal bar, 0 otherwise
      position: same as signal for those bars (for convenience)
      sl: absolute SL price based on entry close and fixed_sl_abs
    """
    df = _ensure_ohlcv(df_ohlcv)

    if params.start_ts is not None:
        start_ts = pd.to_datetime(params.start_ts, utc=True)
        df = df[df["ts"] >= start_ts].copy()

    lb = int(params.lookback)
    if lb < 2:
        raise ValueError("lookback must be >= 2")

    high_line = df["high"].rolling(lb, min_periods=lb).max()
    low_line = df["low"].rolling(lb, min_periods=lb).min()
    rng = (high_line - low_line)

    fib236 = high_line - rng * float(params.fib_236)
    fib5 = high_line - rng * float(params.fib_5)
    fib786 = high_line - rng * float(params.fib_786)

    close = df["close"]

    # Long/short conditions exactly as in the Pine (can_long/can_short)
    long_cond = (close >= fib5) & (close >= fib236)
    short_cond = (close <= fib5) & (close <= fib786)

    # Sticky trend state machine (matches:
    # if can_long -> is_long_trend true / if can_short -> is_short_trend true / else keep)
    trend = np.zeros(len(df), dtype=np.int8)  # -1,0,+1
    signals = np.zeros(len(df), dtype=np.int8)

    cur = 0
    for i in range(len(df)):
        if not np.isfinite(fib5.iat[i]) or not np.isfinite(fib236.iat[i]) or not np.isfinite(fib786.iat[i]):
            trend[i] = cur
            continue

        if bool(long_cond.iat[i]) and cur != 1:
            cur = 1
            signals[i] = 1
        elif bool(short_cond.iat[i]) and cur != -1:
            cur = -1
            signals[i] = -1

        trend[i] = cur

    out = pd.DataFrame(
        {
            "ts": df["ts"].values,
            "signal": signals.astype(int),
        }
    )
    out = out[out["signal"] != 0].copy()
    out["position"] = out["signal"]
    out["source"] = "imba"

    # fixed absolute stop
    sl_abs = float(params.fixed_sl_abs)
    entry = df.set_index("ts")["close"]
    out = out.set_index("ts")
    out["entry"] = entry.reindex(out.index).astype(float)
    out["sl"] = np.where(out["signal"] > 0, out["entry"] - sl_abs, out["entry"] + sl_abs)
    out = out.reset_index().drop(columns=["entry"])

    # Dedup in case we emitted multiple signals on same ts (shouldn't, but safe)
    out = out.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    return out


def write_signals_jsonl(signals: pd.DataFrame, out_path: Path) -> int:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sig = signals.copy()
    sig["ts"] = pd.to_datetime(sig["ts"], utc=True)
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
