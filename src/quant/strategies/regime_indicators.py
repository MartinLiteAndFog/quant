"""Shared regime indicator functions (CHOP, ADX, ER) with hysteresis gate logic.

Ported from quant.backtest.renko_runner so both backtests and live signal workers
use identical computations.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def choppiness(df: pd.DataFrame, n: int) -> pd.Series:
    n = int(n)
    tr = true_range(df)
    sum_tr = tr.rolling(n, min_periods=n).sum()
    hh = df["high"].astype(float).rolling(n, min_periods=n).max()
    ll = df["low"].astype(float).rolling(n, min_periods=n).min()
    denom = (hh - ll).replace(0.0, np.nan)
    return 100.0 * np.log10(sum_tr / denom) / np.log10(float(n))


def wilder_smooth(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / float(n), adjust=False).mean()


def adx(df: pd.DataFrame, n: int) -> pd.Series:
    n = int(n)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    up = high.diff()
    down = -low.diff()

    dm_plus = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    dm_minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    tr = true_range(df)
    atr = wilder_smooth(tr, n)
    sm_plus = wilder_smooth(dm_plus, n)
    sm_minus = wilder_smooth(dm_minus, n)

    di_plus = 100.0 * (sm_plus / atr.replace(0.0, np.nan))
    di_minus = 100.0 * (sm_minus / atr.replace(0.0, np.nan))

    dx = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0.0, np.nan)
    return wilder_smooth(dx, n)


def efficiency_ratio(df: pd.DataFrame, n: int) -> pd.Series:
    n = int(n)
    close = df["close"].astype(float)
    net = (close - close.shift(n)).abs()
    denom = close.diff().abs().rolling(n, min_periods=n).sum()
    er = net / denom.replace(0.0, np.nan)
    return er.clip(lower=0.0, upper=1.0)


def hysteresis_high_on(x: pd.Series, on_th: float, off_th: float) -> pd.Series:
    """ON when x >= on_th, OFF when x <= off_th. (High value = gate ON.)"""
    on_th, off_th = float(on_th), float(off_th)
    state = False
    out = []
    for v in x.values:
        if np.isnan(v):
            out.append(state)
            continue
        if not state and v >= on_th:
            state = True
        elif state and v <= off_th:
            state = False
        out.append(state)
    return pd.Series(out, index=x.index, dtype="bool")


def hysteresis_low_on(
    x: pd.Series, on_th: float, off_th: float, start_on: bool = True
) -> pd.Series:
    """ON when x <= on_th, OFF when x >= off_th. (Low value = gate ON.)"""
    on_th, off_th = float(on_th), float(off_th)
    state = bool(start_on)
    out = []
    for v in x.values:
        if np.isnan(v):
            out.append(state)
            continue
        if state and v >= off_th:
            state = False
        elif (not state) and v <= on_th:
            state = True
        out.append(state)
    return pd.Series(out, index=x.index, dtype="bool")


def build_regime_on(
    bars: pd.DataFrame,
    mode: str,
    chop_len: int = 14,
    chop_on: float = 58.0,
    chop_off: float = 52.0,
    adx_len: int = 14,
    adx_on: float = 18.0,
    adx_off: float = 25.0,
    er_len: int = 40,
    er_on: float = 0.30,
    er_off: float = 0.40,
) -> Optional[pd.Series]:
    """
    Build a boolean regime gate from CHOP/ADX/ER indicators on renko (or OHLC) bars.

    Gate ON = countertrend (IMBA) is active.
    Gate OFF = trendfollower active.

    Returns a boolean Series indexed by bar timestamps, or None if mode is disabled.
    """
    mode = str(mode).strip().lower()
    if mode in ("none", "off", ""):
        return None

    df = bars.copy().reset_index(drop=True)
    parts: List[pd.Series] = []

    if "chop" in mode:
        chop_v = choppiness(df, int(chop_len))
        chop_ok = hysteresis_high_on(chop_v, on_th=float(chop_on), off_th=float(chop_off))
        parts.append(chop_ok)

    if "adx" in mode:
        adx_v = adx(df, int(adx_len))
        adx_ok = hysteresis_low_on(adx_v, on_th=float(adx_on), off_th=float(adx_off), start_on=True)
        parts.append(adx_ok)

    if "er" in mode:
        er_v = efficiency_ratio(df, int(er_len))
        er_ok = hysteresis_low_on(er_v, on_th=float(er_on), off_th=float(er_off), start_on=True)
        parts.append(er_ok)

    if not parts:
        return None

    regime = parts[0]
    for p in parts[1:]:
        regime = regime & p

    ts_index = pd.DatetimeIndex(pd.to_datetime(bars["ts"], utc=True, errors="coerce"))
    regime.index = ts_index
    regime = regime[~regime.index.isna()]
    regime = regime[~regime.index.duplicated(keep="last")]
    return regime
