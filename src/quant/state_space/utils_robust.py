from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_median(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window, min_periods=window).median()


def rolling_mad(x: pd.Series, window: int, eps: float = 1e-8) -> pd.Series:
    med = rolling_median(x, window=window)
    abs_dev = (x - med).abs()
    mad = abs_dev.rolling(window=window, min_periods=window).median()
    return mad + eps


def robust_z(x: pd.Series, window: int, eps: float = 1e-8) -> pd.Series:
    med = rolling_median(x, window=window)
    mad = rolling_mad(x, window=window, eps=eps)
    return (x - med) / (1.4826 * mad)


def clip_series(x: pd.Series, zmax: float) -> pd.Series:
    return x.clip(lower=-zmax, upper=zmax)


def norm_score(z: pd.Series, alpha: float = 1.0, zmax: float = 5.0) -> pd.Series:
    clipped = clip_series(z, zmax=zmax)
    return pd.Series(np.tanh(alpha * clipped), index=z.index, name=z.name)


def rolling_mad_threshold(x: pd.Series, window: int, eps: float = 1e-8) -> pd.Series:
    med = rolling_median(x, window=window)
    mad = (x - med).abs().rolling(window=window, min_periods=window).median()
    return mad + eps
