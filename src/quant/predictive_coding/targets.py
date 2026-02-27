from __future__ import annotations
import numpy as np


def build_obs_features(close: np.ndarray) -> np.ndarray:
    """Build observation features from close prices.

    Returns (N, 5) array: [r_1, r_5, r_15, rv_20, rv_60].
    Invalid lookback rows are NaN.
    """
    n = len(close)
    log_close = np.log(close)
    obs = np.full((n, 5), np.nan)

    for i, k in enumerate([1, 5, 15]):
        obs[k:, i] = log_close[k:] - log_close[:-k]

    r1 = np.full(n, np.nan)
    r1[1:] = log_close[1:] - log_close[:-1]

    for i, w in enumerate([20, 60]):
        col_idx = 3 + i
        for t in range(w, n):
            window = r1[t - w + 1 : t + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= 2:
                obs[t, col_idx] = np.std(valid, ddof=1)
    return obs


def build_targets(close: np.ndarray, horizons: list[int]) -> np.ndarray:
    """Build future log-return targets.

    Returns (N, len(horizons)) array. Rows without sufficient future data are NaN.
    """
    n = len(close)
    log_close = np.log(close)
    targets = np.full((n, len(horizons)), np.nan)
    for i, h in enumerate(horizons):
        if h < n:
            targets[: n - h, i] = log_close[h:] - log_close[:-h]
    return targets


def get_valid_range(
    close: np.ndarray,
    horizons: list[int],
    obs_lookback: int = 60,
) -> tuple[int, int]:
    """Return (start, end) indices where both obs and targets are valid."""
    start = obs_lookback
    end = len(close) - max(horizons)
    return start, end
