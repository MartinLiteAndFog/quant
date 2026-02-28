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

    # Vectorized rolling std using cumulative sums
    r1_filled = np.where(np.isnan(r1), 0.0, r1)
    r1_valid = (~np.isnan(r1)).astype(np.float64)
    cs = np.cumsum(r1_filled)
    cs2 = np.cumsum(r1_filled**2)
    cn = np.cumsum(r1_valid)

    for i, w in enumerate([20, 60]):
        col_idx = 3 + i
        s = cs[w:] - cs[:-w]
        s2 = cs2[w:] - cs2[:-w]
        cnt = cn[w:] - cn[:-w]
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (s2 - s**2 / cnt) / (cnt - 1)
            std = np.sqrt(np.maximum(var, 0.0))
        valid_mask = cnt >= 2
        obs[w:, col_idx] = np.where(valid_mask, std, np.nan)
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
