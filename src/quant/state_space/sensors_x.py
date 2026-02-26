from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StateSpaceConfig
from .utils_robust import norm_score, robust_z


def compute_sensors_x(feat: pd.DataFrame, cfg: StateSpaceConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=feat.index)
    log_close = feat["log_close"]
    close = feat["close"]

    slope_scores = []
    er_scores = []

    for h in cfg.horizons:
        slope_h = (log_close - log_close.shift(h)) / float(h)
        slope_z_h = robust_z(slope_h, window=cfg.window_W, eps=cfg.eps)
        slope_s_h = norm_score(slope_z_h, alpha=cfg.alpha, zmax=cfg.z_max)
        out[f"x_slope_h{h}"] = slope_s_h
        slope_scores.append(slope_s_h)

        num = (close - close.shift(h)).abs()
        den = close.diff().abs().rolling(window=h, min_periods=h).sum()
        er_h = num / (den + cfg.eps)
        er_z_h = robust_z(er_h, window=cfg.window_W, eps=cfg.eps)
        sign_h = np.sign(close - close.shift(h))
        er_s_h = sign_h * norm_score(er_z_h, alpha=cfg.alpha, zmax=cfg.z_max)
        out[f"x_er_h{h}"] = er_s_h
        er_scores.append(er_s_h)

    out["x_slope"] = pd.concat(slope_scores, axis=1).mean(axis=1)
    out["x_er"] = pd.concat(er_scores, axis=1).mean(axis=1)

    long_h = max(cfg.horizons)
    slope_long = (log_close - log_close.shift(long_h)) / float(long_h)
    sign_long = np.sign(slope_long)
    align_cols = []
    for h in cfg.horizons:
        slope_h = (log_close - log_close.shift(h)) / float(h)
        sign_h = np.sign(slope_h)
        aligned = (sign_h == sign_long).astype(float)
        aligned = aligned.where(sign_long != 0.0, np.nan)
        align_cols.append(aligned)
    frac = pd.concat(align_cols, axis=1).mean(axis=1)
    out["x_align"] = 2.0 * frac - 1.0
    return out
