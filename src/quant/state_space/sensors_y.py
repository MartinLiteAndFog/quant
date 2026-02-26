from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StateSpaceConfig
from .utils_robust import norm_score, robust_z


def rolling_ou_beta(delta_p: pd.Series, d_lag: pd.Series, window: int) -> pd.Series:
    x = d_lag
    y = delta_p
    mx = x.rolling(window=window, min_periods=window).mean()
    my = y.rolling(window=window, min_periods=window).mean()
    cov = (x * y).rolling(window=window, min_periods=window).mean() - mx * my
    var = (x * x).rolling(window=window, min_periods=window).mean() - mx * mx
    beta = cov / var.replace(0.0, np.nan)
    return beta


def compute_sensors_y(feat: pd.DataFrame, cfg: StateSpaceConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=feat.index)
    d = feat["d"]
    r = feat["r"]
    delta_p = feat["delta_p"]

    z_dev = robust_z(d, window=cfg.window_W, eps=cfg.eps)
    out["y_dev"] = -norm_score(z_dev, alpha=cfg.alpha, zmax=cfg.z_max)

    beta = rolling_ou_beta(delta_p=delta_p, d_lag=d.shift(1), window=cfg.W_ou)
    out["y_ou_beta_raw"] = beta
    z_beta = robust_z(-beta, window=cfg.window_W, eps=cfg.eps)
    out["y_ou_pull"] = norm_score(z_beta, alpha=cfg.alpha, zmax=cfg.z_max)

    ac1 = r.rolling(window=cfg.W_ac1, min_periods=cfg.W_ac1).corr(r.shift(1))
    out["y_ac1_raw"] = ac1
    z_ac1 = robust_z(ac1, window=cfg.window_W, eps=cfg.eps)
    out["y_ac1"] = -norm_score(z_ac1, alpha=cfg.alpha, zmax=cfg.z_max)
    return out
