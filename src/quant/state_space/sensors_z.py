from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StateSpaceConfig
from .utils_robust import norm_score, rolling_mad_threshold, robust_z


def _rolling_entropy_binary_sign(r: pd.Series, window: int, eps: float) -> pd.Series:
    up = (r > 0.0).astype(float)
    p_up = up.rolling(window=window, min_periods=window).mean()
    p_dn = 1.0 - p_up
    return -(p_up * np.log(p_up + eps) + p_dn * np.log(p_dn + eps))


def compute_sensors_z(feat: pd.DataFrame, cfg: StateSpaceConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=feat.index)
    r = feat["r"]

    mad_r = rolling_mad_threshold(r, window=cfg.W_jump_mad, eps=cfg.eps)
    jump = (r.abs() > (cfg.k_jump * mad_r)).astype(float)
    intensity = jump.ewm(alpha=cfg.lambda_jump, adjust=False, min_periods=1).mean()
    out["z_jump_rate_raw"] = intensity
    out["z_jump_rate"] = norm_score(
        robust_z(intensity, window=cfg.window_W, eps=cfg.eps),
        alpha=cfg.alpha,
        zmax=cfg.z_max,
    )

    rv = (r * r).rolling(window=cfg.W_jv, min_periods=cfg.W_jv).sum()
    bv = (
        (np.pi / 2.0)
        * (r.abs() * r.shift(1).abs()).rolling(window=cfg.W_jv, min_periods=cfg.W_jv).sum()
    )
    jv = (rv - bv).clip(lower=0.0) / (rv + cfg.eps)
    out["z_rv_bv_jumpiness_raw"] = jv
    out["z_rv_bv_jumpiness"] = norm_score(
        robust_z(jv, window=cfg.window_W, eps=cfg.eps),
        alpha=cfg.alpha,
        zmax=cfg.z_max,
    )

    ent = _rolling_entropy_binary_sign(r, window=cfg.W_ent, eps=cfg.eps)
    out["z_entropy_raw"] = ent
    out["z_entropy"] = norm_score(
        robust_z(ent, window=cfg.window_W, eps=cfg.eps),
        alpha=cfg.alpha,
        zmax=cfg.z_max,
    )

    noise = feat["wick_noise_raw"].rolling(window=cfg.W_noise, min_periods=cfg.W_noise).mean()
    out["z_wick_noise_raw"] = noise
    out["z_wick_noise"] = norm_score(
        robust_z(noise, window=cfg.window_W, eps=cfg.eps),
        alpha=cfg.alpha,
        zmax=cfg.z_max,
    )
    return out
