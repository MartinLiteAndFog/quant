from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StateSpaceConfig


REQUIRED_INPUT_COLS = ("ts", "open", "high", "low", "close")


def _require_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe missing required columns: {missing}")


def compute_features(df: pd.DataFrame, cfg: StateSpaceConfig) -> pd.DataFrame:
    _require_columns(df)
    out = df.copy()

    out["open"] = out["open"].astype(float)
    out["high"] = out["high"].astype(float)
    out["low"] = out["low"].astype(float)
    out["close"] = out["close"].astype(float)

    out["typical"] = (out["high"] + out["low"] + out["close"]) / 3.0
    has_volume = "volume" in out.columns
    if has_volume:
        out["volume"] = out["volume"].astype(float)
        num = (out["typical"] * out["volume"]).rolling(cfg.window_W, min_periods=1).sum()
        den = out["volume"].rolling(cfg.window_W, min_periods=1).sum()
        vwap_roll = num / (den + cfg.eps)
        out["eq_base"] = np.where(den > 0.0, vwap_roll, out["typical"])
    else:
        out["eq_base"] = out["typical"]

    out["p_eq"] = out["eq_base"].ewm(alpha=cfg.lambda_eq, adjust=False, min_periods=1).mean()
    out["d"] = out["close"] - out["p_eq"]

    out["log_close"] = np.log(out["close"].clip(lower=cfg.eps))
    out["r"] = out["log_close"].diff()
    out["delta_p"] = out["close"].diff()

    body = (out["close"] - out["open"]).abs()
    upper_wick = (out["high"] - np.maximum(out["open"], out["close"])).clip(lower=0.0)
    lower_wick = (np.minimum(out["open"], out["close"]) - out["low"]).clip(lower=0.0)
    out["wick"] = upper_wick + lower_wick
    out["body"] = body
    out["wick_noise_raw"] = out["wick"] / (out["body"] + cfg.eps)
    return out
