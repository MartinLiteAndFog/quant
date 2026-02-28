from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def _metric_cfg(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    return cfg.get("scales", {}).get(key, {})


def quantile_clip(series: pd.Series, cfg: Dict[str, Any]) -> pd.Series:
    qcfg = _metric_cfg(cfg, "quantile_clip")
    if not qcfg.get("enabled", True):
        return series

    low = float(qcfg.get("low", 0.02))
    high = float(qcfg.get("high", 0.98))

    s = pd.to_numeric(series, errors="coerce")
    if len(s) == 0:
        return s

    lo = float(s.quantile(low))
    hi = float(s.quantile(high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return s

    return s.clip(lower=lo, upper=hi)


def maybe_log1p(series: pd.Series, metric: str, cfg: Dict[str, Any]) -> pd.Series:
    lcfg = _metric_cfg(cfg, "log_scale_metrics")
    if not lcfg.get("enabled", True):
        return series

    metrics = set(lcfg.get("metrics", []))
    if metric not in metrics:
        return series

    s = pd.to_numeric(series, errors="coerce")
    return np.log1p(s.clip(lower=0.0))


def cdf_stretch(series: pd.Series, metric: str, cfg: Dict[str, Any]) -> pd.Series:
    # High-contrast stretch using ranks (empirical CDF).
    ccfg = _metric_cfg(cfg, "cdf_stretch")
    if not ccfg.get("enabled", False):
        return series

    metrics = set(ccfg.get("metrics", []))
    if metric not in metrics:
        return series

    s = pd.to_numeric(series, errors="coerce")
    if len(s) == 0:
        return s

    return s.rank(pct=True, method="average")


def normalize_0_1(series: pd.Series, metric: str, cfg: Dict[str, Any], eps: float = 1e-12) -> pd.Series:
    ncfg = _metric_cfg(cfg, "normalize_to_0_1")
    if not ncfg.get("enabled", True):
        return series

    metrics = set(ncfg.get("metrics", []))
    if metric not in metrics:
        return series

    s = pd.to_numeric(series, errors="coerce")
    if len(s) == 0:
        return s

    mn = float(s.min())
    mx = float(s.max())
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return s * 0.0

    return (s - mn) / (mx - mn + eps)


def scale_metric(series: pd.Series, metric: str, cfg: Dict[str, Any]) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = quantile_clip(s, cfg)
    s = maybe_log1p(s, metric, cfg)
    s = cdf_stretch(s, metric, cfg)
    s = normalize_0_1(s, metric, cfg)
    return s


def to_marker_sizes(series: pd.Series, min_size: float, max_size: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return min_size + (max_size - min_size) * s
