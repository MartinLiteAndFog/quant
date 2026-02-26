from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def aggregate_axis(
    sensor_df: pd.DataFrame, weights: Mapping[str, float] | None = None
) -> pd.DataFrame:
    if sensor_df.empty:
        raise ValueError("sensor_df must contain at least one sensor column")

    cols = list(sensor_df.columns)
    if weights is None:
        w = np.ones(len(cols), dtype=float)
    else:
        w = np.array([float(weights.get(c, 0.0)) for c in cols], dtype=float)
        if np.allclose(w.sum(), 0.0):
            w = np.ones(len(cols), dtype=float)

    w = w / (w.sum() + 1e-12)
    values = sensor_df.to_numpy(dtype=float)
    valid = np.isfinite(values)

    weighted_vals = np.where(valid, values, 0.0) * w
    weight_mask = np.where(valid, 1.0, 0.0) * w
    signal = weighted_vals.sum(axis=1) / np.clip(weight_mask.sum(axis=1), 1e-12, None)
    signal_s = pd.Series(signal, index=sensor_df.index, name="signal")

    disagreement = sensor_df.std(axis=1, ddof=0)
    reliability = (1.0 - disagreement).clip(lower=0.0, upper=1.0)
    conf = (signal_s * reliability).clip(lower=-1.0, upper=1.0)

    out = pd.DataFrame(index=sensor_df.index)
    out["signal"] = signal_s
    out["disagreement"] = disagreement
    out["reliability"] = reliability
    out["conf"] = conf
    return out
