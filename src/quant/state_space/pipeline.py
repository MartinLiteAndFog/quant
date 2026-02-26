from __future__ import annotations

import pandas as pd

from .axes import aggregate_axis
from .basin import voxel_basins
from .config import StateSpaceConfig
from .features import compute_features
from .residualize import residualize_axes
from .sensors_x import compute_sensors_x
from .sensors_y import compute_sensors_y
from .sensors_z import compute_sensors_z


def compute_state_space(df: pd.DataFrame, cfg: StateSpaceConfig | None = None) -> pd.DataFrame:
    cfg = cfg or StateSpaceConfig()
    feat = compute_features(df, cfg)

    sx = compute_sensors_x(feat, cfg)
    sy = compute_sensors_y(feat, cfg)
    sz = compute_sensors_z(feat, cfg)

    x_axis = aggregate_axis(sx[["x_slope", "x_er", "x_align"]], weights=cfg.sensor_weights_x)
    y_axis = aggregate_axis(sy[["y_dev", "y_ou_pull", "y_ac1"]], weights=cfg.sensor_weights_y)
    z_axis = aggregate_axis(
        sz[["z_jump_rate", "z_rv_bv_jumpiness", "z_entropy", "z_wick_noise"]],
        weights=cfg.sensor_weights_z,
    )

    out = pd.DataFrame(index=df.index)
    out["ts"] = df["ts"]

    out["X_raw"] = x_axis["signal"]
    out["Y_raw"] = y_axis["signal"]
    out["Z_raw"] = z_axis["signal"]

    res = residualize_axes(out["X_raw"], out["Y_raw"], out["Z_raw"])
    out["Y_res"] = res["Y_res"]
    out["Z_res"] = res["Z_res"]

    out["signal_x"] = x_axis["signal"]
    out["reliability_x"] = x_axis["reliability"]
    out["disagreement_x"] = x_axis["disagreement"]
    out["conf_x"] = x_axis["conf"]

    out["signal_y"] = y_axis["signal"]
    out["reliability_y"] = y_axis["reliability"]
    out["disagreement_y"] = y_axis["disagreement"]
    out["conf_y"] = y_axis["conf"]

    out["signal_z"] = z_axis["signal"]
    out["reliability_z"] = z_axis["reliability"]
    out["disagreement_z"] = z_axis["disagreement"]
    out["conf_z"] = z_axis["conf"]

    debug_cols = [
        "x_slope",
        "x_er",
        "x_align",
        "y_dev",
        "y_ou_pull",
        "y_ac1",
        "z_jump_rate",
        "z_rv_bv_jumpiness",
        "z_entropy",
        "z_wick_noise",
    ]
    for c in debug_cols:
        if c in sx.columns:
            out[c] = sx[c]
        if c in sy.columns:
            out[c] = sy[c]
        if c in sz.columns:
            out[c] = sz[c]

    basins, labels = voxel_basins(
        out["X_raw"],
        out["Y_res"],
        out["Z_res"],
        bins=cfg.basin_bins,
        top_k=cfg.basin_top_k,
        include_labels=cfg.include_basin_labels,
    )
    out.attrs["basins"] = basins.to_dict(orient="records")
    if labels is not None:
        out["voxel_label"] = labels
    return out
