from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import TransitionConfig


def build_sparse_transition_counts(
    df_vox: pd.DataFrame,
    cfg: TransitionConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_col, y_col, z_col = cfg.axes
    v = df_vox["voxel_id"].to_numpy(dtype=np.int64, copy=False)
    x = df_vox[x_col].to_numpy(dtype=np.float64, copy=False)
    y = df_vox[y_col].to_numpy(dtype=np.float64, copy=False)
    z = df_vox[z_col].to_numpy(dtype=np.float64, copy=False)

    n = v.shape[0]
    if n < 2:
        empty_edges = pd.DataFrame(columns=["from_id", "to_id", "count_eff"])
        empty_occ = pd.DataFrame(columns=["voxel_id", "occ_eff", "support_raw"])
        empty_steps = pd.DataFrame(
            columns=["from_id", "to_id", "w", "dx", "dy", "dz", "cx", "cy", "cz"]
        )
        return empty_edges, empty_occ, empty_steps

    from_id = v[:-1]
    to_id = v[1:]
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    dz = z[1:] - z[:-1]

    valid = (
        (from_id >= 0)
        & (to_id >= 0)
        & np.isfinite(dx)
        & np.isfinite(dy)
        & np.isfinite(dz)
    )

    from_id = from_id[valid]
    to_id = to_id[valid]
    dx = dx[valid]
    dy = dy[valid]
    dz = dz[valid]
    cx = df_vox["cx"].to_numpy(dtype=np.float64, copy=False)[:-1][valid]
    cy = df_vox["cy"].to_numpy(dtype=np.float64, copy=False)[:-1][valid]
    cz = df_vox["cz"].to_numpy(dtype=np.float64, copy=False)[:-1][valid]

    m = from_id.shape[0]
    if cfg.decay_halflife_bars and cfg.decay_halflife_bars > 0:
        lam = np.log(2.0) / float(cfg.decay_halflife_bars)
        # Most recent step gets weight=1.
        w = np.exp(-lam * (m - 1 - np.arange(m, dtype=np.float64)))
    else:
        w = np.ones(m, dtype=np.float64)

    steps = pd.DataFrame(
        {
            "from_id": from_id.astype(np.int32),
            "to_id": to_id.astype(np.int32),
            "w": w,
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "cx": cx,
            "cy": cy,
            "cz": cz,
        }
    )

    edges_counts = (
        steps.groupby(["from_id", "to_id"], as_index=False)["w"]
        .sum()
        .rename(columns={"w": "count_eff"})
    )
    occ_eff = (
        steps.groupby("from_id", as_index=False)["w"]
        .sum()
        .rename(columns={"from_id": "voxel_id", "w": "occ_eff"})
    )
    support_raw = (
        steps.groupby("from_id", as_index=False)
        .size()
        .rename(columns={"from_id": "voxel_id", "size": "support_raw"})
    )
    occ = occ_eff.merge(support_raw, on="voxel_id", how="outer").fillna(0.0)
    occ["voxel_id"] = occ["voxel_id"].astype(np.int32)
    occ["occ_eff"] = occ["occ_eff"].astype(np.float64)
    occ["support_raw"] = occ["support_raw"].astype(np.int64)
    return edges_counts, occ, steps
