from __future__ import annotations

from typing import Tuple

import pandas as pd

from .config import TransitionConfig


def assign_basins_topk_voxels(
    df_vox: pd.DataFrame,
    voxel_stats: pd.DataFrame,
    basin_k: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if voxel_stats.empty:
        out = df_vox.copy()
        out["basin_id"] = -1
        basin_voxels = pd.DataFrame(columns=["basin_id", "voxel_id"])
        return out, basin_voxels

    top = voxel_stats.nlargest(int(max(1, basin_k)), "occ_eff")[["voxel_id"]].copy()
    top = top.reset_index(drop=True)
    top["basin_id"] = top.index.astype("int32")
    basin_voxels = top[["basin_id", "voxel_id"]].copy()

    out = df_vox.merge(basin_voxels, on="voxel_id", how="left")
    out["basin_id"] = out["basin_id"].fillna(-1).astype("int32")
    basin_voxels = basin_voxels[["basin_id", "voxel_id"]].sort_values("basin_id").reset_index(drop=True)
    return out, basin_voxels


def compute_basin_transitions(
    df_vox: pd.DataFrame,
    cfg: TransitionConfig,
    exclude_unassigned: bool = True,
) -> pd.DataFrame:
    if df_vox.shape[0] < 2:
        return pd.DataFrame(columns=["from_basin", "to_basin", "count_eff", "prob"])

    b = df_vox["basin_id"].to_numpy(copy=False)
    from_b = b[:-1]
    to_b = b[1:]

    valid = (from_b >= 0) & (to_b >= 0) if exclude_unassigned else (from_b > -10**12)
    from_b = from_b[valid]
    to_b = to_b[valid]
    if from_b.shape[0] == 0:
        return pd.DataFrame(columns=["from_basin", "to_basin", "count_eff", "prob"])

    if cfg.decay_halflife_bars and cfg.decay_halflife_bars > 0:
        import numpy as np

        m = from_b.shape[0]
        lam = np.log(2.0) / float(cfg.decay_halflife_bars)
        w = np.exp(-lam * (m - 1 - np.arange(m, dtype=float)))
    else:
        import numpy as np

        w = np.ones(from_b.shape[0], dtype=float)

    t = pd.DataFrame({"from_basin": from_b, "to_basin": to_b, "w": w})
    out = (
        t.groupby(["from_basin", "to_basin"], as_index=False)["w"]
        .sum()
        .rename(columns={"w": "count_eff"})
    )
    out["total"] = out.groupby("from_basin")["count_eff"].transform("sum")
    out["prob"] = out["count_eff"] / (out["total"] + cfg.eps)
    return out[["from_basin", "to_basin", "count_eff", "prob"]].sort_values(
        ["from_basin", "prob"], ascending=[True, False]
    )


def compute_basin_stats(
    basin_voxels: pd.DataFrame,
    basin_transitions: pd.DataFrame,
    voxel_stats: pd.DataFrame,
    cfg: TransitionConfig,
) -> pd.DataFrame:
    if basin_voxels.empty:
        return pd.DataFrame(
            columns=[
                "basin_id",
                "occ_share",
                "basin_persistence",
                "avg_entropy",
                "avg_speed",
                "center_x",
                "center_y",
                "center_z",
            ]
        )

    bv = basin_voxels.merge(voxel_stats, on="voxel_id", how="left")
    total_occ = float(voxel_stats["occ_eff"].sum()) if not voxel_stats.empty else 0.0
    basin = bv.groupby("basin_id", as_index=False).agg(
        occ_eff=("occ_eff", "sum"),
        w_entropy=("entropy", lambda s: 0.0),  # placeholder overridden below
    )
    # weighted features
    wsum = bv.groupby("basin_id", as_index=False)["occ_eff"].sum().rename(columns={"occ_eff": "w"})
    w = bv["occ_eff"] + cfg.eps
    bv = bv.copy()
    bv["we"] = w * bv["entropy"].fillna(0.0)
    bv["ws"] = w * bv["speed"].fillna(0.0)
    bv["wx"] = w * bv["center_x"].fillna(0.0)
    bv["wy"] = w * bv["center_y"].fillna(0.0)
    bv["wz"] = w * bv["center_z"].fillna(0.0)
    agg = bv.groupby("basin_id", as_index=False).agg(
        w=("occ_eff", "sum"),
        we=("we", "sum"),
        ws=("ws", "sum"),
        wx=("wx", "sum"),
        wy=("wy", "sum"),
        wz=("wz", "sum"),
    )
    basin = basin.drop(columns=["w_entropy"]).merge(agg, on="basin_id", how="left")
    basin["occ_share"] = basin["occ_eff"] / (total_occ + cfg.eps)
    basin["avg_entropy"] = basin["we"] / (basin["w"] + cfg.eps)
    basin["avg_speed"] = basin["ws"] / (basin["w"] + cfg.eps)
    basin["center_x"] = basin["wx"] / (basin["w"] + cfg.eps)
    basin["center_y"] = basin["wy"] / (basin["w"] + cfg.eps)
    basin["center_z"] = basin["wz"] / (basin["w"] + cfg.eps)

    if basin_transitions.empty:
        basin["basin_persistence"] = 0.0
    else:
        self_t = basin_transitions[basin_transitions["from_basin"] == basin_transitions["to_basin"]][
            ["from_basin", "prob"]
        ].rename(columns={"from_basin": "basin_id", "prob": "basin_persistence"})
        basin = basin.merge(self_t, on="basin_id", how="left")
        basin["basin_persistence"] = basin["basin_persistence"].fillna(0.0)

    cols = [
        "basin_id",
        "occ_share",
        "basin_persistence",
        "avg_entropy",
        "avg_speed",
        "center_x",
        "center_y",
        "center_z",
    ]
    return basin[cols].sort_values("occ_share", ascending=False).reset_index(drop=True)
