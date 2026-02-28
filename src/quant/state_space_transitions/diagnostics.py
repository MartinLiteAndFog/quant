from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TransitionConfig
from .topk import compute_p_self


def build_voxel_stats(
    occ: pd.DataFrame,
    transitions_topk: pd.DataFrame,
    steps: pd.DataFrame,
    cfg: TransitionConfig,
) -> pd.DataFrame:
    if occ.empty:
        return pd.DataFrame(
            columns=[
                "voxel_id",
                "occ_eff",
                "pi",
                "p_self",
                "escape",
                "holding_time",
                "entropy",
                "drift_x",
                "drift_y",
                "drift_z",
                "speed",
                "support_raw",
                "center_x",
                "center_y",
                "center_z",
            ]
        )

    voxel = occ.copy()
    total_occ = float(voxel["occ_eff"].sum())
    voxel["pi"] = voxel["occ_eff"] / (total_occ + cfg.eps)

    p_self = compute_p_self(transitions_topk)
    voxel = voxel.merge(p_self, on="voxel_id", how="left")
    voxel["p_self"] = voxel["p_self"].fillna(0.0)
    voxel["escape"] = 1.0 - voxel["p_self"]
    voxel["holding_time"] = 1.0 / (1.0 - voxel["p_self"] + cfg.eps)

    if transitions_topk.empty:
        entropy = pd.DataFrame({"voxel_id": voxel["voxel_id"], "entropy": 0.0})
    else:
        topk = transitions_topk.copy()
        topk["term"] = topk["prob"] * np.log(topk["prob"] + cfg.eps)
        entropy = (
            topk.groupby("from_id", as_index=False)["term"]
            .sum()
            .rename(columns={"from_id": "voxel_id"})
        )
        entropy["entropy"] = -entropy["term"]
        entropy = entropy[["voxel_id", "entropy"]]
    voxel = voxel.merge(entropy, on="voxel_id", how="left")
    voxel["entropy"] = voxel["entropy"].fillna(0.0)

    if steps.empty:
        drift = pd.DataFrame(
            {
                "voxel_id": voxel["voxel_id"],
                "drift_x": 0.0,
                "drift_y": 0.0,
                "drift_z": 0.0,
                "center_x": np.nan,
                "center_y": np.nan,
                "center_z": np.nan,
            }
        )
    else:
        steps_w = steps.copy()
        for c in ["dx", "dy", "dz", "cx", "cy", "cz"]:
            steps_w[f"w_{c}"] = steps_w["w"] * steps_w[c]
        drift = steps_w.groupby("from_id", as_index=False).agg(
            w_sum=("w", "sum"),
            w_dx=("w_dx", "sum"),
            w_dy=("w_dy", "sum"),
            w_dz=("w_dz", "sum"),
            w_cx=("w_cx", "sum"),
            w_cy=("w_cy", "sum"),
            w_cz=("w_cz", "sum"),
        )
        drift["drift_x"] = drift["w_dx"] / (drift["w_sum"] + cfg.eps)
        drift["drift_y"] = drift["w_dy"] / (drift["w_sum"] + cfg.eps)
        drift["drift_z"] = drift["w_dz"] / (drift["w_sum"] + cfg.eps)
        drift["center_x"] = drift["w_cx"] / (drift["w_sum"] + cfg.eps)
        drift["center_y"] = drift["w_cy"] / (drift["w_sum"] + cfg.eps)
        drift["center_z"] = drift["w_cz"] / (drift["w_sum"] + cfg.eps)
        drift = drift.rename(columns={"from_id": "voxel_id"})
        drift = drift[["voxel_id", "drift_x", "drift_y", "drift_z", "center_x", "center_y", "center_z"]]

    voxel = voxel.merge(drift, on="voxel_id", how="left")
    voxel[["drift_x", "drift_y", "drift_z"]] = voxel[["drift_x", "drift_y", "drift_z"]].fillna(0.0)
    voxel["speed"] = np.sqrt(voxel["drift_x"] ** 2 + voxel["drift_y"] ** 2 + voxel["drift_z"] ** 2)

    if "support_raw" not in voxel.columns:
        voxel["support_raw"] = 0
    voxel["support_raw"] = voxel["support_raw"].fillna(0).astype(np.int64)

    cols = [
        "voxel_id",
        "occ_eff",
        "pi",
        "p_self",
        "escape",
        "holding_time",
        "entropy",
        "drift_x",
        "drift_y",
        "drift_z",
        "speed",
        "support_raw",
        "center_x",
        "center_y",
        "center_z",
    ]
    voxel = voxel[cols].sort_values("occ_eff", ascending=False).reset_index(drop=True)
    return voxel
