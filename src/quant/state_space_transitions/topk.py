from __future__ import annotations

import pandas as pd

from .config import TransitionConfig


def build_topk_transitions(
    edges_counts: pd.DataFrame,
    cfg: TransitionConfig,
) -> pd.DataFrame:
    if edges_counts.empty:
        return pd.DataFrame(columns=["from_id", "to_id", "count_eff", "prob"])

    work = edges_counts.copy()
    work["from_id"] = work["from_id"].astype("int32")
    work["to_id"] = work["to_id"].astype("int32")
    work["count_eff"] = work["count_eff"].astype("float64")

    grp = work.groupby("from_id")["count_eff"]
    work["total_eff"] = grp.transform("sum")
    work["k_out"] = grp.transform("size").astype("float64")
    denom = work["total_eff"] + cfg.laplace_alpha * work["k_out"]
    work["prob"] = (work["count_eff"] + cfg.laplace_alpha) / denom

    # Keep top-k by probability per from_id.
    work = work.sort_values(["from_id", "prob", "count_eff"], ascending=[True, False, False])
    topk = work.groupby("from_id", as_index=False).head(cfg.topk).copy()
    topk = topk[["from_id", "to_id", "count_eff", "prob"]]
    return topk.reset_index(drop=True)


def compute_p_self(transitions_topk: pd.DataFrame) -> pd.DataFrame:
    if transitions_topk.empty:
        return pd.DataFrame(columns=["voxel_id", "p_self"])
    self_rows = transitions_topk[transitions_topk["from_id"] == transitions_topk["to_id"]]
    out = self_rows.groupby("from_id", as_index=False)["prob"].sum()
    out = out.rename(columns={"from_id": "voxel_id", "prob": "p_self"})
    out["voxel_id"] = out["voxel_id"].astype("int32")
    out["p_self"] = out["p_self"].astype("float64")
    return out
