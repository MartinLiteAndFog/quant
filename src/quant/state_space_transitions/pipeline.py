from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .basins import assign_basins_topk_voxels, compute_basin_stats, compute_basin_transitions
from .config import TransitionConfig
from .counts import build_sparse_transition_counts
from .diagnostics import build_voxel_stats
from .topk import build_topk_transitions
from .voxelizer import voxelize_dataframe


def _jsonify_edges(edges_map: Dict[str, object], mids_map: Dict[str, object], n_bins: int) -> Dict[str, object]:
    return {
        "n_bins": int(n_bins),
        "edges": {k: [float(x) for x in v] for k, v in edges_map.items()},
        "mids": {k: [float(x) for x in v] for k, v in mids_map.items()},
    }


def run_pipeline(
    df: pd.DataFrame,
    cfg: TransitionConfig,
    out_dir: Path | str,
    basin_k: int = 30,
) -> Dict[str, int]:
    if "ts" not in df.columns:
        raise ValueError("input dataframe must include 'ts' column")
    for col in cfg.axes:
        if col not in df.columns:
            raise ValueError(f"input dataframe missing required axis column: {col}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    work = df.copy()
    work["ts"] = pd.to_datetime(work["ts"], utc=True, errors="coerce")
    work = work.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    for c in cfg.axes:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    df_vox, edges_map, mids_map = voxelize_dataframe(work, cfg)
    edges_counts, occ, steps = build_sparse_transition_counts(df_vox, cfg)
    transitions_topk = build_topk_transitions(edges_counts, cfg)
    voxel_stats = build_voxel_stats(occ=occ, transitions_topk=transitions_topk, steps=steps, cfg=cfg)
    df_vox_basin, basin_voxels = assign_basins_topk_voxels(df_vox, voxel_stats, basin_k=basin_k)
    basin_transitions = compute_basin_transitions(df_vox_basin, cfg=cfg, exclude_unassigned=True)
    basin_stats = compute_basin_stats(
        basin_voxels=basin_voxels,
        basin_transitions=basin_transitions,
        voxel_stats=voxel_stats,
        cfg=cfg,
    )

    with open(out_path / "edges.json", "w", encoding="utf-8") as f:
        json.dump(_jsonify_edges(edges_map, mids_map, cfg.n_bins), f, ensure_ascii=True, indent=2)

    cols = ["ts", cfg.axes[0], cfg.axes[1], cfg.axes[2], "ix", "iy", "iz", "voxel_id", "basin_id", "cx", "cy", "cz"]
    df_vox_basin[cols].to_parquet(out_path / "df_with_voxels.parquet", index=False)
    edges_counts.to_parquet(out_path / "edges_counts.parquet", index=False)
    transitions_topk.to_parquet(out_path / "transitions_topk.parquet", index=False)
    voxel_stats.to_parquet(out_path / "voxel_stats.parquet", index=False)
    basin_voxels.to_parquet(out_path / "basin_voxels.parquet", index=False)
    basin_transitions.to_parquet(out_path / "basin_transitions.parquet", index=False)
    basin_stats.to_parquet(out_path / "basin_stats.parquet", index=False)

    return {
        "n_rows": int(df_vox_basin.shape[0]),
        "n_edges_counts": int(edges_counts.shape[0]),
        "n_topk": int(transitions_topk.shape[0]),
        "n_voxels": int(voxel_stats.shape[0]),
        "n_basins": int(basin_voxels["basin_id"].nunique() if not basin_voxels.empty else 0),
    }
