# scripts/visual/build_basins_v02.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BuildBasinsV02Config:
    # Quantile threshold on flow_mass = pi(from) * p(from->to)
    flow_mass_quantile: float = 0.90
    # Minimum component size to be considered a basin (else voxel stays noise basin_id=-1)
    min_component_size: int = 2
    # Output filename (relative to in_dir)
    out_name: str = "basins_v02_components.parquet"


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    return pd.read_parquet(path)


def _connected_components_undirected(nodes: np.ndarray, edges_undirected: pd.DataFrame) -> np.ndarray:
    """
    nodes: array of voxel_id (int)
    edges_undirected: columns ['a','b'] as voxel ids
    returns: component id per node index (>=0) or -1 (unassigned)
    """
    nodes_sorted = np.array(sorted(set(int(x) for x in nodes)), dtype=np.int64)
    idx: Dict[int, int] = {int(v): i for i, v in enumerate(nodes_sorted)}

    adj: List[List[int]] = [[] for _ in range(len(nodes_sorted))]
    for a, b in edges_undirected.itertuples(index=False):
        a = int(a)
        b = int(b)
        if a == b:
            continue
        ia = idx.get(a)
        ib = idx.get(b)
        if ia is None or ib is None:
            continue
        adj[ia].append(ib)
        adj[ib].append(ia)

    seen = np.zeros(len(nodes_sorted), dtype=bool)
    comp = -np.ones(len(nodes_sorted), dtype=np.int64)
    cid = 0

    for i in range(len(nodes_sorted)):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        members: List[int] = []
        while stack:
            j = stack.pop()
            members.append(j)
            for k in adj[j]:
                if not seen[k]:
                    seen[k] = True
                    stack.append(k)

        # Label later; caller decides min size.
        if len(members) > 0:
            for j in members:
                comp[j] = cid
            cid += 1

    return nodes_sorted, comp


def build_basins_v02(
    in_dir: Path,
    cfg: BuildBasinsV02Config,
) -> Tuple[Path, pd.DataFrame]:
    """
    Creates basins_v02_components.parquet based on strong-flow connected components.

    Requires in_dir files:
      - voxel_stats.parquet with columns: voxel_id, pi
      - transitions_topk.parquet with columns: from_voxel_id, to_voxel_id, p (or prob)

    Writes:
      - basins_v02_components.parquet with columns: voxel_id, basin_id
        basin_id = 0..K-1 for basins (components size>=min_component_size), else -1
    """
    stats_path = in_dir / "voxel_stats.parquet"
    edges_path = in_dir / "transitions_topk.parquet"

    stats = _read_parquet(stats_path)
    edges = _read_parquet(edges_path)

    # Normalize column names
    if "from_voxel_id" not in edges.columns and "from_id" in edges.columns:
        edges = edges.rename(columns={"from_id": "from_voxel_id"})
    if "to_voxel_id" not in edges.columns and "to_id" in edges.columns:
        edges = edges.rename(columns={"to_id": "to_voxel_id"})
    if "p" not in edges.columns and "prob" in edges.columns:
        edges = edges.rename(columns={"prob": "p"})

    for c in ["from_voxel_id", "to_voxel_id", "p"]:
        if c not in edges.columns:
            raise ValueError(f"transitions_topk missing column '{c}'. Have: {list(edges.columns)}")
    for c in ["voxel_id", "pi"]:
        if c not in stats.columns:
            raise ValueError(f"voxel_stats missing column '{c}'. Have: {list(stats.columns)}")

    edges = edges[["from_voxel_id", "to_voxel_id", "p"]].copy()
    stats_pi = stats[["voxel_id", "pi"]].copy()

    # Ensure int ids
    edges["from_voxel_id"] = edges["from_voxel_id"].astype(np.int64)
    edges["to_voxel_id"] = edges["to_voxel_id"].astype(np.int64)
    stats_pi["voxel_id"] = stats_pi["voxel_id"].astype(np.int64)

    edges = edges.merge(
        stats_pi.rename(columns={"voxel_id": "from_voxel_id", "pi": "pi_from"}),
        on="from_voxel_id",
        how="left",
    )
    edges["flow_mass"] = edges["pi_from"] * edges["p"]
    edges = edges.dropna(subset=["flow_mass"])

    if len(edges) == 0:
        raise ValueError("No edges after flow_mass computation (pi_from or p missing/NaN).")

    thr = float(edges["flow_mass"].quantile(cfg.flow_mass_quantile))
    keep = edges[edges["flow_mass"] >= thr][["from_voxel_id", "to_voxel_id"]].drop_duplicates()

    # Build undirected edge list over ALL voxels (components computed on all nodes)
    # Note: we compute components on full node set, but edges only where strong flow exists.
    und = keep.rename(columns={"from_voxel_id": "a", "to_voxel_id": "b"}).copy()

    nodes = stats_pi["voxel_id"].to_numpy()
    nodes_sorted, comp_raw = _connected_components_undirected(nodes, und)

    # comp_raw currently labels all connected components (including singletons).
    # We need to re-label only those with size >= min_component_size, others -> -1
    # Compute component sizes
    comp_sizes = pd.Series(comp_raw).value_counts().to_dict()

    basin_id = -np.ones_like(comp_raw)
    remap: Dict[int, int] = {}
    next_id = 0
    for c_raw, size in sorted(comp_sizes.items(), key=lambda kv: (-kv[1], kv[0])):
        if int(size) >= int(cfg.min_component_size):
            remap[int(c_raw)] = next_id
            next_id += 1

    for i, c_raw in enumerate(comp_raw):
        c_raw = int(c_raw)
        basin_id[i] = remap.get(c_raw, -1)

    out = pd.DataFrame({"voxel_id": nodes_sorted, "basin_id": basin_id.astype(np.int64)})

    out_path = in_dir / cfg.out_name
    out.to_parquet(out_path, index=False)

    return out_path, out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build v0.2 basin mapping from strong-flow components.")
    ap.add_argument("--in-dir", required=True, help="Folder containing voxel_stats.parquet and transitions_topk.parquet")
    ap.add_argument("--flow-mass-quantile", type=float, default=0.90, help="Quantile for flow_mass threshold (pi*p)")
    ap.add_argument("--min-component-size", type=int, default=2, help="Min size to be considered a basin")
    ap.add_argument("--out-name", default="basins_v02_components.parquet", help="Output parquet name")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    cfg = BuildBasinsV02Config(
        flow_mass_quantile=float(args.flow_mass_quantile),
        min_component_size=int(args.min_component_size),
        out_name=str(args.out_name),
    )

    out_path, out = build_basins_v02(in_dir, cfg)

    n_basins = int(out.loc[out["basin_id"] >= 0, "basin_id"].nunique())
    n_noise = int((out["basin_id"] < 0).sum())
    vc = out["basin_id"].value_counts().head(10)

    print(f"wrote: {out_path}")
    print(f"basins: {n_basins} | noise voxels: {n_noise} | total voxels: {len(out)}")
    print("top basin sizes:")
    print(vc.to_string())


if __name__ == "__main__":
    main()