"""Gate Confidence Projection — state-based gate rates + Markov forward propagation.

Outputs:
    voxel_gate_stats.parquet    — per-voxel gate-on rate
    basin_gate_stats.parquet    — per-basin pi-weighted gate-on rate
    confidence_curve.parquet    — forward-looking E[gate|S_{t+h}] at multiple horizons

Usage:
    python -m scripts.visual.build_gate_confidence \
        --config configs/viz/v0.2.yaml \
        --in-dir data/runs/visual_v02_seed/transitions \
        --out-dir data/runs/visual_v02_seed/gate_confidence
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.visual.common import base_parser, load_cfg, ensure_path
from visual.io import load_contracts


# ---------------------------------------------------------------------------
# Gate loading + alignment
# ---------------------------------------------------------------------------

def load_gate(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load daily gate CSV/Parquet; return DataFrame[ts, gate_on] sorted by ts."""
    gate_cfg = cfg.get("gate", {})
    path_str = str(gate_cfg.get("path", ""))
    if not path_str:
        raise ValueError("gate.path not set in config")

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"gate file not found: {path}")

    ts_col = str(gate_cfg.get("ts_col", "ts"))
    gate_col = str(gate_cfg.get("col", "gate_on_2of3"))

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if ts_col not in df.columns:
        raise ValueError(f"gate file missing ts column '{ts_col}': {list(df.columns)}")
    if gate_col not in df.columns:
        raise ValueError(f"gate file missing gate column '{gate_col}': {list(df.columns)}")

    out = df[[ts_col, gate_col]].copy()
    out = out.rename(columns={ts_col: "ts", gate_col: "gate_on"})
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    out["gate_on"] = pd.to_numeric(out["gate_on"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    return out


def align_gate_to_voxels(
    voxel_map: pd.DataFrame,
    gate: pd.DataFrame,
    direction: str = "backward",
) -> pd.DataFrame:
    """merge_asof: assign each voxel_map row the most recent gate value."""
    vm = voxel_map[["ts", "voxel_id"]].copy()
    vm["ts"] = pd.to_datetime(vm["ts"], utc=True, errors="coerce")
    vm = vm.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    g = gate[["ts", "gate_on"]].sort_values("ts").reset_index(drop=True)

    merged = pd.merge_asof(vm, g, on="ts", direction=direction)
    merged["gate_on"] = merged["gate_on"].fillna(0).astype(int).clip(0, 1)
    return merged


# ---------------------------------------------------------------------------
# Voxel-level gate stats
# ---------------------------------------------------------------------------

def _wilson_ci(n_success: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p_hat = n_success / n
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def build_voxel_gate_stats(aligned: pd.DataFrame) -> pd.DataFrame:
    """Per-voxel gate-on rate with Wilson CI."""
    grouped = aligned.groupby("voxel_id", as_index=False).agg(
        n=("gate_on", "count"),
        gate_on_sum=("gate_on", "sum"),
        last_ts=("ts", "max"),
    )
    grouped["gate_on_rate"] = grouped["gate_on_sum"] / grouped["n"].clip(lower=1)

    ci_lo, ci_hi = [], []
    for _, row in grouped.iterrows():
        lo, hi = _wilson_ci(int(row["gate_on_sum"]), int(row["n"]))
        ci_lo.append(lo)
        ci_hi.append(hi)
    grouped["ci_lo"] = ci_lo
    grouped["ci_hi"] = ci_hi
    grouped = grouped.drop(columns=["gate_on_sum"])
    return grouped


# ---------------------------------------------------------------------------
# Basin-level gate stats
# ---------------------------------------------------------------------------

def build_basin_gate_stats(
    voxel_gate: pd.DataFrame,
    voxel_stats: pd.DataFrame,
    basins: pd.DataFrame,
    *,
    noise_basin_id: int = -1,
) -> pd.DataFrame:
    """pi-weighted gate_on_rate per basin."""
    b = basins[["voxel_id", "basin_id"]].copy()
    b["basin_id"] = pd.to_numeric(b["basin_id"], errors="coerce").fillna(noise_basin_id).astype(int)
    b = b[b["basin_id"] != noise_basin_id].copy()

    pi = voxel_stats[["voxel_id", "pi"]].copy()
    pi["pi"] = pd.to_numeric(pi["pi"], errors="coerce").fillna(0.0)

    m = b.merge(voxel_gate[["voxel_id", "gate_on_rate", "n"]], on="voxel_id", how="left")
    m = m.merge(pi, on="voxel_id", how="left")
    m["gate_on_rate"] = m["gate_on_rate"].fillna(0.0)
    m["pi"] = m["pi"].fillna(0.0)

    rows = []
    for bid, g in m.groupby("basin_id"):
        total_pi = float(g["pi"].sum())
        w = g["pi"].to_numpy()
        r = g["gate_on_rate"].to_numpy()
        wsum = w.sum()
        wrate = float(np.dot(w, r) / max(wsum, 1e-12))
        rows.append({
            "basin_id": int(bid),
            "n_voxels": len(g),
            "mass_pi": total_pi,
            "gate_on_rate": wrate,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("mass_pi", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sparse Markov operator (pure numpy, no scipy)
# ---------------------------------------------------------------------------

def _build_sparse_transition(
    transitions_topk: pd.DataFrame,
    voxel_ids: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, int], List[List[Tuple[int, float]]]]:
    """Build row-stochastic sparse transition from top-k transitions.

    Returns:
        voxel_ids   -- sorted unique voxel IDs (index = position)
        id2idx      -- voxel_id -> index mapping
        adj         -- adj[i] = [(j, p_ij), ...] with sum <= 1; remainder -> self
    """
    n = len(voxel_ids)
    id2idx = {int(v): i for i, v in enumerate(voxel_ids)}

    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]

    for _, row in transitions_topk.iterrows():
        fid = int(row["from_voxel_id"])
        tid = int(row["to_voxel_id"])
        p = float(row["p"])
        if fid in id2idx and tid in id2idx and p > 0:
            adj[id2idx[fid]].append((id2idx[tid], p))

    return voxel_ids, id2idx, adj


def _propagate_sparse(
    adj: List[List[Tuple[int, float]]],
    start_idx: int,
    n_steps: int,
    n: int,
) -> np.ndarray:
    """Iterative sparse Markov propagation from one-hot start.

    v_{t+1}[j] = sum_i v_t[i] * P(i->j)
    Rows not summing to 1 get remainder as self-loop.
    """
    v = np.zeros(n, dtype=np.float64)
    v[start_idx] = 1.0

    for _ in range(n_steps):
        v_new = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if v[i] < 1e-15:
                continue
            mass_out = 0.0
            for j, p in adj[i]:
                transfer = v[i] * p
                v_new[j] += transfer
                mass_out += p
            self_p = max(0.0, 1.0 - mass_out)
            v_new[i] += v[i] * self_p
        v = v_new

    return v


def _build_basin_transition(
    transitions_topk: pd.DataFrame,
    voxel_stats: pd.DataFrame,
    basins: pd.DataFrame,
    *,
    noise_basin_id: int = -1,
) -> Tuple[np.ndarray, Dict[int, int], List[List[Tuple[int, float]]]]:
    """Build row-stochastic basin-level transition (coarse-grained)."""
    b = basins[["voxel_id", "basin_id"]].copy()
    b["basin_id"] = pd.to_numeric(b["basin_id"], errors="coerce").fillna(noise_basin_id).astype(int)
    v2b = dict(zip(b["voxel_id"].astype(int), b["basin_id"].astype(int)))

    pi_map = dict(zip(
        voxel_stats["voxel_id"].astype(int),
        pd.to_numeric(voxel_stats["pi"], errors="coerce").fillna(0.0).astype(float),
    ))

    flow: Dict[Tuple[int, int], float] = {}
    for _, row in transitions_topk.iterrows():
        fid = int(row["from_voxel_id"])
        tid = int(row["to_voxel_id"])
        p = float(row["p"])
        if p <= 0:
            continue
        fb = v2b.get(fid)
        tb = v2b.get(tid)
        if fb is None or tb is None:
            continue
        if fb == noise_basin_id or tb == noise_basin_id:
            continue
        fm = pi_map.get(fid, 0.0) * p
        key = (fb, tb)
        flow[key] = flow.get(key, 0.0) + fm

    basin_ids_set: set = set()
    for (fb, tb) in flow:
        basin_ids_set.add(fb)
        basin_ids_set.add(tb)
    basin_ids = np.array(sorted(basin_ids_set), dtype=int)
    n = len(basin_ids)
    bid2idx = {int(v): i for i, v in enumerate(basin_ids)}

    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    row_sums: Dict[int, float] = {}
    for (fb, tb), fm in flow.items():
        row_sums[fb] = row_sums.get(fb, 0.0) + fm

    for (fb, tb), fm in flow.items():
        rs = row_sums.get(fb, 0.0)
        if rs < 1e-15:
            continue
        adj[bid2idx[fb]].append((bid2idx[tb], fm / rs))

    return basin_ids, bid2idx, adj


# ---------------------------------------------------------------------------
# Confidence curve
# ---------------------------------------------------------------------------

def build_confidence_curve(
    contracts: Dict[str, pd.DataFrame],
    voxel_gate: pd.DataFrame,
    basin_gate: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Compute forward E[gate] at multiple horizons via Markov propagation."""
    conf_cfg = cfg.get("confidence", {})
    horizons_min = conf_cfg.get("horizons_minutes", [5, 30, 120, 240])
    if isinstance(horizons_min, (int, float)):
        horizons_min = [horizons_min]
    horizons_min = [int(h) for h in horizons_min]
    use_basins = bool(conf_cfg.get("use_basins", True))
    noise_id = int(cfg.get("figures", {}).get("basins_view", {}).get("noise_basin_id", -1))

    voxel_map = contracts["voxel_map"]
    voxel_stats = contracts["voxel_stats"]
    transitions_topk = contracts["transitions_topk"]
    basins = contracts["basins"]

    last_row = voxel_map.iloc[-1]
    now_ts = last_row["ts"]
    now_vid = int(last_row["voxel_id"])

    bmap = basins[["voxel_id", "basin_id"]].copy()
    bmap["basin_id"] = pd.to_numeric(bmap["basin_id"], errors="coerce").fillna(noise_id).astype(int)
    now_b = bmap[bmap["voxel_id"] == now_vid]
    now_bid = int(now_b.iloc[0]["basin_id"]) if len(now_b) else noise_id

    vg_rate = dict(zip(
        voxel_gate["voxel_id"].astype(int),
        voxel_gate["gate_on_rate"].astype(float),
    ))

    bg_rate: Dict[int, float] = {}
    if use_basins and len(basin_gate):
        bg_rate = dict(zip(
            basin_gate["basin_id"].astype(int),
            basin_gate["gate_on_rate"].astype(float),
        ))

    # Voxel-level sparse transition
    all_vids = np.sort(voxel_stats["voxel_id"].astype(int).unique())
    _, id2idx, adj_voxel = _build_sparse_transition(transitions_topk, all_vids)
    n_vox = len(all_vids)
    gate_vec = np.array([vg_rate.get(int(v), 0.0) for v in all_vids], dtype=np.float64)

    # Basin-level sparse transition
    basin_ids_arr = np.empty(0, dtype=int)
    bid2idx: Dict[int, int] = {}
    adj_basin: List[List[Tuple[int, float]]] = []
    gate_vec_basin = np.empty(0, dtype=np.float64)
    if use_basins:
        basin_ids_arr, bid2idx, adj_basin = _build_basin_transition(
            transitions_topk, voxel_stats, basins, noise_basin_id=noise_id,
        )
        gate_vec_basin = np.array(
            [bg_rate.get(int(b), 0.0) for b in basin_ids_arr],
            dtype=np.float64,
        )

    results = []
    for h_min in horizons_min:
        k = h_min  # 1 step = 1 minute

        if now_vid in id2idx:
            dist_v = _propagate_sparse(adj_voxel, id2idx[now_vid], k, n_vox)
            conf_voxel = float(np.dot(dist_v, gate_vec))
        else:
            conf_voxel = vg_rate.get(now_vid, 0.0)

        conf_basin = float("nan")
        if use_basins and now_bid in bid2idx:
            dist_b = _propagate_sparse(adj_basin, bid2idx[now_bid], k, len(basin_ids_arr))
            conf_basin = float(np.dot(dist_b, gate_vec_basin))

        results.append({
            "ts_now": now_ts,
            "now_voxel_id": now_vid,
            "now_basin_id": now_bid,
            "horizon_steps": k,
            "horizon_minutes": h_min,
            "conf_voxel": round(conf_voxel, 6),
            "conf_basin": round(conf_basin, 6) if not np.isnan(conf_basin) else None,
        })

    out = pd.DataFrame(results)
    if len(out) >= 2 and "conf_voxel" in out.columns:
        out["conf_delta"] = out["conf_voxel"].diff()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = base_parser()
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    contracts = load_contracts(args.in_dir, cfg)
    out_dir = ensure_path(args.out_dir)

    gate_cfg = cfg.get("gate", {})
    noise_id = int(cfg.get("figures", {}).get("basins_view", {}).get("noise_basin_id", -1))

    # 1) Load + align gate
    print("Loading gate data ...")
    gate = load_gate(cfg)
    print(f"  gate rows: {len(gate)}, range: {gate['ts'].min()} -> {gate['ts'].max()}")

    direction = str(gate_cfg.get("asof_direction", "backward"))
    aligned = align_gate_to_voxels(contracts["voxel_map"], gate, direction=direction)
    print(f"  aligned rows: {len(aligned)}")

    # 2) Voxel gate stats
    print("Building voxel_gate_stats ...")
    voxel_gate = build_voxel_gate_stats(aligned)
    voxel_gate.to_parquet(out_dir / "voxel_gate_stats.parquet", index=False)
    print(f"  voxels with gate data: {len(voxel_gate)}")

    # 3) Basin gate stats
    print("Building basin_gate_stats ...")
    basin_gate = build_basin_gate_stats(
        voxel_gate, contracts["voxel_stats"], contracts["basins"],
        noise_basin_id=noise_id,
    )
    basin_gate.to_parquet(out_dir / "basin_gate_stats.parquet", index=False)
    print(f"  basins with gate data: {len(basin_gate)}")

    # 4) Confidence curve
    print("Computing confidence projections ...")
    curve = build_confidence_curve(contracts, voxel_gate, basin_gate, cfg)
    curve.to_parquet(out_dir / "confidence_curve.parquet", index=False)

    # 5) Summary
    vm = contracts["voxel_map"]
    now_vid = int(vm.iloc[-1]["voxel_id"])
    now_ts = vm.iloc[-1]["ts"]

    bmap = contracts["basins"][["voxel_id", "basin_id"]].copy()
    bmap["basin_id"] = pd.to_numeric(bmap["basin_id"], errors="coerce").fillna(noise_id).astype(int)
    now_b = bmap[bmap["voxel_id"] == now_vid]
    now_bid = int(now_b.iloc[0]["basin_id"]) if len(now_b) else noise_id

    vg_now = voxel_gate[voxel_gate["voxel_id"] == now_vid]
    bg_now = basin_gate[basin_gate["basin_id"] == now_bid]

    print()
    print("=" * 60)
    print("GATE CONFIDENCE SUMMARY")
    print("=" * 60)
    print(f"  NOW ts           : {now_ts}")
    print(f"  NOW voxel_id     : {now_vid}")
    print(f"  NOW basin_id     : {now_bid}")
    if len(vg_now):
        r = vg_now.iloc[0]
        print(f"  gate_on_rate (voxel) : {r['gate_on_rate']:.4f}  [CI: {r['ci_lo']:.3f} - {r['ci_hi']:.3f}]  (n={int(r['n'])})")
    else:
        print("  gate_on_rate (voxel) : N/A")
    if len(bg_now):
        r = bg_now.iloc[0]
        print(f"  gate_on_rate (basin) : {r['gate_on_rate']:.4f}  (mass={r['mass_pi']:.4f}, n_voxels={int(r['n_voxels'])})")
    else:
        print("  gate_on_rate (basin) : N/A")

    print()
    for _, row in curve.iterrows():
        h = int(row["horizon_minutes"])
        cv = row["conf_voxel"]
        cb = row.get("conf_basin")
        cb_s = f"{cb:.4f}" if cb is not None and not (isinstance(cb, float) and np.isnan(cb)) else "N/A"
        label = f"{h}m" if h < 60 else f"{h // 60}h"
        print(f"  conf_{label:>4s}  voxel={cv:.4f}  basin={cb_s}")

    print("=" * 60)
    print(f"\nOutputs in: {out_dir}")
    print("  voxel_gate_stats.parquet")
    print("  basin_gate_stats.parquet")
    print("  confidence_curve.parquet")


if __name__ == "__main__":
    main()
