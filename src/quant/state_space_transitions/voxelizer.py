from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import TransitionConfig


def _strictly_increasing_edges(edges: np.ndarray, eps: float) -> np.ndarray:
    fixed = np.asarray(edges, dtype=np.float64).copy()
    if fixed.ndim != 1:
        raise ValueError("edges must be 1D")
    for i in range(1, fixed.size):
        if not np.isfinite(fixed[i]):
            fixed[i] = fixed[i - 1] + eps
        elif fixed[i] <= fixed[i - 1]:
            fixed[i] = fixed[i - 1] + eps
    return fixed


def _compute_edges(values: np.ndarray, n_bins: int, method: str, eps: float) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        # fallback edges around zero for all-NaN axis
        base = np.linspace(-1.0, 1.0, n_bins + 1, dtype=np.float64)
        return _strictly_increasing_edges(base, eps)
    if method == "quantile":
        q = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
        edges = np.quantile(finite, q)
    elif method == "uniform":
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if hi <= lo:
            hi = lo + eps * n_bins
        edges = np.linspace(lo, hi, n_bins + 1, dtype=np.float64)
    else:
        raise ValueError(f"unknown bin_method='{method}'")
    return _strictly_increasing_edges(edges, eps)


def voxelize_dataframe(
    df: pd.DataFrame,
    cfg: TransitionConfig,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    if len(cfg.axes) != 3:
        raise ValueError("cfg.axes must contain exactly 3 axis columns")
    for col in cfg.axes:
        if col not in df.columns:
            raise ValueError(f"missing required axis column: {col}")

    out = df.copy()
    out[cfg.axes[0]] = pd.to_numeric(out[cfg.axes[0]], errors="coerce")
    out[cfg.axes[1]] = pd.to_numeric(out[cfg.axes[1]], errors="coerce")
    out[cfg.axes[2]] = pd.to_numeric(out[cfg.axes[2]], errors="coerce")

    edges_map: Dict[str, np.ndarray] = {}
    mids_map: Dict[str, np.ndarray] = {}
    bins: Dict[str, np.ndarray] = {}

    for axis in cfg.axes:
        values = out[axis].to_numpy(dtype=np.float64, copy=False)
        edges = _compute_edges(values, cfg.n_bins, cfg.bin_method, cfg.eps)
        edges_map[axis] = edges

        fixed = edges.copy()
        finite = values[np.isfinite(values)]
        if finite.size > 0:
            fixed[0] = float(np.min(finite))
            fixed[-1] = float(np.max(finite))
            fixed = _strictly_increasing_edges(fixed, cfg.eps)
        mids = 0.5 * (fixed[:-1] + fixed[1:])
        mids_map[axis] = mids

        idx = np.full(values.shape[0], -1, dtype=np.int32)
        valid = np.isfinite(values)
        if np.any(valid):
            d = np.digitize(values[valid], edges[1:-1], right=False).astype(np.int32)
            d = np.clip(d, 0, cfg.n_bins - 1)
            idx[valid] = d
        bins[axis] = idx

    ix = bins[cfg.axes[0]]
    iy = bins[cfg.axes[1]]
    iz = bins[cfg.axes[2]]
    voxel = np.full(ix.shape[0], -1, dtype=np.int32)
    valid_voxel = (ix >= 0) & (iy >= 0) & (iz >= 0)
    if np.any(valid_voxel):
        n = int(cfg.n_bins)
        voxel[valid_voxel] = (
            ix[valid_voxel]
            + n * iy[valid_voxel]
            + n * n * iz[valid_voxel]
        ).astype(np.int32)

    cx = np.full(ix.shape[0], np.nan, dtype=np.float64)
    cy = np.full(ix.shape[0], np.nan, dtype=np.float64)
    cz = np.full(ix.shape[0], np.nan, dtype=np.float64)
    if np.any(valid_voxel):
        cx[valid_voxel] = mids_map[cfg.axes[0]][ix[valid_voxel]]
        cy[valid_voxel] = mids_map[cfg.axes[1]][iy[valid_voxel]]
        cz[valid_voxel] = mids_map[cfg.axes[2]][iz[valid_voxel]]

    out["ix"] = ix
    out["iy"] = iy
    out["iz"] = iz
    out["voxel_id"] = voxel
    out["cx"] = cx
    out["cy"] = cy
    out["cz"] = cz
    return out, edges_map, mids_map
