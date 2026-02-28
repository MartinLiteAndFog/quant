from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def decode_voxel_id(voxel_id: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = voxel_id.astype(np.int64)
    ix = v % n_bins
    iy = (v // n_bins) % n_bins
    iz = v // (n_bins * n_bins)
    return ix.astype(np.int32), iy.astype(np.int32), iz.astype(np.int32)


def centers_from_voxel_stats(voxel_stats: pd.DataFrame) -> pd.DataFrame:
    required = {'voxel_id', 'center_x', 'center_y', 'center_z'}
    missing = required - set(voxel_stats.columns)
    if missing:
        raise ValueError(f'missing center columns in voxel_stats: {sorted(missing)}')
    return voxel_stats[['voxel_id', 'center_x', 'center_y', 'center_z']].copy()


def build_edges_geometry(transitions: pd.DataFrame, centers: pd.DataFrame) -> pd.DataFrame:
    req_t = {'from_voxel_id', 'to_voxel_id', 'p'}
    miss_t = req_t - set(transitions.columns)
    if miss_t:
        raise ValueError(f'missing transitions columns: {sorted(miss_t)}')
    req_c = {'voxel_id', 'center_x', 'center_y', 'center_z'}
    miss_c = req_c - set(centers.columns)
    if miss_c:
        raise ValueError(f'missing centers columns: {sorted(miss_c)}')

    c_from = centers.rename(
        columns={'voxel_id': 'from_voxel_id', 'center_x': 'x0', 'center_y': 'y0', 'center_z': 'z0'}
    )
    c_to = centers.rename(
        columns={'voxel_id': 'to_voxel_id', 'center_x': 'x1', 'center_y': 'y1', 'center_z': 'z1'}
    )
    out = transitions.merge(c_from, on='from_voxel_id', how='inner').merge(c_to, on='to_voxel_id', how='inner')
    return out


def build_drift_arrows(voxel_stats: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    need = {'voxel_id', 'center_x', 'center_y', 'center_z', 'drift_dx', 'drift_dy', 'drift_dz'}
    missing = need - set(voxel_stats.columns)
    if missing:
        raise ValueError(f'missing drift columns in voxel_stats: {sorted(missing)}')
    out = voxel_stats[['voxel_id', 'center_x', 'center_y', 'center_z', 'drift_dx', 'drift_dy', 'drift_dz']].copy()
    out['x1'] = out['center_x'] + scale * out['drift_dx']
    out['y1'] = out['center_y'] + scale * out['drift_dy']
    out['z1'] = out['center_z'] + scale * out['drift_dz']
    return out
