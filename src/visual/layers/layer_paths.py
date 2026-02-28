from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from visual.scene import SceneLayer


def _run_length_filter(voxel_ids: np.ndarray, min_dwell: int) -> np.ndarray:
    """Keep indices that belong to runs of length >= min_dwell."""
    n = int(voxel_ids.size)
    if n == 0:
        return np.array([], dtype=bool)
    if min_dwell <= 1:
        return np.ones(n, dtype=bool)

    # identify run boundaries
    change = np.r_[True, voxel_ids[1:] != voxel_ids[:-1]]
    run_starts = np.flatnonzero(change)
    run_ends = np.r_[run_starts[1:], n]
    keep = np.zeros(n, dtype=bool)
    for s, e in zip(run_starts, run_ends):
        if (e - s) >= min_dwell:
            keep[s:e] = True
    return keep


def _compress_points(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    fcfg = cfg.get('figures', {}).get('trajectory_3d', {})
    max_points = int(fcfg.get('max_points', 3000))
    min_dwell = int(fcfg.get('min_dwell_steps', 3))

    if len(df) == 0:
        return df

    vox = pd.to_numeric(df['voxel_id'], errors='coerce').fillna(-1).to_numpy(dtype=np.int64, copy=False)

    # change_only always on for v0.2 trajectory intelligence
    keep_change = np.r_[True, vox[1:] != vox[:-1]]
    df = df.loc[keep_change].copy()
    vox = pd.to_numeric(df['voxel_id'], errors='coerce').fillna(-1).to_numpy(dtype=np.int64, copy=False)

    # suppress flicker: only keep runs with dwell >= min_dwell in the ORIGINAL series.
    # Since we already change-only compressed, approximate by requiring that a voxel appears
    # at least min_dwell times in a short neighborhood is not possible; instead we do proper
    # dwell filtering on the pre-change series before change-only.
    # To keep it simple and deterministic: apply dwell filtering BEFORE change-only.
    # Caller provides full merged df; we do it in one go below.

    # (This function is called after filtering time window but before change-only in make_trajectory_layer)
    return df


def _make_jump_edges(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    fcfg = cfg.get('figures', {}).get('trajectory_3d', {})
    draw_jumps = bool(fcfg.get('draw_jump_edges', True))
    if not draw_jumps or len(df) < 2:
        return pd.DataFrame(columns=['x0','y0','z0','x1','y1','z1','dist'])

    q = float(fcfg.get('jump_quantile', 0.99))
    x = pd.to_numeric(df['X_raw'], errors='coerce').to_numpy(dtype=float, copy=False)
    y = pd.to_numeric(df['Y_res'], errors='coerce').to_numpy(dtype=float, copy=False)
    z = pd.to_numeric(df['Z_res'], errors='coerce').to_numpy(dtype=float, copy=False)

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    dz = z[1:] - z[:-1]
    dist = np.sqrt(dx*dx + dy*dy + dz*dz)
    dist = np.where(np.isfinite(dist), dist, 0.0)

    if dist.size == 0:
        return pd.DataFrame(columns=['x0','y0','z0','x1','y1','z1','dist'])

    cutoff = float(np.quantile(dist, q))
    m = dist >= cutoff
    if not np.any(m):
        return pd.DataFrame(columns=['x0','y0','z0','x1','y1','z1','dist'])

    idx = np.flatnonzero(m)
    return pd.DataFrame({
        'x0': x[idx], 'y0': y[idx], 'z0': z[idx],
        'x1': x[idx+1], 'y1': y[idx+1], 'z1': z[idx+1],
        'dist': dist[idx],
    })


def make_trajectory_layer(
    state_space: pd.DataFrame,
    voxel_map: pd.DataFrame,
    cfg: Dict[str, Any],
    ts_from: Optional[pd.Timestamp] = None,
    ts_to: Optional[pd.Timestamp] = None,
    name: str = 'trajectory',
) -> SceneLayer:
    df = state_space.merge(voxel_map[['ts', 'voxel_id']], on='ts', how='inner').copy()
    if ts_from is not None:
        df = df[df['ts'] >= ts_from].copy()
    if ts_to is not None:
        df = df[df['ts'] <= ts_to].copy()

    if len(df) == 0:
        return SceneLayer(name=name, kind='trajectory_points', data=df, params={})

    fcfg = cfg.get('figures', {}).get('trajectory_3d', {})
    max_points = int(fcfg.get('max_points', 3000))
    min_dwell = int(fcfg.get('min_dwell_steps', 3))
    draw_lines = bool(fcfg.get('draw_lines', False))

    # dwell filtering on full series
    vox_full = pd.to_numeric(df['voxel_id'], errors='coerce').fillna(-1).to_numpy(dtype=np.int64, copy=False)
    keep_dwell = _run_length_filter(vox_full, min_dwell)
    df = df.loc[keep_dwell].copy()
    if len(df) == 0:
        return SceneLayer(name=name, kind='trajectory_points', data=df, params={})

    # change_only compression
    vox = pd.to_numeric(df['voxel_id'], errors='coerce').fillna(-1).to_numpy(dtype=np.int64, copy=False)
    keep_change = np.r_[True, vox[1:] != vox[:-1]]
    df = df.loc[keep_change].copy()

    # downsample to max_points
    if len(df) > max_points:
        step = max(1, len(df) // max_points)
        df = df.iloc[::step].copy()

    df['_t'] = np.linspace(0.0, 1.0, len(df))

    jumps = _make_jump_edges(df, cfg)

    payload = df[['ts', 'X_raw', 'Y_res', 'Z_res', 'voxel_id', '_t']].copy()
    return SceneLayer(
        name=name,
        kind='trajectory_points',
        data=payload,
        params={
            'draw_lines': draw_lines,
            'jump_edges': jumps,
        },
    )
