from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from visual.scales import scale_metric, to_marker_sizes
from visual.scene import SceneLayer


def _filter_top_mass(df: pd.DataFrame, keep_mass: float) -> pd.DataFrame:
    if keep_mass <= 0 or keep_mass >= 1 or 'pi' not in df.columns or len(df) == 0:
        return df

    pi = pd.to_numeric(df['pi'], errors='coerce').fillna(0.0)
    order = pi.sort_values(ascending=False).index
    cum = pi.loc[order].cumsum()
    total = float(pi.sum())
    if total <= 0:
        return df

    cutoff = keep_mass * total
    n_keep = int((cum <= cutoff).sum())
    n_keep = max(1, min(len(df), n_keep + 1))
    keep_idx = order[:n_keep]
    return df.loc[keep_idx].copy()


def make_voxel_layer(
    voxel_stats: pd.DataFrame,
    cfg: Dict[str, Any],
    color_metric: str,
    size_metric: str,
    name: str = 'voxels',
) -> SceneLayer:
    df = voxel_stats.copy()
    fcfg = cfg.get('filters', {}).get('voxels', {})

    keep_mass = fcfg.get('mass_cumsum_keep', None)
    if keep_mass is not None:
        df = _filter_top_mass(df, float(keep_mass))
    else:
        min_q = float(fcfg.get('min_pi_quantile', 0.2))
        if 'pi' in df.columns and len(df):
            cutoff = float(pd.to_numeric(df['pi'], errors='coerce').fillna(0.0).quantile(min_q))
            df = df[pd.to_numeric(df['pi'], errors='coerce').fillna(0.0) >= cutoff].copy()

    max_voxels = int(fcfg.get('max_voxels', len(df)))
    if len(df) > max_voxels:
        df = df.nlargest(max_voxels, 'pi').copy() if 'pi' in df.columns else df.head(max_voxels).copy()

    if color_metric not in df.columns:
        raise ValueError(f'missing color_metric={color_metric} in voxel_stats')
    if size_metric not in df.columns:
        raise ValueError(f'missing size_metric={size_metric} in voxel_stats')

    df['_color'] = scale_metric(df[color_metric], color_metric, cfg)
    df['_size'] = to_marker_sizes(scale_metric(df[size_metric], size_metric, cfg), 8.0, 120.0)

    for c in ['center_x', 'center_y', 'center_z']:
        if c not in df.columns:
            raise ValueError(f'missing voxel center column in voxel_stats: {c}')

    return SceneLayer(
        name=name,
        kind='voxels',
        data=df,
        params={'color_metric': color_metric, 'size_metric': size_metric},
    )
