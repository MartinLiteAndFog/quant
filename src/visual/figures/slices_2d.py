from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from visual.scales import scale_metric
from visual.scene import Scene, SceneLayer


def _pivot_slice(df: pd.DataFrame, x: str, y: str, value: str, method: str) -> pd.DataFrame:
    if method == 'mean':
        p = df.pivot_table(index=y, columns=x, values=value, aggfunc='mean', fill_value=0.0)
    else:
        p = df.pivot_table(index=y, columns=x, values=value, aggfunc='sum', fill_value=0.0)
    return p.sort_index().sort_index(axis=1)


def _contrast_map(mat: pd.DataFrame, metric: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    # apply scaling pipeline (quantile clip / log1p / cdf stretch / normalize)
    flat = pd.Series(mat.values.ravel())
    scaled = scale_metric(flat, metric, cfg)
    out = scaled.to_numpy(dtype=float).reshape(mat.shape)
    return pd.DataFrame(out, index=mat.index, columns=mat.columns)


def build_slice_scenes(voxel_map: pd.DataFrame, voxel_stats: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[Scene, Scene, Scene]:
    if not {'bin_x', 'bin_y', 'bin_z'} <= set(voxel_map.columns):
        raise ValueError('voxel_map.parquet must contain bin_x, bin_y, bin_z for slice plots')

    metric = str(cfg.get('slices', {}).get('aggregations', {}).get('metric', 'pi'))
    method = str(cfg.get('slices', {}).get('aggregations', {}).get('method', 'sum'))

    vm = voxel_map[['voxel_id', 'bin_x', 'bin_y', 'bin_z']].drop_duplicates('voxel_id')
    cols = ['voxel_id', metric] if metric in voxel_stats.columns else ['voxel_id', 'pi']
    df = vm.merge(voxel_stats[cols], on='voxel_id', how='left').fillna(0.0)

    xy = _pivot_slice(df, 'bin_x', 'bin_y', metric, method)
    xz = _pivot_slice(df, 'bin_x', 'bin_z', metric, method)
    yz = _pivot_slice(df, 'bin_y', 'bin_z', metric, method)

    # contrast enhance for visual intelligence
    xy_c = _contrast_map(xy, metric, cfg)
    xz_c = _contrast_map(xz, metric, cfg)
    yz_c = _contrast_map(yz, metric, cfg)

    s_xy = Scene(title='XY Slice (contrast)', settings={'kind': 'heatmap2d'})
    s_xy.add_layer(SceneLayer(name='xy', kind='heatmap2d', data=xy_c, params={'xlabel': 'bin_x', 'ylabel': 'bin_y', 'cbar_label': metric}))

    s_xz = Scene(title='XZ Slice (contrast)', settings={'kind': 'heatmap2d'})
    s_xz.add_layer(SceneLayer(name='xz', kind='heatmap2d', data=xz_c, params={'xlabel': 'bin_x', 'ylabel': 'bin_z', 'cbar_label': metric}))

    s_yz = Scene(title='YZ Slice (contrast)', settings={'kind': 'heatmap2d'})
    s_yz.add_layer(SceneLayer(name='yz', kind='heatmap2d', data=yz_c, params={'xlabel': 'bin_y', 'ylabel': 'bin_z', 'cbar_label': metric}))

    return s_xy, s_xz, s_yz
