from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from visual.scene import SceneLayer


def make_basin_layer(
    voxel_stats: pd.DataFrame,
    basins: pd.DataFrame,
    cfg: Dict[str, Any],
    basin_id: int,
    name: str = 'basins',
) -> SceneLayer:
    df = voxel_stats.merge(basins[['voxel_id', 'basin_id']], on='voxel_id', how='left')
    df['basin_id'] = df['basin_id'].fillna(-1).astype(int)
    out = df[df['basin_id'] == basin_id].copy()
    if out.empty:
        raise ValueError(f'no voxels for basin_id={basin_id}')

    q = float(cfg.get('figures', {}).get('basins_view', {}).get('core_voxels_top_quantile', 0.90))
    cutoff = float(out['pi'].quantile(q))
    out['is_core'] = out['pi'] >= cutoff
    return SceneLayer(name=name, kind='basin_voxels', data=out, params={'basin_id': basin_id})
