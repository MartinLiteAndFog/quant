from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from visual.geometry import build_edges_geometry, centers_from_voxel_stats
from visual.scales import scale_metric
from visual.scene import SceneLayer


def make_edges_layer(
    transitions: pd.DataFrame,
    voxel_stats: pd.DataFrame,
    cfg: Dict[str, Any],
    name: str = 'edges',
) -> SceneLayer:
    t = transitions.copy()
    v = voxel_stats[['voxel_id', 'pi', 'center_x', 'center_y', 'center_z']].copy()
    t = t.merge(v[['voxel_id', 'pi']], left_on='from_voxel_id', right_on='voxel_id', how='left').drop(columns=['voxel_id'])
    t['flow_mass'] = t['pi'].fillna(0.0) * t['p'].fillna(0.0)

    fcfg = cfg.get('filters', {}).get('transitions', {})
    if bool(fcfg.get('drop_self_edges', False)):
        t = t[t['from_voxel_id'] != t['to_voxel_id']].copy()

    min_p = float(fcfg.get('min_p', 0.0))
    t = t[t['p'] >= min_p].copy()

    q = float(fcfg.get('min_flow_mass_quantile', 0.90))
    if len(t):
        cutoff = float(t['flow_mass'].quantile(q))
        t = t[t['flow_mass'] >= cutoff].copy()

    max_edges = int(fcfg.get('max_edges_global', 600))
    if len(t) > max_edges:
        t = t.nlargest(max_edges, 'flow_mass').copy()

    geom = build_edges_geometry(t[['from_voxel_id', 'to_voxel_id', 'p', 'flow_mass']], centers_from_voxel_stats(voxel_stats))
    geom['_width'] = 1.0 + 5.0 * scale_metric(geom['flow_mass'], 'flow_mass', cfg)
    geom['_opacity'] = 0.1 + 0.7 * scale_metric(geom['flow_mass'], 'flow_mass', cfg)
    return SceneLayer(name=name, kind='edges', data=geom, params={})
