from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from visual.layers.layer_edges import make_edges_layer
from visual.layers.layer_voxels import make_voxel_layer
from visual.scene import Scene


def build_transitions_scene(
    voxel_stats: pd.DataFrame,
    transitions_topk: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Scene:
    fcfg = cfg.get('figures', {}).get('transitions_3d', {})
    scene = Scene(title=str(fcfg.get('title', 'Transitions')), settings={'kind': '3d'})
    scene.add_layer(make_voxel_layer(voxel_stats, cfg, color_metric='pi', size_metric='occ_eff', name='voxels_bg'))
    scene.add_layer(make_edges_layer(transitions_topk, voxel_stats, cfg, name='edges_flow'))
    return scene
