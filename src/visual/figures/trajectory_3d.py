from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from visual.layers.layer_paths import make_trajectory_layer
from visual.scene import Scene


def build_trajectory_scene(
    state_space: pd.DataFrame,
    voxel_map: pd.DataFrame,
    cfg: Dict[str, Any],
    ts_from: Optional[pd.Timestamp] = None,
    ts_to: Optional[pd.Timestamp] = None,
) -> Scene:
    fcfg = cfg.get('figures', {}).get('trajectory_3d', {})
    scene = Scene(title=str(fcfg.get('title', 'Trajectory (compressed)')), settings={'kind': '3d'})
    scene.add_layer(make_trajectory_layer(state_space, voxel_map, cfg, ts_from=ts_from, ts_to=ts_to))
    return scene
