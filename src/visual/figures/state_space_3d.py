from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from visual.layers.layer_voxels import make_voxel_layer
from visual.scene import Scene, SceneLayer


def _add_current_position(scene: Scene, current_position: Optional[Dict[str, float]]) -> None:
    if current_position is not None:
        cp = pd.DataFrame([current_position])
        scene.add_layer(
            SceneLayer(name="current_position", kind="current_position", data=cp, params={})
        )


def build_occupancy_scene(
    voxel_stats: pd.DataFrame,
    cfg: Dict[str, Any],
    current_position: Optional[Dict[str, float]] = None,
) -> Scene:
    fcfg = cfg.get("figures", {}).get("state_space_occupancy", {})
    scene = Scene(
        title=str(fcfg.get("title", "State Space Occupancy")),
        settings={"kind": "3d"},
    )
    scene.add_layer(
        make_voxel_layer(
            voxel_stats, cfg,
            color_metric=str(fcfg.get("color_metric", "pi")),
            size_metric=str(fcfg.get("size_metric", "occ_eff")),
            name="occupancy_voxels",
        )
    )
    _add_current_position(scene, current_position)
    return scene


def build_persistence_scene(
    voxel_stats: pd.DataFrame,
    cfg: Dict[str, Any],
    current_position: Optional[Dict[str, float]] = None,
) -> Scene:
    fcfg = cfg.get("figures", {}).get("state_space_persistence", {})
    scene = Scene(
        title=str(fcfg.get("title", "State Space Persistence")),
        settings={"kind": "3d"},
    )
    scene.add_layer(
        make_voxel_layer(
            voxel_stats, cfg,
            color_metric=str(fcfg.get("color_metric", "holding_time")),
            size_metric=str(fcfg.get("size_metric", "occ_eff")),
            name="persistence_voxels",
        )
    )
    _add_current_position(scene, current_position)
    return scene
