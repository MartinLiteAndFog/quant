from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class SceneLayer:
    name: str
    kind: str
    data: pd.DataFrame
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Scene:
    title: str
    layers: List[SceneLayer] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)

    def add_layer(self, layer: SceneLayer) -> None:
        self.layers.append(layer)
