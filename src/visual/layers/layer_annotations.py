from __future__ import annotations

from visual.scene import SceneLayer



def make_annotation_layer(title: str) -> SceneLayer:
    import pandas as pd

    return SceneLayer(name='annotations', kind='annotations', data=pd.DataFrame({'text': [title]}), params={})
