from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class TransitionConfig:
    n_bins: int = 40
    bin_method: str = "quantile"
    n_min_voxel: int = 20
    topk: int = 20
    laplace_alpha: float = 0.01
    decay_halflife_bars: int = 0
    eps: float = 1e-12
    axes: List[str] = field(default_factory=lambda: ["X_raw", "Y_res", "Z_res"])
    run_id: str = "state-space-transitions-v01"
