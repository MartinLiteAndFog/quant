from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class StateSpaceConfig:
    window_W: int = 240
    horizons: Tuple[int, ...] = (5, 15, 60, 240)
    z_max: float = 5.0
    alpha: float = 1.0
    eps: float = 1e-8
    lambda_eq: float = 0.03

    W_ou: int = 240
    W_ac1: int = 240
    W_jump_mad: int = 240
    W_jv: int = 240
    W_ent: int = 240
    W_noise: int = 240

    k_jump: float = 5.0
    lambda_jump: float = 0.1

    basin_bins: int = 40
    basin_top_k: int = 8
    include_basin_labels: bool = True

    sensor_weights_x: dict = field(
        default_factory=lambda: {
            "x_slope": 1.0,
            "x_er": 1.0,
            "x_align": 1.0,
        }
    )
    sensor_weights_y: dict = field(
        default_factory=lambda: {
            "y_dev": 1.0,
            "y_ou_pull": 1.0,
            "y_ac1": 1.0,
        }
    )
    sensor_weights_z: dict = field(
        default_factory=lambda: {
            "z_jump_rate": 1.0,
            "z_rv_bv_jumpiness": 1.0,
            "z_entropy": 1.0,
            "z_wick_noise": 1.0,
        }
    )
