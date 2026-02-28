from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PCConfig:
    # --- Model ---
    d_latent: int = 32
    n_obs: int = 5               # [r_1, r_5, r_15, rv_20, rv_60]
    horizons: List[int] = field(default_factory=lambda: [1, 5, 15, 60])
    n_inference_steps: int = 5
    lr_x: float = 0.05
    lr_A: float = 1e-4
    lr_W: float = 1e-4
    lr_C: float = 1e-4
    lambda_A: float = 1e-4       # shrink-to-identity strength
    alpha_v: float = 0.01        # variance EMA rate
    v_min: float = 1e-12
    v_max: float = 1e2
    eps: float = 1e-12
    pi_ceil: float = 1e6
    tau: float = 0.05            # state carry rate
    beta_obs: float = 0.2        # obs term damping
    k_robust: float = 5.0        # residual clipping at k*sigma
    k_mu_out: float = 2.5        # cap mu used for trading/probabilities
    x_clip: float = 10.0         # element-wise clamp on latent state
    dx_clip: Optional[float] = None
    warmup_bars: int = 200
    v_temporal_init: float = 1e-4
    v_obs_init: float = 1e-3
    v_h_base: float = 1e-6       # v_h_init = h * v_h_base
    seed: int = 0

    # --- Trade logic ---
    fee_bps: float = 7.0
    slippage_bps: float = 2.0
    margin: float = 0.02
    z_min: float = 0.15
    min_edge_bps: float = 5.0
    flip_margin: float = 0.05
    z_flip_min: float = 0.5
    cooldown_bars: int = 3
    sl_pct: float = 0.015
    tp_pct: float = 0.03

    @property
    def total_cost(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 10_000
