from __future__ import annotations
import numpy as np
from quant.predictive_coding.config import PCConfig


class TemporalPCModel:
    def __init__(self, cfg: PCConfig) -> None:
        self.cfg = cfg
        d = cfg.d_latent
        n_obs = cfg.n_obs
        horizons = cfg.horizons

        rng = np.random.default_rng(0)

        self.x = np.zeros(d)
        self.x_prev = np.zeros(d)

        # Transition matrix (near identity)
        self.A = 0.95 * np.eye(d) + 0.01 * rng.standard_normal((d, d))

        # Readout weights per horizon: W_h in R^d, keyed by horizon value
        self.W: dict[int, np.ndarray] = {
            h: 0.01 * rng.standard_normal(d) for h in horizons
        }

        # Decoder: C in R^(n_obs x d)
        self.C = 0.01 * rng.standard_normal((n_obs, d))

        # Variance estimates (init proportional to horizon for targets)
        self.v_h: dict[int, float] = {h: h * 1e-6 for h in horizons}
        self.v_temporal: float = 1e-4
        self.v_obs: np.ndarray = np.full(n_obs, 1e-3)

    def step(
        self,
        obs: np.ndarray,
        targets: dict[int, float],
        is_warmup: bool,
    ) -> tuple[dict[int, float], dict[int, float]]:
        cfg = self.cfg
        d = cfg.d_latent
        eps = 1e-12

        # --- Prior from transition ---
        x_prior = self.A @ self.x_prev
        x = x_prior.copy()

        # --- Inference (K relaxation steps) ---
        for _ in range(cfg.n_inference_steps):
            e_temp = x - x_prior
            pi_temp = 1.0 / (self.v_temporal + eps)

            dx = -pi_temp * e_temp

            for h in cfg.horizons:
                res_h = targets[h] - self.W[h] @ x
                pi_h = 1.0 / (self.v_h[h] + eps)
                dx += pi_h * res_h * self.W[h]

            e_obs = obs - self.C @ x
            pi_obs = 1.0 / (self.v_obs + eps)
            dx += cfg.beta_obs * (self.C.T @ (pi_obs * e_obs))

            x = x + cfg.lr_x * dx

        self.x = x

        # --- Predictions (clipped mu) ---
        mu: dict[int, float] = {}
        sigma: dict[int, float] = {}
        for h in cfg.horizons:
            mu_raw = float(self.W[h] @ x)
            s = float(np.sqrt(self.v_h[h] + eps))
            mu_clip = float(np.clip(mu_raw, -cfg.k_robust * s, cfg.k_robust * s))
            mu[h] = mu_clip
            sigma[h] = s

        # --- Variance updates (always, including warmup) ---
        for h in cfg.horizons:
            res_h = targets[h] - float(self.W[h] @ x)
            s = np.sqrt(self.v_h[h] + eps)
            res_clipped = float(np.clip(res_h, -cfg.k_robust * s, cfg.k_robust * s))
            self.v_h[h] = (
                (1 - cfg.alpha_v) * self.v_h[h]
                + cfg.alpha_v * float(np.clip(res_clipped**2, cfg.v_min, cfg.v_max))
            )

        e_temp_final = x - x_prior
        temp_norm = float(np.sqrt(np.dot(e_temp_final, e_temp_final) / d))
        temp_norm_clipped = min(
            temp_norm, cfg.k_robust * float(np.sqrt(self.v_temporal + eps))
        )
        self.v_temporal = (
            (1 - cfg.alpha_v) * self.v_temporal
            + cfg.alpha_v * float(np.clip(temp_norm_clipped**2, cfg.v_min, cfg.v_max))
        )

        e_obs_final = obs - self.C @ x
        s_obs = np.sqrt(self.v_obs + eps)
        e_obs_clipped = np.clip(
            e_obs_final, -cfg.k_robust * s_obs, cfg.k_robust * s_obs
        )
        self.v_obs = (
            (1 - cfg.alpha_v) * self.v_obs
            + cfg.alpha_v * np.clip(e_obs_clipped**2, cfg.v_min, cfg.v_max)
        )

        # --- Weight learning (skip during warmup) ---
        if not is_warmup:
            pi_temp = 1.0 / (self.v_temporal + eps)
            e_temp_learn = x - x_prior

            # Transition: shrink-to-identity
            dA = cfg.lr_A * (pi_temp * e_temp_learn[:, None] @ self.x_prev[None, :])
            self.A += dA - cfg.lr_A * cfg.lambda_A * (self.A - np.eye(d))

            # Readout per horizon
            for h in cfg.horizons:
                pi_h = 1.0 / (self.v_h[h] + eps)
                res_h = targets[h] - float(self.W[h] @ x)
                self.W[h] += cfg.lr_W * (pi_h * res_h) * x

            # Decoder
            pi_obs = 1.0 / (self.v_obs + eps)
            e_obs_learn = obs - self.C @ x
            self.C += cfg.lr_C * (pi_obs * e_obs_learn)[:, None] @ x[None, :]

        # --- State carry ---
        self.x_prev = (1 - cfg.tau) * self.x_prev + cfg.tau * x

        return mu, sigma
