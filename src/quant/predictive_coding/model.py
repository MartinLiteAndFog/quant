from __future__ import annotations

import math
from collections import deque
from typing import Dict

import numpy as np

from quant.predictive_coding.config import PCConfig


def normal_cdf(z: float) -> float:
    # Phi(z) via erf (no scipy dependency)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class TemporalPCModel:
    """
    Causal Temporal Predictive Coding model with delayed-supervised W_h learning.

    Key rule (no lookahead):
      - Inference each bar uses ONLY:
          temporal prior (A @ x_prev) + current obs features (obs)
      - Target returns y_h (future) are used ONLY when they become available,
        to update W_h and v_h using a stored snapshot x_{t-h}.
    """

    def __init__(self, cfg: PCConfig) -> None:
        self.cfg = cfg
        d = cfg.d_latent
        n_obs = cfg.n_obs

        rng = np.random.default_rng(cfg.seed)

        self.t: int = -1

        # latent state and carry
        self.x = np.zeros(d, dtype=float)
        self.x_prev = np.zeros(d, dtype=float)

        # transition matrix near identity
        self.A = 0.95 * np.eye(d) + 0.01 * rng.standard_normal((d, d))

        # readout weights per horizon (vector in R^d)
        self.W: Dict[int, np.ndarray] = {h: 0.01 * rng.standard_normal(d) for h in cfg.horizons}

        # decoder/generator: obs_hat = C @ x
        self.C = 0.01 * rng.standard_normal((n_obs, d))

        # variance trackers
        self.v_h: Dict[int, float] = {h: float(h) * cfg.v_h_base for h in cfg.horizons}
        self.v_temporal: float = cfg.v_temporal_init
        self.v_obs: np.ndarray = np.full(n_obs, cfg.v_obs_init, dtype=float)

        # ring buffers for delayed targets (store last max_h+1)
        self._max_h = max(cfg.horizons)
        self._buf_price: deque[float] = deque(maxlen=self._max_h + 1)
        self._buf_x: deque[np.ndarray] = deque(maxlen=self._max_h + 1)    # snapshot x_t (after inference)
        self._buf_idx: deque[int] = deque(maxlen=self._max_h + 1)

    def _pi(self, v: float) -> float:
        cfg = self.cfg
        p = 1.0 / (v + cfg.eps)
        return p if p < cfg.pi_ceil else cfg.pi_ceil

    def _pi_vec(self, v: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        p = 1.0 / (v + cfg.eps)
        return np.minimum(p, cfg.pi_ceil)

    def predict_from_state(self, price_now: float) -> Dict[str, Dict[int, float]]:
        """
        Compute mu/sigma/z/p_up/price_level from current x (no updates).
        Returns dict-of-dicts keyed by horizon.
        """
        cfg = self.cfg
        out_mu: Dict[int, float] = {}
        out_sigma: Dict[int, float] = {}
        out_z: Dict[int, float] = {}
        out_pup: Dict[int, float] = {}
        out_level: Dict[int, float] = {}
        out_upper: Dict[int, float] = {}
        out_lower: Dict[int, float] = {}

        for h in cfg.horizons:
            mu_raw = float(self.W[h] @ self.x)
            sigma = float(math.sqrt(self.v_h[h] + cfg.eps))
            mu = float(np.clip(mu_raw, -cfg.k_mu_out * sigma, cfg.k_mu_out * sigma))
            z = mu / sigma if sigma > 1e-15 else 0.0
            p_up = normal_cdf(z)

            out_mu[h] = mu
            out_sigma[h] = sigma
            out_z[h] = z
            out_pup[h] = p_up
            out_level[h] = float(price_now * math.exp(mu))
            out_upper[h] = float(price_now * math.exp(mu + 1.0 * sigma))
            out_lower[h] = float(price_now * math.exp(mu - 1.0 * sigma))

        return {
            "mu": out_mu,
            "sigma": out_sigma,
            "z": out_z,
            "p_up": out_pup,
            "price_level": out_level,
            "price_upper": out_upper,
            "price_lower": out_lower,
        }

    def step(self, price_now: float, obs: np.ndarray) -> Dict[str, Dict[int, float]]:
        """
        One causal bar update. Returns forecasts dict-of-dicts keyed by horizon.
        """
        cfg = self.cfg
        self.t += 1
        t = self.t

        obs = np.asarray(obs, dtype=float)
        if obs.shape != (cfg.n_obs,):
            raise ValueError(f"obs must have shape ({cfg.n_obs},), got {obs.shape}")

        price_now = float(price_now)
        if not np.isfinite(price_now) or price_now <= 0.0:
            raise ValueError("price_now must be finite and > 0 for log-returns")

        # --- prior ---
        x_prior = self.A @ self.x_prev
        np.clip(x_prior, -cfg.x_clip, cfg.x_clip, out=x_prior)
        x = x_prior.copy()

        # --- inference (targets NOT used here) ---
        for _ in range(cfg.n_inference_steps):
            e_temp = x - x_prior
            pi_temp = self._pi(self.v_temporal)
            dx = -pi_temp * e_temp

            # obs decoder term
            e_obs = obs - (self.C @ x)
            pi_obs = self._pi_vec(self.v_obs)
            dx += cfg.beta_obs * (self.C.T @ (pi_obs * e_obs))

            if cfg.dx_clip is not None:
                np.clip(dx, -cfg.dx_clip, cfg.dx_clip, out=dx)

            x = x + cfg.lr_x * dx
            np.clip(x, -cfg.x_clip, cfg.x_clip, out=x)

        if not np.all(np.isfinite(x)):
            x = x_prior.copy()
            np.clip(x, -cfg.x_clip, cfg.x_clip, out=x)

        self.x = x

        # --- variance updates (causal) ---
        # temporal variance from ||x - x_prior||^2 / d
        e_temp = x - x_prior
        temp_mse = float(np.dot(e_temp, e_temp) / cfg.d_latent)
        temp_sigma = float(math.sqrt(self.v_temporal + cfg.eps))
        temp_mse = float(np.clip(temp_mse, cfg.v_min, min(cfg.v_max, (cfg.k_robust * temp_sigma) ** 2)))
        self.v_temporal = (1.0 - cfg.alpha_v) * self.v_temporal + cfg.alpha_v * temp_mse

        # obs variance elementwise
        e_obs = obs - (self.C @ x)
        obs_sigma = np.sqrt(self.v_obs + cfg.eps)
        e_obs_clip = np.clip(e_obs, -cfg.k_robust * obs_sigma, cfg.k_robust * obs_sigma)
        obs_mse = np.clip(e_obs_clip**2, cfg.v_min, cfg.v_max)
        self.v_obs = (1.0 - cfg.alpha_v) * self.v_obs + cfg.alpha_v * obs_mse

        # --- causal weight learning for A and C ---
        if t >= cfg.warmup_bars:
            pi_temp = self._pi(self.v_temporal)
            d = cfg.d_latent
            dA = cfg.lr_A * (pi_temp * e_temp)[:, None] @ self.x_prev[None, :]
            self.A += dA - cfg.lr_A * cfg.lambda_A * (self.A - np.eye(d))
            np.clip(self.A, -5.0, 5.0, out=self.A)

            pi_obs = self._pi_vec(self.v_obs)
            dC = cfg.lr_C * (pi_obs * e_obs_clip)[:, None] @ x[None, :]
            self.C += dC
            np.clip(self.C, -5.0, 5.0, out=self.C)

        # --- push snapshot into buffers ---
        self._buf_price.append(price_now)
        self._buf_x.append(x.copy())
        self._buf_idx.append(t)

        # --- delayed supervised updates for W_h / v_h when targets become available ---
        for h in cfg.horizons:
            if len(self._buf_idx) <= h:
                continue
            idx0 = self._buf_idx[-(h + 1)]
            if idx0 != t - h:
                continue

            price0 = float(self._buf_price[-(h + 1)])
            x0 = self._buf_x[-(h + 1)]
            y = float(math.log(price_now / price0))

            mu0_raw = float(self.W[h] @ x0)
            sigma_h = float(math.sqrt(self.v_h[h] + cfg.eps))
            res = float(np.clip(y - mu0_raw, -cfg.k_robust * sigma_h, cfg.k_robust * sigma_h))

            self.v_h[h] = (1.0 - cfg.alpha_v) * self.v_h[h] + cfg.alpha_v * float(np.clip(res * res, cfg.v_min, cfg.v_max))

            if t >= cfg.warmup_bars:
                pi_h = self._pi(self.v_h[h])
                self.W[h] += cfg.lr_W * (pi_h * res) * x0
                np.clip(self.W[h], -5.0, 5.0, out=self.W[h])

        # --- state carry (EMA) ---
        self.x_prev = (1.0 - cfg.tau) * self.x_prev + cfg.tau * x
        np.clip(self.x_prev, -cfg.x_clip, cfg.x_clip, out=self.x_prev)

        return self.predict_from_state(price_now)