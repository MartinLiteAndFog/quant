# Predictive-Coding Trade Engine – Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an online predictive-coding trade engine that forecasts multi-horizon price levels from raw 1m OHLCV and triggers probability-based trades.

**Architecture:** Temporal PC model with adaptive precision, linear dynamics (shrink-to-identity), decoder for obs features, precision-weighted learning. Standalone module at `src/quant/predictive_coding/` with output-compatible trade/event parquets. CLI runner at `scripts/run_pc_backtest.py`.

**Tech Stack:** Python 3.9+, numpy, pandas, pyarrow (all already in pyproject.toml). No scipy, no torch.

**Design doc:** `docs/plans/2026-02-27-predictive-coding-engine-design.md`

**Data:** `data/raw/exchange=kucoin/symbol=SOL-USDT/timeframe=1m/SOL-USDT_1m_20260207T102718Z.parquet` or `data/clean/SOL-USDC_1m_20260207T143630Z.clean4.parquet`. OHLCV with columns `ts, open, high, low, close`.

---

## Task 1: PCConfig Dataclass

**Files:**
- Create: `src/quant/predictive_coding/__init__.py`
- Create: `src/quant/predictive_coding/config.py`
- Test: `tests/test_predictive_coding.py`

**Step 1: Write the failing test**

```python
# tests/test_predictive_coding.py
from __future__ import annotations
import unittest
from quant.predictive_coding.config import PCConfig

class TestPCConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = PCConfig()
        self.assertEqual(cfg.d_latent, 32)
        self.assertEqual(cfg.horizons, [1, 5, 15, 60])
        self.assertAlmostEqual(cfg.total_cost, (cfg.fee_bps + cfg.slippage_bps) / 10_000)

    def test_override(self):
        cfg = PCConfig(d_latent=64, fee_bps=10.0)
        self.assertEqual(cfg.d_latent, 64)
        self.assertAlmostEqual(cfg.fee_bps, 10.0)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_predictive_coding.py::TestPCConfig -v`
Expected: FAIL (ImportError)

**Step 3: Write minimal implementation**

```python
# src/quant/predictive_coding/__init__.py
(empty)
```

```python
# src/quant/predictive_coding/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

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
    v_min: float = 1e-10
    v_max: float = 1e-2
    tau: float = 0.05            # state carry rate
    beta_obs: float = 0.2        # obs term damping
    k_robust: float = 5.0        # residual clipping at k*sigma
    warmup_bars: int = 200

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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_predictive_coding.py::TestPCConfig -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/quant/predictive_coding/__init__.py src/quant/predictive_coding/config.py tests/test_predictive_coding.py
git commit -m "feat(pc): add PCConfig dataclass with all hyperparameters"
```

---

## Task 2: Targets & Obs Features

**Files:**
- Create: `src/quant/predictive_coding/targets.py`
- Test: `tests/test_predictive_coding.py` (append)

**Step 1: Write the failing test**

```python
class TestTargets(unittest.TestCase):
    def _make_close(self, n=500):
        import numpy as np
        np.random.seed(42)
        return 100.0 * np.exp(np.cumsum(np.random.randn(n) * 0.001))

    def test_build_obs_features_shape(self):
        from quant.predictive_coding.targets import build_obs_features
        close = self._make_close(500)
        obs = build_obs_features(close)
        # first 60 bars are NaN (max lookback for rv_60)
        self.assertEqual(obs.shape, (500, 5))
        self.assertTrue(np.isnan(obs[0, 0]))    # r_1 at t=0 is NaN
        self.assertFalse(np.isnan(obs[60, :]).any())  # all valid after lookback

    def test_build_targets_shape(self):
        from quant.predictive_coding.targets import build_targets
        close = self._make_close(500)
        targets = build_targets(close, horizons=[1, 5, 15, 60])
        self.assertEqual(targets.shape, (500, 4))
        # last 60 rows have NaN for h=60
        self.assertTrue(np.isnan(targets[-1, 3]))
        # but not for h=1
        self.assertFalse(np.isnan(targets[-2, 0]))

    def test_valid_mask(self):
        from quant.predictive_coding.targets import get_valid_range
        close = self._make_close(500)
        start, end = get_valid_range(close, horizons=[1, 5, 15, 60], obs_lookback=60)
        self.assertEqual(start, 60)
        self.assertEqual(end, 500 - 60)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_predictive_coding.py::TestTargets -v`
Expected: FAIL (ImportError)

**Step 3: Write minimal implementation**

```python
# src/quant/predictive_coding/targets.py
from __future__ import annotations
import numpy as np

def build_obs_features(close: np.ndarray) -> np.ndarray:
    """Build observation features from close prices.

    Returns (N, 5) array: [r_1, r_5, r_15, rv_20, rv_60].
    Invalid lookback rows are NaN.
    """
    n = len(close)
    log_close = np.log(close)
    obs = np.full((n, 5), np.nan)

    # Past log-returns
    for i, k in enumerate([1, 5, 15]):
        obs[k:, i] = log_close[k:] - log_close[:-k]

    # r_1 for rolling vol computation
    r1 = np.full(n, np.nan)
    r1[1:] = log_close[1:] - log_close[:-1]

    # Realized vol (rolling std of r_1)
    for i, w in enumerate([20, 60]):
        col_idx = 3 + i
        for t in range(w, n):
            window = r1[t - w + 1 : t + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= 2:
                obs[t, col_idx] = np.std(valid, ddof=1)
    return obs


def build_targets(close: np.ndarray, horizons: list[int]) -> np.ndarray:
    """Build future log-return targets.

    Returns (N, len(horizons)) array. Rows without sufficient future data are NaN.
    """
    n = len(close)
    log_close = np.log(close)
    targets = np.full((n, len(horizons)), np.nan)
    for i, h in enumerate(horizons):
        if h < n:
            targets[: n - h, i] = log_close[h:] - log_close[:-h]
    return targets


def get_valid_range(
    close: np.ndarray,
    horizons: list[int],
    obs_lookback: int = 60,
) -> tuple[int, int]:
    """Return (start, end) indices where both obs and targets are valid."""
    start = obs_lookback
    end = len(close) - max(horizons)
    return start, end
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_predictive_coding.py::TestTargets -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/quant/predictive_coding/targets.py tests/test_predictive_coding.py
git commit -m "feat(pc): add targets and obs feature builders"
```

---

## Task 3: Temporal PC Model Core

**Files:**
- Create: `src/quant/predictive_coding/model.py`
- Test: `tests/test_predictive_coding.py` (append)

**Step 1: Write the failing test**

```python
class TestTemporalPCModel(unittest.TestCase):
    def test_init_shapes(self):
        from quant.predictive_coding.model import TemporalPCModel
        from quant.predictive_coding.config import PCConfig
        cfg = PCConfig(d_latent=16)
        m = TemporalPCModel(cfg)
        self.assertEqual(m.x.shape, (16,))
        self.assertEqual(m.A.shape, (16, 16))
        self.assertEqual(len(m.W), 4)
        self.assertEqual(m.W[0].shape, (16,))
        self.assertEqual(m.C.shape, (5, 16))

    def test_step_returns_mu_and_sigma(self):
        from quant.predictive_coding.model import TemporalPCModel
        from quant.predictive_coding.config import PCConfig
        import numpy as np
        cfg = PCConfig(d_latent=16, warmup_bars=0)
        m = TemporalPCModel(cfg)
        obs = np.random.randn(5) * 0.01
        targets = {1: 0.001, 5: 0.002, 15: 0.003, 60: 0.005}
        mu, sigma = m.step(obs, targets, is_warmup=False)
        self.assertEqual(len(mu), 4)
        self.assertEqual(len(sigma), 4)
        for s in sigma.values():
            self.assertGreater(s, 0)

    def test_variance_updates_during_warmup(self):
        from quant.predictive_coding.model import TemporalPCModel
        from quant.predictive_coding.config import PCConfig
        import numpy as np
        cfg = PCConfig(d_latent=8)
        m = TemporalPCModel(cfg)
        v_before = {h: m.v_h[h] for h in cfg.horizons}
        obs = np.random.randn(5) * 0.01
        targets = {1: 0.01, 5: 0.02, 15: 0.03, 60: 0.05}
        m.step(obs, targets, is_warmup=True)
        # Variance should change even during warmup
        changed = any(m.v_h[h] != v_before[h] for h in cfg.horizons)
        self.assertTrue(changed)

    def test_weights_frozen_during_warmup(self):
        from quant.predictive_coding.model import TemporalPCModel
        from quant.predictive_coding.config import PCConfig
        import numpy as np
        cfg = PCConfig(d_latent=8)
        m = TemporalPCModel(cfg)
        A_before = m.A.copy()
        obs = np.random.randn(5) * 0.01
        targets = {1: 0.001, 5: 0.002, 15: 0.003, 60: 0.005}
        m.step(obs, targets, is_warmup=True)
        np.testing.assert_array_equal(m.A, A_before)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_predictive_coding.py::TestTemporalPCModel -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

```python
# src/quant/predictive_coding/model.py
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

        # Latent state
        self.x = np.zeros(d)
        self.x_prev = np.zeros(d)

        # Transition matrix (near identity)
        self.A = 0.95 * np.eye(d) + 0.01 * rng.standard_normal((d, d))

        # Readout weights per horizon: W_h ∈ R^d
        self.W: dict[int, np.ndarray] = {
            h: 0.01 * rng.standard_normal(d) for h in horizons
        }

        # Decoder: C ∈ R^(n_obs × d)
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

        # --- Predictions ---
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
                + cfg.alpha_v * np.clip(res_clipped**2, cfg.v_min, cfg.v_max)
            )

        e_temp = x - x_prior
        self.v_temporal = (
            (1 - cfg.alpha_v) * self.v_temporal
            + cfg.alpha_v * np.clip(float(np.dot(e_temp, e_temp) / d), cfg.v_min, cfg.v_max)
        )

        e_obs = obs - self.C @ x
        self.v_obs = (
            (1 - cfg.alpha_v) * self.v_obs
            + cfg.alpha_v * np.clip(e_obs**2, cfg.v_min, cfg.v_max)
        )

        # --- Weight learning (skip during warmup) ---
        if not is_warmup:
            pi_temp = 1.0 / (self.v_temporal + eps)

            # Transition: shrink-to-identity
            e_temp = x - x_prior
            dA = cfg.lr_A * (pi_temp * e_temp[:, None] @ self.x_prev[None, :])
            self.A += dA - cfg.lr_A * cfg.lambda_A * (self.A - np.eye(d))

            # Readout per horizon
            for h in cfg.horizons:
                pi_h = 1.0 / (self.v_h[h] + eps)
                res_h = targets[h] - float(self.W[h] @ x)
                self.W[h] += cfg.lr_W * (pi_h * res_h) * x

            # Decoder
            pi_obs = 1.0 / (self.v_obs + eps)
            e_obs = obs - self.C @ x
            self.C += cfg.lr_C * (pi_obs * e_obs)[:, None] @ x[None, :]

        # --- State carry ---
        self.x_prev = (1 - cfg.tau) * self.x_prev + cfg.tau * x

        return mu, sigma
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_predictive_coding.py::TestTemporalPCModel -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/quant/predictive_coding/model.py tests/test_predictive_coding.py
git commit -m "feat(pc): add TemporalPCModel with inference, learning, adaptive precision"
```

---

## Task 4: Probability Layer

**Files:**
- Create: `src/quant/predictive_coding/probability.py`
- Test: `tests/test_predictive_coding.py` (append)

**Step 1: Write the failing test**

```python
class TestProbabilityLayer(unittest.TestCase):
    def test_compute_basics(self):
        from quant.predictive_coding.probability import compute_probabilities
        mu = {1: 0.001, 5: -0.002, 15: 0.0, 60: 0.005}
        sigma = {1: 0.001, 5: 0.001, 15: 0.001, 60: 0.002}
        price = 100.0
        result = compute_probabilities(mu, sigma, price)
        # p_up for positive mu should be > 0.5
        self.assertGreater(result[1]["p_up"], 0.5)
        # p_up for negative mu should be < 0.5
        self.assertLess(result[5]["p_up"], 0.5)
        # p_up for zero mu should be ~0.5
        self.assertAlmostEqual(result[15]["p_up"], 0.5, places=3)
        # price levels should be close to price
        self.assertAlmostEqual(result[1]["price_level"], 100.0 * np.exp(0.001), places=4)

    def test_bands(self):
        from quant.predictive_coding.probability import compute_probabilities
        mu = {1: 0.001}
        sigma = {1: 0.002}
        result = compute_probabilities(mu, sigma, 100.0)
        self.assertGreater(result[1]["price_upper"], result[1]["price_level"])
        self.assertLess(result[1]["price_lower"], result[1]["price_level"])
```

**Step 2: Run test → FAIL**

**Step 3: Write implementation**

```python
# src/quant/predictive_coding/probability.py
from __future__ import annotations
import math
import numpy as np

_SQRT2 = math.sqrt(2.0)

def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / _SQRT2))

def compute_probabilities(
    mu: dict[int, float],
    sigma: dict[int, float],
    price: float,
) -> dict[int, dict[str, float]]:
    """Compute p_up, price levels, and ±1σ bands per horizon."""
    out: dict[int, dict[str, float]] = {}
    for h in mu:
        m = mu[h]
        s = sigma[h]
        z = m / s if s > 1e-15 else 0.0
        p_up = _normal_cdf(z)
        out[h] = {
            "mu": m,
            "sigma": s,
            "z": z,
            "p_up": p_up,
            "price_level": price * math.exp(m),
            "price_upper": price * math.exp(m + s),
            "price_lower": price * math.exp(m - s),
        }
    return out
```

**Step 4: Run test → PASS**

**Step 5: Commit**

```bash
git add src/quant/predictive_coding/probability.py tests/test_predictive_coding.py
git commit -m "feat(pc): add probability layer with CDF, price levels, bands"
```

---

## Task 5: Trade Decision Layer

**Files:**
- Create: `src/quant/predictive_coding/trade_logic.py`
- Test: `tests/test_predictive_coding.py` (append)

**Step 1: Write the failing test**

```python
class TestTradeDecisionLayer(unittest.TestCase):
    def test_no_trade_during_cooldown(self):
        from quant.predictive_coding.trade_logic import TradeDecisionLayer
        from quant.predictive_coding.config import PCConfig
        cfg = PCConfig(cooldown_bars=3)
        tdl = TradeDecisionLayer(cfg)
        # Simulate a trade exit, then check cooldown
        tdl.position = 0
        tdl.cooldown_remaining = 3
        probs = {5: {"mu": 0.01, "sigma": 0.001, "z": 10.0, "p_up": 0.99,
                      "price_level": 101.0, "price_upper": 102.0, "price_lower": 100.0}}
        signal, events = tdl.update(probs, 100.0, bar_idx=10)
        self.assertEqual(signal, 0)
        self.assertEqual(tdl.cooldown_remaining, 2)

    def test_long_entry(self):
        from quant.predictive_coding.trade_logic import TradeDecisionLayer
        from quant.predictive_coding.config import PCConfig
        cfg = PCConfig(margin=0.02, z_min=0.1, min_edge_bps=0.0, fee_bps=0.0, slippage_bps=0.0)
        tdl = TradeDecisionLayer(cfg)
        probs = {}
        for h in [1, 5, 15, 60]:
            probs[h] = {"mu": 0.005, "sigma": 0.001, "z": 5.0, "p_up": 0.99,
                        "price_level": 100.5, "price_upper": 101.0, "price_lower": 100.0}
        signal, events = tdl.update(probs, 100.0, bar_idx=10)
        self.assertEqual(signal, 1)
        self.assertEqual(tdl.position, 1)

    def test_stop_loss(self):
        from quant.predictive_coding.trade_logic import TradeDecisionLayer
        from quant.predictive_coding.config import PCConfig
        cfg = PCConfig(sl_pct=0.01, fee_bps=0.0, slippage_bps=0.0, margin=0.0, z_min=0.0, min_edge_bps=0.0)
        tdl = TradeDecisionLayer(cfg)
        # Force into a long position
        tdl.position = 1
        tdl.entry_price = 100.0
        tdl.entry_bar = 0
        tdl.chosen_horizon = 5
        # Price drops below SL
        probs = {h: {"mu": -0.02, "sigma": 0.01, "z": -2.0, "p_up": 0.02,
                      "price_level": 98.0, "price_upper": 99.0, "price_lower": 97.0}
                 for h in [1, 5, 15, 60]}
        signal, events = tdl.update(probs, 98.5, bar_idx=5)
        self.assertEqual(tdl.position, 0)
        self.assertTrue(any(e["event"] == "sl_exit" for e in events))
```

**Step 2: Run test → FAIL**

**Step 3: Write implementation**

```python
# src/quant/predictive_coding/trade_logic.py
from __future__ import annotations
from quant.predictive_coding.config import PCConfig


class TradeDecisionLayer:
    def __init__(self, cfg: PCConfig) -> None:
        self.cfg = cfg
        self.position: int = 0       # -1, 0, 1
        self.entry_price: float = 0.0
        self.entry_bar: int = 0
        self.chosen_horizon: int = 0
        self.cooldown_remaining: int = 0
        self.trade_seq: int = 0

    def update(
        self,
        probs: dict[int, dict[str, float]],
        price: float,
        bar_idx: int,
    ) -> tuple[int, list[dict]]:
        cfg = self.cfg
        events: list[dict] = []
        cost = cfg.total_cost
        min_edge = cfg.min_edge_bps / 10_000

        # --- Cooldown ---
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            # Still check exits if in position
            if self.position != 0:
                events.extend(self._check_exits(price, bar_idx, probs))
            return self.position, events

        # --- If in position: check exits first ---
        if self.position != 0:
            exit_events = self._check_exits(price, bar_idx, probs)
            events.extend(exit_events)
            if self.position == 0:
                return 0, events

            # Check flip
            flip_events = self._check_flip(price, bar_idx, probs)
            events.extend(flip_events)
            return self.position, events

        # --- If flat: check entry ---
        best_score = -1.0
        best_dir = 0
        best_h = 0

        for h, p in probs.items():
            mu = p["mu"]
            z = p["z"]
            p_up = p["p_up"]

            # Long candidate
            if mu > cost + min_edge and p_up > 0.5 + cfg.margin and z > cfg.z_min:
                score = (mu - cost) * (2 * p_up - 1)
                if score > best_score:
                    best_score = score
                    best_dir = 1
                    best_h = h

            # Short candidate
            if -mu > cost + min_edge and p_up < 0.5 - cfg.margin and z < -cfg.z_min:
                score = (-mu - cost) * (1 - 2 * p_up)
                if score > best_score:
                    best_score = score
                    best_dir = -1
                    best_h = h

        if best_dir != 0:
            self.position = best_dir
            self.entry_price = price
            self.entry_bar = bar_idx
            self.chosen_horizon = best_h
            self.trade_seq += 1
            events.append({
                "event": "entry",
                "side": best_dir,
                "price": price,
                "bar_idx": bar_idx,
                "horizon": best_h,
                "edge": best_score,
                "seq": self.trade_seq,
                "pnl_pct": 0.0,
            })

        return self.position, events

    def _check_exits(
        self, price: float, bar_idx: int, probs: dict[int, dict[str, float]]
    ) -> list[dict]:
        cfg = self.cfg
        events: list[dict] = []
        if self.position == 0:
            return events

        pnl_pct = self.position * (price - self.entry_price) / self.entry_price
        bars_held = bar_idx - self.entry_bar
        timeout = self.chosen_horizon if self.chosen_horizon > 0 else 60

        exit_event = None
        if pnl_pct < -cfg.sl_pct:
            exit_event = "sl_exit"
        elif pnl_pct > cfg.tp_pct:
            exit_event = "tp_exit"
        elif bars_held >= timeout:
            exit_event = "timeout_exit"

        if exit_event is not None:
            events.append({
                "event": exit_event,
                "side": self.position,
                "price": price,
                "bar_idx": bar_idx,
                "pnl_pct": pnl_pct,
                "seq": self.trade_seq,
                "horizon": self.chosen_horizon,
                "edge": 0.0,
            })
            self.position = 0
            self.entry_price = 0.0
            self.cooldown_remaining = cfg.cooldown_bars

        return events

    def _check_flip(
        self, price: float, bar_idx: int, probs: dict[int, dict[str, float]]
    ) -> list[dict]:
        cfg = self.cfg
        events: list[dict] = []
        cost = cfg.total_cost
        min_edge = cfg.min_edge_bps / 10_000

        for h, p in probs.items():
            mu = p["mu"]
            z = p["z"]
            p_up = p["p_up"]

            should_flip = False
            new_dir = 0

            if self.position == 1:
                if (
                    -mu > cost + min_edge
                    and p_up < 0.5 - cfg.flip_margin
                    and z < -cfg.z_flip_min
                ):
                    should_flip = True
                    new_dir = -1

            elif self.position == -1:
                if (
                    mu > cost + min_edge
                    and p_up > 0.5 + cfg.flip_margin
                    and z > cfg.z_flip_min
                ):
                    should_flip = True
                    new_dir = 1

            if should_flip:
                pnl_pct = self.position * (price - self.entry_price) / self.entry_price
                events.append({
                    "event": "flip_exit",
                    "side": self.position,
                    "price": price,
                    "bar_idx": bar_idx,
                    "pnl_pct": pnl_pct,
                    "seq": self.trade_seq,
                    "horizon": self.chosen_horizon,
                    "edge": 0.0,
                })
                self.position = new_dir
                self.entry_price = price
                self.entry_bar = bar_idx
                self.chosen_horizon = h
                self.trade_seq += 1
                events.append({
                    "event": "entry",
                    "side": new_dir,
                    "price": price,
                    "bar_idx": bar_idx,
                    "horizon": h,
                    "edge": 0.0,
                    "seq": self.trade_seq,
                    "pnl_pct": 0.0,
                })
                self.cooldown_remaining = cfg.cooldown_bars
                break

        return events
```

**Step 4: Run test → PASS**

**Step 5: Commit**

```bash
git add src/quant/predictive_coding/trade_logic.py tests/test_predictive_coding.py
git commit -m "feat(pc): add trade decision layer with edge scoring, exits, flip"
```

---

## Task 6: Backtest Runner + Logger

**Files:**
- Create: `scripts/run_pc_backtest.py`
- Test: manual run with data

**Step 1: Write the runner**

The runner:
1. Loads OHLCV parquet (reuse `_read_ohlcv_parquet` pattern from `renko_runner.py`)
2. Builds obs features and targets
3. Runs bar-by-bar: model.step → probability → trade_logic.update
4. Logs predictions, events, trades, equity
5. Exports to `data/runs/<run-id>/pc/`
6. Prints summary stats

```python
# scripts/run_pc_backtest.py
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quant.predictive_coding.config import PCConfig
from quant.predictive_coding.model import TemporalPCModel
from quant.predictive_coding.probability import compute_probabilities
from quant.predictive_coding.targets import build_obs_features, build_targets, get_valid_range
from quant.predictive_coding.trade_logic import TradeDecisionLayer


def _read_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts"})
        else:
            raise ValueError("parquet missing 'ts' column and not datetime-indexed")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


def run_backtest(df: pd.DataFrame, cfg: PCConfig) -> dict:
    close = df["close"].values.astype(np.float64)
    ts = df["ts"].values
    n = len(close)

    obs_all = build_obs_features(close)
    targets_all = build_targets(close, cfg.horizons)
    start, end = get_valid_range(close, cfg.horizons, obs_lookback=60)

    model = TemporalPCModel(cfg)
    trade_logic = TradeDecisionLayer(cfg)

    pred_rows = []
    event_rows = []
    trade_pairs = []
    equity_rows = []

    equity = 1.0
    peak_equity = 1.0
    open_trade = None

    for t in range(start, end):
        obs = obs_all[t]
        tgt = {h: targets_all[t, i] for i, h in enumerate(cfg.horizons)}

        if np.any(np.isnan(obs)) or any(np.isnan(v) for v in tgt.values()):
            continue

        is_warmup = t < start + cfg.warmup_bars
        mu, sigma = model.step(obs, tgt, is_warmup=is_warmup)
        probs = compute_probabilities(mu, sigma, float(close[t]))

        if is_warmup:
            continue

        signal, events = trade_logic.update(probs, float(close[t]), t)

        # Log prediction
        row = {"ts": ts[t], "close": float(close[t])}
        for h in cfg.horizons:
            p = probs[h]
            row[f"mu_{h}"] = p["mu"]
            row[f"sigma_{h}"] = p["sigma"]
            row[f"p_up_{h}"] = p["p_up"]
            row[f"price_level_{h}"] = p["price_level"]
        row["signal"] = signal
        pred_rows.append(row)

        # Process events
        for ev in events:
            ev_row = {
                "ts": ts[t],
                "event": ev["event"],
                "side": ev["side"],
                "price": ev["price"],
                "pnl_pct": ev["pnl_pct"],
                "seq": ev["seq"],
                "horizon": ev.get("horizon", 0),
            }
            event_rows.append(ev_row)

            if ev["event"] == "entry":
                open_trade = {
                    "entry_ts": ts[t],
                    "entry_px": ev["price"],
                    "side": ev["side"],
                    "horizon": ev.get("horizon", 0),
                    "edge": ev.get("edge", 0.0),
                    "p_at_entry": probs.get(ev.get("horizon", 1), {}).get("p_up", 0.5),
                }
            elif ev["event"] in ("sl_exit", "tp_exit", "timeout_exit", "flip_exit"):
                if open_trade is not None:
                    pnl = ev["pnl_pct"] - cfg.total_cost
                    equity *= (1 + pnl)
                    trade_pairs.append({
                        **open_trade,
                        "exit_ts": ts[t],
                        "exit_px": ev["price"],
                        "pnl_pct": pnl,
                        "exit_event": ev["event"],
                    })
                    open_trade = None
                    if ev["event"] == "flip_exit":
                        pass  # next entry event in same bar handles open_trade

        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        equity_rows.append({"ts": ts[t], "equity": equity, "drawdown": dd})

    # Build DataFrames
    predictions_df = pd.DataFrame(pred_rows)
    events_df = pd.DataFrame(event_rows) if event_rows else pd.DataFrame()
    trades_df = pd.DataFrame(trade_pairs) if trade_pairs else pd.DataFrame()
    equity_df = pd.DataFrame(equity_rows)

    # Stats
    n_trades = len(trades_df)
    stats = {
        "total_return_pct": (equity - 1) * 100,
        "max_drawdown_pct": float(equity_df["drawdown"].max() * 100) if len(equity_df) else 0.0,
        "trade_count": n_trades,
        "hit_rate": float((trades_df["pnl_pct"] > 0).mean() * 100) if n_trades else 0.0,
        "avg_trade_pnl_bps": float(trades_df["pnl_pct"].mean() * 10_000) if n_trades else 0.0,
        "avg_winner_bps": float(trades_df.loc[trades_df["pnl_pct"] > 0, "pnl_pct"].mean() * 10_000) if n_trades and (trades_df["pnl_pct"] > 0).any() else 0.0,
        "avg_loser_bps": float(trades_df.loc[trades_df["pnl_pct"] <= 0, "pnl_pct"].mean() * 10_000) if n_trades and (trades_df["pnl_pct"] <= 0).any() else 0.0,
        "fee_bps": cfg.fee_bps,
        "slippage_bps": cfg.slippage_bps,
        "total_fee_drag_bps": n_trades * (cfg.fee_bps + cfg.slippage_bps),
        "bars_processed": end - start - cfg.warmup_bars,
        "warmup_bars": cfg.warmup_bars,
    }

    return {
        "predictions": predictions_df,
        "events": events_df,
        "trades": trades_df,
        "equity": equity_df,
        "stats": stats,
    }


def main():
    ap = argparse.ArgumentParser(description="Predictive-Coding Backtest Runner")
    ap.add_argument("--input", required=True, help="Path to OHLCV parquet")
    ap.add_argument("--run-id", default=None)

    # Model params
    ap.add_argument("--d-latent", type=int, default=32)
    ap.add_argument("--n-inference-steps", type=int, default=5)
    ap.add_argument("--lr-x", type=float, default=0.05)
    ap.add_argument("--lr-A", type=float, default=1e-4)
    ap.add_argument("--lr-W", type=float, default=1e-4)
    ap.add_argument("--lr-C", type=float, default=1e-4)
    ap.add_argument("--warmup-bars", type=int, default=200)
    ap.add_argument("--beta-obs", type=float, default=0.2)

    # Trade params
    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--z-min", type=float, default=0.15)
    ap.add_argument("--min-edge-bps", type=float, default=5.0)
    ap.add_argument("--sl-pct", type=float, default=0.015)
    ap.add_argument("--tp-pct", type=float, default=0.03)
    ap.add_argument("--cooldown-bars", type=int, default=3)

    args = ap.parse_args()

    cfg = PCConfig(
        d_latent=args.d_latent,
        n_inference_steps=args.n_inference_steps,
        lr_x=args.lr_x,
        lr_A=args.lr_A,
        lr_W=args.lr_W,
        lr_C=args.lr_C,
        warmup_bars=args.warmup_bars,
        beta_obs=args.beta_obs,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        margin=args.margin,
        z_min=args.z_min,
        min_edge_bps=args.min_edge_bps,
        sl_pct=args.sl_pct,
        tp_pct=args.tp_pct,
        cooldown_bars=args.cooldown_bars,
    )

    print(f"[PC] Loading {args.input}")
    df = _read_ohlcv(args.input)
    print(f"[PC] Loaded {len(df)} bars, {df['ts'].iloc[0]} to {df['ts'].iloc[-1]}")

    results = run_backtest(df, cfg)

    # Print stats
    stats = results["stats"]
    print("\n=== Predictive-Coding Backtest Results ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Export
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("data/runs") / run_id / "pc"
    out_dir.mkdir(parents=True, exist_ok=True)

    results["predictions"].to_parquet(out_dir / "predictions.parquet", index=False)
    if len(results["events"]):
        results["events"].to_parquet(out_dir / "events.parquet", index=False)
    if len(results["trades"]):
        results["trades"].to_parquet(out_dir / "trades.parquet", index=False)
    results["equity"].to_parquet(out_dir / "equity.parquet", index=False)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\n[PC] Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
```

**Step 2: Run end-to-end test**

```bash
python scripts/run_pc_backtest.py \
    --input data/clean/SOL-USDC_1m_20260207T143630Z.clean4.parquet \
    --run-id pc_baseline_test \
    --warmup-bars 200
```

Expected: prints stats, writes to `data/runs/pc_baseline_test/pc/`.

**Step 3: Commit**

```bash
git add scripts/run_pc_backtest.py
git commit -m "feat(pc): add backtest runner with CLI, logging, export"
```

---

## Task 7: Integration Smoke Test

**Files:**
- Modify: `tests/test_predictive_coding.py` (append)

**Step 1: Write integration test with synthetic data**

```python
class TestIntegration(unittest.TestCase):
    def test_end_to_end_synthetic(self):
        """Full pipeline on synthetic trending data should produce trades."""
        from quant.predictive_coding.config import PCConfig
        from quant.predictive_coding.model import TemporalPCModel
        from quant.predictive_coding.probability import compute_probabilities
        from quant.predictive_coding.targets import build_obs_features, build_targets, get_valid_range
        from quant.predictive_coding.trade_logic import TradeDecisionLayer

        np.random.seed(42)
        n = 2000
        trend = np.cumsum(np.random.randn(n) * 0.001 + 0.0001)  # slight uptrend
        close = 100.0 * np.exp(trend)

        cfg = PCConfig(
            d_latent=8,
            n_inference_steps=3,
            warmup_bars=100,
            fee_bps=0.0,
            slippage_bps=0.0,
            margin=0.01,
            z_min=0.05,
            min_edge_bps=0.0,
        )
        obs_all = build_obs_features(close)
        targets_all = build_targets(close, cfg.horizons)
        start, end = get_valid_range(close, cfg.horizons)

        model = TemporalPCModel(cfg)
        tdl = TradeDecisionLayer(cfg)
        n_signals = 0

        for t in range(start, end):
            obs = obs_all[t]
            tgt = {h: targets_all[t, i] for i, h in enumerate(cfg.horizons)}
            if np.any(np.isnan(obs)) or any(np.isnan(v) for v in tgt.values()):
                continue
            is_warmup = t < start + cfg.warmup_bars
            mu, sigma = model.step(obs, tgt, is_warmup=is_warmup)
            if is_warmup:
                continue
            probs = compute_probabilities(mu, sigma, float(close[t]))
            signal, _ = tdl.update(probs, float(close[t]), t)
            if signal != 0:
                n_signals += 1

        # Should have produced at least some signals on trending data
        self.assertGreater(n_signals, 0)
```

**Step 2: Run all tests**

```bash
python -m pytest tests/test_predictive_coding.py -v
```

**Step 3: Commit**

```bash
git add tests/test_predictive_coding.py
git commit -m "test(pc): add integration smoke test on synthetic data"
```
