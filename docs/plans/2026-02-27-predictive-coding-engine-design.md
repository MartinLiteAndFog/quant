# Predictive-Coding Trade Engine – Design Document

**Date:** 2026-02-27
**Status:** Approved
**Approach:** Linear Temporal PC with Adaptive Precision (Approach C)

## Goal

A local, online-learning trading component based on predictive coding / energy minimization.
Input: raw 1-minute OHLCV (close-based).
Output: price-level forecasts and probabilities for horizons 1m, 5m, 15m, 60m, plus actionable trade signals with risk management.

## Architecture

```
src/quant/predictive_coding/
├── __init__.py
├── config.py          # PCConfig dataclass
├── model.py           # TemporalPCModel
├── probability.py     # ProbabilityLayer
├── trade_logic.py     # TradeDecisionLayer
└── targets.py         # multi-horizon log-return targets

scripts/
└── run_pc_backtest.py  # CLI runner
```

### Data Flow Per Bar

```
close_t → targets.py (future log-returns h={1,5,15,60})
        → obs features (r_1, r_5, r_15, rv_20, rv_60)
                ↓
         model.py (infer x_t via relaxation, predict mu_h, learn weights)
                ↓
         probability.py (sigma_h, p_up_h, price_levels, ±1σ bands)
                ↓
         trade_logic.py (directional edge, z-gate, signal, position update, SL/TP/timeout)
                ↓
         runner logs: predictions, signals, trades, equity
```

### Integration Model

Standalone module with output-compatible format.
Trade/event parquets match existing `renko_runner.py` column conventions.
Output directory: `data/runs/<run-id>/pc/`.

## Predictive Coding Core (`model.py`)

### State Variables

| Variable | Shape | Description |
|----------|-------|-------------|
| `x` | R^d | Latent state (d=32 default) |
| `x_prev` | R^d | Previous latent for temporal prediction |
| `A` | R^(d×d) | Transition matrix (init: 0.95*I + 0.01*randn) |
| `W_h` | R^(1×d) × 4 | Readout weights per horizon |
| `C` | R^(n_obs×d) | Decoder/generator (obs_hat = C @ x), n_obs=5 |
| `v_h` | R × 4 | Variance per horizon (init: h * 1e-6) |
| `v_temporal` | R | Temporal variance |
| `v_obs` | R^n_obs | Obs decoder variance (init: 1e-3, keeps obs term subordinate) |

### Observation Features

Derived from close prices only: `obs_t = [r_1, r_5, r_15, rv_20, rv_60]`.
- `r_k = log(close_t / close_{t-k})`: past log-returns
- `rv_k = rolling_std(r_1, k)`: realized volatility proxy
- No r_60 in obs (avoids target leakage / double-counting with y_60)

### Per-Bar Inference (K relaxation steps)

```python
x_prior = A @ x_prev
x = x_prior.copy()

for k in range(K):
    e_temp = x - x_prior
    res_h = y_h - W_h @ x               # per horizon, scalar
    e_obs = obs_t - C @ x               # R^n_obs

    pi_temp = 1.0 / (v_temporal + eps)
    pi_h = 1.0 / (v_h + eps)
    pi_obs = 1.0 / (v_obs + eps)

    dx = (-pi_temp * e_temp
          + sum_h(pi_h * res_h * W_h.T)
          + beta_obs * C.T @ (pi_obs * e_obs))

    x = x + lr_x * dx
```

`beta_obs = 0.2` dampens decoder influence so latent primarily serves prediction, not obs reconstruction.

### Per-Bar Learning (skipped during warmup)

```python
pi_temp = 1.0 / (v_temporal + eps)

# Transition: shrink-to-identity regularization
dA = lr_A * (pi_temp * (x - x_prior))[:,None] @ x_prev[None,:]
A += dA - lr_A * lambda_A * (A - I)

# Readout per horizon
for h in horizons:
    pi_h = 1.0 / (v_h + eps)
    res_h = y_h - W_h @ x
    W_h += lr_W * (pi_h * res_h) * x

# Decoder
pi_obs = 1.0 / (v_obs + eps)
e_obs = obs_t - C @ x
C += lr_C * (pi_obs * e_obs)[:,None] @ x[None,:]
```

### Variance Updates (always run, including warmup)

```python
# Robust clipping of residuals before variance update
res_h = clip(res_h, -k_robust * sqrt(v_h + eps), k_robust * sqrt(v_h + eps))

v_h = (1 - alpha_v) * v_h + alpha_v * clip(res_h**2, v_min, v_max)
v_temporal = (1 - alpha_v) * v_temporal + alpha_v * clip(norm(e_temp)**2 / d, v_min, v_max)
v_obs = (1 - alpha_v) * v_obs + alpha_v * clip(e_obs**2, v_min, v_max)
```

`k_robust = 5`: clips at 5σ to prevent fat-tail contamination of variance estimates.

### State Carry

```python
x_prev = (1 - tau) * x_prev + tau * x
```

`tau = 0.05`: 5% new info per bar. Keeps inference initialization stable without post-hoc pullback.

## Probability & Price Level Layer (`probability.py`)

```python
sigma_h = sqrt(v_h + eps)
mu_h = mu_clip_h                                  # clipped prediction
z_h = mu_h / sigma_h
p_up_h = 0.5 * (1 + erf(z_h / sqrt(2)))          # no scipy needed

price_level_h = P_t * exp(mu_h)
price_upper_h = P_t * exp(mu_h + sigma_h)         # +1σ
price_lower_h = P_t * exp(mu_h - sigma_h)         # -1σ
```

All downstream computations use `mu_clip` consistently.

## Trade Decision Layer (`trade_logic.py`)

### Cost Model

- `fee_bps = 7.0` (roundtrip taker)
- `slippage_bps = 2.0`
- `total_cost = (fee_bps + slippage_bps) / 10_000 = 0.0009`

### Entry Logic Per Horizon

```python
mu = mu_clip_h
sigma = sqrt(v_h + eps)
z = mu / sigma
p = Phi(z)
cost = total_cost
min_edge = min_edge_bps / 10_000

# Long candidate
long_ok = (mu > cost + min_edge) and (p > 0.5 + margin) and (z > z_min)
score_long = (mu - cost) * (2*p - 1)  if long_ok else -inf

# Short candidate
short_ok = (-mu > cost + min_edge) and (p < 0.5 - margin) and (z < -z_min)
score_short = ((-mu) - cost) * (1 - 2*p)  if short_ok else -inf
```

Uses directional mu (not abs), cost-aware z-gate, and confidence-weighted score.

### Horizon Selection

Pick horizon with max score across all horizons and both directions.

### Position Management

States: FLAT, LONG, SHORT.

Transitions:
- FLAT + signal → enter
- In position + exit trigger → FLAT
- In position + strong opposite signal → flip (stricter conditions)

### Exit Rules

| Rule | Condition | Priority |
|------|-----------|----------|
| Stop-loss | unrealized PnL < -sl_pct | 1 |
| Take-profit | unrealized PnL > tp_pct | 2 |
| Timeout | bars_in_trade > timeout_bars (default = chosen horizon) | 3 |
| Flip | opposite signal meeting flip conditions | 4 |

### Flip Conditions (stricter than entry)

- `p_opposite > 0.5 + flip_margin`
- `dir_mu_opposite > cost + min_edge`
- `|z| > z_flip_min`

### Cooldown

After any exit or flip, no new entry/flip for `cooldown_bars` (default 3).

## Backtest Runner (`scripts/run_pc_backtest.py`)

### CLI

```bash
python scripts/run_pc_backtest.py \
    --input data/ohlcv/solusdt_1m.parquet \
    --run-id pc_baseline_01 \
    --fee-bps 7 --slippage-bps 2 \
    --d-latent 32 --n-inference-steps 5 \
    --warmup-bars 200 \
    --margin 0.02 --z-min 0.15 --min-edge-bps 5
```

### Main Loop

```python
for t in range(max_lookback, len(close) - 60):
    obs_t = obs_features[t]
    targets_t = {h: future_logret[h][t]}
    mu, sigma, v = model.step(obs_t, targets_t, is_warmup=(t < warmup))
    probs = probability.compute(mu, sigma, close[t])
    signal, events = trade_logic.update(probs, close[t], t)
    logger.log_bar(t, mu, sigma, probs, signal, events)
```

### Outputs

| File | Key Columns |
|------|-------------|
| `predictions.parquet` | ts, close, mu_{1,5,15,60}, sigma_{1,5,15,60}, p_up_{1,5,15,60}, price_level_{1,5,15,60} |
| `trades.parquet` | entry_ts, exit_ts, entry_px, exit_px, side, pnl_pct, exit_event, horizon, edge, p_at_entry |
| `events.parquet` | ts, event, side, price, pnl_pct, note, seq |
| `equity.parquet` | ts, equity, drawdown |
| `stats.json` | total_return, max_dd, hit_rate, avg_trade_pnl, trade_count, sharpe_approx, fee_total |

## Hyperparameter Defaults

### Model

| Param | Default | Purpose |
|-------|---------|---------|
| `d_latent` | 32 | Latent dimensionality |
| `n_inference_steps` | 5 | Relaxation steps per bar |
| `lr_x` | 0.05 | Inference learning rate |
| `lr_A` | 1e-4 | Transition weight LR |
| `lr_W` | 1e-4 | Readout LR |
| `lr_C` | 1e-4 | Decoder LR |
| `lambda_A` | 1e-4 | Shrink-to-identity strength |
| `alpha_v` | 0.01 | Variance EMA rate |
| `v_min` | 1e-10 | Variance floor |
| `v_max` | 1e-2 | Variance ceiling |
| `tau` | 0.05 | State carry rate |
| `beta_obs` | 0.2 | Obs term damping |
| `k_robust` | 5.0 | Residual clipping at k*sigma |
| `warmup_bars` | 200 | Bars before weight learning |

### Trade Logic

| Param | Default | Purpose |
|-------|---------|---------|
| `fee_bps` | 7.0 | Roundtrip fee |
| `slippage_bps` | 2.0 | Expected slippage |
| `margin` | 0.02 | p threshold above/below 0.5 |
| `z_min` | 0.15 | Min z-score for entry |
| `min_edge_bps` | 5.0 | Noise buffer |
| `flip_margin` | 0.05 | Stricter p for flip |
| `z_flip_min` | 0.5 | Min z for flip |
| `cooldown_bars` | 3 | Bars after exit before re-entry |
| `sl_pct` | 0.015 | Stop-loss |
| `tp_pct` | 0.03 | Take-profit |

## Build Order

1. **A) Data + targets** – load OHLCV, build obs features, build future log-return targets
2. **B) PC core** – TemporalPCModel with inference, learning, variance updates
3. **C) Probability layer** – sigma, p_up, price levels
4. **D) Trade logic** – entry/exit/flip state machine with edge scoring
5. **E) Runner** – CLI, bar-by-bar loop, logging, export, stats
6. **F) Evaluation** – run on SOL data, report baseline, iterate thresholds
