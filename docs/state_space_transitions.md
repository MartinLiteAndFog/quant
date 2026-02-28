# State Space Transitions (v0.1)

## What this is
We build a **3-axis market state space** `S_t = (X_raw, Y_res, Z_res)` and then discretize it into **voxels** to estimate a **bar-to-bar transition model**:
- `voxel_id(t) -> voxel_id(t+1)`
- from this we derive interpretable diagnostics like:
  - persistence (self-transition)
  - escape probability
  - expected holding time
  - transition entropy (approx)
  - drift vector field in state space
  - basin summaries (v0.1: top-density voxels)

This converts a continuous, noisy market process into a reproducible, inspectable **Markov-ish** model over a small state graph.

## Why we do it
Trading questions we can answer:
- **Where am I right now?** (current voxel/basin)
- **How stable is this state?** (p_self, holding_time, entropy)
- **Where does it tend to move next?** (top-k transitions, drift vector)
- **Is this a “home regime” or a transition zone?** (occupancy, basin share)
- **How should risk sizing adapt?** (low entropy + high persistence => larger size)

## Inputs (data contract)
The transition pipeline expects a *state space* table with at least:
- `ts` (datetime or int; must be sortable)
- `X_raw`, `Y_res`, `Z_res` (floats)

Recommended additional debug columns:
- `signal_*`, `reliability_*`, `disagreement_*`, raw sensor columns, etc.

Example (current SOLUSDT):
`data/state_space/solusdt_ohlcv_state_space_v01_LAST6M.parquet`
with columns:
`ts, X_raw, Y_res, Z_res, ...`

## Core steps
1) **Integrity check**
   - `ts` exists, no nulls
   - sortable (datetime or int)
   - stable sort by `ts`
   - duplicates removed or flagged (policy: dedup keep last)

2) **Voxelization**
   - `n_bins` per axis (default 5 => 125 voxels)
   - bin edges:
     - default: `quantile` per axis (equal occupancy)
     - fallback: `uniform` (not recommended for skewed distributions)
   - edge fix: enforce strict monotonic edges if quantiles collide
   - compute:
     - `ix, iy, iz`
     - `voxel_id = ix + n*iy + n^2*iz`
     - voxel center coordinates `cx, cy, cz`

3) **Transition counting (sparse)**
   - bar-to-bar edges `(from_id -> to_id)`
   - optional exponential decay weighting with `decay_halflife_bars`
   - store long-form counts table (not dense VxV)

4) **Top-k outgoing transitions (smoothed)**
   - per `from_id`, keep `topk` edges by effective count
   - smoothed probability:
     `p = (count_eff + alpha) / (occ_eff + alpha*K)`
   - avoids huge dense matrices and stays inspectable

5) **Diagnostics per voxel**
   - `occ_eff`, `pi`
   - `p_self`, `escape`, `holding_time = 1/(1-p_self)`
   - entropy approx (using top-k)
   - drift vector + speed

6) **Basins v0.1**
   - pragmatic: pick top `basin_k` voxels by `occ_eff`
   - label them as basins (id 0..)
   - compute basin transitions + basin stats

## Outputs (run folder layout)
The CLI writes to:
`data/runs/<run-id>/transitions/`

Artifacts:
- `edges.json` (bin edges, config, schema version)
- `df_with_voxels.parquet` (original df + ix/iy/iz/voxel_id)
- `edges_counts.parquet` (long-form counts: from,to,count_eff,occ_eff, etc.)
- `transitions_topk.parquet` (top-k outgoing edges per from_id)
- `voxel_stats.parquet` (diagnostics per voxel)
- `basin_voxels.parquet`
- `basin_transitions.parquet`
- `basin_stats.parquet`

## Interpretation notes
- Axis values are normalized using rolling robust statistics; **0 ~ locally typical**, not “global median”.
- Negative `Z` means “below baseline instability” (calmer than usual), not “negative stability”.
- Persistence can be inflated if the state space itself is very smooth (rolling windows). Use:
  - decay-weighted counts
  - sensitivity checks on `n_bins`
  - optionally subsample to reduce autocorrelation

## Known limitations (v0.1)
- Basin definition is density-based (top-k), not connected components.
- The Markov assumption is approximate (path dependence exists).
- Nonstationarity: addressed via exponential decay but still imperfect.

## Repro commands (examples)
Integrity:
`python3 scripts/check_ts_integrity.py --input data/state_space/...parquet --ts-col ts`

Transitions:
`python3 scripts/run_state_space_transitions_v01.py --input ... --run-id ... --n-bins 5 --topk 12 --decay-halflife-bars 3000 --basin-k 30`