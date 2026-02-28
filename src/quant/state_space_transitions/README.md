# quant.state_space_transitions

This package builds a sparse, inspectable **voxelized transition model** from a 3-axis market state space.

## Entry points
- CLI: `scripts/run_state_space_transitions_v01.py`
- Core orchestration: `quant.state_space_transitions.pipeline`

## Expected input columns
Required:
- `ts`
- `X_raw`, `Y_res`, `Z_res`

Configurable column names via CLI flags.

## Outputs
Written to `data/runs/<run-id>/transitions/` (see `docs/state_space_transitions.md`).

## Design goals
- pure functions
- no global state
- sparse outputs (no dense VxV matrix)
- artifacts are KG-friendly: explicit schema-ish JSON + parquet tables

## Versioning
- v0.1: top-k transitions + basic voxel diagnostics + density basins
- future: connected-component basins, hysteresis/soft voxel assignment, rolling transition models