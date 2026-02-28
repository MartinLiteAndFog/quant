# Visual Style Guide

## Scaling and Clipping
- All metric scaling goes through `src/visual/scales.py`.
- Apply quantile clipping before normalization.
- Apply `log1p` only to heavy-tailed metrics.

## Naming
- `*_scene`: returns `Scene` object.
- `make_*_layer`: returns `SceneLayer`.
- `render_*`: side-effect functions writing output files.

## Plot Hygiene
- Avoid spaghetti: decimate trajectories (`change_only` or `stride`).
- Limit transitions with global filters (`flow_mass` quantile, max edges).
- Keep legends and axes explicit (`X_raw`, `Y_res`, `Z_res`).

## Contracts
- Fail fast with clear messages if artifacts or required columns are missing.
- Enforce sorted + unique `ts` for `state_space` and `voxel_map`.
