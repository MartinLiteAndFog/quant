# Plot Catalog

## state_space_occupancy
- 3D voxel cloud
- size: `occ_eff`
- color: `pi`
- use: where market spends time

## state_space_persistence
- 3D voxel cloud
- size: `occ_eff`
- color: `holding_time`
- use: stable vs unstable local states

## slices_xy/xz/yz
- 2D heatmaps aggregated over third axis
- metric default: `pi`
- use: quick structural overview without 3D interaction

## transitions
- 3D edge overlay
- filtered by flow mass quantile
- use: dominant directional transitions

## trajectory
- 3D path over time with decimation
- use: readable temporal movement in state space

## basins + basin_flow_matrix
- basin voxel highlight + core voxels
- basin-to-basin flow heatmap
- use: attractor and regime-shift overview
