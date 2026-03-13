# Other — viz

# Visual Layer Configuration Documentation

## Overview
The Visual Layer (v0.2) configuration module defines settings for generating visualizations and analysis outputs from state space data. It provides a standardized, configurable way to create consistent visualizations across different analyses.

## Key Components

### Data Sources
The configuration expects several key input artifacts:
- `state_space.parquet`: Contains raw trajectory data (timestamps, coordinates)
- `voxel_map.parquet`: Discretized state space mapping
- `transitions_topk.parquet`: Transition probabilities between voxels
- `basins.parquet`: Basin classification data
- `basin_flows.parquet`: Inter-basin transition flows

### Visualization Settings

#### Global Render Settings
```yaml
render:
  backend_default: "plotly"  # Controls default visualization backend
  write_png: true
  write_html: true
  dpi: 150
```

The module supports both Plotly and Matplotlib backends, with configurable output formats and quality settings.

#### Data Scaling Pipeline
The configuration implements a robust data scaling pipeline:

1. Quantile clipping (removes outliers)
2. Log scaling for heavy-tailed metrics
3. CDF stretching for high-contrast visualization
4. Final 0-1 normalization

This pipeline ensures consistent and visually meaningful representations across different metrics.

### Key Visualization Types

1. **State Space Views**
   - Occupancy plots showing voxel density and importance
   - Persistence views highlighting temporal characteristics
   - Support for both 2D slices and 3D interactive plots

2. **Transition Analysis**
   - Filtered transition network visualization
   - Configurable edge filtering based on flow mass
   - Maximum edge limits to prevent visual clutter

3. **Basin Analysis**
   - Basin-centric views with configurable policies
   - Flow matrices between basins
   - Support for noise basin handling

4. **Trajectory Visualization**
   - Compressed state-wander representation
   - Anti-spaghetti measures for clarity
   - Configurable point density and jump detection

## Usage Guidelines

### Filtering Configuration
The module provides several filtering mechanisms:

```yaml
filters:
  voxels:
    mass_cumsum_keep: 0.60  # Keep top 60% by cumulative mass
    max_voxels: 125        # Hard limit on voxel count
    
  transitions:
    min_flow_mass_quantile: 0.90
    max_edges_global: 600
    topk_per_from: 5
```

These filters help maintain visual clarity while preserving important features.

### Export Capabilities
The configuration supports standardized exports for:
- Node/edge data for network analysis
- Basin-level aggregations
- Documentation generation
- Slice visualizations

## Best Practices

1. **Performance Considerations**
   - Use `max_points` and `max_edges` limits appropriate for your visualization target
   - Enable `write_png` for static sharing, `write_html` for interactive exploration

2. **Visual Clarity**
   - Leverage the scaling pipeline for consistent visualization
   - Use appropriate filters to reduce visual clutter
   - Consider using different metrics for size vs. color encoding

3. **Documentation**
   - Enable the `docs` section to generate supporting documentation
   - Use consistent naming conventions for output files
   - Maintain version compatibility (current: v0.2)

## Integration Notes
This configuration module is typically used in conjunction with visualization execution code. While it defines the settings, it requires corresponding visualization logic to interpret and apply these settings.