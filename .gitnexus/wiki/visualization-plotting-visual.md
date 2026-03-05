# Visualization & Plotting — visual

# Visualization & Plotting Module

The `visual` module provides a flexible framework for creating interactive and static visualizations of state space data, trajectories, transitions, and basins. It uses a scene-based architecture that separates data preparation, scene composition, and rendering.

## Core Concepts

### Scene Architecture

The module uses a scene-based architecture built around two key classes:

```python
@dataclass
class SceneLayer:
    name: str
    kind: str 
    data: pd.DataFrame
    params: Dict[str, Any]

@dataclass 
class Scene:
    title: str
    layers: List[SceneLayer]
    settings: Dict[str, Any]
```

A Scene represents a complete visualization and contains multiple layers. Each SceneLayer represents a distinct visual element (points, lines, text, etc.) with its own data and rendering parameters.

### Key Components

1. **Scene Building Functions** (`figures/`)
   - `build_basins_scene()` - Basin visualization with flow arrows
   - `build_trajectory_scene()` - 3D trajectory visualization 
   - `build_transitions_scene()` - State transition network
   - `build_slice_scenes()` - 2D slice views
   - `build_occupancy_scene()` - State space occupancy

2. **Layer Generators** (`layers/`)
   - `make_voxel_layer()` - Point clouds for state space
   - `make_edges_layer()` - Transition edges
   - `make_trajectory_layer()` - Trajectory paths
   - `make_basin_layer()` - Basin regions

3. **Renderers** (`render/`)
   - `render_matplotlib()` - Static plots using matplotlib
   - `render_plotly()` - Interactive plots using Plotly

4. **Data Loading** (`io.py`)
   - Loads and validates input data files
   - Handles configuration parsing
   - Enforces data format requirements

## Usage Example

```python
from visual.figures import build_basins_scene
from visual.render import render_plotly

# Load data
contracts = load_contracts("data/", config)

# Build scene
scene = build_basins_scene(
    voxel_stats=contracts["voxel_stats"],
    basins=contracts["basins"],
    cfg=config
)

# Render visualization
render_plotly(scene, "output/basins.html")
```

## Data Requirements

The module expects several key data files:

- `state_space.parquet` - Raw trajectory data
- `voxel_map.parquet` - Voxel discretization mapping
- `voxel_stats.parquet` - Voxel-level statistics
- `transitions_topk.parquet` - Top-K transitions between voxels
- `basins.parquet` - Basin assignments for voxels

Each file must contain specific required columns as defined in the configuration.

## Configuration

The module uses a YAML configuration file to control:

- Input/output file paths
- Column mappings
- Visual styling parameters
- Filtering thresholds
- Scaling behavior

## Key Features

- **Flexible Rendering**: Supports both static (matplotlib) and interactive (Plotly) output
- **Layered Architecture**: Modular design allows mixing different visual elements
- **Scale Transformations**: Configurable scaling pipelines for metrics
- **Filtering Controls**: Multiple filtering options to handle large datasets
- **Basin Analysis**: Specialized tools for visualizing basin structure and flows

## Integration Points

The module primarily integrates with:

- Data processing pipeline outputs (parquet files)
- Configuration management system
- Visualization scripts in `scripts/visual/`

## Error Handling

The module includes validation for:

- Required columns in input data
- Data type consistency
- Configuration completeness
- Timestamp parsing and ordering

## Performance Considerations

For large datasets, the module implements:

- Configurable downsampling
- Quantile-based filtering
- Efficient data structures
- Lazy loading of large files

The module is designed to handle datasets with millions of points while maintaining interactive performance through intelligent filtering and sampling.