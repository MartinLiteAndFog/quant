from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.visual.common import base_parser, load_cfg
from visual.figures.basins_view import (
    build_basins_scene,
    build_basin_flow_matrices,
    build_basins_stats_table,
    compute_basin_flows_v02,
)
from visual.io import load_contracts
from visual.render.mpl import render_matplotlib
from visual.render.plotly import render_plotly


def _write_basin_flows_v02(out_dir: Path, cfg: dict, flows: pd.DataFrame) -> Path:
    fcfg = cfg.get('figures', {}).get('basins_view', {})
    fname = str(fcfg.get('basin_flows_out_parquet', 'basin_flows_v02.parquet'))
    out_path = out_dir / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flows.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    ap = base_parser()
    ap.add_argument('--basin-id', type=int, default=None)
    ap.add_argument('--basins-file', default='basins_v02_components.parquet')
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    cfg.setdefault('paths', {})['basins'] = str(args.basins_file)

    contracts = load_contracts(args.in_dir, cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Basins plot (decluttered Top-N + NOW)
    cur_pos = None
    try:
        vm = contracts.get('voxel_map')
        vs = contracts.get('voxel_stats')
        if vm is not None and vs is not None and len(vm):
            vid = int(vm.iloc[-1]['voxel_id'])
            row = vs[vs['voxel_id'] == vid]
            if len(row):
                r = row.iloc[0]
                cur_pos = {
                    'center_x': float(r['center_x']),
                    'center_y': float(r['center_y']),
                    'center_z': float(r['center_z']),
                    'voxel_id': int(vid),
                }
    except Exception:
        cur_pos = None

    scene = build_basins_scene(
        contracts['voxel_stats'],
        contracts['basins'],
        cfg,
        basin_id=args.basin_id,
        current_position=cur_pos,
    )

    fcfg = cfg.get('figures', {}).get('basins_view', {})
    render_plotly(scene, out_dir / str(fcfg.get('out_html', 'basins.html')))
    render_matplotlib(scene, out_dir / str(fcfg.get('out_png', 'basins.png')))

    # Basin flows v0.2 (recomputed reliably) + write parquet
    flows = compute_basin_flows_v02(contracts, cfg)
    flows_path = _write_basin_flows_v02(out_dir, cfg, flows)

    # Matrices (Top-N only)
    mats = build_basin_flow_matrices(flows, contracts['voxel_stats'], contracts['basins'], cfg)

    if 'row_prob' in mats:
        render_matplotlib(mats['row_prob'], out_dir / str(fcfg.get('basin_flow_matrix_rowprob_png', 'basin_flow_matrix_rowprob.png')))

    if 'log_flow' in mats:
        render_matplotlib(mats['log_flow'], out_dir / str(fcfg.get('basin_flow_matrix_logflow_png', 'basin_flow_matrix_logflow.png')))

    stats = build_basins_stats_table(contracts['voxel_stats'], contracts['basins'], cfg)
    if not stats.empty:
        print('
Top basins (by mass):')
        print(stats.to_string(index=False))
        print(f'
Wrote basin flows: {flows_path}')


if __name__ == '__main__':
    main()
