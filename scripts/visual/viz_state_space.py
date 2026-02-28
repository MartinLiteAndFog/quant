from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.visual.common import base_parser, load_cfg
from visual.figures.state_space_3d import build_occupancy_scene, build_persistence_scene
from visual.io import load_contracts
from visual.render.mpl import render_matplotlib
from visual.render.plotly import render_plotly


def _get_current_position(contracts: dict) -> dict | None:
    vm = contracts.get('voxel_map')
    vs = contracts.get('voxel_stats')
    if vm is None or vs is None or len(vm) == 0:
        return None
    try:
        vid = int(vm.iloc[-1]['voxel_id'])
        row = vs[vs['voxel_id'] == vid]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            'center_x': float(r['center_x']),
            'center_y': float(r['center_y']),
            'center_z': float(r['center_z']),
            'voxel_id': vid,
        }
    except Exception:
        return None


def main() -> None:
    ap = base_parser()
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    contracts = load_contracts(args.in_dir, cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cur_pos = _get_current_position(contracts)

    occ_scene = build_occupancy_scene(contracts['voxel_stats'], cfg, current_position=cur_pos)
    per_scene = build_persistence_scene(contracts['voxel_stats'], cfg, current_position=cur_pos)

    occ_cfg = cfg.get('figures', {}).get('state_space_occupancy', {})
    per_cfg = cfg.get('figures', {}).get('state_space_persistence', {})

    render_plotly(occ_scene, out_dir / str(occ_cfg.get('out_html', 'state_space_occupancy.html')))
    render_matplotlib(occ_scene, out_dir / str(occ_cfg.get('out_png', 'state_space_occupancy.png')))

    render_plotly(per_scene, out_dir / str(per_cfg.get('out_html', 'state_space_persistence.html')))
    render_matplotlib(per_scene, out_dir / str(per_cfg.get('out_png', 'state_space_persistence.png')))


if __name__ == '__main__':
    main()
