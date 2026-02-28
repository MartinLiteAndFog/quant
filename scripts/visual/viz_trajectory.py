from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.visual.common import base_parser, load_cfg
from visual.figures.trajectory_3d import build_trajectory_scene
from visual.io import load_contracts
from visual.render.mpl import render_matplotlib
from visual.render.plotly import render_plotly


def main() -> None:
    ap = base_parser()
    ap.add_argument('--from', dest='ts_from', default=None)
    ap.add_argument('--to', dest='ts_to', default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    contracts = load_contracts(args.in_dir, cfg)

    ts_from = pd.to_datetime(args.ts_from, utc=True, errors='coerce') if args.ts_from else None
    ts_to = pd.to_datetime(args.ts_to, utc=True, errors='coerce') if args.ts_to else None

    if ts_from is None and ts_to is None:
        last_n = int(cfg.get('figures', {}).get('trajectory_3d', {}).get('default_last_n_rows', 200000))
        state = contracts['state_space'].iloc[-last_n:].copy()
        vox = contracts['voxel_map'].iloc[-last_n:].copy()
    else:
        state = contracts['state_space']
        vox = contracts['voxel_map']

    scene = build_trajectory_scene(state, vox, cfg, ts_from=ts_from, ts_to=ts_to)
    fcfg = cfg.get('figures', {}).get('trajectory_3d', {})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    render_plotly(scene, out_dir / str(fcfg.get('out_html', 'trajectory.html')))
    render_matplotlib(scene, out_dir / str(fcfg.get('out_png', 'trajectory.png')))


if __name__ == '__main__':
    main()
