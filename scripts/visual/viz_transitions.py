from __future__ import annotations

from pathlib import Path

from scripts.visual.common import base_parser, load_cfg
from visual.figures.transitions_3d import build_transitions_scene
from visual.io import load_contracts
from visual.render.mpl import render_matplotlib
from visual.render.plotly import render_plotly


def main() -> None:
    ap = base_parser()
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    contracts = load_contracts(args.in_dir, cfg)

    scene = build_transitions_scene(contracts['voxel_stats'], contracts['transitions_topk'], cfg)
    fcfg = cfg.get('figures', {}).get('transitions_3d', {})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    render_plotly(scene, out_dir / str(fcfg.get('out_html', 'transitions.html')))
    render_matplotlib(scene, out_dir / str(fcfg.get('out_png', 'transitions.png')))


if __name__ == '__main__':
    main()
