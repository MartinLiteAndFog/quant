from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.visual.common import base_parser, load_cfg
from visual.export.export_assets import ensure_output_dirs
from visual.export.export_graph import export_graph
from visual.figures.basins_view import (
    build_basins_scene,
    build_basin_flow_matrices,
    compute_basin_flows_v02,
)
from visual.figures.slices_2d import build_slice_scenes
from visual.figures.state_space_3d import build_occupancy_scene, build_persistence_scene
from visual.figures.trajectory_3d import build_trajectory_scene
from visual.figures.transitions_3d import build_transitions_scene
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
    ap.add_argument(
        '--build-gate-confidence', action='store_true', default=False,
        help='Also run gate confidence projection (requires gate section in config)',
    )
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    contracts = load_contracts(args.in_dir, cfg)

    paths = ensure_output_dirs(args.out_dir, cfg)
    fig_dir = paths['figures']
    exp_dir = paths['exports']

    cur_pos = _get_current_position(contracts)
    if cur_pos is not None:
        print(
            f"Current position: voxel={cur_pos['voxel_id']}, "
            f"X={cur_pos['center_x']:.3f}, Y={cur_pos['center_y']:.3f}, Z={cur_pos['center_z']:.3f}"
        )

    # occupancy + persistence
    occ_scene = build_occupancy_scene(contracts['voxel_stats'], cfg, current_position=cur_pos)
    per_scene = build_persistence_scene(contracts['voxel_stats'], cfg, current_position=cur_pos)
    occ_cfg = cfg.get('figures', {}).get('state_space_occupancy', {})
    per_cfg = cfg.get('figures', {}).get('state_space_persistence', {})
    render_plotly(occ_scene, fig_dir / str(occ_cfg.get('out_html', 'state_space_occupancy.html')))
    render_matplotlib(occ_scene, fig_dir / str(occ_cfg.get('out_png', 'state_space_occupancy.png')))
    render_plotly(per_scene, fig_dir / str(per_cfg.get('out_html', 'state_space_persistence.html')))
    render_matplotlib(per_scene, fig_dir / str(per_cfg.get('out_png', 'state_space_persistence.png')))

    # slices
    s_xy, s_xz, s_yz = build_slice_scenes(contracts['voxel_map'], contracts['voxel_stats'], cfg)
    slices_cfg = cfg.get('slices', {}).get('output', {})
    render_matplotlib(s_xy, fig_dir / str(slices_cfg.get('xy', 'slices_xy.png')))
    render_matplotlib(s_xz, fig_dir / str(slices_cfg.get('xz', 'slices_xz.png')))
    render_matplotlib(s_yz, fig_dir / str(slices_cfg.get('yz', 'slices_yz.png')))

    # transitions + NOW
    tr_scene = build_transitions_scene(contracts['voxel_stats'], contracts['transitions_topk'], cfg)
    if cur_pos is not None:
        from visual.scene import SceneLayer
        tr_scene.add_layer(SceneLayer(
            name='current_position', kind='current_position',
            data=pd.DataFrame([cur_pos]), params={},
        ))
    tr_cfg = cfg.get('figures', {}).get('transitions_3d', {})
    render_plotly(tr_scene, fig_dir / str(tr_cfg.get('out_html', 'transitions.html')))
    render_matplotlib(tr_scene, fig_dir / str(tr_cfg.get('out_png', 'transitions.png')))

    # trajectory
    tcfg = cfg.get('figures', {}).get('trajectory_3d', {})
    last_n = int(tcfg.get('default_last_n_rows', 200000))
    traj_scene = build_trajectory_scene(
        contracts['state_space'].iloc[-last_n:].copy(),
        contracts['voxel_map'].iloc[-last_n:].copy(),
        cfg,
    )
    render_plotly(traj_scene, fig_dir / str(tcfg.get('out_html', 'trajectory.html')))
    render_matplotlib(traj_scene, fig_dir / str(tcfg.get('out_png', 'trajectory.png')))

    # basin flows v02 (compute BEFORE building basins scene)
    flows = compute_basin_flows_v02(contracts, cfg)
    flows_path = fig_dir / str(cfg.get('figures', {}).get('basins_view', {}).get('basin_flows_out_parquet', 'basin_flows_v02.parquet'))
    flows.to_parquet(flows_path, index=False)

    # basins (with dynamic flow arrows)
    bcfg = cfg.get('figures', {}).get('basins_view', {})
    basin_scene = build_basins_scene(
        contracts['voxel_stats'],
        contracts['basins'],
        cfg,
        basin_id=None,
        current_position=cur_pos,
        basin_flows=flows,
    )
    render_plotly(basin_scene, fig_dir / str(bcfg.get('out_html', 'basins.html')))
    render_matplotlib(basin_scene, fig_dir / str(bcfg.get('out_png', 'basins.png')))

    mats = build_basin_flow_matrices(flows, contracts['voxel_stats'], contracts['basins'], cfg)
    if 'row_prob' in mats:
        render_matplotlib(mats['row_prob'], fig_dir / str(bcfg.get('basin_flow_matrix_rowprob_png', 'basin_flow_matrix_rowprob.png')))
    if 'log_flow' in mats:
        render_matplotlib(mats['log_flow'], fig_dir / str(bcfg.get('basin_flow_matrix_logflow_png', 'basin_flow_matrix_logflow.png')))

    # graph exports
    export_graph(
        out_dir=exp_dir,
        voxel_stats=contracts['voxel_stats'],
        transitions_topk=contracts['transitions_topk'],
        basins=contracts['basins'],
        basin_flows=flows,
        cfg=cfg,
    )

    # gate confidence (optional)
    if args.build_gate_confidence:
        gate_cfg = cfg.get('gate', {})
        if not gate_cfg.get('path'):
            print('WARNING: --build-gate-confidence requested but gate.path not set in config; skipping')
        else:
            from scripts.visual.build_gate_confidence import (
                load_gate,
                align_gate_to_voxels,
                build_voxel_gate_stats,
                build_basin_gate_stats,
                build_confidence_curve,
            )

            noise_id = int(cfg.get('figures', {}).get('basins_view', {}).get('noise_basin_id', -1))

            print('\n--- Gate Confidence ---')
            gate = load_gate(cfg)
            print(f"  gate rows: {len(gate)}, range: {gate['ts'].min()} -> {gate['ts'].max()}")

            direction = str(gate_cfg.get('asof_direction', 'backward'))
            aligned = align_gate_to_voxels(contracts['voxel_map'], gate, direction=direction)

            voxel_gate = build_voxel_gate_stats(aligned)
            voxel_gate.to_parquet(fig_dir / 'voxel_gate_stats.parquet', index=False)

            basin_gate = build_basin_gate_stats(
                voxel_gate, contracts['voxel_stats'], contracts['basins'],
                noise_basin_id=noise_id,
            )
            basin_gate.to_parquet(fig_dir / 'basin_gate_stats.parquet', index=False)

            curve = build_confidence_curve(contracts, voxel_gate, basin_gate, cfg)
            curve.to_parquet(fig_dir / 'confidence_curve.parquet', index=False)

            print(f"  gate confidence artifacts written to {fig_dir}")
            for _, row in curve.iterrows():
                h = int(row['horizon_minutes'])
                cv = row['conf_voxel']
                cb = row.get('conf_basin')
                cb_s = f"{cb:.4f}" if cb is not None and not (isinstance(cb, float) and cb != cb) else 'N/A'
                label = f"{h}m" if h < 60 else f"{h // 60}h"
                print(f"    conf_{label:>4s}  voxel={cv:.4f}  basin={cb_s}")

    print('Build complete')
    print('Figures:', fig_dir)
    print('Graph exports:', exp_dir)


if __name__ == '__main__':
    main()
