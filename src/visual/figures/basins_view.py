from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from visual.export.export_graph import basin_flows_compatible, basin_flows_from_voxels_v02, basin_stats_from_voxels
from visual.scene import Scene, SceneLayer


@dataclass(frozen=True)
class BasinViewSelection:
    top_basin_ids: List[int]
    noise_basin_id: int


def _basin_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get('figures', {}).get('basins_view', {})


def _select_top_basins(
    voxel_stats: pd.DataFrame,
    basins: pd.DataFrame,
    cfg: Dict[str, Any],
) -> BasinViewSelection:
    fcfg = _basin_cfg(cfg)
    top_n = int(fcfg.get('top_n_basins', 6))
    noise_id = int(fcfg.get('noise_basin_id', -1))

    df = basins[['voxel_id', 'basin_id']].merge(voxel_stats[['voxel_id', 'pi']], on='voxel_id', how='left')
    df['basin_id'] = pd.to_numeric(df['basin_id'], errors='coerce').fillna(noise_id).astype(int)
    df['pi'] = pd.to_numeric(df['pi'], errors='coerce').fillna(0.0)

    mass = df[df['basin_id'] != noise_id].groupby('basin_id', as_index=False)['pi'].sum()
    mass = mass.sort_values('pi', ascending=False)
    top_ids = [int(x) for x in mass['basin_id'].head(max(0, top_n)).tolist()]
    return BasinViewSelection(top_basin_ids=top_ids, noise_basin_id=noise_id)


def _basin_core_center(df: pd.DataFrame, basin_id: int) -> Optional[Dict[str, float]]:
    sub = df[df['basin_id'] == int(basin_id)]
    if sub.empty:
        return None
    core = sub.loc[sub['pi'].idxmax()]
    return {
        'center_x': float(core['center_x']),
        'center_y': float(core['center_y']),
        'center_z': float(core['center_z']),
    }


PALETTE = [
    '#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#a65628',
    '#f781bf', '#66c2a5', '#999999', '#ffd92f',
]


def build_basins_scene(
    voxel_stats: pd.DataFrame,
    basins: pd.DataFrame,
    cfg: Dict[str, Any],
    basin_id: Optional[int] = None,
    current_position: Optional[Dict[str, float]] = None,
    basin_flows: Optional[pd.DataFrame] = None,
) -> Scene:
    fcfg = _basin_cfg(cfg)
    noise_id = int(fcfg.get('noise_basin_id', -1))
    noise_color = str(fcfg.get('noise_color', '#dddddd'))
    label_cores = bool(fcfg.get('label_cores', True))

    scene = Scene(title=str(fcfg.get('title', 'Basins v0.2')), settings={'kind': '3d'})

    df = voxel_stats.merge(basins[['voxel_id', 'basin_id']], on='voxel_id', how='left')
    df['basin_id'] = pd.to_numeric(df['basin_id'], errors='coerce').fillna(noise_id).astype(int)
    df['pi'] = pd.to_numeric(df.get('pi', 0.0), errors='coerce').fillna(0.0)

    for c in ['center_x', 'center_y', 'center_z']:
        if c not in df.columns:
            raise ValueError(f'missing voxel center column in voxel_stats: {c}')

    # Determine NOW basin
    now_basin_id = None
    if current_position is not None and 'voxel_id' in current_position:
        vid = int(current_position['voxel_id'])
        row = df[df['voxel_id'] == vid]
        if len(row):
            now_basin_id = int(row.iloc[0]['basin_id'])
            if now_basin_id == noise_id:
                now_basin_id = None

    # Determine top-3 next basins from flows
    top3_next: List[Tuple[int, float]] = []
    if now_basin_id is not None and basin_flows is not None and len(basin_flows):
        outgoing = basin_flows[basin_flows['from_basin_id'] == now_basin_id].copy()
        if 'row_prob' in outgoing.columns:
            outgoing = outgoing[outgoing['to_basin_id'] != now_basin_id]
            outgoing = outgoing.sort_values('row_prob', ascending=False).head(3)
            top3_next = [(int(r['to_basin_id']), float(r['row_prob'])) for _, r in outgoing.iterrows()]

    top3_ids = set(tid for tid, _ in top3_next)

    if basin_id is not None:
        show = df[df['basin_id'] == int(basin_id)].copy()
        show['_group'] = f'B{int(basin_id)}'
        show['_color'] = '#377eb8'
        show['_size'] = 80.0
        scene.add_layer(SceneLayer(name=f'basin_{basin_id}', kind='basins_v02_points', data=show, params={}))
    else:
        sel = _select_top_basins(voxel_stats, basins, cfg)

        # Ensure NOW basin and top3 targets are in the visible set
        visible_ids = set(sel.top_basin_ids)
        if now_basin_id is not None:
            visible_ids.add(now_basin_id)
        visible_ids |= top3_ids

        # Build color map: NOW basin gets index 0 (bold blue), top3 get next slots
        color_order = []
        if now_basin_id is not None:
            color_order.append(now_basin_id)
        for tid, _ in top3_next:
            if tid not in color_order:
                color_order.append(tid)
        for bid in sel.top_basin_ids:
            if bid not in color_order:
                color_order.append(bid)

        bid_to_color = {}
        for k, bid in enumerate(color_order):
            bid_to_color[bid] = PALETTE[k % len(PALETTE)]

        # noise
        noise = df[df['basin_id'] == noise_id].copy()
        if len(noise):
            noise['_group'] = 'noise'
            noise['_color'] = noise_color
            noise['_size'] = 8.0
            noise['_legend'] = False
            scene.add_layer(SceneLayer(name='noise', kind='basins_v02_points', data=noise, params={}))

        # non-visible basins -> "others" (very transparent gray)
        others = df[(df['basin_id'] != noise_id) & (~df['basin_id'].isin(list(visible_ids)))].copy()
        if len(others):
            others['_group'] = 'others'
            others['_color'] = '#cccccc'
            others['_size'] = 14.0
            others['_alpha'] = 0.18
            others['_legend'] = True
            scene.add_layer(SceneLayer(name='others', kind='basins_v02_points', data=others, params={}))

        # visible basins that are NOT now_basin and NOT top3 targets -> muted
        muted_ids = visible_ids - {now_basin_id} - top3_ids if now_basin_id else visible_ids - top3_ids
        for bid in sorted(muted_ids):
            if bid is None:
                continue
            sub = df[df['basin_id'] == int(bid)].copy()
            if sub.empty:
                continue
            sub['_group'] = f'B{int(bid)}'
            sub['_color'] = bid_to_color.get(bid, '#aaaaaa')
            sub['_size'] = 28.0
            sub['_alpha'] = 0.35
            sub['_legend'] = True
            scene.add_layer(SceneLayer(name=f'basin_B{bid}', kind='basins_v02_points', data=sub, params={'basin_id': int(bid)}))
            if label_cores:
                core = sub.loc[[sub['pi'].idxmax()]].copy()
                core['_group'] = f'B{int(bid)}_core'
                core['_color'] = bid_to_color.get(bid, '#aaaaaa')
                core['_size'] = 80.0
                core['_label'] = f'B{int(bid)}'
                core['_legend'] = False
                scene.add_layer(SceneLayer(name=f'core_B{bid}', kind='basins_v02_core', data=core, params={'basin_id': int(bid)}))

        # top-3 target basins (medium bold)
        for tid, prob in top3_next:
            sub = df[df['basin_id'] == int(tid)].copy()
            if sub.empty:
                continue
            lbl = f'B{int(tid)}  P={prob:.3f}'
            sub['_group'] = lbl
            sub['_color'] = bid_to_color.get(tid, '#888888')
            sub['_size'] = 50.0
            sub['_alpha'] = 0.85
            sub['_legend'] = True
            scene.add_layer(SceneLayer(name=f'basin_B{tid}', kind='basins_v02_points', data=sub, params={'basin_id': int(tid)}))
            if label_cores:
                core = sub.loc[[sub['pi'].idxmax()]].copy()
                core['_group'] = f'B{int(tid)}_core'
                core['_color'] = bid_to_color.get(tid, '#888888')
                core['_size'] = 120.0
                core['_label'] = f'B{int(tid)}'
                core['_legend'] = False
                scene.add_layer(SceneLayer(name=f'core_B{tid}', kind='basins_v02_core', data=core, params={'basin_id': int(tid)}))

        # NOW basin (bold, large, fully opaque)
        if now_basin_id is not None:
            sub = df[df['basin_id'] == int(now_basin_id)].copy()
            if not sub.empty:
                lbl = f'B{int(now_basin_id)} (NOW)'
                sub['_group'] = lbl
                sub['_color'] = bid_to_color.get(now_basin_id, '#377eb8')
                sub['_size'] = 80.0
                sub['_alpha'] = 1.0
                sub['_legend'] = True
                scene.add_layer(SceneLayer(name=f'basin_now_B{now_basin_id}', kind='basins_v02_points', data=sub, params={'basin_id': int(now_basin_id)}))
                if label_cores:
                    core = sub.loc[[sub['pi'].idxmax()]].copy()
                    core['_group'] = f'B{int(now_basin_id)}_core'
                    core['_color'] = bid_to_color.get(now_basin_id, '#377eb8')
                    core['_size'] = 200.0
                    core['_label'] = f'B{int(now_basin_id)}'
                    core['_legend'] = False
                    scene.add_layer(SceneLayer(name=f'core_now_B{now_basin_id}', kind='basins_v02_core', data=core, params={'basin_id': int(now_basin_id)}))

        # Flow arrows: NOW basin core -> top-3 target basin cores
        if now_basin_id is not None and top3_next:
            now_center = _basin_core_center(df, now_basin_id)
            if now_center is not None:
                arrow_rows = []
                for tid, prob in top3_next:
                    tgt = _basin_core_center(df, tid)
                    if tgt is None:
                        continue
                    arrow_rows.append({
                        'x0': now_center['center_x'], 'y0': now_center['center_y'], 'z0': now_center['center_z'],
                        'x1': tgt['center_x'], 'y1': tgt['center_y'], 'z1': tgt['center_z'],
                        'prob': prob,
                        'from_basin': now_basin_id,
                        'to_basin': tid,
                        'color': bid_to_color.get(tid, '#888888'),
                        'label': f'P({now_basin_id}→{tid})={prob:.3f}',
                    })
                if arrow_rows:
                    arrows_df = pd.DataFrame(arrow_rows)
                    scene.add_layer(SceneLayer(
                        name='basin_flow_arrows',
                        kind='basin_flow_arrows',
                        data=arrows_df,
                        params={'now_basin_id': now_basin_id},
                    ))

    # current position marker
    if current_position is not None:
        cp = pd.DataFrame([current_position])
        scene.add_layer(SceneLayer(name='current_position', kind='current_position', data=cp, params={}))

    return scene


def compute_basin_flows_v02(
    contracts: Dict[str, pd.DataFrame],
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    fcfg = _basin_cfg(cfg)
    noise_id = int(fcfg.get('noise_basin_id', -1))

    basins = contracts['basins']
    if basin_flows_compatible(contracts.get('basin_flows'), basins, noise_basin_id=noise_id):
        flows = contracts['basin_flows'].copy()
        if 'row_prob' not in flows.columns and {'from_basin_id', 'flow_mass'} <= set(flows.columns):
            row = flows.groupby('from_basin_id', as_index=False)['flow_mass'].sum().rename(columns={'flow_mass': '_row_sum'})
            flows = flows.merge(row, on='from_basin_id', how='left')
            flows['row_prob'] = flows['flow_mass'] / (flows['_row_sum'].fillna(0.0) + 1e-12)
            flows = flows.drop(columns=['_row_sum'])
        return flows

    return basin_flows_from_voxels_v02(
        transitions_topk=contracts['transitions_topk'],
        voxel_stats=contracts['voxel_stats'],
        basins=basins,
        noise_basin_id=noise_id,
    )


def _pivot_topn_matrix(
    flows: pd.DataFrame,
    basin_order: List[int],
    value_col: str,
) -> pd.DataFrame:
    mat = flows.pivot_table(
        index='from_basin_id',
        columns='to_basin_id',
        values=value_col,
        aggfunc='sum',
        fill_value=0.0,
    )
    mat = mat.reindex(index=basin_order, columns=basin_order, fill_value=0.0)
    mat.index = [f'B{int(x)}' for x in mat.index]
    mat.columns = [f'B{int(x)}' for x in mat.columns]
    return mat


def build_basin_flow_matrices(
    basin_flows: pd.DataFrame,
    voxel_stats: pd.DataFrame,
    basins: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Dict[str, Scene]:
    fcfg = _basin_cfg(cfg)
    sel = _select_top_basins(voxel_stats, basins, cfg)

    if not sel.top_basin_ids:
        raise ValueError('no basins to build flow matrices (top_n_basins resolved to empty)')

    include_diag = bool(fcfg.get('matrix_include_diag', True))
    clip_q = float(fcfg.get('matrix_clip_quantile', 0.99))
    scenes: Dict[str, Scene] = {}

    if 'row_prob' not in basin_flows.columns:
        raise ValueError('basin_flows missing row_prob (expected v0.2 computed flows)')

    mat_prob = _pivot_topn_matrix(basin_flows, sel.top_basin_ids, 'row_prob')
    if not include_diag:
        np.fill_diagonal(mat_prob.values, 0.0)

    scenes['row_prob'] = Scene(title='Basin Flow Matrix (P(to | from))', settings={'kind': 'heatmap2d'})
    scenes['row_prob'].add_layer(SceneLayer(
        name='basin_flow_rowprob', kind='heatmap2d', data=mat_prob,
        params={'xlabel': 'to_basin', 'ylabel': 'from_basin', 'cbar_label': 'P(to | from)', 'clip_quantile': clip_q},
    ))

    mat_flow = _pivot_topn_matrix(basin_flows, sel.top_basin_ids, 'flow_mass')
    if not include_diag:
        np.fill_diagonal(mat_flow.values, 0.0)
    log_flow = np.log1p(mat_flow.values)
    mat_log = pd.DataFrame(log_flow, index=mat_flow.index, columns=mat_flow.columns)

    scenes['log_flow'] = Scene(title='Basin Flow Matrix (log1p(flow_mass))', settings={'kind': 'heatmap2d'})
    scenes['log_flow'].add_layer(SceneLayer(
        name='basin_flow_logflow', kind='heatmap2d', data=mat_log,
        params={'xlabel': 'to_basin', 'ylabel': 'from_basin', 'cbar_label': 'log1p(flow_mass)', 'clip_quantile': clip_q},
    ))

    return scenes


def build_basins_stats_table(
    voxel_stats: pd.DataFrame,
    basins: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    fcfg = _basin_cfg(cfg)
    noise_id = int(fcfg.get('noise_basin_id', -1))
    top_n = int(fcfg.get('top_n_basins', 6))

    stats = basin_stats_from_voxels(voxel_stats, basins, noise_basin_id=noise_id)
    if stats.empty:
        return stats

    return stats.head(max(1, top_n)).copy()
