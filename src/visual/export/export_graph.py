from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _weighted_mean(values: pd.Series, weights: pd.Series, eps: float = 1e-12) -> float:
    v = pd.to_numeric(values, errors='coerce').to_numpy(dtype=float, copy=False)
    w = pd.to_numeric(weights, errors='coerce').to_numpy(dtype=float, copy=False)
    m = np.isfinite(v) & np.isfinite(w) & (w >= 0)
    if not np.any(m):
        return float('nan')
    vv = v[m]
    ww = w[m]
    return float(np.sum(vv * ww) / (np.sum(ww) + eps))


def basin_stats_from_voxels(
    voxel_stats: pd.DataFrame,
    basins: pd.DataFrame,
    *,
    noise_basin_id: int = -1,
) -> pd.DataFrame:
    b = basins[['voxel_id', 'basin_id']].copy()
    b['basin_id'] = pd.to_numeric(b['basin_id'], errors='coerce')
    b = b.dropna(subset=['basin_id']).copy()
    b['basin_id'] = b['basin_id'].astype(int)
    b = b[b['basin_id'] != int(noise_basin_id)].copy()

    v = voxel_stats[['voxel_id', 'pi', 'holding_time', 'escape']].copy()
    v['pi'] = pd.to_numeric(v['pi'], errors='coerce').fillna(0.0)

    merged = b.merge(v, on='voxel_id', how='left')
    merged['pi'] = pd.to_numeric(merged['pi'], errors='coerce').fillna(0.0)

    rows = []
    for bid, g in merged.groupby('basin_id', sort=False):
        rows.append(
            {
                'basin_id': int(bid),
                'n_voxels': int(g['voxel_id'].nunique()),
                'mass': float(g['pi'].sum()),
                'mean_holding_time': _weighted_mean(g['holding_time'], g['pi']),
                'mean_escape': _weighted_mean(g['escape'], g['pi']),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(['mass', 'basin_id'], ascending=[False, True]).reset_index(drop=True)


def basin_flows_from_voxels_v02(
    transitions_topk: pd.DataFrame,
    voxel_stats: pd.DataFrame,
    basins: pd.DataFrame,
    *,
    noise_basin_id: int = -1,
    eps: float = 1e-12,
) -> pd.DataFrame:
    # flow_mass(from->to) = pi(from_voxel) * p(from_voxel->to_voxel)
    t = transitions_topk[['from_voxel_id', 'to_voxel_id', 'p']].copy()
    t['p'] = pd.to_numeric(t['p'], errors='coerce').fillna(0.0)

    pi = voxel_stats[['voxel_id', 'pi']].copy()
    pi['pi'] = pd.to_numeric(pi['pi'], errors='coerce').fillna(0.0)

    t = t.merge(pi, left_on='from_voxel_id', right_on='voxel_id', how='left').drop(columns=['voxel_id'])
    t['pi'] = pd.to_numeric(t['pi'], errors='coerce').fillna(0.0)
    t['flow_mass'] = t['pi'] * t['p']

    b = basins[['voxel_id', 'basin_id']].copy()
    b['basin_id'] = pd.to_numeric(b['basin_id'], errors='coerce')
    b = b.dropna(subset=['basin_id']).copy()
    b['basin_id'] = b['basin_id'].astype(int)

    b_from = b.rename(columns={'voxel_id': 'from_voxel_id', 'basin_id': 'from_basin_id'})
    b_to = b.rename(columns={'voxel_id': 'to_voxel_id', 'basin_id': 'to_basin_id'})

    out = t.merge(b_from, on='from_voxel_id', how='left').merge(b_to, on='to_voxel_id', how='left')
    out = out.dropna(subset=['from_basin_id', 'to_basin_id']).copy()
    out['from_basin_id'] = pd.to_numeric(out['from_basin_id'], errors='coerce').astype(int)
    out['to_basin_id'] = pd.to_numeric(out['to_basin_id'], errors='coerce').astype(int)

    out = out[(out['from_basin_id'] != int(noise_basin_id)) & (out['to_basin_id'] != int(noise_basin_id))].copy()

    flows = out.groupby(['from_basin_id', 'to_basin_id'], as_index=False)['flow_mass'].sum()
    if flows.empty:
        flows['row_prob'] = []
        return flows

    row_sum = flows.groupby('from_basin_id', as_index=False)['flow_mass'].sum().rename(columns={'flow_mass': '_row_sum'})
    flows = flows.merge(row_sum, on='from_basin_id', how='left')
    flows['row_prob'] = flows['flow_mass'] / (flows['_row_sum'].fillna(0.0) + eps)
    flows = flows.drop(columns=['_row_sum'])

    return flows.sort_values(['from_basin_id', 'to_basin_id']).reset_index(drop=True)


def basin_flows_compatible(
    basin_flows: pd.DataFrame | None,
    basins: pd.DataFrame,
    *,
    noise_basin_id: int = -1,
) -> bool:
    if basin_flows is None or basin_flows.empty:
        return False
    if not {'from_basin_id', 'to_basin_id', 'flow_mass'} <= set(basin_flows.columns):
        return False

    b = basins[['basin_id']].copy()
    b['basin_id'] = pd.to_numeric(b['basin_id'], errors='coerce')
    b = b.dropna().copy()
    bset = set(int(x) for x in b['basin_id'].astype(int).unique().tolist())
    bset.discard(int(noise_basin_id))
    if not bset:
        return False

    f = basin_flows[['from_basin_id', 'to_basin_id']].copy()
    f['from_basin_id'] = pd.to_numeric(f['from_basin_id'], errors='coerce')
    f['to_basin_id'] = pd.to_numeric(f['to_basin_id'], errors='coerce')
    f = f.dropna().copy()
    fids = set(int(x) for x in pd.concat([f['from_basin_id'], f['to_basin_id']]).astype(int).unique().tolist())
    fids.discard(int(noise_basin_id))

    return bool(fids) and fids.issubset(bset)


def export_graph(
    out_dir: str | Path,
    voxel_stats: pd.DataFrame,
    transitions_topk: pd.DataFrame,
    basins: pd.DataFrame,
    basin_flows: pd.DataFrame | None,
    cfg: Dict,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    vox_cols = ['voxel_id', 'center_x', 'center_y', 'center_z', 'pi', 'holding_time', 'escape', 'entropy', 'speed']
    voxels_nodes = voxel_stats[vox_cols].copy()
    if 'basin_id' in basins.columns:
        voxels_nodes = voxels_nodes.merge(basins[['voxel_id', 'basin_id']], on='voxel_id', how='left')
    voxels_nodes.to_parquet(out / str(cfg['exports'].get('voxels_nodes', 'voxels_nodes.parquet')), index=False)

    t = transitions_topk[['from_voxel_id', 'to_voxel_id', 'p', 'rank']].copy()
    t = t.merge(voxel_stats[['voxel_id', 'pi']], left_on='from_voxel_id', right_on='voxel_id', how='left').drop(columns=['voxel_id'])
    t['p'] = pd.to_numeric(t['p'], errors='coerce').fillna(0.0)
    t['pi'] = pd.to_numeric(t['pi'], errors='coerce').fillna(0.0)
    t['flow_mass'] = t['pi'] * t['p']
    t[['from_voxel_id', 'to_voxel_id', 'p', 'flow_mass', 'rank']].to_parquet(
        out / str(cfg['exports'].get('transitions_edges', 'transitions_edges.parquet')),
        index=False,
    )

    noise_id = int(cfg.get('figures', {}).get('basins_view', {}).get('noise_basin_id', -1))
    if not basin_flows_compatible(basin_flows, basins, noise_basin_id=noise_id):
        basin_flows = basin_flows_from_voxels_v02(
            transitions_topk=transitions_topk,
            voxel_stats=voxel_stats,
            basins=basins,
            noise_basin_id=noise_id,
        )
    else:
        if 'row_prob' not in basin_flows.columns:
            tmp = basin_flows.copy()
            row_sum = tmp.groupby('from_basin_id', as_index=False)['flow_mass'].sum().rename(columns={'flow_mass': '_row_sum'})
            tmp = tmp.merge(row_sum, on='from_basin_id', how='left')
            tmp['row_prob'] = tmp['flow_mass'] / (tmp['_row_sum'].fillna(0.0) + 1e-12)
            basin_flows = tmp.drop(columns=['_row_sum'])

    basin_nodes = basin_stats_from_voxels(voxel_stats, basins, noise_basin_id=noise_id)
    basin_nodes.to_parquet(out / str(cfg['exports'].get('basin_nodes', 'basin_nodes.parquet')), index=False)

    basin_flows[['from_basin_id', 'to_basin_id', 'flow_mass']].to_parquet(
        out / str(cfg['exports'].get('basin_edges', 'basin_edges.parquet')),
        index=False,
    )
