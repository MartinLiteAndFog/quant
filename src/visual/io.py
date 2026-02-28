from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _strip_comment_preserving_quotes(raw: str) -> str:
    """Strip YAML comments (# ...) but keep # inside quoted strings."""
    out = []
    in_single = False
    in_double = False
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == "'" and not in_double:
            in_single = not in_single
            out.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            out.append(ch)
            i += 1
            continue
        if ch == '#' and not in_single and not in_double:
            break
        out.append(ch)
        i += 1
    return ''.join(out)


def _parse_scalar(value: str) -> Any:
    v = value.strip()
    if v == '':
        return ''
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    if v.lower() in {'true', 'false'}:
        return v.lower() == 'true'
    if v.lower() in {'null', 'none'}:
        return None
    if v.startswith('[') and v.endswith(']'):
        inner = v[1:-1].strip()
        if not inner:
            return []
        parts = [p.strip() for p in inner.split(',')]
        return [_parse_scalar(p) for p in parts]
    try:
        if any(ch in v for ch in ['.', 'e', 'E']):
            return float(v)
        return int(v)
    except ValueError:
        return v


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Minimal YAML parser for nested key/value maps + inline lists.

    Note: Supports quoted strings containing '#', e.g. "#dddddd".
    """
    p = Path(path)
    text = p.read_text(encoding='utf-8')
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, root)]

    for raw in text.splitlines():
        line = _strip_comment_preserving_quotes(raw)
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(' '))
        content = line.strip()
        if ':' not in content:
            raise ValueError(f'invalid yaml line (expected key: value): {raw}')

        key, val = content.split(':', 1)
        key = key.strip()
        val = val.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f'invalid indentation near line: {raw}')
        parent = stack[-1][1]

        if val == '':
            node: Dict[str, Any] = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = _parse_scalar(val)

    return root


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{name} missing required columns: {sorted(missing)}')


def _parse_ts(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors='coerce')
    if ts.isna().all():
        raise ValueError('ts parsing failed: all values became NaT')
    return ts


def _validate_ts(df: pd.DataFrame, ts_col: str, name: str) -> pd.DataFrame:
    out = df.copy()
    out[ts_col] = _parse_ts(out[ts_col])
    if out[ts_col].isna().any():
        n = int(out[ts_col].isna().sum())
        raise ValueError(f'{name} contains {n} unparseable ts values')
    if not out[ts_col].is_monotonic_increasing:
        raise ValueError(f'{name} ts column is not sorted ascending')
    if out[ts_col].duplicated().any():
        n = int(out[ts_col].duplicated().sum())
        raise ValueError(f'{name} ts column contains duplicate timestamps: {n}')
    return out


def _resolve_path(in_dir: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else in_dir / p


def load_state_space(path: Path, cols: Dict[str, str]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    req = {cols['ts'], cols['x'], cols['y'], cols['z']}
    _require_columns(df, req, 'state_space.parquet')
    df = df.rename(
        columns={
            cols['ts']: 'ts',
            cols['x']: 'X_raw',
            cols['y']: 'Y_res',
            cols['z']: 'Z_res',
        }
    )
    return _validate_ts(df[['ts', 'X_raw', 'Y_res', 'Z_res']], 'ts', 'state_space.parquet')


def load_voxel_map(path: Path, cols: Dict[str, str]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    req = {cols['ts'], cols['voxel_id']}
    _require_columns(df, req, 'voxel_map.parquet')

    rename = {cols['ts']: 'ts', cols['voxel_id']: 'voxel_id'}
    for c in ['bin_x', 'bin_y', 'bin_z']:
        if c in df.columns:
            rename[c] = c

    out = df.rename(columns=rename)
    keep = ['ts', 'voxel_id'] + [c for c in ['bin_x', 'bin_y', 'bin_z'] if c in out.columns]
    out = _validate_ts(out[keep], 'ts', 'voxel_map.parquet')
    out['voxel_id'] = pd.to_numeric(out['voxel_id'], errors='coerce').astype('Int64')
    return out


def load_voxel_stats(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    req = {'voxel_id', 'occ_eff', 'pi'}
    _require_columns(df, req, 'voxel_stats.parquet')

    out = df.copy()

    rename: Dict[str, str] = {}
    if 'drift_x' in out.columns and 'drift_dx' not in out.columns:
        rename['drift_x'] = 'drift_dx'
    if 'drift_y' in out.columns and 'drift_dy' not in out.columns:
        rename['drift_y'] = 'drift_dy'
    if 'drift_z' in out.columns and 'drift_dz' not in out.columns:
        rename['drift_z'] = 'drift_dz'
    if 'from_id' in out.columns:
        rename['from_id'] = 'voxel_id'

    out = out.rename(columns=rename)

    for c in ['p_self', 'escape', 'holding_time', 'entropy', 'speed', 'drift_dx', 'drift_dy', 'drift_dz']:
        if c not in out.columns:
            out[c] = np.nan

    if {'center_x', 'center_y', 'center_z'} - set(out.columns):
        for c in ['center_x', 'center_y', 'center_z']:
            if c not in out.columns:
                out[c] = np.nan

    return out


def load_transitions_topk(path: Path, cols: Dict[str, str]) -> pd.DataFrame:
    df = pd.read_parquet(path)

    rename: Dict[str, str] = {}
    if cols['from_voxel_id'] in df.columns:
        rename[cols['from_voxel_id']] = 'from_voxel_id'
    elif 'from_id' in df.columns:
        rename['from_id'] = 'from_voxel_id'

    if cols['to_voxel_id'] in df.columns:
        rename[cols['to_voxel_id']] = 'to_voxel_id'
    elif 'to_id' in df.columns:
        rename['to_id'] = 'to_voxel_id'

    out = df.rename(columns=rename)

    required = {'from_voxel_id', 'to_voxel_id'}
    _require_columns(out, required, 'transitions_topk.parquet')

    if 'p' not in out.columns:
        if 'prob' in out.columns:
            out = out.rename(columns={'prob': 'p'})
        else:
            raise ValueError("transitions_topk.parquet missing probability column 'p' or 'prob'")

    if 'rank' not in out.columns:
        out = out.sort_values(['from_voxel_id', 'p'], ascending=[True, False]).copy()
        out['rank'] = out.groupby('from_voxel_id').cumcount() + 1

    return out[['from_voxel_id', 'to_voxel_id', 'p', 'rank']].copy()


def load_basins(path: Path, cols: Dict[str, str]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    req = {cols['voxel_id'], cols['basin_id']}
    _require_columns(df, req, 'basins.parquet')
    return df.rename(columns={cols['voxel_id']: 'voxel_id', cols['basin_id']: 'basin_id'})


def load_basin_flows(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None

    df = pd.read_parquet(path)
    req = {'from_basin_id', 'to_basin_id', 'flow_mass'}
    _require_columns(df, req, 'basin_flows.parquet')
    return df[['from_basin_id', 'to_basin_id', 'flow_mass']].copy()


def load_contracts(in_dir: str | Path, cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    base = Path(in_dir)
    paths = cfg.get('paths', {})
    cols = cfg.get('columns', {})

    required_cols = {'ts', 'x', 'y', 'z', 'voxel_id', 'from_voxel_id', 'to_voxel_id', 'basin_id'}
    missing = required_cols - set(cols.keys())
    if missing:
        raise ValueError(f'config.columns missing keys: {sorted(missing)}')

    state_space_path = _resolve_path(base, str(paths.get('state_space', 'state_space.parquet')))
    voxel_map_path = _resolve_path(base, str(paths.get('voxel_map', 'voxel_map.parquet')))
    voxel_stats_path = _resolve_path(base, str(paths.get('voxel_stats', 'voxel_stats.parquet')))
    transitions_path = _resolve_path(base, str(paths.get('transitions_topk', 'transitions_topk.parquet')))

    basins_candidates = []
    if 'basins' in paths:
        basins_candidates.append(_resolve_path(base, str(paths.get('basins'))))
    if 'basins_v02' in paths:
        basins_candidates.append(_resolve_path(base, str(paths.get('basins_v02'))))
    basins_candidates.extend([
        _resolve_path(base, 'basins_v02_components.parquet'),
        _resolve_path(base, 'basins.parquet'),
    ])

    seen = set()
    dedup_candidates = []
    for p in basins_candidates:
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            dedup_candidates.append(p)

    basins_path = None
    for p in dedup_candidates:
        if p.exists():
            basins_path = p
            break
    if basins_path is None:
        raise FileNotFoundError(
            'missing required basins artifact. tried: ' + ', '.join(str(p) for p in dedup_candidates)
        )

    basin_flows_path = _resolve_path(base, str(paths.get('basin_flows', 'basin_flows.parquet')))

    for p in [state_space_path, voxel_map_path, voxel_stats_path, transitions_path]:
        if not p.exists():
            raise FileNotFoundError(f'missing required artifact: {p}')

    contracts = {
        'state_space': load_state_space(state_space_path, cols),
        'voxel_map': load_voxel_map(voxel_map_path, cols),
        'voxel_stats': load_voxel_stats(voxel_stats_path),
        'transitions_topk': load_transitions_topk(transitions_path, cols),
        'basins': load_basins(basins_path, cols),
    }

    contracts['basin_flows'] = load_basin_flows(basin_flows_path)
    return contracts
