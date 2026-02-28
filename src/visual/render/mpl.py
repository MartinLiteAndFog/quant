from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from visual.scene import Scene


def _set_3d_style(ax) -> None:
    ax.set_xlabel('X  (Drift)', labelpad=10, fontsize=11)
    ax.set_ylabel('Y  (Elasticity)', labelpad=10, fontsize=11)
    ax.set_zlabel('Z  (Instability)', labelpad=10, fontsize=11)
    ax.grid(True, alpha=0.18)
    ax.view_init(elev=24, azim=40)


def _heatmap_style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _apply_clip_quantile(values: np.ndarray, q: float) -> tuple[np.ndarray, float]:
    if values.size == 0:
        return values, 1.0
    vv = values[np.isfinite(values)]
    if vv.size == 0:
        return values, 1.0
    hi = float(np.quantile(vv, q))
    if not np.isfinite(hi) or hi <= 0:
        return values, float(np.nanmax(vv))
    return np.clip(values, 0.0, hi), hi


def render_matplotlib(scene: Scene, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    kind = scene.settings.get('kind', '3d')

    if kind == 'heatmap2d':
        fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
        for layer in scene.layers:
            if layer.kind != 'heatmap2d':
                continue
            d = layer.data
            vals = np.asarray(d.values, dtype=float)
            clip_q = float(layer.params.get('clip_quantile', 1.0))
            if clip_q < 1.0:
                vals, _ = _apply_clip_quantile(vals, clip_q)
            im = ax.imshow(vals, origin='lower', aspect='auto', cmap='magma')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(str(layer.params.get('cbar_label', 'value')))
            _heatmap_style(ax, scene.title, str(layer.params.get('xlabel', 'x')), str(layer.params.get('ylabel', 'y')))
            ax.set_xticks(range(len(d.columns)))
            ax.set_yticks(range(len(d.index)))
            ax.set_xticklabels([str(c) for c in d.columns], rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels([str(i) for i in d.index], fontsize=8)
        fig.tight_layout()
        fig.savefig(out, dpi=160, bbox_inches='tight')
        plt.close(fig)
        return

    fig = plt.figure(figsize=(12, 9), dpi=160)
    ax = fig.add_subplot(111, projection='3d')

    mappable = None
    legend_handles = []
    legend_labels = []

    for layer in scene.layers:
        if layer.kind == 'voxels':
            d = layer.data
            sc = ax.scatter(
                d['center_x'], d['center_y'], d['center_z'],
                s=d['_size'], c=d['_color'], cmap='viridis',
                alpha=0.90, linewidths=0.25, edgecolors='k',
            )
            mappable = sc

        elif layer.kind == 'basins_v02_points':
            d = layer.data
            show_legend = bool(d['_legend'].iloc[0]) if '_legend' in d.columns and len(d) else True
            lbl = str(d['_group'].iloc[0]) if '_group' in d.columns and len(d) else layer.name
            alpha = float(d['_alpha'].iloc[0]) if '_alpha' in d.columns and len(d) else (0.25 if lbl == 'noise' else 0.92)

            sc = ax.scatter(
                d['center_x'], d['center_y'], d['center_z'],
                s=d.get('_size', 32.0),
                c=d.get('_color', 'lightgray'),
                alpha=alpha,
                linewidths=0.3 if alpha > 0.5 else 0.0,
                edgecolors='k' if alpha > 0.5 else 'none',
                label=lbl if show_legend else None,
            )
            if show_legend:
                legend_handles.append(sc)
                legend_labels.append(lbl)

        elif layer.kind == 'basins_v02_core':
            d = layer.data
            sc = ax.scatter(
                d['center_x'], d['center_y'], d['center_z'],
                s=d.get('_size', 140.0),
                c=d.get('_color', '#377eb8'),
                alpha=0.98,
                linewidths=0.8,
                edgecolors='k',
            )
            if '_label' in d.columns and len(d):
                ax.text(
                    float(d['center_x'].iloc[0]) + 0.03,
                    float(d['center_y'].iloc[0]) + 0.03,
                    float(d['center_z'].iloc[0]) + 0.03,
                    str(d['_label'].iloc[0]),
                    fontsize=10, fontweight='bold', color='black',
                )

        elif layer.kind == 'basin_flow_arrows':
            d = layer.data
            for _, r in d.iterrows():
                dx = r['x1'] - r['x0']
                dy = r['y1'] - r['y0']
                dz = r['z1'] - r['z0']
                prob = float(r['prob'])
                color = str(r['color'])
                lw = 1.5 + 4.0 * prob
                ax.quiver(
                    r['x0'], r['y0'], r['z0'],
                    dx, dy, dz,
                    color=color, alpha=0.85,
                    linewidth=lw,
                    arrow_length_ratio=0.12,
                )
                mx = r['x0'] + 0.55 * dx
                my = r['y0'] + 0.55 * dy
                mz = r['z0'] + 0.55 * dz
                ax.text(
                    mx, my, mz,
                    str(r['label']),
                    fontsize=7, fontweight='bold', color=color,
                    ha='center',
                )

        elif layer.kind == 'edges':
            d = layer.data
            for _, r in d.iterrows():
                ax.plot(
                    [r['x0'], r['x1']], [r['y0'], r['y1']], [r['z0'], r['z1']],
                    color='dimgray',
                    alpha=float(r.get('_opacity', 0.25)),
                    linewidth=max(0.5, float(r.get('_width', 1.0)) * 0.35),
                )

        elif layer.kind == 'trajectory_points':
            d = layer.data
            ax.scatter(
                d['X_raw'], d['Y_res'], d['Z_res'],
                c=d['_t'] if '_t' in d.columns else None,
                cmap='plasma', s=8, alpha=0.95, linewidths=0,
            )
            if bool(layer.params.get('draw_lines', False)) and len(d) >= 2:
                ax.plot(d['X_raw'], d['Y_res'], d['Z_res'], color='black', alpha=0.25, linewidth=0.6)
            jumps = layer.params.get('jump_edges')
            if isinstance(jumps, (list, tuple)):
                jumps = None
            if jumps is not None and hasattr(jumps, 'iterrows'):
                for _, r in jumps.iterrows():
                    ax.plot([r['x0'], r['x1']], [r['y0'], r['y1']], [r['z0'], r['z1']], color='black', alpha=0.15, linewidth=1.0)
            mappable = ScalarMappable(norm=Normalize(0, 1), cmap='plasma')

        elif layer.kind == 'current_position':
            d = layer.data
            sc = ax.scatter(
                d['center_x'], d['center_y'], d['center_z'],
                s=280, c='red', marker='*', zorder=100,
                linewidths=0.8, edgecolors='darkred', label='NOW',
            )
            legend_handles.append(sc)
            legend_labels.append('NOW')
            ax.text(
                float(d['center_x'].iloc[0]) + 0.04,
                float(d['center_y'].iloc[0]) + 0.04,
                float(d['center_z'].iloc[0]) + 0.04,
                'NOW', fontsize=10, fontweight='bold', color='darkred',
            )

    _set_3d_style(ax)
    ax.set_title(scene.title, pad=16, fontsize=13)

    if mappable is not None:
        fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.08)

    if legend_handles:
        ax.legend(
            legend_handles, legend_labels,
            loc='upper left', fontsize=8, markerscale=0.8,
            ncol=1, framealpha=0.8,
        )

    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
