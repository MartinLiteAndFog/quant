from __future__ import annotations

from pathlib import Path

from visual.scene import Scene


def render_plotly(scene: Scene, out_path_html: str | Path) -> None:
    out = Path(out_path_html)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go
    except Exception:
        out.write_text(
            '<html><body><h3>Plotly not installed</h3>'
            f'<p>Requested scene: {scene.title}</p>'
            '</body></html>',
            encoding='utf-8',
        )
        return

    fig = go.Figure()

    for layer in scene.layers:
        if layer.kind == 'voxels':
            d = layer.data
            fig.add_trace(go.Scatter3d(
                x=d['center_x'], y=d['center_y'], z=d['center_z'],
                mode='markers',
                marker=dict(
                    size=d['_size'], color=d['_color'],
                    colorscale='Viridis', opacity=0.88,
                    colorbar=dict(title=str(layer.params.get('color_metric', 'metric'))),
                    line=dict(width=0.3, color='rgba(30,30,30,0.45)'),
                ),
                text=[f"voxel={v}" for v in d.get('voxel_id', [])],
                hovertemplate='x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>%{text}<extra></extra>',
                name=layer.name,
            ))

        elif layer.kind in ('basins_v02_points', 'basins_v02_core'):
            d = layer.data
            show_legend = bool(d['_legend'].iloc[0]) if '_legend' in d.columns and len(d) else True
            name = str(d['_group'].iloc[0]) if '_group' in d.columns and len(d) else layer.name
            alpha = float(d['_alpha'].iloc[0]) if '_alpha' in d.columns and len(d) else 0.90

            fig.add_trace(go.Scatter3d(
                x=d['center_x'], y=d['center_y'], z=d['center_z'],
                mode='markers',
                marker=dict(
                    size=d.get('_size', 40.0),
                    color=d.get('_color', 'lightgray'),
                    opacity=alpha,
                    line=dict(width=0.5 if alpha > 0.5 else 0.0, color='rgba(0,0,0,0.5)'),
                ),
                name=name,
                showlegend=show_legend,
                hovertemplate=f"{name}<br>" + 'x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>',
            ))

            if layer.kind == 'basins_v02_core' and '_label' in d.columns and len(d):
                fig.add_trace(go.Scatter3d(
                    x=d['center_x'], y=d['center_y'], z=d['center_z'],
                    mode='text',
                    text=[str(d['_label'].iloc[0])],
                    textposition='top center',
                    showlegend=False,
                ))

        elif layer.kind == 'basin_flow_arrows':
            d = layer.data
            for _, r in d.iterrows():
                fig.add_trace(go.Scatter3d(
                    x=[r['x0'], r['x1']], y=[r['y0'], r['y1']], z=[r['z0'], r['z1']],
                    mode='lines+text',
                    line=dict(width=3.0 + 8.0 * float(r['prob']), color=str(r['color'])),
                    text=[None, str(r['label'])],
                    textposition='top center',
                    textfont=dict(size=10, color=str(r['color'])),
                    name=str(r['label']),
                    showlegend=True,
                    hovertemplate=str(r['label']) + '<extra></extra>',
                ))
                # arrowhead cone at destination
                dx = r['x1'] - r['x0']
                dy = r['y1'] - r['y0']
                dz = r['z1'] - r['z0']
                fig.add_trace(go.Cone(
                    x=[r['x1']], y=[r['y1']], z=[r['z1']],
                    u=[dx], v=[dy], w=[dz],
                    sizemode='absolute', sizeref=0.06,
                    anchor='tip', showscale=False, showlegend=False,
                    colorscale=[[0, str(r['color'])], [1, str(r['color'])]],
                    opacity=0.85,
                ))

        elif layer.kind == 'edges':
            d = layer.data
            for _, r in d.iterrows():
                fig.add_trace(go.Scatter3d(
                    x=[r['x0'], r['x1']], y=[r['y0'], r['y1']], z=[r['z0'], r['z1']],
                    mode='lines',
                    line=dict(width=max(1.0, float(r.get('_width', 1)) * 0.9), color='rgba(70,70,70,0.33)'),
                    showlegend=False,
                ))

        elif layer.kind == 'trajectory_points':
            d = layer.data
            fig.add_trace(go.Scatter3d(
                x=d['X_raw'], y=d['Y_res'], z=d['Z_res'],
                mode='markers',
                marker=dict(
                    size=3.0,
                    color=d['_t'] if '_t' in d.columns else None,
                    colorscale='Turbo', opacity=0.95,
                    colorbar=dict(title='time') if '_t' in d.columns else None,
                ),
                name=layer.name, showlegend=False,
            ))
            if bool(layer.params.get('draw_lines', False)) and len(d) >= 2:
                fig.add_trace(go.Scatter3d(
                    x=d['X_raw'], y=d['Y_res'], z=d['Z_res'],
                    mode='lines',
                    line=dict(width=1.0, color='rgba(25,25,25,0.25)'),
                    showlegend=False,
                ))
            jumps = layer.params.get('jump_edges')
            if jumps is not None and hasattr(jumps, 'iterrows'):
                for _, r in jumps.iterrows():
                    fig.add_trace(go.Scatter3d(
                        x=[r['x0'], r['x1']], y=[r['y0'], r['y1']], z=[r['z0'], r['z1']],
                        mode='lines',
                        line=dict(width=2.0, color='rgba(0,0,0,0.12)'),
                        showlegend=False,
                    ))

        elif layer.kind == 'current_position':
            d = layer.data
            fig.add_trace(go.Scatter3d(
                x=d['center_x'], y=d['center_y'], z=d['center_z'],
                mode='markers+text',
                marker=dict(size=14, color='red', symbol='diamond', line=dict(width=2, color='darkred')),
                text=['NOW'], textposition='top center',
                textfont=dict(size=13, color='darkred'),
                name='NOW',
            ))

    fig.update_layout(
        title=scene.title,
        template='plotly_white',
        margin=dict(l=0, r=0, t=55, b=0),
        scene=dict(
            xaxis_title='X (Drift)', yaxis_title='Y (Elasticity)', zaxis_title='Z (Instability)',
            xaxis=dict(showbackground=True, backgroundcolor='rgba(245,245,245,0.5)'),
            yaxis=dict(showbackground=True, backgroundcolor='rgba(245,245,245,0.5)'),
            zaxis=dict(showbackground=True, backgroundcolor='rgba(245,245,245,0.5)'),
            camera=dict(eye=dict(x=1.55, y=1.45, z=1.2)),
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0.01, font=dict(size=10)),
    )
    fig.write_html(str(out), include_plotlyjs='cdn')
