from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def _load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input must be parquet or csv")


def _crosshair_2d(ax, cx, cy, color="red", lw=0.9, alpha=0.55):
    ax.axhline(cy, color=color, lw=lw, ls="--", alpha=alpha, zorder=5)
    ax.axvline(cx, color=color, lw=lw, ls="--", alpha=alpha, zorder=5)


def _mark_current_2d(ax, cx, cy):
    ax.scatter([cx], [cy], s=320, marker="*", c="red", edgecolors="white", linewidths=1.2, zorder=12)
    ax.scatter([cx], [cy], s=50, marker="o", c="yellow", edgecolors="red", linewidths=0.8, zorder=13)


def _mark_start_2d(ax, sx, sy):
    ax.scatter([sx], [sy], s=80, marker="o", c="limegreen", edgecolors="black", linewidths=0.7, zorder=11)


def _draw_axis_bar(ax, y_pos, value, color, label_left, label_right):
    ax.plot([0.08, 0.92], [y_pos, y_pos], lw=3, color="#2a2e38",
            transform=ax.transAxes, zorder=1, solid_capstyle="round")
    bar_end = 0.50 + value * 0.38
    ax.plot([0.50, bar_end], [y_pos, y_pos], lw=5, color=color,
            transform=ax.transAxes, zorder=2, solid_capstyle="round")
    ax.plot([0.50], [y_pos], marker="|", color="#666", markersize=10,
            transform=ax.transAxes, zorder=3)
    ax.text(0.04, y_pos, label_left, transform=ax.transAxes, fontsize=8.5,
            color=color, va="center", ha="left", fontweight="bold")
    ax.text(0.96, y_pos, label_right, transform=ax.transAxes, fontsize=8.5,
            color="#ccc", va="center", ha="right", family="monospace")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot state space v0.1 outputs")
    p.add_argument("--state", required=True, help="State space parquet/csv")
    p.add_argument("--basins", required=False, help="Basins csv path")
    p.add_argument("--out", required=True, help="Output png path")
    p.add_argument("--tail", type=int, default=0, help="Plot last N points; 0 = full history")
    p.add_argument("--max-traj-points", type=int, default=8000)
    p.add_argument("--title", default="SOL-USDT Renko State Space")
    args = p.parse_args()

    state_path = Path(args.state)
    out_path = Path(args.out)
    basins_path = Path(args.basins) if args.basins else None

    df = _load(state_path).copy()
    extra_cols = ["conf_x", "conf_y", "conf_z"]
    req = ["X_raw", "Y_res", "Z_res"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required state columns: {missing}")

    pts = df[req + [c for c in extra_cols if c in df.columns]].dropna(subset=req).copy()
    if args.tail > 0:
        pts = pts.tail(args.tail)
    if pts.empty:
        raise ValueError("No finite points to plot")

    basins = None
    if basins_path is not None and basins_path.exists():
        basins = pd.read_csv(basins_path)

    x = pts["X_raw"].to_numpy(dtype=float)
    y = pts["Y_res"].to_numpy(dtype=float)
    z = pts["Z_res"].to_numpy(dtype=float)
    n = len(pts)

    start = pts.iloc[0]
    last = pts.iloc[-1]
    cx, cy, cz = float(last["X_raw"]), float(last["Y_res"]), float(last["Z_res"])
    sx, sy, sz = float(start["X_raw"]), float(start["Y_res"]), float(start["Z_res"])

    if n <= args.max_traj_points:
        ki = np.arange(n, dtype=int)
    else:
        ki = np.unique(np.linspace(0, n - 1, num=args.max_traj_points, dtype=int))
    xt, yt, zt = x[ki], y[ki], z[ki]

    # ── Surface grid ──
    bins = 28
    xedges = np.linspace(-1.0, 1.0, bins + 1)
    yedges = np.linspace(-1.0, 1.0, bins + 1)
    xi = np.clip(np.digitize(x, xedges) - 1, 0, bins - 1)
    yi = np.clip(np.digitize(y, yedges) - 1, 0, bins - 1)

    z_sum = np.zeros((bins, bins))
    z_cnt = np.zeros((bins, bins))
    den = np.zeros((bins, bins))
    for i in range(n):
        z_sum[xi[i], yi[i]] += z[i]
        z_cnt[xi[i], yi[i]] += 1.0
        den[xi[i], yi[i]] += 1.0
    z_mean = np.divide(z_sum, z_cnt, out=np.full_like(z_sum, np.nan), where=z_cnt > 0)
    den_norm = den / max(float(den.max()), 1.0)

    xc = (xedges[:-1] + xedges[1:]) / 2.0
    yc = (yedges[:-1] + yedges[1:]) / 2.0
    Xg, Yg = np.meshgrid(xc, yc, indexing="ij")
    z_plot = np.nan_to_num(z_mean, nan=0.0)

    # ── Figure: 3 rows x 3 cols ──
    #   row 0-1, col 0:  3D surface (tall)
    #   row 0,   col 1:  Drift vs Elasticity
    #   row 0,   col 2:  Drift vs Instability
    #   row 1,   col 1:  Elasticity vs Instability
    #   row 1,   col 2:  Current State card
    #   row 2:           legend + colorbar strip
    fig = plt.figure(figsize=(18, 13), facecolor="#0e1117")
    gs = fig.add_gridspec(
        3, 3,
        width_ratios=[1.35, 1, 1],
        height_ratios=[1, 1, 0.06],
        wspace=0.28, hspace=0.32,
        left=0.04, right=0.96, top=0.93, bottom=0.06,
    )
    ax3d = fig.add_subplot(gs[0:2, 0], projection="3d")
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xz = fig.add_subplot(gs[0, 2])
    ax_yz = fig.add_subplot(gs[1, 1])
    ax_info = fig.add_subplot(gs[1, 2])
    ax_cbar = fig.add_subplot(gs[2, 1:3])

    for ax in [ax_xy, ax_xz, ax_yz]:
        ax.set_facecolor("#181c24")
        ax.tick_params(colors="#aaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#555")

    ax_info.set_facecolor("#141820")
    ax_info.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax_info.spines.values():
        spine.set_color("#444")

    ax3d.set_facecolor("#0e1117")
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor("#333")
    ax3d.yaxis.pane.set_edgecolor("#333")
    ax3d.zaxis.pane.set_edgecolor("#333")
    ax3d.tick_params(colors="#aaa", labelsize=7)

    x_label = "X: Drift / Trend"
    y_label = "Y: Elasticity (res.)"
    z_label = "Z: Instability (res.)"

    # ── 3D surface ──
    facecolors = plt.cm.inferno(den_norm.T)
    facecolors[..., 3] = 0.65
    ax3d.plot_surface(Xg, Yg, z_plot, facecolors=facecolors, linewidth=0,
                      antialiased=True, shade=False)
    ax3d.plot(xt, yt, zt, linewidth=0.6, alpha=0.30, c="#88aacc")

    ax3d.plot([cx, cx], [cy, cy], [-1, cz], color="red", lw=1.4, ls=":", alpha=0.8, zorder=8)
    ax3d.plot([-1, 1], [cy, cy], [cz, cz], color="red", lw=0.7, ls="--", alpha=0.35, zorder=7)
    ax3d.plot([cx, cx], [-1, 1], [cz, cz], color="red", lw=0.7, ls="--", alpha=0.35, zorder=7)

    ax3d.scatter([sx], [sy], [sz], s=90, marker="o", c="limegreen", edgecolors="white",
                 linewidths=0.7, zorder=9, depthshade=False)
    ax3d.scatter([cx], [cy], [cz], s=420, marker="*", c="red", edgecolors="white",
                 linewidths=1.0, zorder=15, depthshade=False)
    ax3d.scatter([cx], [cy], [cz], s=80, marker="o", c="yellow", edgecolors="red",
                 linewidths=0.8, zorder=16, depthshade=False)
    ax3d.text(cx + 0.05, cy + 0.05, cz + 0.09, "NOW", color="white",
              fontsize=11, fontweight="bold", zorder=17)

    ax3d.set_xlabel(x_label, color="#ccc", fontsize=8, labelpad=6)
    ax3d.set_ylabel(y_label, color="#ccc", fontsize=8, labelpad=6)
    ax3d.set_zlabel(z_label, color="#ccc", fontsize=8, labelpad=6)
    ax3d.set_xlim(-1, 1); ax3d.set_ylim(-1, 1); ax3d.set_zlim(-1, 1)
    ax3d.set_title("3D State Surface  (height = avg Z, color = density)",
                    color="white", fontsize=10, pad=10)
    ax3d.view_init(elev=28, azim=-55)

    # ── 2D panels ──
    ax_xy.hexbin(x, y, gridsize=55, extent=(-1, 1, -1, 1), cmap="inferno", mincnt=1, bins="log")
    _crosshair_2d(ax_xy, cx, cy); _mark_start_2d(ax_xy, sx, sy); _mark_current_2d(ax_xy, cx, cy)
    ax_xy.set_title("Drift vs Elasticity", color="white", fontsize=10)
    ax_xy.set_xlabel(x_label, color="#ccc", fontsize=8)
    ax_xy.set_ylabel(y_label, color="#ccc", fontsize=8)
    ax_xy.set_xlim(-1, 1); ax_xy.set_ylim(-1, 1); ax_xy.grid(alpha=0.12, color="#555")

    ax_xz.hexbin(x, z, gridsize=55, extent=(-1, 1, -1, 1), cmap="inferno", mincnt=1, bins="log")
    _crosshair_2d(ax_xz, cx, cz); _mark_start_2d(ax_xz, sx, sz); _mark_current_2d(ax_xz, cx, cz)
    ax_xz.set_title("Drift vs Instability", color="white", fontsize=10)
    ax_xz.set_xlabel(x_label, color="#ccc", fontsize=8)
    ax_xz.set_ylabel(z_label, color="#ccc", fontsize=8)
    ax_xz.set_xlim(-1, 1); ax_xz.set_ylim(-1, 1); ax_xz.grid(alpha=0.12, color="#555")

    hb = ax_yz.hexbin(y, z, gridsize=55, extent=(-1, 1, -1, 1), cmap="inferno", mincnt=1, bins="log")
    _crosshair_2d(ax_yz, cy, cz); _mark_start_2d(ax_yz, sy, sz); _mark_current_2d(ax_yz, cy, cz)
    ax_yz.set_title("Elasticity vs Instability", color="white", fontsize=10)
    ax_yz.set_xlabel(y_label, color="#ccc", fontsize=8)
    ax_yz.set_ylabel(z_label, color="#ccc", fontsize=8)
    ax_yz.set_xlim(-1, 1); ax_yz.set_ylim(-1, 1); ax_yz.grid(alpha=0.12, color="#555")

    # ── Basins ──
    if basins is not None and {"voxel_x", "voxel_y", "voxel_z"}.issubset(set(basins.columns)):
        b = basins.head(8)
        ax3d.scatter(b["voxel_x"], b["voxel_y"], b["voxel_z"], s=100, marker="D",
                     c="cyan", edgecolors="white", linewidths=0.6, alpha=0.9,
                     zorder=10, depthshade=False)
        for a, bx, by in [(ax_xy, "voxel_x", "voxel_y"),
                           (ax_xz, "voxel_x", "voxel_z"),
                           (ax_yz, "voxel_y", "voxel_z")]:
            a.scatter(b[bx], b[by], s=55, marker="D", c="cyan",
                      edgecolors="white", linewidths=0.6, alpha=0.9, zorder=10)

    # ── Current State card ──
    ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)

    conf_x = float(last.get("conf_x", 0.0)) if "conf_x" in last.index else float("nan")
    conf_y = float(last.get("conf_y", 0.0)) if "conf_y" in last.index else float("nan")
    conf_z = float(last.get("conf_z", 0.0)) if "conf_z" in last.index else float("nan")

    ax_info.set_title("Current State", color="white", fontsize=11, pad=8)

    _draw_axis_bar(ax_info, 0.82, cx, "#ff6644", "X Drift",
                   f"{cx:+.3f}  conf {conf_x:+.3f}")
    _draw_axis_bar(ax_info, 0.62, cy, "#44bbff", "Y Elast.",
                   f"{cy:+.3f}  conf {conf_y:+.3f}")
    _draw_axis_bar(ax_info, 0.42, cz, "#ffcc33", "Z Instab.",
                   f"{cz:+.3f}  conf {conf_z:+.3f}")

    ax_info.text(0.50, 0.22, f"Points: {n:,}", transform=ax_info.transAxes,
                 fontsize=9, color="#888", ha="center", va="center")
    ax_info.plot([0.08, 0.92], [0.32, 0.32], color="#333", lw=0.8,
                transform=ax_info.transAxes)

    note = "Z bands = Renko box quantization" if n > 50_000 else ""
    if note:
        ax_info.text(0.50, 0.12, note, transform=ax_info.transAxes,
                     fontsize=7.5, color="#666", ha="center", va="center", style="italic")

    # ── Horizontal colorbar in bottom strip ──
    cb = fig.colorbar(hb, cax=ax_cbar, orientation="horizontal")
    cb.set_label("Occupancy density (log count per hex)", color="#ccc", fontsize=8)
    cb.ax.tick_params(colors="#aaa", labelsize=7)
    ax_cbar.set_facecolor("#0e1117")

    # ── Legend ──
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="limegreen",
               markeredgecolor="white", markersize=8, label="Start"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
               markeredgecolor="white", markersize=14, label="Current (NOW)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="cyan",
               markeredgecolor="white", markersize=7, label="Basin center"),
        Line2D([0], [0], color="red", ls="--", lw=0.9, alpha=0.6,
               label="Current crosshair"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               frameon=False, fontsize=9, labelcolor="white",
               bbox_to_anchor=(0.5, 0.005))

    tail_text = "full history" if args.tail <= 0 else f"tail {args.tail}"
    fig.suptitle(f"{args.title}  ({tail_text}, {n:,} pts)",
                 fontsize=14, color="white", y=0.97)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(str(out_path))


if __name__ == "__main__":
    main()
