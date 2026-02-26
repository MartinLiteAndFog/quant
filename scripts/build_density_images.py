"""Build background density PNG images for dashboard state-space heatmaps.

Pre-computes hexbin density plots from the full historical state-space parquet,
one per axis pair (XY, XZ, YZ).  Intended to run daily via cron / Railway.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

AXIS_PAIRS: Sequence[Tuple[str, str, str, str, str]] = [
    ("xy", "X_raw", "Y_res", "X: Drift", "Y: Elasticity"),
    ("xz", "X_raw", "Z_res", "X: Drift", "Z: Instability"),
    ("yz", "Y_res", "Z_res", "Y: Elasticity", "Z: Instability"),
]

BG_COLOR = "#181c24"


def build_density_images(
    state_space_path: str | Path,
    out_dir: str | Path,
    gridsize: int = 55,
    dpi: int = 150,
    figsize: Tuple[float, float] = (4, 4),
) -> None:
    """Generate one density background PNG per axis pair."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(state_space_path)

    for tag, col_x, col_y, label_x, label_y in AXIS_PAIRS:
        subset = df[[col_x, col_y]].dropna()
        x, y = subset[col_x].values, subset[col_y].values

        fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        ax.hexbin(
            x, y,
            gridsize=gridsize,
            extent=(-1, 1, -1, 1),
            cmap="inferno",
            mincnt=1,
            bins="log",
        )

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel(label_x, color="#ccc", fontsize=8)
        ax.set_ylabel(label_y, color="#ccc", fontsize=8)
        ax.tick_params(colors="#888", labelsize=7)
        ax.grid(alpha=0.12, color="#555")
        for spine in ax.spines.values():
            spine.set_color("#333")

        fig.savefig(
            out_dir / f"density_bg_{tag}.png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build density background PNGs for dashboard heatmaps",
    )
    parser.add_argument("--state-space", required=True, help="Path to state-space parquet")
    parser.add_argument("--out-dir", default="data/live/density", help="Output directory")
    parser.add_argument("--gridsize", type=int, default=55)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    build_density_images(
        state_space_path=args.state_space,
        out_dir=args.out_dir,
        gridsize=args.gridsize,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
