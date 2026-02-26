from __future__ import annotations

import numpy as np
import pandas as pd


def _voxel_centers(edges: np.ndarray) -> np.ndarray:
    return (edges[:-1] + edges[1:]) / 2.0


def voxel_basins(
    x: pd.Series,
    y: pd.Series,
    z: pd.Series,
    bins: int = 40,
    top_k: int = 8,
    include_labels: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None]:
    valid = x.notna() & y.notna() & z.notna()
    if valid.sum() == 0:
        empty_basins = pd.DataFrame(
            columns=["voxel_x", "voxel_y", "voxel_z", "density", "count", "ix", "iy", "iz"]
        )
        labels = pd.Series(index=x.index, dtype="float64", name="voxel_label") if include_labels else None
        return empty_basins, labels

    xv = x[valid].to_numpy(dtype=float)
    yv = y[valid].to_numpy(dtype=float)
    zv = z[valid].to_numpy(dtype=float)
    pts = np.column_stack([xv, yv, zv])

    hist, edges = np.histogramdd(pts, bins=[bins, bins, bins], range=[[-1, 1], [-1, 1], [-1, 1]])
    total = float(hist.sum())
    if total <= 0.0:
        empty_basins = pd.DataFrame(
            columns=["voxel_x", "voxel_y", "voxel_z", "density", "count", "ix", "iy", "iz"]
        )
        labels = pd.Series(index=x.index, dtype="float64", name="voxel_label") if include_labels else None
        return empty_basins, labels

    nz = np.argwhere(hist > 0)
    counts = hist[hist > 0]
    cx = _voxel_centers(edges[0])
    cy = _voxel_centers(edges[1])
    cz = _voxel_centers(edges[2])

    rows = []
    for (ix, iy, iz), c in zip(nz, counts):
        rows.append(
            {
                "voxel_x": float(cx[ix]),
                "voxel_y": float(cy[iy]),
                "voxel_z": float(cz[iz]),
                "density": float(c / total),
                "count": int(c),
                "ix": int(ix),
                "iy": int(iy),
                "iz": int(iz),
            }
        )
    basins = pd.DataFrame(rows).sort_values(["density", "count"], ascending=False).head(top_k).reset_index(drop=True)

    labels = None
    if include_labels:
        ix = np.clip(np.digitize(x.to_numpy(dtype=float), edges[0]) - 1, 0, bins - 1)
        iy = np.clip(np.digitize(y.to_numpy(dtype=float), edges[1]) - 1, 0, bins - 1)
        iz = np.clip(np.digitize(z.to_numpy(dtype=float), edges[2]) - 1, 0, bins - 1)
        flat = np.ravel_multi_index((ix, iy, iz), dims=(bins, bins, bins))
        labels = pd.Series(flat, index=x.index, name="voxel_label")
        labels = labels.where(valid, np.nan)

    return basins, labels
