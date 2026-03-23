from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from quant.state_space.pipeline import compute_state_space


RENKO_PATH = Path("/Users/martinpeter/Desktop/quant/data/renko/SOL-USDT_renko_box0.03_last7d_binance_ns.parquet")
GATE_PATH = Path("/Users/martinpeter/Desktop/quant-main/data/regimes/pc_3axis_gate_FULLRANGE_nolookahead_v2.csv")
OUT_PATH = Path("/Users/martinpeter/Desktop/quant-main/data/runs/pc_axes_spectral_last7d.png")


def normalize_signed(x: pd.Series) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce").astype(float)
    vmax = np.nanmax(np.abs(v.values))
    if not np.isfinite(vmax) or vmax <= 1e-12:
        return pd.Series(np.zeros(len(v)), index=v.index, dtype=float)
    return v / vmax


def add_background_heatmap(
    ax: plt.Axes,
    ts: pd.Series,
    close: pd.Series,
    intensity: pd.Series,
    title: str,
    cmap: str = "coolwarm",
    alpha: float = 0.30,
) -> None:
    x_num = mdates.date2num(pd.to_datetime(ts, utc=True).dt.to_pydatetime())
    y = pd.to_numeric(close, errors="coerce").astype(float).values
    z = pd.to_numeric(intensity, errors="coerce").astype(float).fillna(0.0).values

    if len(x_num) < 2:
        raise ValueError("Need at least 2 timestamps to render heatmap background")

    y_min = np.nanmin(y)
    y_max = np.nanmax(y)
    pad = max((y_max - y_min) * 0.03, 1e-9)
    y_min -= pad
    y_max += pad

    # Build a vertical strip image so the axis intensity becomes a full background field.
    heat = np.tile(z, (80, 1))

    im = ax.imshow(
        heat,
        extent=[x_num[0], x_num[-1], y_min, y_max],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        alpha=alpha,
        interpolation="nearest",
        zorder=0,
    )

    ax.plot(ts, close, color="black", linewidth=0.9, alpha=0.9, zorder=2)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title)
    ax.grid(True, alpha=0.18)

    cbar = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
    cbar.set_label("normalized intensity", rotation=90)
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])


def add_gate_background(ax: plt.Axes, ts: pd.Series, close: pd.Series, gate_on: np.ndarray) -> None:
    ax.plot(ts, close, color="black", linewidth=0.9, alpha=0.95, zorder=2)

    start_idx = None
    for i, g in enumerate(gate_on):
        if g == 1 and start_idx is None:
            start_idx = i
        if g == 0 and start_idx is not None:
            ax.axvspan(ts.iloc[start_idx], ts.iloc[i], color="blue", alpha=0.16, zorder=0)
            start_idx = None

    if start_idx is not None:
        ax.axvspan(ts.iloc[start_idx], ts.iloc[len(ts) - 1], color="blue", alpha=0.16, zorder=0)

    ax.set_title("Gate ON/OFF (blue background = gate ON)")
    ax.grid(True, alpha=0.18)


def main() -> None:
    if not RENKO_PATH.exists():
        raise FileNotFoundError(f"Missing renko parquet: {RENKO_PATH}")
    if not GATE_PATH.exists():
        raise FileNotFoundError(f"Missing gate csv: {GATE_PATH}")

    df = pd.read_parquet(RENKO_PATH).copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts").reset_index(drop=True)

    ss = compute_state_space(df[["ts", "open", "high", "low", "close"]].copy()).copy()
    ss["ts"] = pd.to_datetime(ss["ts"], utc=True, errors="coerce")
    ss = ss.sort_values("ts").reset_index(drop=True)

    # Requested mapping:
    # X axis -> X_raw
    # Y axis -> Y_res
    # Z axis -> Z_res
    ss["X_norm"] = normalize_signed(ss["X_raw"])
    ss["Y_norm"] = normalize_signed(ss["Y_res"])
    ss["Z_norm"] = normalize_signed(ss["Z_res"])

    gate = pd.read_csv(GATE_PATH).copy()
    gate["ts"] = pd.to_datetime(gate["ts"], utc=True, errors="coerce")
    gate = gate.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    if "gate_base_3of3" not in gate.columns:
        raise ValueError("Expected gate_base_3of3 in gate csv")

    gate["gate_on"] = pd.to_numeric(gate["gate_base_3of3"], errors="coerce").fillna(0).astype(int)

    base = df[["ts", "close"]].copy().sort_values("ts").reset_index(drop=True)

    gate_aligned = pd.merge_asof(
        base[["ts"]].sort_values("ts"),
        gate[["ts", "gate_on"]].sort_values("ts"),
        on="ts",
        direction="backward",
        allow_exact_matches=True,
    )
    gate_aligned["gate_on"] = gate_aligned["gate_on"].fillna(0).astype(int)

    plot_df = pd.merge_asof(
        base.sort_values("ts"),
        ss[["ts", "X_norm", "Y_norm", "Z_norm"]].sort_values("ts"),
        on="ts",
        direction="nearest",
        allow_exact_matches=True,
    )

    plot_df["X_norm"] = plot_df["X_norm"].fillna(0.0)
    plot_df["Y_norm"] = plot_df["Y_norm"].fillna(0.0)
    plot_df["Z_norm"] = plot_df["Z_norm"].fillna(0.0)
    plot_df["gate_on"] = gate_aligned["gate_on"].values

    ts = plot_df["ts"]
    close = plot_df["close"]

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(18, 16),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1]},
    )

    panel_specs = [
        ("X axis background heatmap (X_raw)", plot_df["X_norm"], "coolwarm"),
        ("Y axis background heatmap (Y_res)", plot_df["Y_norm"], "PiYG"),
        ("Z axis background heatmap (Z_res)", plot_df["Z_norm"], "PuOr"),
    ]

    for ax, (title, intensity, cmap) in zip(axes[:3], panel_specs):
        add_background_heatmap(
            ax=ax,
            ts=ts,
            close=close,
            intensity=intensity,
            title=title,
            cmap=cmap,
            alpha=0.30,
        )

    add_gate_background(
        ax=axes[3],
        ts=ts,
        close=close,
        gate_on=plot_df["gate_on"].astype(int).values,
    )

    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter("%m-%d %H:%M", tz=ts.dt.tz)
    )

    fig.suptitle("State-space axes as background heatmaps + gate overlay", fontsize=15, y=0.995)
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
    print(f"WROTE: {OUT_PATH}")


if __name__ == "__main__":
    main()