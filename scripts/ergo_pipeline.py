#!/usr/bin/env python3
"""
Ergodicity Economics Pipeline
=============================
Visualises the core insight of Ergodicity Economics: for multiplicative
dynamics the ensemble average (across N realisations) diverges from
the time average (single trajectory).  As N grows the ensemble average
converges to the theoretical expected value — but NO single agent
actually experiences that growth.

Simulation: x(t+1) = x(t) * r(t), coin-flip multiplicative gamble
(×1.5 or ×0.6 each step).  Positive expected value, negative median.

Pipeline:
  1. DATA  — simulate multiplicative random walks for each N.
  2. VISUAL — dual-panel static snapshot (linear + log scale).

Based on: quant-traderr-lab/Ergo/Ergo Pipeline.py

Usage:
  python scripts/ergo_pipeline.py                        # defaults
  python scripts/ergo_pipeline.py --out artifacts/ergo/  # custom dir
  python scripts/ergo_pipeline.py --T 100 --seed 7       # override
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULTS = {
    "T": 50,
    "N_VALUES": [1, 100, 10_000, 1_000_000],
    "UP_MULT": 1.5,
    "DOWN_MULT": 0.6,
    "X0": 1.0,
    "SEED": 42,
}

THEME = {
    "BG": "#0b0b0b",
    "PANEL_BG": "#0e0e0e",
    "GRID": "#1f1f1f",
    "TEXT": "#ffffff",
    "TEXT_MUTED": "#aaaaaa",
    "COLORS": {
        1: "#00bfff",
        100: "#00ff41",
        10_000: "#ff3333",
        1_000_000: "#ffffff",
    },
    "LINEWIDTHS": {
        1: 1.8,
        100: 1.6,
        10_000: 1.4,
        1_000_000: 2.0,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

# ---------------------------------------------------------------------------
# Module 1 — Data: simulate multiplicative random walks
# ---------------------------------------------------------------------------

def simulate_ensemble_averages(
    T: int,
    n_values: list[int],
    up: float,
    down: float,
    seed: int,
) -> dict[int, np.ndarray]:
    """Return {N: ensemble_avg_array} for each N in *n_values*."""
    _log("[Data] Simulating multiplicative random walks …")
    rng = np.random.default_rng(seed)
    results: dict[int, np.ndarray] = {}

    for N in n_values:
        _log(f"[Data] N={N:>10,d} — simulating …")
        flips = rng.choice([up, down], size=(N, T))
        cum_prod = np.cumprod(flips, axis=1)
        paths = np.hstack([np.ones((N, 1)), cum_prod])
        ensemble_avg = paths.mean(axis=0)
        results[N] = ensemble_avg
        _log(f"[Data] N={N:>10,d} — final ⟨x⟩ = {ensemble_avg[-1]:.4f}")

    _log("[Data] Simulation complete.")
    return results

# ---------------------------------------------------------------------------
# Module 2 — Visualisation: dual-panel static snapshot
# ---------------------------------------------------------------------------

def visualize(
    data: dict[int, np.ndarray],
    T: int,
    up: float,
    down: float,
    out_path: Path,
) -> Path:
    """Generate a dual-panel (linear + log) PNG and return its path."""
    _log("[Visual] Generating static snapshot …")

    t_range = np.arange(T + 1)
    fig, (ax_lin, ax_log) = plt.subplots(
        1, 2, figsize=(19.2, 10.8), dpi=100, facecolor=THEME["BG"],
    )

    for ax, is_log, label in [(ax_lin, False, "(A)"), (ax_log, True, "(B)")]:
        ax.set_facecolor(THEME["PANEL_BG"])

        for N, y in data.items():
            color = THEME["COLORS"][N]
            lw = THEME["LINEWIDTHS"][N]
            ax.step(
                t_range, y, where="post",
                color=color, linewidth=lw,
                label=f"N={N:,}", alpha=0.95,
                zorder=3 if N == 1_000_000 else 2,
            )

        if is_log:
            ax.set_yscale("log")
            ax.set_ylim(0.08, 15)
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.set_yticks([0.1, 1, 10])
            ax.set_yticklabels(["10\u207b\u00b9", "10\u2070", "10\u00b9"])
        else:
            ax.set_ylim(-0.3, 10.5)
            ax.set_yticks(range(0, 11))

        ax.set_xlim(0, T + 1)
        ax.set_xticks(range(0, T + 5, 5))
        ax.set_xlabel("t", color=THEME["TEXT"], fontsize=13)
        ax.set_ylabel(r"$\langle x(t) \rangle_N$", color=THEME["TEXT"], fontsize=13)

        ax.grid(True, color=THEME["GRID"], linewidth=0.5, alpha=0.7)
        ax.tick_params(colors=THEME["TEXT_MUTED"], labelsize=10)
        for spine in ax.spines.values():
            spine.set_color(THEME["GRID"])

        ax.legend(
            loc="upper left", fontsize=9,
            facecolor="#111111", edgecolor=THEME["GRID"],
            labelcolor=THEME["TEXT"], framealpha=0.9,
        )

        ax.text(
            0.5, -0.10, label, transform=ax.transAxes, fontsize=14,
            ha="center", va="top", color=THEME["TEXT_MUTED"],
        )

    fig.suptitle(
        "ERGODICITY ECONOMICS \u2014 Ensemble Average vs Time Average",
        fontsize=16, color=THEME["TEXT"], fontweight="bold", y=0.96,
    )
    fig.text(
        0.5, 0.915,
        f"Multiplicative gamble: \u00d7{up} or \u00d7{down} | "
        f"E[r] = {(up + down) / 2:.2f} | T = {T} steps",
        fontsize=11, color=THEME["TEXT_MUTED"], ha="center",
    )
    fig.text(
        0.5, 0.02, "@quant.traderr", fontsize=10, color="#333333",
        ha="center", style="italic",
    )

    plt.subplots_adjust(left=0.06, right=0.97, top=0.87, bottom=0.13, wspace=0.22)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, facecolor=THEME["BG"], dpi=100)
    plt.close(fig)
    _log(f"[Visual] Saved → {out_path}")
    return out_path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ergodicity Economics: ensemble vs time average visualisation",
    )
    ap.add_argument("--T", type=int, default=DEFAULTS["T"], help="Time steps")
    ap.add_argument("--seed", type=int, default=DEFAULTS["SEED"], help="RNG seed")
    ap.add_argument("--up", type=float, default=DEFAULTS["UP_MULT"], help="Win multiplier")
    ap.add_argument("--down", type=float, default=DEFAULTS["DOWN_MULT"], help="Loss multiplier")
    ap.add_argument(
        "--out", default="artifacts/ergo",
        help="Output directory (default: artifacts/ergo)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log("=== ERGODICITY ECONOMICS PIPELINE ===")

    data = simulate_ensemble_averages(
        T=args.T,
        n_values=DEFAULTS["N_VALUES"],
        up=args.up,
        down=args.down,
        seed=args.seed,
    )

    visualize(
        data,
        T=args.T,
        up=args.up,
        down=args.down,
        out_path=out_dir / "ergodicity_static.png",
    )

    _log("=== PIPELINE FINISHED ===")


if __name__ == "__main__":
    main()
