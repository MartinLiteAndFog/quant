# Dashboard V2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand the live dashboard with 2D state space heatmaps, a continuous regime gradient band, fading trajectory overlay, and axis status bars.

**Architecture:** New backend module `dashboard_statespace.py` computes state space features from live Renko data and serves them via a new API endpoint. Daily cron script pre-computes background density PNGs. Frontend renders everything via HTML5 Canvas alongside the existing Lightweight Charts Renko chart.

**Tech Stack:** Python (pandas, numpy, matplotlib for density PNGs), FastAPI (API endpoint), HTML5 Canvas + vanilla JS (frontend rendering)

---

### Task 1: State Space Feature Writer Module

**Files:**
- Create: `src/quant/execution/dashboard_statespace.py`
- Test: `tests/test_dashboard_statespace.py`
- Reference: `src/quant/state_space/pipeline.py` (the `compute_state_space` function)

**Step 1: Write the failing test**

```python
# tests/test_dashboard_statespace.py
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class TestStateSpaceWriter(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # Build a minimal Renko parquet with enough rows for feature computation
        n = 500
        ts = pd.date_range("2026-02-01", periods=n, freq="5min", tz="UTC")
        close = 100.0 + np.cumsum(np.random.default_rng(42).normal(0, 0.05, n))
        renko = pd.DataFrame({
            "ts": ts,
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.3,
            "close": close,
        })
        renko.to_parquet(self.root / "renko.parquet", index=False)
        os.environ["DASHBOARD_RENKO_PARQUET"] = str(self.root / "renko.parquet")
        os.environ["DASHBOARD_STATESPACE_PARQUET"] = str(self.root / "state_space.parquet")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_refresh_writes_parquet(self) -> None:
        from quant.execution.dashboard_statespace import refresh_state_space_cache
        info = refresh_state_space_cache()
        self.assertTrue(info["ok"])
        out_path = Path(os.environ["DASHBOARD_STATESPACE_PARQUET"])
        self.assertTrue(out_path.exists())
        df = pd.read_parquet(out_path)
        for col in ("ts", "X_raw", "Y_res", "Z_res", "conf_x", "conf_y", "conf_z"):
            self.assertIn(col, df.columns, f"Missing column: {col}")
        self.assertGreater(len(df), 0)
        # Values should be in [-1, 1] range (signals are clipped)
        for col in ("X_raw", "Y_res", "Z_res"):
            self.assertTrue(df[col].dropna().between(-2.0, 2.0).all(), f"{col} out of range")

    def test_load_trajectory_filters_by_window(self) -> None:
        from quant.execution.dashboard_statespace import refresh_state_space_cache, load_state_space_trajectory
        refresh_state_space_cache()
        # Full trajectory
        full = load_state_space_trajectory(window_hours=9999)
        # Short window should return fewer points
        short = load_state_space_trajectory(window_hours=1)
        self.assertLessEqual(len(short["trajectory"]), len(full["trajectory"]))
        self.assertIn("current", full)
        for key in ("x", "y", "z", "conf_x", "conf_y", "conf_z"):
            self.assertIn(key, full["current"])

    def test_load_trajectory_returns_empty_when_no_data(self) -> None:
        os.environ["DASHBOARD_STATESPACE_PARQUET"] = str(self.root / "nonexistent.parquet")
        from quant.execution.dashboard_statespace import load_state_space_trajectory
        result = load_state_space_trajectory(window_hours=8)
        self.assertEqual(len(result["trajectory"]), 0)
        self.assertIsNone(result["current"])


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dashboard_statespace.py -v`
Expected: FAIL (module does not exist yet)

**Step 3: Write the implementation**

```python
# src/quant/execution/dashboard_statespace.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from quant.state_space.pipeline import compute_state_space
from quant.state_space.config import StateSpaceConfig


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default))


def refresh_state_space_cache() -> Dict[str, Any]:
    """Compute state space features from live Renko data and write to parquet."""
    renko_path = _env_path("DASHBOARD_RENKO_PARQUET", "data/live/renko_latest.parquet")
    out_path = _env_path("DASHBOARD_STATESPACE_PARQUET", "data/live/state_space_latest.parquet")

    if not renko_path.exists():
        return {"ok": False, "reason": "renko parquet not found"}

    try:
        df = pd.read_parquet(renko_path)
    except Exception as e:
        return {"ok": False, "reason": f"failed to read renko: {e}"}

    if "ts" not in df.columns:
        return {"ok": False, "reason": "renko parquet missing ts column"}
    if len(df) < 50:
        return {"ok": False, "reason": f"insufficient renko rows: {len(df)}"}

    cfg = StateSpaceConfig()
    ss = compute_state_space(df, cfg)

    keep = ["ts", "X_raw", "Y_res", "Z_res", "conf_x", "conf_y", "conf_z"]
    out = ss[[c for c in keep if c in ss.columns]].copy()
    out = out.dropna(subset=["X_raw", "Y_res", "Z_res"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return {"ok": True, "rows": len(out), "path": str(out_path)}


def _read_state_space_df() -> pd.DataFrame:
    p = _env_path("DASHBOARD_STATESPACE_PARQUET", "data/live/state_space_latest.parquet")
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(p)
    except Exception:
        return pd.DataFrame()
    if "ts" not in df.columns:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)


def load_state_space_trajectory(window_hours: float = 8.0) -> Dict[str, Any]:
    """Load trajectory and current position for the dashboard."""
    df = _read_state_space_df()
    if df.empty:
        return {"trajectory": [], "current": None}

    now = df["ts"].iloc[-1]
    cutoff = now - pd.Timedelta(hours=max(0.1, float(window_hours)))
    window = df[df["ts"] >= cutoff].copy()

    trajectory: List[Dict[str, Any]] = []
    for _, r in window.iterrows():
        trajectory.append({
            "ts": int(pd.Timestamp(r["ts"]).timestamp()),
            "x": round(float(r["X_raw"]), 6),
            "y": round(float(r["Y_res"]), 6),
            "z": round(float(r["Z_res"]), 6),
        })

    last = df.iloc[-1]
    current = {
        "x": round(float(last["X_raw"]), 6),
        "y": round(float(last["Y_res"]), 6),
        "z": round(float(last["Z_res"]), 6),
        "conf_x": round(float(last.get("conf_x", 0.0)), 6),
        "conf_y": round(float(last.get("conf_y", 0.0)), 6),
        "conf_z": round(float(last.get("conf_z", 0.0)), 6),
    }
    return {"trajectory": trajectory, "current": current}
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dashboard_statespace.py -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add src/quant/execution/dashboard_statespace.py tests/test_dashboard_statespace.py
git commit -m "feat: add state space feature writer and trajectory loader for dashboard"
```

---

### Task 2: Background Density Builder

**Files:**
- Create: `scripts/build_density_images.py`
- Test: `tests/test_build_density_images.py`
- Reference: `scripts/plot_state_space_v01.py` (hexbin rendering logic)

**Step 1: Write the failing test**

```python
# tests/test_build_density_images.py
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class TestDensityBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        n = 500
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "ts": pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"),
            "X_raw": rng.normal(0, 0.3, n).clip(-1, 1),
            "Y_res": rng.normal(0, 0.3, n).clip(-1, 1),
            "Z_res": rng.normal(0, 0.3, n).clip(-1, 1),
        })
        df.to_parquet(self.root / "state_space.parquet", index=False)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_build_density_pngs(self) -> None:
        from scripts.build_density_images import build_density_images
        out_dir = self.root / "density"
        build_density_images(
            state_space_path=self.root / "state_space.parquet",
            out_dir=out_dir,
        )
        for name in ("density_bg_xy.png", "density_bg_xz.png", "density_bg_yz.png"):
            p = out_dir / name
            self.assertTrue(p.exists(), f"Missing: {name}")
            self.assertGreater(p.stat().st_size, 100, f"Empty: {name}")


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_build_density_images.py -v`
Expected: FAIL

**Step 3: Write the implementation**

```python
# scripts/build_density_images.py
"""Build background density PNG images for the dashboard state space heatmaps.

Intended to run daily (cron / scheduler). Reads the full historical state
space parquet and writes three density images: XY, XZ, YZ.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_PAIRS = [
    ("xy", "X_raw", "Y_res", "X: Drift", "Y: Elasticity"),
    ("xz", "X_raw", "Z_res", "X: Drift", "Z: Instability"),
    ("yz", "Y_res", "Z_res", "Y: Elasticity", "Z: Instability"),
]


def build_density_images(
    state_space_path: Path,
    out_dir: Path,
    gridsize: int = 55,
    dpi: int = 150,
    figsize: tuple = (4, 4),
) -> None:
    df = pd.read_parquet(state_space_path)
    for col in ("X_raw", "Y_res", "Z_res"):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    out_dir.mkdir(parents=True, exist_ok=True)

    for tag, xcol, ycol, xlabel, ylabel in _PAIRS:
        x = df[xcol].dropna().to_numpy(dtype=float)
        y = df[ycol].reindex(df[xcol].dropna().index).to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]

        fig, ax = plt.subplots(figsize=figsize, facecolor="#181c24")
        ax.set_facecolor("#181c24")
        ax.hexbin(x, y, gridsize=gridsize, extent=(-1, 1, -1, 1),
                  cmap="inferno", mincnt=1, bins="log")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel(xlabel, color="#ccc", fontsize=8)
        ax.set_ylabel(ylabel, color="#ccc", fontsize=8)
        ax.tick_params(colors="#aaa", labelsize=6)
        for spine in ax.spines.values():
            spine.set_color("#555")
        ax.grid(alpha=0.12, color="#555")

        out_path = out_dir / f"density_bg_{tag}.png"
        fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(),
                    bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Build density background images for dashboard")
    p.add_argument("--state-space", required=True, help="Path to state_space.parquet")
    p.add_argument("--out-dir", default="data/live/density", help="Output directory")
    p.add_argument("--gridsize", type=int, default=55)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    build_density_images(
        state_space_path=Path(args.state_space),
        out_dir=Path(args.out_dir),
        gridsize=args.gridsize,
        dpi=args.dpi,
    )
    print(f"Density images written to {args.out_dir}")


if __name__ == "__main__":
    main()
```

**Step 4: Run test**

Run: `python -m pytest tests/test_build_density_images.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/build_density_images.py tests/test_build_density_images.py
git commit -m "feat: add daily density image builder for dashboard heatmap backgrounds"
```

---

### Task 3: Recent Density Computation

**Files:**
- Modify: `src/quant/execution/dashboard_statespace.py`
- Modify: `tests/test_dashboard_statespace.py`

**Step 1: Add failing test**

Add to `tests/test_dashboard_statespace.py`:

```python
def test_compute_recent_density(self) -> None:
    from quant.execution.dashboard_statespace import refresh_state_space_cache, compute_recent_density
    refresh_state_space_cache()
    density = compute_recent_density(hours=48)
    self.assertIn("xy", density)
    self.assertIn("xz", density)
    self.assertIn("yz", density)
    # Each entry is a list of [bin_a, bin_b, count] triples
    for key in ("xy", "xz", "yz"):
        self.assertIsInstance(density[key], list)
        if density[key]:
            self.assertEqual(len(density[key][0]), 3)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dashboard_statespace.py::TestStateSpaceWriter::test_compute_recent_density -v`
Expected: FAIL

**Step 3: Add `compute_recent_density` to `dashboard_statespace.py`**

Add this function to `src/quant/execution/dashboard_statespace.py`:

```python
def compute_recent_density(hours: float = 4.0, bins: int = 28) -> Dict[str, List]:
    """Compute binned density for recent state space data (for overlay on heatmaps)."""
    df = _read_state_space_df()
    if df.empty:
        return {"xy": [], "xz": [], "yz": []}

    now = df["ts"].iloc[-1]
    cutoff = now - pd.Timedelta(hours=max(0.1, float(hours)))
    recent = df[df["ts"] >= cutoff]
    if recent.empty:
        return {"xy": [], "xz": [], "yz": []}

    import numpy as np

    edges = np.linspace(-1.0, 1.0, bins + 1)
    centers = ((edges[:-1] + edges[1:]) / 2.0).tolist()

    pairs = [
        ("xy", "X_raw", "Y_res"),
        ("xz", "X_raw", "Z_res"),
        ("yz", "Y_res", "Z_res"),
    ]
    result: Dict[str, List] = {}
    for tag, acol, bcol in pairs:
        a = recent[acol].to_numpy(dtype=float)
        b = recent[bcol].to_numpy(dtype=float)
        valid = np.isfinite(a) & np.isfinite(b)
        a, b = a[valid], b[valid]
        if len(a) == 0:
            result[tag] = []
            continue
        ai = np.clip(np.digitize(a, edges) - 1, 0, bins - 1)
        bi = np.clip(np.digitize(b, edges) - 1, 0, bins - 1)
        grid = np.zeros((bins, bins), dtype=int)
        for k in range(len(a)):
            grid[ai[k], bi[k]] += 1
        cells = []
        for i in range(bins):
            for j in range(bins):
                if grid[i, j] > 0:
                    cells.append([round(centers[i], 4), round(centers[j], 4), int(grid[i, j])])
        result[tag] = cells
    return result
```

**Step 4: Run test**

Run: `python -m pytest tests/test_dashboard_statespace.py -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add src/quant/execution/dashboard_statespace.py tests/test_dashboard_statespace.py
git commit -m "feat: add recent density computation for dashboard heatmap overlay"
```

---

### Task 4: State Space API Endpoint

**Files:**
- Modify: `src/quant/execution/webhook_server.py` (add `/api/dashboard/statespace` route)
- Modify: `src/quant/execution/dashboard_state.py` (add regime_scores to chart payload)
- Modify: `tests/test_webhook_dashboard_api.py` (add test)

**Step 1: Add failing test**

Add to `tests/test_webhook_dashboard_api.py`:

```python
def test_statespace_endpoint_shape(self) -> None:
    body = api_dashboard_statespace(window_hours=48)
    self.assertIn("trajectory", body)
    self.assertIn("current", body)
    self.assertIn("recent_density", body)
    self.assertIn("density_bg", body)
```

**Step 2: Run to verify fail**

Run: `python -m pytest tests/test_webhook_dashboard_api.py::WebhookDashboardApiTests::test_statespace_endpoint_shape -v`
Expected: FAIL

**Step 3: Implement the endpoint**

In `src/quant/execution/webhook_server.py`, add import:

```python
from quant.execution.dashboard_statespace import (
    load_state_space_trajectory,
    compute_recent_density,
)
```

Add the endpoint function (after `api_dashboard_chart`):

```python
@app.get("/api/dashboard/statespace")
def api_dashboard_statespace(window_hours: float = 8.0) -> Dict[str, Any]:
    """State space heatmap data: trajectory, current position, density layers."""
    try:
        traj = load_state_space_trajectory(window_hours=float(max(0.1, window_hours)))
        recent = compute_recent_density(hours=min(window_hours, 12.0))
        density_bg = _load_density_bg_images()
        return {
            "ok": True,
            "trajectory": traj.get("trajectory", []),
            "current": traj.get("current"),
            "recent_density": recent,
            "density_bg": density_bg,
            "window_hours": window_hours,
        }
    except Exception as e:
        return {"ok": False, "trajectory": [], "current": None,
                "recent_density": {"xy": [], "xz": [], "yz": []},
                "density_bg": {"xy": None, "xz": None, "yz": None},
                "error": str(e)}


def _load_density_bg_images() -> Dict[str, Optional[str]]:
    """Load pre-computed density PNGs as base64 strings."""
    import base64
    density_dir = Path(os.getenv("DASHBOARD_DENSITY_DIR", "data/live/density"))
    out: Dict[str, Optional[str]] = {}
    for tag in ("xy", "xz", "yz"):
        p = density_dir / f"density_bg_{tag}.png"
        if p.exists():
            data = p.read_bytes()
            out[tag] = f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"
        else:
            out[tag] = None
    return out
```

Also add `regime_scores` and `regime_forecast` to the existing `api_dashboard_chart` return payload. In `src/quant/execution/dashboard_state.py`, add a function:

```python
def build_regime_scores(symbol: str, hours: int = 24 * 14) -> Dict[str, List]:
    """Extract regime_score time series and forward projection for the gradient band."""
    store = RegimeStore()
    end_ts = pd.Timestamp.now("UTC")
    start_ts = end_ts - pd.Timedelta(hours=int(max(1, hours)))
    rows = store.get_history(symbol=symbol, start_ts=start_ts.isoformat(), end_ts=end_ts.isoformat(), limit=20000)
    if not rows:
        return {"scores": [], "forecast": []}

    scores = []
    for r in rows:
        ts = pd.to_datetime(r.get("ts"), utc=True, errors="coerce")
        rs = pd.to_numeric(r.get("regime_score"), errors="coerce")
        if pd.notna(ts) and pd.notna(rs):
            scores.append({"time": int(ts.timestamp()), "score": round(float(rs), 4)})

    return {"scores": scores, "forecast": []}
```

The `forecast` array will be populated using gate confidence horizons data from `get_live_gate_confidence()` in the `api_dashboard_chart` handler. Convert each horizon's `p_trend_voxel` to a score via `2 * p_trend - 1` and append to the forecast list with projected timestamps.

**Step 4: Run tests**

Run: `python -m pytest tests/test_webhook_dashboard_api.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/quant/execution/webhook_server.py src/quant/execution/dashboard_state.py tests/test_webhook_dashboard_api.py
git commit -m "feat: add /api/dashboard/statespace endpoint and regime_scores to chart API"
```

---

### Task 5: Frontend — Regime Gradient Band

**Files:**
- Modify: `src/quant/execution/webhook_server.py` (DASHBOARD_HTML)

Replace the `#shade` canvas overlay with a regime gradient band canvas below the chart.

**Step 1: Update HTML/CSS layout**

In `DASHBOARD_HTML`, change the layout from single-row to two-row grid:

```css
.layout { display: grid; grid-template-columns: 1fr 320px; gap: 1rem; align-items: start; }
.bottom-row { display: grid; grid-template-columns: 1fr 1fr 1fr 280px; gap: 1rem; margin-top: 1rem; }
.regime-band { height: 35px; border-radius: 6px; overflow: hidden; margin-top: 0.5rem; position: relative; }
#regime-canvas { width: 100%; height: 100%; }
```

**Step 2: Add regime gradient canvas element**

Replace the `<canvas id="shade"></canvas>` with a regime band element below the chart card:

```html
<div class="regime-band card" style="grid-column: 1 / -1;">
  <canvas id="regime-canvas"></canvas>
</div>
```

**Step 3: Implement `drawRegimeBand()` JS function**

```javascript
function scoreToColor(score) {
  // score in [-1, 1] -> red(-1) through yellow(0) to green(+1)
  const t = (score + 1.0) / 2.0; // 0..1
  let r, g, b;
  if (t < 0.5) {
    // red to yellow
    const u = t / 0.5;
    r = 247; g = Math.round(118 + (204 - 118) * u); b = Math.round(142 * (1 - u));
  } else {
    // yellow to green
    const u = (t - 0.5) / 0.5;
    r = Math.round(247 * (1 - u) + 46 * u); g = Math.round(204); b = Math.round(113 * u);
  }
  return `rgb(${r}, ${g}, ${b})`;
}

function drawRegimeBand() {
  const canvas = document.getElementById('regime-canvas');
  if (!canvas || !latestPayload) return;
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.parentElement.clientWidth;
  canvas.height = canvas.parentElement.clientHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const scores = latestPayload.regime_scores || [];
  const forecast = latestPayload.regime_forecast || [];
  if (!scores.length) return;

  const tscale = chart.timeScale();
  // Historical scores
  for (let i = 0; i < scores.length - 1; i++) {
    const x0 = tscale.timeToCoordinate(mapTimeForChart(scores[i].time));
    const x1 = tscale.timeToCoordinate(mapTimeForChart(scores[i + 1].time));
    if (x0 == null || x1 == null) continue;
    ctx.fillStyle = scoreToColor(scores[i].score);
    ctx.fillRect(Math.min(x0, x1), 0, Math.max(1, Math.abs(x1 - x0)), canvas.height);
  }

  // Future forecast (fading toward gray)
  for (let i = 0; i < forecast.length; i++) {
    const x0 = tscale.timeToCoordinate(mapTimeForChart(forecast[i].time));
    const x1 = i + 1 < forecast.length
      ? tscale.timeToCoordinate(mapTimeForChart(forecast[i + 1].time))
      : x0 != null ? x0 + 20 : null;
    if (x0 == null || x1 == null) continue;
    const fade = 1.0 - (i / Math.max(1, forecast.length));
    ctx.globalAlpha = 0.3 + 0.7 * fade;
    ctx.fillStyle = scoreToColor(forecast[i].score);
    ctx.fillRect(Math.min(x0, x1), 0, Math.max(1, Math.abs(x1 - x0)), canvas.height);
  }
  ctx.globalAlpha = 1.0;
}
```

**Step 4: Remove old `drawGateShading` and `#shade` canvas**

Delete the `drawGateShading()` function, the `resizeShade()` function, the `confAlpha()` function, and the `liveShadeColor()` function. Remove `<canvas id="shade"></canvas>` from the HTML. Replace calls to `drawGateShading()` with `drawRegimeBand()`.

**Step 5: Commit**

```bash
git add src/quant/execution/webhook_server.py
git commit -m "feat: replace background shading with continuous regime gradient band"
```

---

### Task 6: Frontend — State Space Heatmap Panels

**Files:**
- Modify: `src/quant/execution/webhook_server.py` (DASHBOARD_HTML)

**Step 1: Add bottom row HTML structure**

Add below the regime band:

```html
<div class="bottom-row">
  <div class="card heatmap-card">
    <div class="heatmap-title">Drift vs Elasticity</div>
    <canvas id="heatmap-xy" width="300" height="300"></canvas>
  </div>
  <div class="card heatmap-card">
    <div class="heatmap-title">Drift vs Instability</div>
    <canvas id="heatmap-xz" width="300" height="300"></canvas>
  </div>
  <div class="card heatmap-card">
    <div class="heatmap-title">Elasticity vs Instability</div>
    <canvas id="heatmap-yz" width="300" height="300"></canvas>
  </div>
  <div class="card" id="axis-bars-card">
    <!-- axis bars go here, Task 7 -->
  </div>
</div>
```

Add trajectory window dropdown above the bottom row:

```html
<div style="display:flex; align-items:center; gap:0.5rem; margin-top:0.5rem;">
  <span class="label" style="font-size:0.85rem;">Trajectory:</span>
  <select id="traj-window" class="mono" style="background:var(--card);color:var(--text);border:1px solid #2a3044;border-radius:4px;padding:2px 6px;font-size:0.85rem;">
    <option value="1">1h</option>
    <option value="4">4h</option>
    <option value="8" selected>8h</option>
    <option value="12">12h</option>
    <option value="24">24h</option>
    <option value="48">48h</option>
  </select>
</div>
```

**Step 2: Implement heatmap drawing JS**

```javascript
const bgImages = { xy: null, xz: null, yz: null };
let ssPayload = null;

function loadBgImage(tag, dataUrl) {
  if (!dataUrl) return;
  const img = new Image();
  img.onload = () => { bgImages[tag] = img; };
  img.src = dataUrl;
}

function valToCanvas(val, size) {
  // map [-1, 1] -> [0, size]
  return ((val + 1.0) / 2.0) * size;
}

function drawHeatmap(canvasId, tag, xKey, yKey) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !ssPayload) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // Layer 1: background density image
  if (bgImages[tag]) {
    ctx.drawImage(bgImages[tag], 0, 0, w, h);
  } else {
    ctx.fillStyle = '#181c24';
    ctx.fillRect(0, 0, w, h);
  }

  // Layer 2: recent density overlay
  const rd = (ssPayload.recent_density || {})[tag] || [];
  if (rd.length) {
    const maxCount = Math.max(...rd.map(c => c[2]));
    const binW = w / 28;
    const binH = h / 28;
    for (const [a, b, count] of rd) {
      const alpha = 0.1 + 0.5 * (count / Math.max(1, maxCount));
      ctx.fillStyle = `rgba(120, 180, 255, ${alpha})`;
      const cx = valToCanvas(a, w) - binW / 2;
      const cy = h - valToCanvas(b, h) - binH / 2;
      ctx.fillRect(cx, cy, binW, binH);
    }
  }

  // Layer 3: trajectory
  const traj = ssPayload.trajectory || [];
  if (traj.length > 1) {
    for (let i = 1; i < traj.length; i++) {
      const alpha = 0.05 + 0.95 * (i / traj.length);
      ctx.strokeStyle = `rgba(100, 160, 255, ${alpha})`;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(valToCanvas(traj[i-1][xKey], w), h - valToCanvas(traj[i-1][yKey], h));
      ctx.lineTo(valToCanvas(traj[i][xKey], w), h - valToCanvas(traj[i][yKey], h));
      ctx.stroke();
    }
  }

  // Layer 4: crosshair + NOW marker
  const cur = ssPayload.current;
  if (cur) {
    const cx = valToCanvas(cur[xKey], w);
    const cy = h - valToCanvas(cur[yKey], h);
    // Crosshairs
    ctx.strokeStyle = 'rgba(255, 80, 60, 0.55)';
    ctx.lineWidth = 0.9;
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(w, cy); ctx.stroke();
    ctx.setLineDash([]);
    // Star marker
    ctx.fillStyle = 'red';
    ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = 'yellow';
    ctx.beginPath(); ctx.arc(cx, cy, 3, 0, Math.PI * 2); ctx.fill();
  }
}

function drawAllHeatmaps() {
  drawHeatmap('heatmap-xy', 'xy', 'x', 'y');
  drawHeatmap('heatmap-xz', 'xz', 'x', 'z');
  drawHeatmap('heatmap-yz', 'yz', 'y', 'z');
}
```

**Step 3: Add polling for state space data**

```javascript
async function loadStateSpace() {
  const windowH = document.getElementById('traj-window').value || '8';
  const data = await fetch(`/api/dashboard/statespace?window_hours=${windowH}`).then(r => r.json());
  if (!data.ok) return;
  ssPayload = data;
  // Load background images (only on first load or if null)
  const bg = data.density_bg || {};
  for (const tag of ['xy', 'xz', 'yz']) {
    if (bg[tag] && !bgImages[tag]) loadBgImage(tag, bg[tag]);
  }
  drawAllHeatmaps();
  drawAxisBars();
}

document.getElementById('traj-window').addEventListener('change', loadStateSpace);

// Poll state space every 30s (separate from the 10s chart poll)
loadStateSpace();
setInterval(loadStateSpace, 30000);
```

**Step 4: Commit**

```bash
git add src/quant/execution/webhook_server.py
git commit -m "feat: add state space heatmap panels with trajectory and density overlay"
```

---

### Task 7: Frontend — Axis Status Bars

**Files:**
- Modify: `src/quant/execution/webhook_server.py` (DASHBOARD_HTML)

**Step 1: Add HTML for axis bars card**

Inside `#axis-bars-card`:

```html
<div style="padding: 0.5rem;">
  <div style="color:var(--text);font-size:0.95rem;font-weight:600;margin-bottom:0.75rem;">Current State</div>
  <canvas id="axis-bars" width="250" height="200"></canvas>
</div>
```

**Step 2: Implement `drawAxisBars()` JS**

```javascript
function drawAxisBars() {
  const canvas = document.getElementById('axis-bars');
  if (!canvas || !ssPayload || !ssPayload.current) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const cur = ssPayload.current;
  const axes = [
    { label: 'X Drift', value: cur.x, conf: cur.conf_x, color: '#ff6644' },
    { label: 'Y Elast.', value: cur.y, conf: cur.conf_y, color: '#44bbff' },
    { label: 'Z Instab.', value: cur.z, conf: cur.conf_z, color: '#ffcc33' },
  ];

  const barH = 18;
  const gap = 50;
  const trackLeft = 70;
  const trackRight = w - 60;
  const trackW = trackRight - trackLeft;
  const mid = trackLeft + trackW / 2;

  axes.forEach((a, i) => {
    const y = 30 + i * gap;

    // Label
    ctx.fillStyle = a.color;
    ctx.font = 'bold 11px system-ui';
    ctx.textAlign = 'left';
    ctx.fillText(a.label, 4, y + barH / 2 + 4);

    // Track background
    ctx.fillStyle = '#2a2e38';
    ctx.beginPath();
    ctx.roundRect(trackLeft, y, trackW, barH, 4);
    ctx.fill();

    // Center line
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(mid, y);
    ctx.lineTo(mid, y + barH);
    ctx.stroke();

    // Value bar
    const barEnd = mid + a.value * (trackW / 2);
    ctx.fillStyle = a.color;
    const barX = Math.min(mid, barEnd);
    const barW = Math.abs(barEnd - mid);
    ctx.beginPath();
    ctx.roundRect(barX, y + 2, barW, barH - 4, 3);
    ctx.fill();

    // Value text
    ctx.fillStyle = '#ccc';
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`${a.value >= 0 ? '+' : ''}${a.value.toFixed(3)}  c:${a.conf >= 0 ? '+' : ''}${a.conf.toFixed(3)}`, trackRight + 4, y + barH / 2 + 4);
  });
}
```

**Step 3: Commit**

```bash
git add src/quant/execution/webhook_server.py
git commit -m "feat: add axis status bars panel to dashboard"
```

---

### Task 8: Regime Forecast Data

**Files:**
- Modify: `src/quant/execution/webhook_server.py` (populate `regime_forecast` in chart payload)
- Modify: `tests/test_webhook_dashboard_api.py`

**Step 1: Add failing test**

```python
def test_chart_payload_includes_regime_scores(self) -> None:
    body = api_dashboard_chart(symbol="SOL-USDT", hours=48, max_points=1000)
    self.assertTrue(body.get("ok"))
    self.assertIn("regime_scores", body)
    self.assertIn("regime_forecast", body)
    self.assertIsInstance(body["regime_scores"], list)
    self.assertIsInstance(body["regime_forecast"], list)
```

**Step 2: Run to verify fail**

Run: `python -m pytest tests/test_webhook_dashboard_api.py::WebhookDashboardApiTests::test_chart_payload_includes_regime_scores -v`
Expected: FAIL

**Step 3: Add regime_scores and regime_forecast to chart API response**

In `api_dashboard_chart`, after computing `live_gc`, add:

```python
from quant.execution.dashboard_state import build_regime_scores

regime_score_data = build_regime_scores(symbol=symbol, hours=int(max(1, hours)))

# Build forecast from gate confidence horizons
regime_forecast = []
if live_gc and "horizons" in live_gc:
    now_ts = pd.Timestamp.now("UTC")
    for h in live_gc["horizons"]:
        minutes = h.get("minutes", 0)
        p_trend = h.get("p_trend_voxel")
        if p_trend is not None:
            score = round(2.0 * float(p_trend) - 1.0, 4)
            forecast_ts = int((now_ts + pd.Timedelta(minutes=minutes)).timestamp())
            regime_forecast.append({"time": forecast_ts, "score": score})
```

Add to the return dict: `"regime_scores": regime_score_data.get("scores", [])` and `"regime_forecast": regime_forecast`.

**Step 4: Run tests**

Run: `python -m pytest tests/test_webhook_dashboard_api.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/quant/execution/webhook_server.py src/quant/execution/dashboard_state.py tests/test_webhook_dashboard_api.py
git commit -m "feat: add regime_scores and regime_forecast to chart API for gradient band"
```

---

### Task 9: Integration Test & Polish

**Files:**
- Modify: `src/quant/execution/webhook_server.py` (CSS tweaks, responsive behavior)
- Run: full test suite

**Step 1: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 2: CSS polish**

Ensure the bottom row heatmaps have consistent sizing. Add responsive breakpoint so on narrow screens the bottom row stacks vertically:

```css
.heatmap-card { position: relative; }
.heatmap-card canvas { width: 100%; aspect-ratio: 1; display: block; }
.heatmap-title { color: var(--text); font-size: 0.85rem; text-align: center; margin-bottom: 0.25rem; }
@media (max-width: 1200px) {
  .bottom-row { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 800px) {
  .bottom-row { grid-template-columns: 1fr; }
}
```

**Step 3: Remove old shade hint text**

Remove: `<div class="hint">Gate ON is green, Gate OFF is red. Intensity follows confidence.</div>`

Replace with: `<div class="hint">Regime band: red = countertrend, green = trend. Right side = projected.</div>`

**Step 4: Final test**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/quant/execution/webhook_server.py
git commit -m "polish: responsive layout, CSS cleanup, remove old shade overlay"
```

---

### Task 10: State Space Refresh Scheduler

**Files:**
- Modify: `src/quant/execution/webhook_server.py` (add background thread for periodic state space refresh)

**Step 1: Add background refresh thread**

In `webhook_server.py`, add a startup event that runs `refresh_state_space_cache()` periodically:

```python
import threading
from quant.execution.dashboard_statespace import refresh_state_space_cache

def _state_space_refresh_loop() -> None:
    interval = int(os.getenv("DASHBOARD_SS_REFRESH_SEC", "300"))
    while True:
        try:
            refresh_state_space_cache()
        except Exception as e:
            log.warning("state space refresh failed: %s", e)
        time.sleep(max(60, interval))

@app.on_event("startup")
def _start_state_space_refresh() -> None:
    t = threading.Thread(target=_state_space_refresh_loop, daemon=True)
    t.start()
```

**Step 2: Commit**

```bash
git add src/quant/execution/webhook_server.py
git commit -m "feat: add background state space refresh thread (5min interval)"
```
