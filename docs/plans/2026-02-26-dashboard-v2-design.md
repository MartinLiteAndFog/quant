# Dashboard V2 — State Space Heatmaps, Regime Gradient, Trajectory

**Date:** 2026-02-26
**Status:** Approved

## Problem

The current dashboard shows a Renko chart with binary green/red background shading for regime state and a sidebar with scalar status values. It lacks:

- Visibility into the 3-axis state space position (Drift, Elasticity, Instability)
- Historical trajectory through state space
- Continuous regime score (currently only binary gate ON/OFF)
- Forward-projected regime expectations

## Design

### Layout: Two-Row Grid

```
┌──────────────────────────────────┬──────────┐
│         Renko Chart (wide)       │ Sidebar  │
│   candlesticks + markers + SL/  │ API      │
│   TTP/TP1/TP2 level lines       │ Ticker   │
│                                  │ Position │
│                                  │ Gate     │
│                                  │ Regime   │
│                                  │ Conf     │
│                                  │ Levels   │
├──────────────────────────────────┴──────────┤
│  Regime Gradient Band (~35px, full width)    │
│  red ◄──── yellow ────► green │ future ░░░  │
├────────────┬────────────┬───────┬───────────┤
│  Heatmap   │  Heatmap   │ Heat- │ Axis Bars │
│  X vs Y    │  X vs Z    │ Y vs Z│ X Drift   │
│  Drift vs  │  Drift vs  │ Elast │ Y Elast   │
│  Elasticity│  Instab.   │ vs    │ Z Instab  │
│            │            │ Instab│           │
└────────────┴────────────┴───────┴───────────┘
```

### 1. Regime Gradient Band

Replaces the full-chart green/red background overlay. Thin horizontal band between the Renko chart and the heatmap row.

- **Encoding:** Continuous `regime_score` (-1 to +1) mapped to a smooth red → yellow → green gradient. NOT binary gate ON/OFF.
- **Historical (left of NOW):** Each pixel column corresponds to a regime score at that time point.
- **Future (right of NOW):** Markov forward-projected `p_trend` from `gate_confidence_live`, converted to score via `2 * p_trend - 1`. Color fades toward neutral gray as uncertainty grows with projection horizon. Visually marked (reduced saturation or subtle dashed border).
- **Synced** to the Renko chart's time axis — scrolling/zooming the chart scrolls the regime band.

### 2. State Space Heatmaps (3 panels)

Three 2D projections of the state space, rendered as Canvas panels in the bottom row.

**Panels:**
- Drift vs Elasticity (X vs Y)
- Drift vs Instability (X vs Z)
- Elasticity vs Instability (Y vs Z)

**Three visual layers per panel (back to front):**

1. **Background density (pre-computed daily):** Full historical hexbin density, inferno colormap. Rebuilt once daily from the complete state space parquet. Served as base64 PNG via the dashboard API. Provides the "landscape" of where the system has historically spent time.

2. **Recent density overlay:** Faint semi-transparent heat overlay computed from the last few hours of state space data. Shows where the system has been clustering recently relative to the long-term distribution. Computed server-side on each API call from the live state space parquet.

3. **Trajectory line:** Fading blue polyline showing the path through state space over a configurable time window. Per-segment alpha fades linearly from transparent (oldest) to solid blue (newest). Ends with:
   - Red crosshair lines through current position
   - Red star + yellow inner dot (NOW marker)

**Trajectory window control:** Dropdown selector on the heatmap panel row with options: 1h / 4h / **8h** (default) / 12h / 24h / 48h.

### 3. Axis Status Bars

A card panel alongside the three heatmaps showing the current state vector as horizontal bars.

- **X Drift** (orange `#ff6644`): Bar extends from center proportional to value (-1 to +1)
- **Y Elasticity** (blue `#44bbff`): Same
- **Z Instability** (yellow `#ffcc33`): Same
- Each bar shows the numeric value and confidence score
- Center line at 0, track from -1 to +1
- Updates on each state space API refresh

### 4. Data Pipeline

#### State Space Feature Writer

Background task (runs every ~5 minutes):
1. Reads live Renko parquet (`data/live/renko_latest.parquet`)
2. Computes features via `compute_features()` + `sensors_x/y/z` + `aggregate_axis()`
3. Writes `data/live/state_space_latest.parquet` with columns:
   `ts, X_raw, Y_res, Z_res, conf_x, conf_y, conf_z`

This parquet is also available for backtesting and offline analysis.

#### Background Density Builder

Daily scheduled job:
1. Reads full historical state space parquet
2. Computes hexbin density for all three 2D projections (XY, XZ, YZ)
3. Writes PNG images to `data/live/density_bg_{xy,xz,yz}.png`
4. Dashboard API serves these as base64-encoded images

#### New API Endpoint: `/api/dashboard/statespace`

Returns JSON:
```json
{
  "ok": true,
  "current": { "x": 0.12, "y": -0.34, "z": 0.56, "conf_x": 0.08, "conf_y": -0.21, "conf_z": 0.33 },
  "trajectory": [
    { "ts": 1709000000, "x": 0.10, "y": -0.30, "z": 0.50 },
    ...
  ],
  "recent_density": {
    "xy": [[bin_x, bin_y, count], ...],
    "xz": [[bin_x, bin_z, count], ...],
    "yz": [[bin_y, bin_z, count], ...]
  },
  "density_bg": {
    "xy": "data:image/png;base64,...",
    "xz": "data:image/png;base64,...",
    "yz": "data:image/png;base64,..."
  }
}
```

Query params: `?window_hours=8` (trajectory window)

#### Extended Chart API

`/api/dashboard/chart` response gains:
```json
{
  "regime_scores": [{ "time": 1709000000, "score": 0.45 }, ...],
  "regime_forecast": [{ "time": 1709100000, "score": 0.30 }, ...]
}
```

### 5. Frontend Rendering

All new panels rendered via HTML5 Canvas — no additional JS library dependencies.

**Heatmap panels:**
- Draw pre-computed density PNG as `drawImage()` background
- Overlay recent density as semi-transparent colored rectangles via grid binning
- Draw trajectory as a polyline with per-segment alpha (linearly interpolated)
- Draw crosshairs + NOW marker on top

**Regime band:**
- Canvas element synced to Lightweight Charts time scale
- Each pixel column filled with the gradient color for that time point's score
- Future section drawn with reduced opacity / desaturation
- Responds to chart scroll/zoom events via `subscribeVisibleTimeRangeChange`

**Refresh cadence:**
- Renko chart: polls every 10s (existing, can be increased later)
- State space / heatmaps: polls every 30-60s (separate interval)
- Background density images: cached in browser, refreshed on page load

### 6. Migration

- Remove the full-chart background shade canvas (`#shade` + `drawGateShading`)
- Replace with the regime gradient band below the chart
- Existing sidebar stays unchanged
- All new panels are additive — no existing functionality removed besides the background shading
