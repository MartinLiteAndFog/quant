# Dashboard Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Svelte + Vite trading dashboard at `/dashboard2` alongside the existing dashboard, with a 3-column layout, black theme, polar chart v1, regime band, and animated renko brick filling.

**Architecture:** Svelte SPA served by FastAPI at `/dashboard2`. All data comes from existing `/api/*` endpoints via polling. The existing `/dashboard` remains unchanged.

**Tech Stack:** Svelte 5, Vite, lightweight-charts (npm), Canvas API

**Spec:** `docs/superpowers/specs/2026-04-01-dashboard-redesign-design.md`

---

## File Structure

```
dashboard/
├── package.json                    # deps: svelte, vite, @sveltejs/vite-plugin-svelte, lightweight-charts
├── vite.config.js                  # base: '/dashboard2/', proxy /api -> backend
├── svelte.config.js                # minimal svelte config
├── index.html                      # entry HTML (mount point)
├── src/
│   ├── main.js                     # mount App.svelte to #app
│   ├── app.css                     # CSS variables, reset, global styles
│   ├── App.svelte                  # root 3-column grid layout + refresh logic
│   ├── lib/
│   │   ├── api.js                  # fetch wrappers with error handling
│   │   ├── stores.js               # chartStore, statusStore, statespaceStore, equityEventsStore
│   │   ├── chartHelpers.js         # brick-mode time mapping, level lines, TTP trail, segment mapping
│   │   └── colors.js               # scoreToColor, theme constants
│   └── components/
│       ├── StatsPanel.svelte       # left sidebar stats
│       ├── PriceChart.svelte       # Lightweight Charts candlestick + all overlays
│       ├── RegimeBand.svelte       # 20px canvas strip below chart
│       ├── AxisBars.svelte         # 3 horizontal center-zero bars
│       ├── PolarChart.svelte       # radar chart with trail
│       └── EquityCurve.svelte      # full-width equity with time tabs
└── public/
    └── favicon.ico
```

Backend change: one small addition to `src/quant/execution/webhook_server.py` — mount static files and SPA route at `/dashboard2`.

---

## Task 1: Scaffold Svelte + Vite project

**Files:**
- Create: `dashboard/package.json`
- Create: `dashboard/vite.config.js`
- Create: `dashboard/svelte.config.js`
- Create: `dashboard/index.html`
- Create: `dashboard/src/main.js`
- Create: `dashboard/src/app.css`
- Create: `dashboard/src/App.svelte`
- Create: `dashboard/public/favicon.ico`

- [ ] **Step 1: Create `dashboard/package.json`**

```json
{
  "name": "quant-dashboard",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "lightweight-charts": "^4.2.0"
  },
  "devDependencies": {
    "@sveltejs/vite-plugin-svelte": "^4.0.0",
    "svelte": "^5.0.0",
    "vite": "^6.0.0"
  }
}
```

- [ ] **Step 2: Create `dashboard/vite.config.js`**

```js
import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  base: '/dashboard2/',
  server: {
    proxy: {
      '/api': 'http://localhost:8000'
    }
  }
});
```

- [ ] **Step 3: Create `dashboard/svelte.config.js`**

```js
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

export default {
  preprocess: vitePreprocess()
};
```

- [ ] **Step 4: Create `dashboard/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Quant Dashboard</title>
</head>
<body>
  <div id="app"></div>
  <script type="module" src="/src/main.js"></script>
</body>
</html>
```

- [ ] **Step 5: Create `dashboard/src/app.css`**

```css
:root {
  --bg: #000000;
  --text: #e0e0e8;
  --muted: #555555;
  --green: #2ecc71;
  --red: #f7768e;
  --blue: #7aa2f7;
  --purple: #bb9af7;
  --amber: #e0af68;
  --grid: #111118;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body {
  height: 100%;
  background: var(--bg);
  color: var(--text);
  font-family: system-ui, -apple-system, sans-serif;
  overflow: hidden;
}

.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
```

- [ ] **Step 6: Create `dashboard/src/main.js`**

```js
import './app.css';
import App from './App.svelte';
import { mount } from 'svelte';

const app = mount(App, { target: document.getElementById('app') });

export default app;
```

- [ ] **Step 7: Create `dashboard/src/App.svelte`**

Skeleton layout with 3-column grid + bottom equity bar. Components will be placeholder `<div>` elements that get replaced in later tasks.

```svelte
<script>
  import StatsPanel from './components/StatsPanel.svelte';
  import PriceChart from './components/PriceChart.svelte';
  import RegimeBand from './components/RegimeBand.svelte';
  import AxisBars from './components/AxisBars.svelte';
  import PolarChart from './components/PolarChart.svelte';
  import EquityCurve from './components/EquityCurve.svelte';
</script>

<div class="dashboard">
  <div class="stats-panel">
    <StatsPanel />
  </div>
  <div class="chart-area">
    <PriceChart />
    <RegimeBand />
  </div>
  <div class="right-panel">
    <AxisBars />
    <PolarChart />
  </div>
  <div class="equity-bar">
    <EquityCurve />
  </div>
</div>

<style>
  .dashboard {
    display: grid;
    grid-template-columns: 220px 1fr 240px;
    grid-template-rows: 1fr auto;
    height: 100vh;
    gap: 0;
  }
  .stats-panel { grid-column: 1; grid-row: 1; overflow-y: auto; }
  .chart-area { grid-column: 2; grid-row: 1; display: flex; flex-direction: column; min-height: 0; }
  .right-panel { grid-column: 3; grid-row: 1; display: flex; flex-direction: column; }
  .equity-bar { grid-column: 1 / -1; grid-row: 2; }

  @media (max-width: 1200px) {
    .dashboard {
      grid-template-columns: 220px 1fr;
      grid-template-rows: 1fr auto auto;
    }
    .right-panel { grid-column: 1 / -1; grid-row: 2; flex-direction: row; }
    .equity-bar { grid-row: 3; }
  }
  @media (max-width: 800px) {
    .dashboard {
      grid-template-columns: 1fr;
      grid-template-rows: auto 1fr auto auto;
      overflow-y: auto;
    }
    .stats-panel { grid-column: 1; grid-row: 1; }
    .chart-area { grid-column: 1; grid-row: 2; min-height: 400px; }
    .right-panel { grid-column: 1; grid-row: 3; flex-direction: row; }
    .equity-bar { grid-column: 1; grid-row: 4; }
  }
</style>
```

- [ ] **Step 8: Create placeholder component files**

Create all 6 component files with minimal placeholder content so the app compiles:

Each file (`StatsPanel.svelte`, `PriceChart.svelte`, `RegimeBand.svelte`, `AxisBars.svelte`, `PolarChart.svelte`, `EquityCurve.svelte`) contains:

```svelte
<div class="placeholder">ComponentName</div>
<style>
  .placeholder { color: #555; font-size: 0.8rem; padding: 1rem; }
</style>
```

- [ ] **Step 9: Install dependencies and verify dev server**

```bash
cd dashboard && npm install
```

Run: `npm run dev`
Expected: Vite dev server starts, page loads at `http://localhost:5173/dashboard2/` showing 3-column grid with placeholder text.

- [ ] **Step 10: Commit**

```bash
git add dashboard/
git commit -m "feat: scaffold Svelte + Vite dashboard project"
```

---

## Task 2: API layer and Svelte stores

**Files:**
- Create: `dashboard/src/lib/api.js`
- Create: `dashboard/src/lib/stores.js`
- Create: `dashboard/src/lib/colors.js`

- [ ] **Step 1: Create `dashboard/src/lib/api.js`**

```js
const BASE = '';

export async function fetchJson(url) {
  const res = await fetch(BASE + url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export function fetchChart(hours = 24 * 14, maxPoints = 4000) {
  return fetchJson(`/api/dashboard/chart?hours=${hours}&max_points=${maxPoints}`);
}

export function fetchStatus() {
  return fetchJson('/api/status');
}

export function fetchPosition() {
  return fetchJson('/api/position');
}

export function fetchPerformance(venue = 'kucoin') {
  return fetchJson(`/api/dashboard/performance?venue=${venue}`);
}

export function fetchStatespace(windowHours = 8) {
  return fetchJson(`/api/dashboard/statespace?window_hours=${windowHours}`);
}

export function fetchEquityEvents(range = '7d') {
  return fetchJson(`/api/equity/events?range=${range}`);
}
```

- [ ] **Step 2: Create `dashboard/src/lib/stores.js`**

```js
import { writable } from 'svelte/store';
import { fetchChart, fetchStatus, fetchPosition, fetchPerformance, fetchStatespace, fetchEquityEvents } from './api.js';

export const chartStore = writable(null);
export const statusStore = writable({ status: null, position: null, performance: null });
export const statespaceStore = writable(null);
export const equityEventsStore = writable(null);

// Hardcoded for v1. In production, inject via window.__DASHBOARD_CONFIG__
// or a /api/config endpoint matching spec §Environment Variables.
const CHART_MS = 4000;
const STATUS_MS = 10000;
const SS_MS = 10000;
const EQUITY_MS = 30000;

let chartInFlight = false;
let statusInFlight = false;
let ssInFlight = false;

export async function refreshChart() {
  if (chartInFlight) return;
  chartInFlight = true;
  try {
    const data = await fetchChart();
    if (data.ok) chartStore.set(data);
  } catch (e) { /* silent */ }
  finally { chartInFlight = false; }
}

export async function refreshStatus() {
  if (statusInFlight) return;
  statusInFlight = true;
  try {
    const [s, p, perf] = await Promise.all([
      fetchStatus(), fetchPosition(), fetchPerformance()
    ]);
    statusStore.set({ status: s, position: p, performance: perf });
  } catch (e) { /* silent */ }
  finally { statusInFlight = false; }
}

export async function refreshStatespace() {
  if (ssInFlight) return;
  ssInFlight = true;
  try {
    const data = await fetchStatespace();
    if (data.ok) statespaceStore.set(data);
  } catch (e) { /* silent */ }
  finally { ssInFlight = false; }
}

export async function refreshEquityEvents(range = '7d') {
  try {
    const data = await fetchEquityEvents(range);
    if (data.ok) equityEventsStore.set(data);
  } catch (e) { /* silent */ }
}

export async function refreshAll() {
  await Promise.all([refreshChart(), refreshStatus(), refreshStatespace()]);
}

export function startPolling() {
  refreshAll();
  const t1 = setInterval(refreshChart, CHART_MS);
  const t2 = setInterval(refreshStatus, STATUS_MS);
  const t3 = setInterval(refreshStatespace, SS_MS);
  return () => { clearInterval(t1); clearInterval(t2); clearInterval(t3); };
}
```

- [ ] **Step 3: Create `dashboard/src/lib/colors.js`**

```js
export function scoreToColor(score, alpha = 1.0) {
  const t = (Math.max(-1, Math.min(1, score)) + 1.0) / 2.0;
  let r, g, b;
  if (t < 0.5) {
    const u = t / 0.5;
    r = 247; g = Math.round(118 + 86 * u); b = Math.round(142 * (1 - u));
  } else {
    const u = (t - 0.5) / 0.5;
    r = Math.round(247 * (1 - u) + 46 * u); g = 204; b = Math.round(113 * u);
  }
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
```

- [ ] **Step 4: Wire stores into App.svelte**

Update `App.svelte` `<script>` block to call `startPolling()` on mount and set up visibility/focus/pull-to-refresh handlers.

```svelte
<script>
  import { onMount, onDestroy } from 'svelte';
  import { startPolling, refreshAll } from './lib/stores.js';
  // ... component imports ...

  let stopPolling;
  let pullStartY = null;
  let pullTriggered = false;
  let showRefreshIndicator = false;

  onMount(() => {
    stopPolling = startPolling();

    document.addEventListener('visibilitychange', onVisibility);
    window.addEventListener('focus', onFocus);
    window.addEventListener('pageshow', onPageshow);
    window.addEventListener('touchstart', onTouchStart, { passive: true });
    window.addEventListener('touchmove', onTouchMove, { passive: true });
    window.addEventListener('touchend', onTouchEnd, { passive: true });
  });

  onDestroy(() => {
    if (stopPolling) stopPolling();
    document.removeEventListener('visibilitychange', onVisibility);
    window.removeEventListener('focus', onFocus);
    window.removeEventListener('pageshow', onPageshow);
    window.removeEventListener('touchstart', onTouchStart);
    window.removeEventListener('touchmove', onTouchMove);
    window.removeEventListener('touchend', onTouchEnd);
  });

  function onVisibility() {
    if (document.visibilityState === 'visible') refreshAll();
  }
  function onFocus() { refreshAll(); }
  function onPageshow() { refreshAll(); }

  function onTouchStart(ev) {
    if (!ev.touches || ev.touches.length !== 1) return;
    const top = window.scrollY || document.documentElement.scrollTop || 0;
    if (top <= 2) { pullStartY = ev.touches[0].clientY; pullTriggered = false; }
  }
  function onTouchMove(ev) {
    if (pullStartY == null || pullTriggered) return;
    if (!ev.touches || ev.touches.length !== 1) return;
    const dy = ev.touches[0].clientY - pullStartY;
    if (dy >= 90) {
      pullTriggered = true;
      showRefreshIndicator = true;
      refreshAll().then(() => { setTimeout(() => { showRefreshIndicator = false; }, 600); });
    }
  }
  function onTouchEnd() { pullStartY = null; pullTriggered = false; }
</script>
```

Add a refresh indicator bar at the top of the template:

```svelte
{#if showRefreshIndicator}
  <div class="refresh-bar"></div>
{/if}
```

```css
.refresh-bar {
  position: fixed; top: 0; left: 0; right: 0;
  height: 3px; background: var(--blue);
  animation: fadeOut 0.6s ease forwards;
  z-index: 999;
}
@keyframes fadeOut { to { opacity: 0; } }
```

- [ ] **Step 5: Verify stores load data**

Run dev server, open browser, check Network tab — should see `/api/dashboard/chart`, `/api/status`, `/api/position`, `/api/dashboard/performance`, `/api/dashboard/statespace` requests firing at correct intervals.

- [ ] **Step 6: Commit**

```bash
git add dashboard/src/lib/
git commit -m "feat: add API layer, stores, and polling with refresh"
```

---

## Task 3: Chart helpers (brick-mode mapping)

**Files:**
- Create: `dashboard/src/lib/chartHelpers.js`

- [ ] **Step 1: Create `dashboard/src/lib/chartHelpers.js`**

Port all chart helper functions from the current inline JS. This file contains pure functions with no DOM dependency.

Functions to port (exact logic from `webhook_server.py` lines 2075–2239):
- `buildTimeMapFromBars(bars)` → returns `{ map, timeAxis }`
- `mapTimeForChart(t, timeMap, barsRaw, brickBaseTs)` → mapped time or null
- `mapTimeAsOfForChart(t, timeAxis, brickBaseTs)` → mapped time or null
- `mapBarsForChart(bars, brickBaseTs)` → bars with remapped time
- `mapMarkersForChart(markers, timeAxis, brickBaseTs)` → markers with remapped time
- `mapLineForChart(points, timeMap, barsRaw, brickBaseTs)` → line data with remapped time
- `mapSegmentForChart(seg, timeMap, barsRaw, brickBaseTs)` → 2-point line data
- `levelLineData(bars, level)` → full-span horizontal line
- `levelLineFromEntry(bars, level, levels, timeMap, barsRaw, brickBaseTs)` → line from entry time
- `buildUnifiedExitLine(bars, levels, timeMap, barsRaw, brickBaseTs)` → `{data, mode}`
- `buildTTPTrail(bars, levels, ttpTrailPct, timeMap, barsRaw, brickBaseTs)` → trailing stop points
- `fmtNum(v)` → formatted string

Also port:
- `liveRegimeScore(payload)` → score from gate_confidence.selected_p_trend, returns `[-1,1]` or null

Constants: `BRICK_BASE_TS = 1704067200`

Each function takes explicit parameters instead of relying on module-level state. The caller (PriceChart.svelte) manages `timeMap`, `barsRaw`, and `brickBaseTs`.

- [ ] **Step 2: Verify helpers are importable**

Add a temporary `console.log` in `App.svelte` importing and calling `mapBarsForChart([], BRICK_BASE_TS)` to verify no errors.

- [ ] **Step 3: Commit**

```bash
git add dashboard/src/lib/chartHelpers.js
git commit -m "feat: port brick-mode chart helper functions"
```

---

## Task 4: StatsPanel component

**Files:**
- Modify: `dashboard/src/components/StatsPanel.svelte`

- [ ] **Step 1: Implement StatsPanel.svelte**

Subscribe to `chartStore` and `statusStore`. Render 5 sections: STATUS, POSITION, CAPITAL, PERFORMANCE, LEVELS. No card backgrounds, muted uppercase headers, monospace values.

Data mapping (from spec):
- STATUS: price from last bar, Kraken from `kraken_metrics`, regime from `day_regime_state`
- POSITION: from `statusStore.position`, Kraken from `kraken_metrics`
- CAPITAL: balance from `statusStore.status.balance`, Kraken from `kraken_metrics`
- PERFORMANCE: all fields from `statusStore.performance`
- LEVELS: from `chartStore.levels`

Positive values green, negative red. "—" for missing data.

- [ ] **Step 2: Verify in browser**

Dev server running, data appears in left sidebar with correct values.

- [ ] **Step 3: Commit**

```bash
git add dashboard/src/components/StatsPanel.svelte
git commit -m "feat: implement StatsPanel with STATUS/POSITION/CAPITAL/PERFORMANCE/LEVELS"
```

---

## Task 5: PriceChart component

**Files:**
- Modify: `dashboard/src/components/PriceChart.svelte`

- [ ] **Step 1: Implement PriceChart.svelte**

This is the largest component. Steps:

1. Import `lightweight-charts` and `chartHelpers`
2. On mount: create chart with black background config, add all series (candlestick, SL, TTP, Entry, TP1, TP2, fibLong, fibMid, fibShort, priceLine)
3. Use `ResizeObserver` on container for responsive sizing
4. Subscribe to `chartStore` — on each update:
   - Build time map from bars
   - Map bars for chart (brick mode)
   - Set candlestick data + markers
   - Set level lines (SL, TTP trail, Entry, TP1, TP2)
   - Set fibo lines (long, mid, short)
   - Set live mid-price line
   - Manage trade segment series (add/remove dynamically based on signature change)
   - Preserve visible range on updates (first load fits content, subsequent preserves range)

5. Time axis: brick-index formatter mapping back to real timestamps
6. Export chart's `timeScale` reference for RegimeBand synchronization

Chart config from spec: black bg `#000000`, grid `#111118`, crosshair magnet mode, all series colors matching current dashboard.

- [ ] **Step 2: Implement animated last-brick filling**

Add a canvas overlay (positioned absolute over the chart) that draws a semi-transparent fill on the last candlestick area showing brick completion progress.

Renko brick fill logic (directional):
- Get the last completed brick from `bars[-1]` and the live price from `statusStore` ticker mid or last bar close
- Determine brick direction from the last bar: if `close > open` it's an up-brick, else down-brick
- Brick size: `abs(bars[-1].close - bars[-1].open)` (renko bars have uniform size)
- For up-brick (next brick completes upward): `progress = clamp((livePrice - lastClose) / brickSize, 0, 1)`
- For down-brick (next brick completes downward): `progress = clamp((lastClose - livePrice) / brickSize, 0, 1)`
- Draw as a semi-transparent vertical fill bar (green for up, red for down, alpha 0.25) over the last candle's x-position, height proportional to `progress`

Use `chart.timeScale().timeToCoordinate()` and `chart.priceScale('right').priceToCoordinate()` to position the overlay correctly. Update on each poll cycle.

- [ ] **Step 3: Verify in browser**

Chart renders with all overlays — candles, level lines, fibo lines, trade segments, markers. Animated brick fill visible on last candle.

- [ ] **Step 4: Commit**

```bash
git add dashboard/src/components/PriceChart.svelte
git commit -m "feat: implement PriceChart with all overlays and brick animation"
```

---

## Task 6: RegimeBand component

**Files:**
- Modify: `dashboard/src/components/RegimeBand.svelte`

- [ ] **Step 1: Implement RegimeBand.svelte**

- 20px tall `<canvas>`, full width of chart column
- Subscribe to `chartStore` for `regime_scores` and `regime_forecast`
- Need access to chart's `timeScale()` for coordinate mapping — accept as a prop or bind from parent
- On each update: clear canvas, draw regime score bars aligned with chart time axis
- Draw forecast with fading opacity: `alpha = 0.3 + 0.7 * (1 - i / forecast.length)` where `i=0` is nearest forecast point
- Use `scoreToColor` from `colors.js`
- `ResizeObserver` on parent for width changes
- Re-draw on chart visible range changes (subscribe to `timeScale().subscribeVisibleTimeRangeChange`)

- [ ] **Step 2: Wire RegimeBand to PriceChart's timeScale**

In `App.svelte`, pass chart's timeScale reference down to RegimeBand via a binding or store.

- [ ] **Step 3: Verify in browser**

Regime band renders below chart, colors align with chart time axis, scrolling chart scrolls regime band.

- [ ] **Step 4: Commit**

```bash
git add dashboard/src/components/RegimeBand.svelte
git commit -m "feat: implement RegimeBand with time-synced regime scores"
```

---

## Task 7: AxisBars component

**Files:**
- Modify: `dashboard/src/components/AxisBars.svelte`

- [ ] **Step 1: Implement AxisBars.svelte**

- Subscribe to `statespaceStore`
- Draw 3 horizontal center-zero bars on a `<canvas>`:
  - X Drift: `#ff6644`
  - Y Elasticity: `#44bbff`
  - Z Instability: `#ffcc33`
- Each bar: 18px tall, center line at zero, fill extends left/right proportional to value `[-1, 1]`
- Value + confidence text to the right of each bar
- Total height ~80px
- Port drawing logic from current `drawAxisBars()` (webhook_server.py lines 1755-1799)

- [ ] **Step 2: Verify in browser**

Axis bars appear at top of right column with correct values.

- [ ] **Step 3: Commit**

```bash
git add dashboard/src/components/AxisBars.svelte
git commit -m "feat: implement AxisBars with drift/elasticity/instability"
```

---

## Task 8: PolarChart component

**Files:**
- Modify: `dashboard/src/components/PolarChart.svelte`

- [ ] **Step 1: Implement PolarChart.svelte**

- Square canvas, fills available width in right column
- Subscribe to `statespaceStore`
- Draw radar chart with 3 spokes at 120° intervals:
  - Spoke 0 (top, 270°): Drift (x)
  - Spoke 1 (bottom-right, 30°): Elasticity (y)
  - Spoke 2 (bottom-left, 150°): Instability (z)
- Concentric guide circles at r=0.25, 0.5, 0.75, 1.0 — color `#1a1a2e`
- Spoke lines from center to edge — color `#1a1a2e`
- Axis labels at spoke endpoints

Current position:
- Map each value: `r = abs(value)`, clamped to [0, 1]
- Convert to canvas coordinates: `cx + r * cos(angle)`, `cy + r * sin(angle)`
- Draw filled polygon connecting 3 points, semi-transparent fill (alpha 0.3), stroke alpha 0.8
- Per-axis vertex coloring: positive values use axis color (Drift `#ff6644`, Elast `#44bbff`, Instab `#ffcc33`), negative values use a desaturated/muted version (reduce saturation by 50%, lower brightness)
- Overall polygon fill tint: derived from regime score via `scoreToColor` from `chartStore.day_regime_state` or `liveRegimeScore` — green-ish for trend, red-ish for countertrend

Trail (last 20 trajectory points):
- `statespaceStore.trajectory.slice(-20)`
- Each drawn as polygon with opacity: `alpha = 0.05 + 0.15 * (i / N)` where i=0 is oldest
- No animation in v1

- [ ] **Step 2: Verify in browser**

Polar chart shows radar with 3 axes, current position polygon, and fading trail.

- [ ] **Step 3: Commit**

```bash
git add dashboard/src/components/PolarChart.svelte
git commit -m "feat: implement PolarChart v1 with radar axes and trail"
```

---

## Task 9: EquityCurve component

**Files:**
- Modify: `dashboard/src/components/EquityCurve.svelte`

- [ ] **Step 1: Implement EquityCurve.svelte**

- Full-width, ~130px tall
- Canvas-based line chart
- Subscribe to `chartStore` for default equity data (`equity_total`, `equity_components`)

Default view:
- Combined equity line from `equity_total` (fallback to `equity_combined` if `equity_total` is empty/missing — API may send either field name)
- Venue stacked fills: KuCoin `rgba(122,162,247,0.3)`, Kraken `rgba(255,158,100,0.35)`
- Percentage change badge top-right
- Venue values as labels: "KuCoin: $X | Kraken: $Y"

Time-range tabs:
- Buttons: 24h, 7d, 30d, All
- On click: fetch `/api/equity/events?range={range}` via `refreshEquityEvents(range)`
- Switch to event-based rendering: line chart from events array
- Clicking active tab deselects → returns to default view

Mouse hover: show detail at cursor position (time, equity value).

- [ ] **Step 2: Verify in browser**

Equity curve renders at bottom, venue fills visible, tabs switch data source. Hover shows details.

- [ ] **Step 3: Commit**

```bash
git add dashboard/src/components/EquityCurve.svelte
git commit -m "feat: implement EquityCurve with venue breakdown and range tabs"
```

---

## Task 10: Backend — serve dashboard2

**Files:**
- Modify: `src/quant/execution/webhook_server.py` (add ~15 lines near the existing `/dashboard` endpoint)

- [ ] **Step 1: Run impact analysis**

```bash
# Check what depends on the dashboard endpoint area
```

Run `gitnexus_impact` on `dashboard` function before modifying.

- [ ] **Step 2: Add static mount and SPA route**

Add after the existing `dashboard()` endpoint (around line 2612):

```python
from starlette.responses import FileResponse

_DASHBOARD2_DIST = Path(os.getenv(
    "DASHBOARD2_DIST",
    str(Path(__file__).resolve().parent.parent.parent.parent / "dashboard" / "dist"),
))

if _DASHBOARD2_DIST.exists() and (_DASHBOARD2_DIST / "assets").exists():
    app.mount(
        "/dashboard2/assets",
        StaticFiles(directory=str(_DASHBOARD2_DIST / "assets")),
        name="dashboard2-assets",
    )

@app.get("/dashboard2")
@app.get("/dashboard2/{path:path}")
def dashboard2(path: str = ""):
    index = _DASHBOARD2_DIST / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return HTMLResponse("<h1>Dashboard not built</h1><p>Run: cd dashboard && npm run build</p>", status_code=404)
```

Note: `Path(__file__)` resolves relative to `webhook_server.py` at `src/quant/execution/webhook_server.py`, so `.parent.parent.parent.parent` reaches the repo root. `DASHBOARD2_DIST` env var overrides for non-standard deployments.

- [ ] **Step 3: Build and test production serving**

```bash
cd dashboard && npm run build
```

Start the Python backend, navigate to `http://localhost:8000/dashboard2` — should load the Svelte SPA. Verify existing `http://localhost:8000/dashboard` still works.

- [ ] **Step 4: Commit**

```bash
git add src/quant/execution/webhook_server.py
git commit -m "feat: serve Svelte dashboard at /dashboard2 alongside existing dashboard"
```

Note: Do NOT commit `dashboard/dist/` — it is a build artifact. Add `dashboard/dist/` to `.gitignore`. In production, run `cd dashboard && npm run build` as a deploy step.

---

## Task 11: Integration testing and polish

- [ ] **Step 1: Full integration test**

With Python backend running:
1. Open `/dashboard` — existing dashboard works unchanged
2. Open `/dashboard2` — new dashboard loads with all components
3. Verify all data populates: stats, chart, regime band, axis bars, polar chart, equity curve
4. Verify polling: data refreshes every 4s (chart), 10s (status, statespace), 30s (equity)
5. Test visibility refresh: switch tabs away and back — data refreshes immediately
6. Test equity range tabs: click 24h, 7d, 30d, All — data changes
7. Test responsive: resize browser to <1200px and <800px — layout adapts

- [ ] **Step 2: Fix any visual issues**

Adjust spacing, font sizes, colors as needed to match the spec's clean black aesthetic.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "fix: integration polish for dashboard2"
```
