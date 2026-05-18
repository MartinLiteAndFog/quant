# Dashboard Redesign — Svelte SPA

**Date:** 2026-04-01
**Status:** Draft

## Goal

Build a new Svelte + Vite frontend alongside the existing inline dashboard. The new dashboard lives at `/dashboard2` while the current `/dashboard` remains operational. Once the new dashboard is validated, the old one can be removed in a future step. The new dashboard is a professional, high-performance trading dashboard with a clean black aesthetic, no visual clutter, and a foundation for future animation and visualization work.

## Scope

### In Scope

- Svelte + Vite project scaffolding in `dashboard/`
- 3-column layout: stats (left) | chart (center) | state-space (right) + equity (bottom)
- Remove circular chart mask — rectangular, edge-to-edge chart
- Unified black background (`#000`) across entire app and chart
- Stats panel moved to left sidebar (220px), no card backgrounds
- Chart preserves all current series: candlesticks, SL/TTP/Entry/TP1/TP2 lines, Fibonacci overlays, trade segment lines, trade markers, live mid-price line
- Renko brick chart with animated "filling" of the last incomplete brick
- Regime band v1 — thin 20px strip below chart using existing regime scores
- Polar chart v1 — radar chart combining 3 state-space axes with simple trail
- Axis bars — compact horizontal bars at top of right column
- Equity curve — full-width bottom section with time-range tabs (24h, 7d, 30d, All)
- Pull-to-refresh and visibility/focus-based refresh (improved from current broken implementation)
- FastAPI static serving of built Svelte output at `/dashboard2`
- Vite dev proxy to Python backend
- Existing `/dashboard` remains unchanged and operational during development

### Out of Scope

- Full animated radial chart with temporal particle trails (future iteration)
- EUR conversion for equity
- Circular design adaptation (later)
- Recent fills list (removed from dashboard for now)
- Manual orders panel (removed for now)
- Trajectory controls — chart range, trajectory window, time cursor slider (removed for now)
- 3 separate 2D density heatmaps (replaced by polar chart v1)
- Density background PNGs (no longer needed)
- Query-string overrides for refresh intervals (defer)
- WebSocket real-time data (polling works for now)
- Removing the old `/dashboard` endpoint (kept operational alongside new dashboard)

## Architecture

### Project Structure

```
quant-main/
├── src/quant/execution/
│   └── webhook_server.py          # keeps API endpoints, loses DASHBOARD_HTML
├── dashboard/                     # Svelte app
│   ├── package.json
│   ├── vite.config.js
│   ├── svelte.config.js
│   ├── src/
│   │   ├── App.svelte             # root layout (3-column grid + bottom bar)
│   │   ├── app.css                # global theme (CSS variables, reset)
│   │   ├── main.js                # mount point
│   │   ├── lib/
│   │   │   ├── api.js             # fetch helpers for /api/* endpoints
│   │   │   └── stores.js          # Svelte stores for shared state
│   │   └── components/
│   │       ├── StatsPanel.svelte
│   │       ├── PriceChart.svelte
│   │       ├── RegimeBand.svelte
│   │       ├── PolarChart.svelte
│   │       ├── AxisBars.svelte
│   │       └── EquityCurve.svelte
│   └── public/
│       └── favicon.ico
```

### Data Flow — Stores and API Endpoints

Three Svelte writable stores drive the dashboard. Each polls a specific API endpoint.

#### `chartStore`
- **Endpoint:** `GET /api/dashboard/chart?hours={hours}&max_points=3000`
- **Poll interval:** ~4000ms (configurable via env `DASHBOARD_UI_REFRESH_MS`)
- **Response fields used:**
  - `bars` — array of `{time, open, high, low, close}` for candlestick chart
  - `markers` — array of `{time, position, shape, color, text}` for trade markers
  - `levels` — object with `side`, `entry_px`, `sl`, `ttp`, `tp1`, `tp2`, `mode`, `entry_bar_ts`
  - `ttp_trail_pct` — float for trailing take-profit percentage
  - `regime` — `{spans, latest: {regime_state, confidence, gate_on}}`
  - `regime_scores` — array of `{time, score}` for regime band rendering
  - `regime_forecast` — array of `{time, score}` for regime forecast rendering
  - `confidence` — float 0-1
  - `day_regime_state` — string (e.g. "trend", "countertrend")
  - `fibo` — `{long: [{time, value}], mid: [{time, value}], short: [{time, value}]}` for Fibonacci overlays
  - `segments` — array of `{from_time, to_time, from_price, to_price, color, positive}` for trade segment lines
  - `equity_components` — array of `{key, label, kind, points: [{time, equity}]}` (kucoin, kraken)
  - `equity_total` / `equity_combined` — array of `{time, equity}` for combined equity line
  - `open_position` — `{side, entry_time, entry_price, sl, mode}` or null
  - `kraken_metrics` — object with kraken position/capital info
  - `renko_health` — `{ok, bars, last_ts, age_sec}` for renko data freshness
- **Consumed by:** PriceChart (bars, markers, levels, fibo, segments, renko_health), RegimeBand (regime_scores, regime_forecast), StatsPanel (levels, regime, confidence, open_position, kraken_metrics), EquityCurve (equity_components, equity_total)

#### `statusStore`
- **Endpoints:** `GET /api/status` + `GET /api/position` + `GET /api/dashboard/performance`
- **Poll interval:** ~10000ms (status/position change slowly)
- **Response fields used:**
  - From `/api/status`: `api_configured`, `balance` (USDT), `ok`
  - From `/api/position`: `position` (qty), `side`, `leverage`
  - From `/api/dashboard/performance`: `pnl_pct`, `winrate`, `monthly_growth`, `average_gain`, `trade_count`, `winning_trade_count`, `losing_trade_count`
- **Consumed by:** StatsPanel (STATUS, POSITION, CAPITAL, PERFORMANCE sections)
- **Store shape:** `{ status: {...}, position: {...}, performance: {...} }` — three fetches merged into one store object
- **Note:** Kraken status/position is derived from `kraken_metrics` in `chartStore`. There is no separate Kraken status API. If `kraken_metrics` is empty/null, Kraken columns show "—".
- **Note:** PERFORMANCE section reflects KuCoin venue only (`/api/dashboard/performance` defaults `venue=kucoin`). Multi-venue performance aggregation is out of scope for v1.

#### `statespaceStore`
- **Endpoint:** `GET /api/dashboard/statespace?window_hours=8`
- **Poll interval:** ~10000ms
- **Response fields used:**
  - `current` — object with `x` (drift), `y` (elasticity), `z` (instability), `conf_x`, `conf_y`, `conf_z`
  - `trajectory` — array of objects with same fields, ordered by time
- **Consumed by:** AxisBars, PolarChart
- **Note:** `recent_density` and `density_bg` fields are ignored (heatmaps removed)

#### `equityEventsStore` (for time-range tabs)
- **Endpoint:** `GET /api/equity/events?range={range}`
- **Poll interval:** ~30000ms, re-fetched on range change
- **Range values:** `24h`, `7d`, `30d`. For "All": pass `range=all` — the backend treats unknown range keys as unbounded (no time filter), since `range_sec` resolves to `None` and `where_time` stays empty.
- **Response fields used:**
  - `events` — array of `{ts, venue, equity, side, event_type}`
- **Consumed by:** EquityCurve (when user selects a time-range tab)
- **Note:** This is the only endpoint that supports `range` filtering. The default equity view uses `equity_components` / `equity_total` from `chartStore`. When a specific range tab is selected, EquityCurve switches to `equityEventsStore` data instead.

### Serving

#### Development
- `vite dev` runs on port 5173 (default)
- Vite config proxies `/api/*` to the Python backend (e.g. `http://localhost:8000`)
- `vite.config.js`:
  ```js
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

#### Production
- `vite build` outputs to `dashboard/dist/`
- `vite.config.js` sets `base: '/dashboard2/'` — all asset paths prefixed accordingly
- FastAPI changes in `webhook_server.py`:
  1. Mount static assets: `app.mount("/dashboard2/assets", StaticFiles(directory="dashboard/dist/assets"), name="dashboard2-assets")`
  2. SPA route: `@app.get("/dashboard2/{path:path}")` returning `FileResponse("dashboard/dist/index.html")` plus `@app.get("/dashboard2")` for the bare path
  3. Route order: `/api/*` routes first, then `/dashboard` (existing, unchanged), then `/dashboard2/*` routes
  4. The existing `DASHBOARD_HTML` constant and `/dashboard` endpoint remain unchanged

### Backend Changes

Existing `/api/*` handler contracts remain unchanged — no modifications to API request/response shapes.
Existing `/dashboard` endpoint and `DASHBOARD_HTML` constant remain unchanged.

New routing additions only:
- Add `StaticFiles` mount at `/dashboard2/assets` pointing to `dashboard/dist/assets`
- Add `@app.get("/dashboard2")` and `@app.get("/dashboard2/{path:path}")` returning `FileResponse("dashboard/dist/index.html")`

## Layout

3-column grid + full-width bottom bar. Fills viewport height, no scrolling at normal resolution.

```
┌─────────────────────────────────────────────────────────────────┐
│  LEFT (220px)          │     CENTER (flex)          │ RIGHT     │
│                        │                            │ (240px)   │
│  STATUS                │  ┌──────────────────────┐  │ AXIS BARS │
│  KuCoin: 83.34         │  │                      │  │ (~80px)   │
│  Kraken: —             │  │  Candlestick Chart   │  │ ─ ─ ─ ─  │
│  Regime: trend         │  │  (black background)  │  │ POLAR     │
│  ─ ─ ─ ─ ─ ─ ─ ─     │  │                      │  │ CHART     │
│  POSITION              │  │  Lightweight Charts   │  │ v1        │
│  KuCoin: Long 8.1     │  │                      │  │ (square)  │
│  Kraken: Long 13.8    │  │                      │  │           │
│  ─ ─ ─ ─ ─ ─ ─ ─     │  │                      │  │           │
│  CAPITAL               │  │                      │  │           │
│  $244.76  $143.88      │  ├──────────────────────┤  │           │
│  ─ ─ ─ ─ ─ ─ ─ ─     │  │ Regime Band (20px)   │  │           │
│  PERFORMANCE           │  └──────────────────────┘  │           │
│  PnL: -28.02%         │                            │           │
│  Winrate: 36%          │                            │           │
│  ─ ─ ─ ─ ─ ─ ─ ─     │                            │           │
│  LEVELS                │                            │           │
│  Side: Long            │                            │           │
│  Entry / SL / TP1 / TP2│                            │           │
├─────────────────────────┴────────────────────────────┴───────────┤
│  EQUITY CURVE (full width, ~130px)            [24h][7d][30d][All]│
└──────────────────────────────────────────────────────────────────┘
```

CSS grid definition for `App.svelte`:
```css
.dashboard {
  display: grid;
  grid-template-columns: 220px 1fr 240px;
  grid-template-rows: 1fr auto;
  height: 100vh;
  gap: 0;
}
.stats-panel { grid-column: 1; grid-row: 1; }
.chart-area  { grid-column: 2; grid-row: 1; }
.right-panel { grid-column: 3; grid-row: 1; }
.equity-bar  { grid-column: 1 / -1; grid-row: 2; }
```

## Components

### StatsPanel.svelte

- Fixed 220px width, left column
- 5 sections: STATUS, POSITION, CAPITAL, PERFORMANCE, LEVELS
- No card backgrounds — sections sit on black, separated by 16px vertical spacing
- Section headers: muted uppercase, `font-size: 0.7rem`, `letter-spacing: 0.05em`, color `var(--muted)`
- Values: monospace, `var(--text)` (bright white)
- Positive values: `var(--green)`, negative: `var(--red)`
- Padding: `1rem` on all sides

**Data mapping:**

| Section | Field | Source |
|---------|-------|--------|
| STATUS — KuCoin price | Latest bar close from `chartStore.bars[-1].close` | chartStore |
| STATUS — Kraken price | `chartStore.kraken_metrics.price` or "—" | chartStore |
| STATUS — Regime | `chartStore.day_regime_state` | chartStore |
| POSITION — KuCoin | `statusStore.position.side` + `position.position` (qty) | statusStore |
| POSITION — Kraken | `chartStore.kraken_metrics.position` or "Flat" | chartStore |
| CAPITAL — KuCoin | `statusStore.status.balance` | statusStore |
| CAPITAL — Kraken | `chartStore.kraken_metrics.balance` or "—" | chartStore |
| PERFORMANCE — PnL % | `statusStore.performance.pnl_pct` | statusStore |
| PERFORMANCE — Winrate | `statusStore.performance.winrate` | statusStore |
| PERFORMANCE — Monthly growth | `statusStore.performance.monthly_growth` | statusStore |
| PERFORMANCE — Avg trade | `statusStore.performance.average_gain` | statusStore |
| PERFORMANCE — Trades | `statusStore.performance.trade_count` | statusStore |
| PERFORMANCE — Wins | `statusStore.performance.winning_trade_count` | statusStore |
| PERFORMANCE — Losses | `statusStore.performance.losing_trade_count` | statusStore |
| LEVELS — Side | `chartStore.levels.side` → "Long" / "Short" | chartStore |
| LEVELS — Mode | `chartStore.levels.mode` | chartStore |
| LEVELS — Entry | `chartStore.levels.entry_px` | chartStore |
| LEVELS — SL | `chartStore.levels.sl` | chartStore |
| LEVELS — TP1 | `chartStore.levels.tp1` | chartStore |
| LEVELS — TP2 | `chartStore.levels.tp2` | chartStore |

### PriceChart.svelte

- Fills center column, no border-radius, no wrapper card
- Uses `lightweight-charts` npm package (replaces CDN `<script>` tag)
- Chart config:
  - `layout.background.color: '#000000'`
  - `layout.textColor: '#e0e0e8'`
  - `grid.vertLines.color: '#111118'`
  - `grid.horzLines.color: '#111118'`
  - `rightPriceScale.borderColor: '#111118'`
  - `timeScale.borderColor: '#111118'`
  - `crosshair.mode: CrosshairMode.Magnet`

**All series from the current chart are preserved:**
- Candlestick: up `#2ecc71`, down `#f7768e`
- SL line: `#f7768e`, width 2, solid
- TTP line: `#e0af68`, width 2, dashed (lineStyle 1)
- Entry line: `#ffffff`, width 1, solid
- TP1 line: `#7aa2f7`, width 2
- TP2 line: `#bb9af7`, width 2
- Fibonacci overlays: `fibLongSeries` (`#2ecc71`, width 2), `fibMidSeries` (`#ffffff`, width 1, dashed), `fibShortSeries` (`#f7768e`, width 2) — from `chartStore.fibo.long`, `.mid`, `.short`
- Trade segment lines: dynamic series created/removed per `chartStore.segments` array — each segment is a 2-point line (from→to) colored by `seg.color`, labeled "Trade +" or "Trade -"
- Live mid-price line: `#9aa5b1`, width 1, dashed — horizontal line at latest ticker price when position is open
- Trade markers: from `chartStore.markers`

**Renko brick chart:**
- Time axis uses brick-index mapping (same `brickBaseTs + idx * 60` logic as current)
- `tickMarkFormatter` and `timeFormatter` map brick indices back to real timestamps via `barsRawRef`
- Animated last brick: the final candlestick represents the in-progress renko brick. Use a CSS overlay or canvas layer to show a semi-transparent "fill" indicator showing how close the current price is to completing the next brick (distance from current close to next brick threshold as a percentage fill). Updates on each poll cycle.

**Other:**
- Uses `ResizeObserver` to fill container
- Data: subscribes to `chartStore`

### RegimeBand.svelte

- 20px tall `<canvas>`, flush below chart, same column width
- Renders `regime_scores` from `chartStore` using `scoreToColor`:
  - Score range: -1 (countertrend/red) to +1 (trend/green)
  - Color interpolation: `score < 0` → red channel dominant, `score > 0` → green channel dominant
  - Formula (from current codebase):
    ```
    t = (clamp(score, -1, 1) + 1) / 2
    if t < 0.5: r=247, g=118+86*u, b=142*(1-u) where u=t/0.5
    if t >= 0.5: r=247*(1-u)+46*u, g=204, b=113*u where u=(t-0.5)/0.5
    ```
- Includes `regime_forecast` with fading opacity: `alpha = 0.3 + 0.7 * (1 - i/len)`
- Time-synchronized: uses chart `timeScale().timeToCoordinate()` to align regime pixels with chart bars
- No label, no border
- Data: subscribes to `chartStore`

### AxisBars.svelte

- Top of right column, ~80px total height
- 3 horizontal center-zero bars:
  - X Drift: color `#ff6644`
  - Y Elasticity: color `#44bbff`
  - Z Instability: color `#ffcc33`
- Each bar: 18px tall, with a center line (zero), fill extends left or right proportional to value
- Value range: `[-1, 1]` — values are already normalized in the API response
- Text to the right: value with sign (e.g. `+0.847`) and confidence (`c:+0.275`)
- Canvas-based, same drawing logic as current `drawAxisBars()`
- Data: `statespaceStore.current` fields: `x`, `y`, `z`, `conf_x`, `conf_y`, `conf_z`

### PolarChart.svelte

- Right column, below axis bars, square aspect ratio (fills available width, height = width)
- Canvas-based radar chart

**Coordinate system:**
- 3 spokes at 120-degree intervals:
  - Spoke 0 (top, 270°): **Drift** (x)
  - Spoke 1 (bottom-right, 30°): **Elasticity** (y)
  - Spoke 2 (bottom-left, 150°): **Instability** (z)
- Spoke angles: `angle_i = (2π * i / 3) - π/2` (starting from top, clockwise)

**Value mapping:**
- API values are in range `[-1, 1]`
- Map to radial distance: `r = abs(value)` — distance from center (0) to edge (1)
- Sign is indicated by fill color: positive values use the axis color, negative use a desaturated/muted version
- Guide circles at `r = 0.25, 0.5, 0.75, 1.0` — drawn as thin concentric circles, color `#1a1a2e`

**Rendering:**
- Current position: filled polygon connecting the 3 axis points, semi-transparent fill
- Fill color: derived from overall regime score (green-ish for trend, red-ish for countertrend), alpha 0.3
- Polygon stroke: same color, alpha 0.8, width 1.5px
- Axis labels: small text at spoke endpoints ("Drift", "Elast.", "Instab.")

**Trail:**
- Render last 20 positions from `statespaceStore.trajectory` (most recent 20 entries, `trajectory[-20:]`)
- Each as a polygon with increasing opacity: `alpha = 0.05 + 0.15 * (i / N)` where `i=0` is oldest, `i=N-1` is newest
- Newest trail entry is most opaque, oldest is nearly invisible
- No animation in v1 — just static trail on each data refresh

**Data:** `statespaceStore.current` for current polygon, `statespaceStore.trajectory` for trail

### EquityCurve.svelte

- Full-width bottom section, ~130px tall, spans all 3 columns
- Canvas-based line chart

**Default view (no range selected):**
- Combined equity line from `chartStore.equity_total` — array of `{time, equity}`
- Venue breakdown as stacked semi-transparent fills:
  - KuCoin: `rgba(122, 162, 247, 0.3)`
  - Kraken: `rgba(255, 158, 100, 0.35)`
- Venue values as labels: "KuCoin: $244.76 | Kraken: $143.88" (latest point from each `equity_components` entry)
- Percentage change: `((last - first) / first * 100)` displayed top-right, green if positive, red if negative

**Range tab view:**
- Tabs: `24h`, `7d`, `30d`, `All` — small buttons in top-right corner
- Default: no tab selected (shows data from `chartStore`)
- When a tab is clicked: fetch `GET /api/equity/events?range={range}` (for "All", pass `range=all` — the backend treats unknown range values as unbounded)
- Render events as a line chart: x = `ts`, y = `equity`, grouped/colored by `venue`
- Clicking the already-active tab deselects it, returning to default view

**Data:** `chartStore` (default), `equityEventsStore` (when range tab active)

## Theme

| Token | Value | Usage |
|-------|-------|-------|
| `--bg` | `#000000` | body, chart, all sections |
| `--text` | `#e0e0e8` | primary text, values |
| `--muted` | `#555555` | section headers, secondary info |
| `--green` | `#2ecc71` | positive values, buy, trend |
| `--red` | `#f7768e` | negative values, sell, countertrend |
| `--blue` | `#7aa2f7` | TP1, accent, interactive |
| `--purple` | `#bb9af7` | TP2 |
| `--amber` | `#e0af68` | TTP line |
| `--grid` | `#111118` | chart grid lines, subtle borders |

- Font labels: `system-ui, -apple-system, sans-serif`
- Font values: `ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`
- No border-radius on major elements — sharp edges
- No shadows, no gradients on UI chrome
- No card backgrounds — content directly on black

## Responsive Behavior

- `< 1200px`: right column (polar chart + axis bars) moves below the chart area. Layout becomes 2-column (stats | chart) + right below + equity bottom.
- `< 800px`: left sidebar collapses above the chart. Single-column stack: stats → chart → regime band → axis bars → polar chart → equity.

## Dependencies

- `svelte` — framework (latest stable, v4 or v5)
- `@sveltejs/vite-plugin-svelte` — Svelte/Vite integration
- `vite` — build tool
- `lightweight-charts` — candlestick chart (npm package, replaces CDN script tag)

No other external dependencies for v1. Canvas rendering is all custom. GSAP or D3 can be added later for the advanced polar chart iteration.

## Refresh Behavior

**Polling:** All stores poll at their configured intervals (see Environment Variables).

**Visibility/focus refresh:** When the browser tab becomes visible or gains focus, immediately trigger a refresh of all stores. This ensures data is fresh when the user switches back to the dashboard.
- `document.addEventListener('visibilitychange', ...)` — refresh when `visibilityState === 'visible'`
- `window.addEventListener('focus', ...)` — refresh on window focus
- `window.addEventListener('pageshow', ...)` — refresh on page show (back/forward navigation)

**Pull-to-refresh (mobile/touch):** Improved implementation:
- On `touchstart` at scroll position ≤ 2px from top, record start Y
- On `touchmove`, if delta Y ≥ 90px, trigger refresh of all stores and show a brief visual indicator (e.g. a thin accent-colored bar at the top that fades out)
- On `touchend`, reset state
- Guard against multiple simultaneous refreshes (debounce — skip if a refresh is already in flight)
- The current implementation doesn't provide visual feedback and has no debounce — the new version fixes both

**Deduplication:** All refresh triggers (poll, visibility, focus, pull) go through a single `refreshAll()` function that checks `inFlight` flags per store to avoid duplicate concurrent requests.

## Environment Variables

| Variable | Default | Usage |
|----------|---------|-------|
| `DASHBOARD_UI_REFRESH_MS` | `4000` | Chart store poll interval (ms) |
| `DASHBOARD_STATESPACE_REFRESH_MS` | `10000` | Statespace store poll interval (ms) |
| `DASHBOARD_STATUS_REFRESH_MS` | `10000` | Status/position/performance poll interval (ms) |
| `DASHBOARD_EQUITY_REFRESH_MS` | `30000` | Equity events poll interval (ms) |

These are read client-side. In production, inject them into the HTML template or a `/api/config` endpoint. In development, hardcode defaults.
