<script>
  import { onMount, onDestroy } from 'svelte';
  import { startPolling, refreshAll } from './lib/stores.js';
  import StatsPanel from './components/StatsPanel.svelte';
  import PriceChart from './components/PriceChart.svelte';
  import RegimeBand from './components/RegimeBand.svelte';
  import AxisBars from './components/AxisBars.svelte';
  import PolarChart from './components/PolarChart.svelte';
  import EquityCurve from './components/EquityCurve.svelte';

  /** @type {(() => void) | null} */
  let stopPolling = null;

  /** @type {import('./components/PriceChart.svelte').default | null} */
  let priceChartRef = $state(null);

  let refreshBarActive = $state(false);

  /** @type {number | null} */
  let pullStartY = null;
  let pullGestureFired = false;

  function atTop() {
    return window.scrollY <= 2;
  }

  function handleVisibility() {
    if (document.visibilityState === 'visible') refreshAll();
  }

  function handleFocus() {
    refreshAll();
  }

  function handlePageShow() {
    refreshAll();
  }

  function handleTouchStart(e) {
    if (!atTop()) {
      pullStartY = null;
      return;
    }
    pullStartY = e.touches[0]?.clientY ?? null;
    pullGestureFired = false;
  }

  function handleTouchMove(e) {
    if (pullStartY == null || pullGestureFired) return;
    const y = e.touches[0]?.clientY;
    if (y == null) return;
    const dy = y - pullStartY;
    if (dy >= 90) {
      pullGestureFired = true;
      refreshBarActive = true;
      refreshAll().finally(() => {
        setTimeout(() => {
          refreshBarActive = false;
        }, 350);
      });
    }
  }

  function handleTouchEnd() {
    pullStartY = null;
    pullGestureFired = false;
  }

  onMount(() => {
    stopPolling = startPolling();
    document.addEventListener('visibilitychange', handleVisibility);
    window.addEventListener('focus', handleFocus);
    window.addEventListener('pageshow', handlePageShow);
    window.addEventListener('touchstart', handleTouchStart, { passive: true });
    window.addEventListener('touchmove', handleTouchMove, { passive: true });
    window.addEventListener('touchend', handleTouchEnd);
    window.addEventListener('touchcancel', handleTouchEnd);
  });

  onDestroy(() => {
    stopPolling?.();
    document.removeEventListener('visibilitychange', handleVisibility);
    window.removeEventListener('focus', handleFocus);
    window.removeEventListener('pageshow', handlePageShow);
    window.removeEventListener('touchstart', handleTouchStart);
    window.removeEventListener('touchmove', handleTouchMove);
    window.removeEventListener('touchend', handleTouchEnd);
    window.removeEventListener('touchcancel', handleTouchEnd);
  });
</script>

<div
  class="refresh-indicator"
  class:active={refreshBarActive}
  aria-hidden="true"
></div>

<div class="dashboard">
  <div class="stats-panel">
    <StatsPanel />
  </div>
  <div class="chart-area">
    <PriceChart bind:this={priceChartRef} />
    <RegimeBand chartComponent={priceChartRef} />
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
  .refresh-indicator {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: #2563eb;
    opacity: 0;
    pointer-events: none;
    z-index: 9999;
    transition: opacity 0.35s ease;
  }

  .refresh-indicator.active {
    opacity: 1;
  }

  .dashboard {
    display: grid;
    grid-template-columns: 220px 1fr 240px;
    grid-template-rows: 1fr auto;
    height: 100vh;
    gap: 0;
  }
  .stats-panel { grid-column: 1; grid-row: 1; overflow-y: auto; }
  .chart-area {
    grid-column: 2;
    grid-row: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
    border-left: 1px solid var(--border, #2a3040);
    border-right: 1px solid var(--border, #2a3040);
  }
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
