<script>
  import { onMount, onDestroy } from 'svelte';
  import { chartStore } from '../lib/stores.js';
  import { scoreToColor } from '../lib/colors.js';
  import {
    buildTimeMapFromBars,
    mapTimeForChart,
    BRICK_BASE_TS,
  } from '../lib/chartHelpers.js';

  /** @type {{ getTimeScale?: () => import('lightweight-charts').ITimeScaleApi | null } | null | undefined} */
  export let chartComponent = null;

  let containerEl;
  let canvasEl;

  /** @type {import('lightweight-charts').ITimeScaleApi | null} */
  let timeScaleRef = null;

  /** @type {(() => void) | null} */
  let unsubVisible = null;

  /** @type {unknown} */
  let latestChart = null;

  /** @type {ResizeObserver | null} */
  let resizeObs = null;

  /** @type {(() => void) | undefined} */
  let unsubChart;

  function layoutCanvas() {
    if (!containerEl || !canvasEl) return null;
    const w = containerEl.clientWidth;
    const h = 20;
    if (w <= 0) return null;
    const dpr = window.devicePixelRatio || 1;
    const cw = Math.max(1, Math.floor(w * dpr));
    const ch = Math.max(1, Math.floor(h * dpr));
    if (canvasEl.width !== cw || canvasEl.height !== ch) {
      canvasEl.width = cw;
      canvasEl.height = ch;
      canvasEl.style.width = `${w}px`;
      canvasEl.style.height = `${h}px`;
    }
    const ctx = canvasEl.getContext('2d');
    if (!ctx) return null;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  function draw() {
    const layout = layoutCanvas();
    if (!layout) return;
    const { ctx, w, h } = layout;

    const ts = chartComponent?.getTimeScale?.() ?? null;
    if (!ts) {
      ctx.clearRect(0, 0, w, h);
      return;
    }

    const data = latestChart;
    const barsRaw = data?.bars ?? [];
    const { map: timeMap } = buildTimeMapFromBars(barsRaw);

    ctx.clearRect(0, 0, w, h);

    const scores = Array.isArray(data?.regime_scores) ? data.regime_scores : [];
    for (let i = 0; i < scores.length; i++) {
      const s = scores[i];
      const mapped0 = mapTimeForChart(s?.time, timeMap, barsRaw, BRICK_BASE_TS);
      if (mapped0 == null) continue;
      const x0 = ts.timeToCoordinate(mapped0);
      if (x0 == null) continue;
      let x1;
      if (i + 1 < scores.length) {
        const mapped1 = mapTimeForChart(scores[i + 1]?.time, timeMap, barsRaw, BRICK_BASE_TS);
        x1 = mapped1 != null ? ts.timeToCoordinate(mapped1) : null;
      } else {
        x1 = null;
      }
      if (x1 == null) x1 = x0 + 2;
      const left = Math.min(x0, x1);
      const width = Math.max(1, Math.abs(x1 - x0));
      ctx.fillStyle = scoreToColor(Number(s?.score));
      ctx.fillRect(left, 0, width, h);
    }

    const forecast = Array.isArray(data?.regime_forecast) ? data.regime_forecast : [];
    const flen = forecast.length;
    for (let i = 0; i < flen; i++) {
      const f = forecast[i];
      const alpha = 0.3 + 0.7 * (1 - i / flen);
      const mapped0 = mapTimeForChart(f?.time, timeMap, barsRaw, BRICK_BASE_TS);
      if (mapped0 == null) continue;
      const x0 = ts.timeToCoordinate(mapped0);
      if (x0 == null) continue;
      let x1;
      if (i + 1 < flen) {
        const mapped1 = mapTimeForChart(forecast[i + 1]?.time, timeMap, barsRaw, BRICK_BASE_TS);
        x1 = mapped1 != null ? ts.timeToCoordinate(mapped1) : null;
      } else {
        x1 = null;
      }
      if (x1 == null) x1 = x0 + 2;
      const left = Math.min(x0, x1);
      const width = Math.max(1, Math.abs(x1 - x0));
      ctx.fillStyle = scoreToColor(Number(f?.score), alpha);
      ctx.fillRect(left, 0, width, h);
    }
  }

  function attachVisibleRange(ts) {
    if (timeScaleRef === ts && unsubVisible) return;
    if (unsubVisible) {
      unsubVisible();
      unsubVisible = null;
    }
    timeScaleRef = ts;
    if (!ts) return;
    const handler = () => draw();
    ts.subscribeVisibleTimeRangeChange(handler);
    unsubVisible = () => {
      ts.unsubscribeVisibleTimeRangeChange(handler);
      unsubVisible = null;
    };
  }

  $: {
    const ts = chartComponent?.getTimeScale?.() ?? null;
    attachVisibleRange(ts);
    draw();
  }

  function onWindowResize() {
    draw();
  }

  onMount(() => {
    unsubChart = chartStore.subscribe((d) => {
      latestChart = d;
      draw();
    });
    resizeObs = new ResizeObserver(() => draw());
    if (containerEl) resizeObs.observe(containerEl);
    window.addEventListener('resize', onWindowResize);
    draw();
  });

  onDestroy(() => {
    unsubChart?.();
    if (unsubVisible) unsubVisible();
    resizeObs?.disconnect();
    window.removeEventListener('resize', onWindowResize);
  });
</script>

<div class="regime-band" bind:this={containerEl}>
  <canvas bind:this={canvasEl}></canvas>
</div>

<style>
  .regime-band {
    height: 20px;
    width: 100%;
    position: relative;
  }
  .regime-band canvas {
    width: 100%;
    height: 100%;
    display: block;
  }
</style>
