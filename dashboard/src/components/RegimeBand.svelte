<script>
  import { onMount, onDestroy } from 'svelte';
  import { chartStore } from '../lib/stores.js';
  import { scoreToColor, palette } from '../lib/colors.js';
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

  const BAND_HEIGHT = 32;

  function layoutCanvas() {
    if (!containerEl || !canvasEl) return null;
    const w = containerEl.clientWidth;
    const h = BAND_HEIGHT;
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

  /**
   * Draw a smooth, gradient-filled regime band.
   * Historical scores use a continuous gradient with soft edges.
   * Forecast scores fade toward neutral with decreasing alpha.
   */
  function draw() {
    const layout = layoutCanvas();
    if (!layout) return;
    const { ctx, w, h } = layout;

    const ts = chartComponent?.getTimeScale?.() ?? null;

    // Dark fill background
    ctx.fillStyle = palette.bg;
    ctx.fillRect(0, 0, w, h);

    if (!ts) return;

    const data = latestChart;
    const barsRaw = data?.bars ?? [];
    const { map: timeMap } = buildTimeMapFromBars(barsRaw);

    // ── Collect pixel columns from scores ──
    const scores = Array.isArray(data?.regime_scores) ? data.regime_scores : [];

    if (scores.length > 0) {
      drawSegments(ctx, ts, scores, timeMap, barsRaw, w, h, 1.0);
    }

    // ── Forecast overlay with fading alpha ──
    const forecast = Array.isArray(data?.regime_forecast) ? data.regime_forecast : [];
    const flen = forecast.length;
    if (flen > 0) {
      drawSegments(ctx, ts, forecast, timeMap, barsRaw, w, h, -1);
    }

    // ── Top highlight line (subtle glow) ──
    const topGrad = ctx.createLinearGradient(0, 0, 0, 3);
    topGrad.addColorStop(0, 'rgba(255, 255, 255, 0.06)');
    topGrad.addColorStop(1, 'rgba(255, 255, 255, 0)');
    ctx.fillStyle = topGrad;
    ctx.fillRect(0, 0, w, 3);

    // ── Bottom shadow ──
    const botGrad = ctx.createLinearGradient(0, h - 4, 0, h);
    botGrad.addColorStop(0, 'rgba(0, 0, 0, 0)');
    botGrad.addColorStop(1, 'rgba(0, 0, 0, 0.25)');
    ctx.fillStyle = botGrad;
    ctx.fillRect(0, h - 4, w, 4);
  }

  /**
   * Draw a set of score segments with smooth inter-column gradients.
   * @param {CanvasRenderingContext2D} ctx
   * @param {*} ts  - lightweight-charts timeScale API
   * @param {Array<{time: unknown, score: unknown}>} items
   * @param {Map<number,number>} timeMap
   * @param {unknown} barsRaw
   * @param {number} w
   * @param {number} h
   * @param {number} alphaMode  - if >= 0, use this as constant alpha; if -1, use forecast fading
   */
  function drawSegments(ctx, ts, items, timeMap, barsRaw, w, h, alphaMode) {
    const len = items.length;
    if (!len) return;

    // Pre-compute x positions and scores
    const coords = [];
    for (let i = 0; i < len; i++) {
      const s = items[i];
      const mapped = mapTimeForChart(s?.time, timeMap, barsRaw, BRICK_BASE_TS);
      if (mapped == null) continue;
      const x = ts.timeToCoordinate(mapped);
      if (x == null) continue;
      const score = Number(s?.score);
      if (!Number.isFinite(score)) continue;
      coords.push({ x, score, idx: i });
    }

    if (coords.length < 2) {
      // Single point — draw a thin rect
      if (coords.length === 1) {
        const { x, score, idx } = coords[0];
        const alpha = alphaMode >= 0 ? alphaMode : 0.3 + 0.7 * (1 - idx / len);
        ctx.fillStyle = scoreToColor(score, alpha * 0.85);
        ctx.fillRect(x - 1, 0, 3, h);
      }
      return;
    }

    // Draw smooth gradient between consecutive points
    for (let i = 0; i < coords.length - 1; i++) {
      const c0 = coords[i];
      const c1 = coords[i + 1];

      const left = Math.floor(Math.min(c0.x, c1.x));
      const right = Math.ceil(Math.max(c0.x, c1.x));
      const width = Math.max(1, right - left);

      const a0 = alphaMode >= 0 ? alphaMode : 0.3 + 0.7 * (1 - c0.idx / len);
      const a1 = alphaMode >= 0 ? alphaMode : 0.3 + 0.7 * (1 - c1.idx / len);

      // Create a horizontal gradient for each segment
      const grad = ctx.createLinearGradient(left, 0, right, 0);
      grad.addColorStop(0, scoreToColor(c0.score, a0 * 0.85));
      grad.addColorStop(1, scoreToColor(c1.score, a1 * 0.85));

      ctx.fillStyle = grad;
      ctx.fillRect(left, 0, width + 1, h);

      // Soft inner glow — brighter core strip
      const glowGrad = ctx.createLinearGradient(0, 0, 0, h);
      glowGrad.addColorStop(0, 'rgba(255, 255, 255, 0)');
      glowGrad.addColorStop(0.35, 'rgba(255, 255, 255, 0.04)');
      glowGrad.addColorStop(0.65, 'rgba(255, 255, 255, 0.04)');
      glowGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = glowGrad;
      ctx.fillRect(left, 0, width + 1, h);
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
    height: 32px;
    width: 100%;
    position: relative;
    border-top: 1px solid rgba(42, 48, 64, 0.6);
    border-bottom: 1px solid rgba(42, 48, 64, 0.6);
    overflow: hidden;
  }
  .regime-band canvas {
    width: 100%;
    height: 100%;
    display: block;
  }
</style>
