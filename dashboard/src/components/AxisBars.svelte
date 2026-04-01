<script>
  import { onMount, onDestroy } from 'svelte';
  import { get } from 'svelte/store';
  import { statespaceStore } from '../lib/stores.js';

  /** @type {HTMLDivElement | undefined} */
  let containerEl;
  /** @type {HTMLCanvasElement | undefined} */
  let canvasEl;

  const BAR_H = 18;
  const GAP = 50;
  const TRACK_LEFT = 70;
  const TRACK_RIGHT_PAD = 60;
  const CANVAS_H = 200;

  /** @type {ResizeObserver | null} */
  let resizeObs = null;

  /** @param {number} v */
  function clampUnit(v) {
    return Math.max(-1, Math.min(1, Number(v) || 0));
  }

  /** @param {number} v */
  function fmtSigned(v) {
    const x = Number(v);
    if (!Number.isFinite(x)) return '+0.000';
    const s = x >= 0 ? '+' : '';
    return s + x.toFixed(3);
  }

  /** @param {number} c */
  function fmtConf(c) {
    const x = Number(c);
    if (!Number.isFinite(x)) return 'c:+0.000';
    const s = x >= 0 ? '+' : '';
    return `c:${s}${x.toFixed(3)}`;
  }

  /**
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} w
   * @param {number} h
   * @param {any} payload
   */
  function draw(ctx, w, h, payload) {
    ctx.clearRect(0, 0, w, h);
    const cur = payload?.current;
    const trackRight = w - TRACK_RIGHT_PAD;
    const trackW = Math.max(1, trackRight - TRACK_LEFT);
    const centerX = TRACK_LEFT + trackW / 2;
    const half = trackW / 2;

    const axes = [
      { key: 'x', confKey: 'conf_x', label: 'X Drift', color: '#ff6644' },
      { key: 'y', confKey: 'conf_y', label: 'Y Elasticity', color: '#44bbff' },
      { key: 'z', confKey: 'conf_z', label: 'Z Instability', color: '#ffcc33' },
    ];

    for (let i = 0; i < axes.length; i++) {
      const rowY = i * (BAR_H + GAP);
      const ax = axes[i];
      const value = cur ? clampUnit(cur[ax.key]) : 0;
      const conf = cur ? Number(cur[ax.confKey]) : 0;
      const confN = Number.isFinite(conf) ? conf : 0;

      // Axis label (left)
      ctx.font = 'bold 11px system-ui, sans-serif';
      ctx.fillStyle = ax.color;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(ax.label, 4, rowY + BAR_H / 2);

      // Track background
      ctx.fillStyle = '#2a2e38';
      ctx.fillRect(TRACK_LEFT, rowY, trackW, BAR_H);

      // Center line
      ctx.beginPath();
      ctx.strokeStyle = '#666';
      ctx.lineWidth = 1;
      ctx.moveTo(centerX + 0.5, rowY);
      ctx.lineTo(centerX + 0.5, rowY + BAR_H);
      ctx.stroke();

      // Value fill
      const tipX = centerX + value * half;
      const x0 = Math.min(centerX, tipX);
      const x1 = Math.max(centerX, tipX);
      ctx.fillStyle = ax.color;
      ctx.fillRect(x0, rowY, Math.max(1, x1 - x0), BAR_H);

      // Value + confidence (right of track)
      ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
      ctx.fillStyle = '#ccc';
      ctx.textAlign = 'left';
      const textX = trackRight + 6;
      ctx.fillText(`${fmtSigned(value)}  ${fmtConf(confN)}`, textX, rowY + BAR_H / 2);
    }
  }

  /**
   * @param {any} [payload]
   */
  function paint(payload) {
    if (!canvasEl || !containerEl) return;
    const data = payload !== undefined ? payload : get(statespaceStore);
    const w = Math.max(200, containerEl.clientWidth || 200);
    const dpr = window.devicePixelRatio || 1;
    canvasEl.width = Math.round(w * dpr);
    canvasEl.height = Math.round(CANVAS_H * dpr);
    canvasEl.style.width = `${w}px`;
    canvasEl.style.height = `${CANVAS_H}px`;
    const ctx = canvasEl.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw(ctx, w, CANVAS_H, data);
  }

  $effect(() => {
    const ss = $statespaceStore;
    paint(ss);
  });

  onMount(() => {
    paint();
    resizeObs = new ResizeObserver(() => paint());
    if (containerEl) resizeObs.observe(containerEl);
  });

  onDestroy(() => {
    resizeObs?.disconnect();
    resizeObs = null;
  });
</script>

<div class="axis-bars" bind:this={containerEl}>
  <canvas bind:this={canvasEl} width="250" height="200"></canvas>
</div>

<style>
  .axis-bars {
    padding: 0.5rem;
  }
  canvas {
    width: 100%;
    display: block;
  }
</style>
