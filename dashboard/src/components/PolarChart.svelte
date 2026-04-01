<script>
  import { onMount, onDestroy } from 'svelte';
  import { get } from 'svelte/store';
  import { statespaceStore } from '../lib/stores.js';
  import { scoreToColor } from '../lib/colors.js';

  /** @type {HTMLDivElement | undefined} */
  let containerEl;
  /** @type {HTMLCanvasElement | undefined} */
  let canvasEl;

  /** @type {ResizeObserver | null} */
  let resizeObs = null;

  const LABELS = ['Drift', 'Elast.', 'Instab.'];

  /** @param {number} i */
  function spokeAngle(i) {
    return (2 * Math.PI * i) / 3 - Math.PI / 2;
  }

  /**
   * @param {CanvasRenderingContext2D} ctx
   * @param {{ x: number, y: number }[]} pts
   */
  function strokeFillPoly(ctx, pts, fillStyle, strokeStyle, lineWidth = 1) {
    if (pts.length < 3) return;
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let k = 1; k < pts.length; k++) ctx.lineTo(pts[k].x, pts[k].y);
    ctx.closePath();
    ctx.fillStyle = fillStyle;
    ctx.fill();
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
  }

  /**
   * @param {any} pt
   * @param {number} cx
   * @param {number} cy
   * @param {number} maxR
   */
  function pointsForState(pt, cx, cy, maxR) {
    if (!pt) return [];
    const vals = [pt.x, pt.y, pt.z];
    const out = [];
    for (let i = 0; i < 3; i++) {
      const ang = spokeAngle(i);
      const r = Math.min(1, Math.max(0, Math.abs(Number(vals[i]) || 0)));
      out.push({
        x: cx + r * maxR * Math.cos(ang),
        y: cy + r * maxR * Math.sin(ang),
      });
    }
    return out;
  }

  /**
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} w
   * @param {number} h
   * @param {any} payload
   */
  function draw(ctx, w, h, payload) {
    ctx.clearRect(0, 0, w, h);
    const cx = w / 2;
    const cy = h / 2;
    const maxR = Math.max(20, Math.min(w, h) / 2 - 30);

    // Guide circles
    ctx.strokeStyle = '#1a1a2e';
    ctx.lineWidth = 0.5;
    for (const frac of [0.25, 0.5, 0.75, 1.0]) {
      ctx.beginPath();
      ctx.arc(cx, cy, maxR * frac, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Spokes + axis labels
    ctx.font = '10px system-ui, sans-serif';
    ctx.fillStyle = '#555';
    for (let i = 0; i < 3; i++) {
      const ang = spokeAngle(i);
      const xe = cx + maxR * Math.cos(ang);
      const ye = cy + maxR * Math.sin(ang);
      ctx.beginPath();
      ctx.strokeStyle = '#1a1a2e';
      ctx.lineWidth = 0.5;
      ctx.moveTo(cx, cy);
      ctx.lineTo(xe, ye);
      ctx.stroke();

      const lx = cx + (maxR + 12) * Math.cos(ang);
      const ly = cy + (maxR + 12) * Math.sin(ang);
      if (i === 0) {
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(LABELS[i], lx, ly - 2);
      } else if (i === 1) {
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(LABELS[i], lx + 4, ly);
      } else {
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText(LABELS[i], lx - 4, ly);
      }
    }

    const trail = (payload?.trajectory ?? []).slice(-20);
    const N = trail.length;
    for (let i = 0; i < N; i++) {
      const alpha = N > 0 ? 0.05 + 0.15 * (i / N) : 0.05;
      const pts = pointsForState(trail[i], cx, cy, maxR);
      if (pts.length === 3) {
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        ctx.lineTo(pts[1].x, pts[1].y);
        ctx.lineTo(pts[2].x, pts[2].y);
        ctx.closePath();
        ctx.fillStyle = `rgba(100, 160, 255, ${alpha})`;
        ctx.fill();
      }
    }

    const cur = payload?.current;
    if (cur) {
      const pts = pointsForState(cur, cx, cy, maxR);
      if (pts.length === 3) {
        const mix = (Number(cur.x) + Number(cur.y) + Number(cur.z)) / 3;
        const fillC = scoreToColor(mix, 0.3);
        const strokeC = scoreToColor(mix, 0.8);
        strokeFillPoly(ctx, pts, fillC, strokeC, 1.25);
      }
    }
  }

  /**
   * @param {any} [payload]
   */
  function paint(payload) {
    if (!canvasEl || !containerEl) return;
    const data = payload !== undefined ? payload : get(statespaceStore);
    const wBox = containerEl.clientWidth || 200;
    const hBox = containerEl.clientHeight || wBox;
    const side = Math.max(120, Math.floor(Math.min(wBox, hBox)));
    const dpr = window.devicePixelRatio || 1;
    canvasEl.width = Math.round(side * dpr);
    canvasEl.height = Math.round(side * dpr);
    canvasEl.style.width = `${side}px`;
    canvasEl.style.height = `${side}px`;
    const ctx = canvasEl.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw(ctx, side, side, data);
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

<div class="polar-chart" bind:this={containerEl}>
  <canvas bind:this={canvasEl}></canvas>
</div>

<style>
  .polar-chart {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0.5rem;
  }
  canvas {
    display: block;
  }
</style>
