<script>
  import { onMount, onDestroy } from 'svelte';
  import { chartStore, equityEventsStore, refreshEquityEvents } from '../lib/stores.js';

  const ranges = [
    { value: '24h', label: '24h' },
    { value: '7d', label: '7d' },
    { value: '30d', label: '30d' },
    { value: 'all', label: 'All' },
  ];

  const KU_FILL = 'rgba(122, 162, 247, 0.3)';
  const KU_LINE = 'rgb(122, 162, 247)';

  /** @type {HTMLCanvasElement | undefined} */
  let canvasEl;
  /** @type {HTMLDivElement | undefined} */
  let containerEl;

  /** @type {string | null} */
  let activeRange = null;

  let venueLabel = '';
  let pctText = '';
  let pctClass = '';

  /** @type {{ x: number, y: number, text: string } | null} */
  let tooltip = null;

  /** @type {ResizeObserver | null} */
  let resizeObs = null;

  /** @type {(() => void) | undefined} */
  let unsubChart;
  /** @type {(() => void) | undefined} */
  let unsubEq;

  /** @type {unknown} */
  let latestChart = null;
  /** @type {unknown} */
  let latestEqEv = null;

  /** @type {Array<{ cx: number, cy: number, text: string }>} */
  let hitPoints = [];

  /**
   * @param {unknown} p
   * @returns {number}
   */
  function pointTimeSec(p) {
    const o = /** @type {Record<string, unknown>} */ (p);
    const raw = o?.time ?? o?.ts;
    if (typeof raw === 'number' && Number.isFinite(raw)) return raw;
    const n = Number(raw);
    if (Number.isFinite(n)) return n;
    if (typeof raw === 'string') {
      const ms = Date.parse(raw);
      if (!Number.isNaN(ms)) return ms / 1000;
    }
    return 0;
  }

  /**
   * @param {unknown} p
   * @returns {{ time: number, equity: number } | null}
   */
  function normalizePoint(p) {
    const o = /** @type {Record<string, unknown>} */ (p);
    const eq = Number(o?.equity);
    if (!Number.isFinite(eq)) return null;
    const time = pointTimeSec(p);
    return { time, equity: eq };
  }

  /**
   * @param {Array<{ time: number, equity: number }>} sortedPts
   * @param {number} t
   */
  function locfAt(sortedPts, t) {
    let last = 0;
    let found = false;
    for (let i = 0; i < sortedPts.length; i++) {
      const p = sortedPts[i];
      if (p.time > t) break;
      last = p.equity;
      found = true;
    }
    return found ? last : 0;
  }

  /**
   * @param {unknown[] | undefined} raw
   * @returns {Array<{ time: number, equity: number }>}
   */
  function sortedSeries(raw) {
    if (!Array.isArray(raw)) return [];
    const out = [];
    for (const p of raw) {
      const n = normalizePoint(p);
      if (n) out.push(n);
    }
    out.sort((a, b) => a.time - b.time);
    return out;
  }

  /** @param {number} x */
  function fmtMoney(x) {
    if (!Number.isFinite(x)) return '—';
    return `$${x.toFixed(2)}`;
  }

  /** @param {number} t */
  function fmtTime(t) {
    if (!Number.isFinite(t) || t <= 0) return '';
    const d = new Date(t * 1000);
    return d.toISOString().slice(0, 16).replace('T', ' ') + ' UTC';
  }

  function layoutCanvas() {
    if (!containerEl || !canvasEl) return null;
    const w = containerEl.clientWidth;
    const h = 130;
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
   * @param {number[]} ys
   * @param {number} h
   */
  function yScale(ys, h) {
    let minE = Math.min(...ys);
    let maxE = Math.max(...ys);
    if (!Number.isFinite(minE) || !Number.isFinite(maxE)) {
      minE = 0;
      maxE = 1;
    }
    if (minE === maxE) {
      minE -= 1;
      maxE += 1;
    }
    const pad = (maxE - minE) * 0.06;
    minE -= pad;
    maxE += pad;
    const span = maxE - minE || 1;
    /** @param {number} v */
    return (v) => h - ((v - minE) / span) * h;
  }

  /**
   * @param {CanvasRenderingContext2D} ctx
   * @param {number[]} xs
   * @param {number[]} y0
   * @param {number[]} y1
   * @param {string} fill
   */
  function fillBetween(ctx, xs, y0, y1, fill) {
    if (xs.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(xs[0], y0[0]);
    for (let i = 1; i < xs.length; i++) ctx.lineTo(xs[i], y0[i]);
    for (let i = xs.length - 1; i >= 0; i--) ctx.lineTo(xs[i], y1[i]);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
  }

  /**
   * @param {CanvasRenderingContext2D} ctx
   * @param {number[]} xs
   * @param {number[]} ys
   * @param {string} stroke
   */
  function strokeLine(ctx, xs, ys, stroke) {
    if (xs.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(xs[0], ys[0]);
    for (let i = 1; i < xs.length; i++) ctx.lineTo(xs[i], ys[i]);
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  function drawDefault(layout, data) {
    const { ctx, w, h } = layout;
    const raw =
      (Array.isArray(data?.equity_total) && data.equity_total) ||
      (Array.isArray(data?.equity_combined) && data.equity_combined) ||
      [];
    const points = sortedSeries(raw);
    if (points.length < 2) {
      venueLabel = '';
      pctText = '';
      pctClass = '';
      hitPoints = [];
      return;
    }

    const comps = Array.isArray(data?.equity_components) ? data.equity_components : [];
    let kuPts = [];
    for (const c of comps) {
      const key = String(c?.key || '').toLowerCase();
      const pts = sortedSeries(c?.points);
      if (key === 'kucoin') kuPts = pts;
    }

    const n = points.length;
    const xs = points.map((_, i) => (n <= 1 ? w / 2 : (i / (n - 1)) * w));

    const kuStack = points.map((p) => locfAt(kuPts, p.time));

    const totals = points.map((p, i) => {
      const t = p.equity;
      const sum = kuStack[i];
      if (sum > 0 && Math.abs(t - sum) / sum > 0.15) return t;
      return sum > 0 ? sum : t;
    });

    const yMinData = Math.min(0, ...totals, ...kuStack, ...points.map((p) => p.equity));
    const yMaxData = Math.max(...totals, ...points.map((p) => p.equity), 1e-9);
    const pad = (yMaxData - yMinData) * 0.06 || 1;
    const minE = yMinData - pad;
    const maxE = yMaxData + pad;
    const span = maxE - minE || 1;
    /** @param {number} v */
    const yMap = (v) => h - ((v - minE) / span) * h;

    const xsf = xs;
    const yKu0 = kuStack.map(() => yMap(0));
    const yKu1 = kuStack.map((v) => yMap(v));

    fillBetween(ctx, xsf, yKu0, yKu1, KU_FILL);

    const lineYs = points.map((p) => yMap(p.equity));
    const firstEq = points[0].equity;
    const lastEq = points[points.length - 1].equity;
    const lineStroke = lastEq >= firstEq ? '#2ecc71' : '#f7768e';
    strokeLine(ctx, xs, lineYs, lineStroke);

    if (Number.isFinite(firstEq) && firstEq !== 0) {
      const pct = ((lastEq - firstEq) / firstEq) * 100;
      pctText = `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`;
      pctClass = pct >= 0 ? 'pos' : 'neg';
    } else {
      pctText = '';
      pctClass = '';
    }

    venueLabel = `KuCoin: ${fmtMoney(kuPts.length ? kuPts[kuPts.length - 1].equity : 0)}`;

    hitPoints = points.map((p, i) => ({
      cx: xs[i],
      cy: lineYs[i],
      text: `${fmtTime(p.time)}  ${fmtMoney(p.equity)}`,
    }));
  }

  /**
   * @param {unknown} ev
   */
  function eventTsSec(ev) {
    const o = /** @type {Record<string, unknown>} */ (ev);
    const raw = o?.ts;
    if (typeof raw === 'string') {
      const ms = Date.parse(raw);
      if (!Number.isNaN(ms)) return ms / 1000;
    }
    return pointTimeSec(ev);
  }

  function drawEvents(layout, eqPayload) {
    const { ctx, w, h } = layout;
    const events = Array.isArray(eqPayload?.events) ? eqPayload.events : [];
    if (events.length < 1) {
      venueLabel = '';
      pctText = '';
      pctClass = '';
      hitPoints = [];
      return;
    }

    /** @type {Map<string, Array<{ t: number, eq: number }>>} */
    const byVenue = new Map();
    for (const ev of events) {
      const o = /** @type {Record<string, unknown>} */ (ev);
      const venue = String(o?.venue || '').toLowerCase();
      const t = eventTsSec(ev);
      const eq = Number(o?.equity);
      if (!Number.isFinite(eq)) continue;
      if (!byVenue.has(venue)) byVenue.set(venue, []);
      byVenue.get(venue).push({ t, eq });
    }

    for (const arr of byVenue.values()) {
      arr.sort((a, b) => a.t - b.t);
    }

    const kuKey = [...byVenue.keys()].find((k) => k.includes('kucoin')) || 'kucoin';
    const allTs = new Set();
    for (const arr of byVenue.values()) {
      for (const row of arr) allTs.add(row.t);
    }
    const uniqueTs = [...allTs].sort((a, b) => a - b);
    if (uniqueTs.length < 2) {
      const midX = w / 2;
      let yOff = 0;
      for (const [venue, arr] of byVenue) {
        if (!arr.length) continue;
        const color = venue.includes('kucoin') ? KU_LINE : '#888';
        const cy = h / 2 + yOff;
        yOff += 14;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(midX, cy, 3, 0, Math.PI * 2);
        ctx.fill();
      }
      const kuLast = (byVenue.get(kuKey) || []).at(-1)?.eq ?? 0;
      venueLabel = `KuCoin: ${fmtMoney(kuLast)}`;
      pctText = '';
      pctClass = '';
      hitPoints = [];
      return;
    }

    const n = uniqueTs.length;
    const xs = uniqueTs.map((_, i) => (n <= 1 ? w / 2 : (i / (n - 1)) * w));

    /** @param {string} key */
    function seriesForVenue(key) {
      const arr = byVenue.get(key) || [];
      let j = 0;
      let last = 0;
      let ok = false;
      return uniqueTs.map((t) => {
        while (j < arr.length && arr[j].t <= t) {
          last = arr[j].eq;
          ok = true;
          j++;
        }
        return ok ? last : 0;
      });
    }

    const kuSeries = seriesForVenue(kuKey);

    const totals = uniqueTs.map((_, i) => kuSeries[i]);
    const ys = [...totals, ...kuSeries];
    const yMap = yScale(ys, h);

    const kuYs = kuSeries.map((v) => yMap(v));

    strokeLine(ctx, xs, kuYs, KU_LINE);

    const firstT = totals[0];
    const lastT = totals[totals.length - 1];
    if (Number.isFinite(firstT) && firstT !== 0) {
      const pct = ((lastT - firstT) / firstT) * 100;
      pctText = `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`;
      pctClass = pct >= 0 ? 'pos' : 'neg';
    } else {
      pctText = '';
      pctClass = '';
    }

    const kuLast = kuSeries[kuSeries.length - 1] ?? 0;
    venueLabel = `KuCoin: ${fmtMoney(kuLast)}`;

    hitPoints = uniqueTs.map((t, i) => ({
      cx: xs[i],
      cy: yMap(totals[i]),
      text: `${fmtTime(t)}  total ${fmtMoney(totals[i])}`,
    }));
  }

  function draw() {
    const layout = layoutCanvas();
    if (!layout) return;
    const { ctx, w, h } = layout;
    ctx.clearRect(0, 0, w, h);
    hitPoints = [];

    if (activeRange) {
      drawEvents(layout, latestEqEv);
    } else {
      drawDefault(layout, latestChart);
    }
  }

  /**
   * @param {string} r
   */
  function toggleRange(r) {
    if (activeRange === r) {
      activeRange = null;
      tooltip = null;
      draw();
      return;
    }
    activeRange = r;
    refreshEquityEvents(r);
    tooltip = null;
    draw();
  }

  /** @param {MouseEvent} e */
  function onMove(e) {
    if (!canvasEl || hitPoints.length === 0) {
      tooltip = null;
      return;
    }
    const rect = canvasEl.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let best = hitPoints[0];
    let bestD = Math.abs(mx - best.cx);
    for (let i = 1; i < hitPoints.length; i++) {
      const d = Math.abs(mx - hitPoints[i].cx);
      if (d < bestD) {
        bestD = d;
        best = hitPoints[i];
      }
    }
    tooltip = {
      x: e.clientX - rect.left + 8,
      y: e.clientY - rect.top + 8,
      text: best.text,
    };
  }

  function onLeave() {
    tooltip = null;
  }

  onMount(() => {
    unsubChart = chartStore.subscribe((d) => {
      latestChart = d;
      draw();
    });
    unsubEq = equityEventsStore.subscribe((d) => {
      latestEqEv = d;
      draw();
    });
    resizeObs = new ResizeObserver(() => draw());
    if (containerEl) resizeObs.observe(containerEl);
    window.addEventListener('resize', draw);
    draw();
  });

  onDestroy(() => {
    unsubChart?.();
    unsubEq?.();
    resizeObs?.disconnect();
    window.removeEventListener('resize', draw);
  });
</script>

<div class="equity-bar" bind:this={containerEl}>
  <div class="equity-header">
    <span class="venue-labels">{venueLabel}</span>
    <div class="equity-header-right">
      <div class="range-tabs">
        {#each ranges as r}
          <button
            type="button"
            class:active={activeRange === r.value}
            on:click={() => toggleRange(r.value)}
          >{r.label}</button>
        {/each}
      </div>
      {#if pctText}
        <span class="pct-badge" class:pos={pctClass === 'pos'} class:neg={pctClass === 'neg'}>{pctText}</span>
      {/if}
    </div>
  </div>
  <canvas bind:this={canvasEl} on:mousemove={onMove} on:mouseleave={onLeave} aria-label="Equity curve"></canvas>
  {#if tooltip}
    <div class="equity-tooltip" style="left:{tooltip.x}px;top:{tooltip.y}px;">{tooltip.text}</div>
  {/if}
</div>

<style>
  .equity-bar {
    position: relative;
    height: 130px;
    width: 100%;
  }
  .equity-header {
    position: absolute;
    top: 4px;
    left: 8px;
    right: 8px;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    z-index: 1;
    pointer-events: none;
  }
  .equity-header-right {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 2px;
    pointer-events: auto;
  }
  .venue-labels {
    font-size: 0.72rem;
    color: var(--muted);
    font-family: ui-monospace, monospace;
  }
  .range-tabs {
    display: flex;
    gap: 2px;
  }
  .range-tabs button {
    background: #111118;
    color: var(--text);
    border: 1px solid #222;
    border-radius: 3px;
    padding: 2px 8px;
    font-size: 0.7rem;
    cursor: pointer;
  }
  .range-tabs button.active {
    background: var(--blue);
    color: white;
  }
  .pct-badge {
    font-size: 11px;
    font-weight: 700;
    line-height: 1.2;
  }
  .pct-badge.pos {
    color: var(--green);
  }
  .pct-badge.neg {
    color: var(--red);
  }
  canvas {
    width: 100%;
    height: 100%;
    display: block;
  }
  .equity-tooltip {
    position: absolute;
    z-index: 2;
    pointer-events: none;
    font-size: 0.65rem;
    font-family: ui-monospace, monospace;
    color: var(--text);
    background: rgba(17, 17, 24, 0.95);
    border: 1px solid #222;
    border-radius: 3px;
    padding: 3px 6px;
    white-space: nowrap;
    max-width: 280px;
    overflow: hidden;
    text-overflow: ellipsis;
  }
</style>
