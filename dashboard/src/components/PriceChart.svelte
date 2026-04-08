<script>
  import { onMount, onDestroy } from 'svelte';
  import { createChart, CrosshairMode } from 'lightweight-charts';
  import { palette } from '../lib/colors.js';
  import {
    BRICK_BASE_TS,
    buildTimeMapFromBars,
    mapBarsForChart,
    mapMarkersForChart,
    mapLineForChart,
    mapSegmentForChart,
    buildTTPTrail,
    levelLineData,
    levelLineFromEntry,
  } from '../lib/chartHelpers.js';
  import { chartStore, statusStore } from '../lib/stores.js';

  let containerEl;
  let chartEl;

  /** @type {import('lightweight-charts').IChartApi | null} */
  let chart = null;
  let candle, slSeries, ttpSeries, entrySeries, tp1Series, tp2Series;
  let fibLongSeries, fibMidSeries, fibShortSeries, priceLineSeries;

  /** @type {import('lightweight-charts').ISeriesApi<'Line'>[]} */
  let tradeSegmentSeries = [];
  let lastSegmentsSig = '';

  let barsRawRef = [];
  let hasFittedOnce = false;

  /** @type {ResizeObserver | null} */
  let resizeObs = null;

  function formatBrickTime(time) {
    const t = Number(time);
    if (!Number.isFinite(t)) return '';
    if (!Array.isArray(barsRawRef) || !barsRawRef.length) {
      const d = new Date(t * 1000);
      return fmtUTC(d);
    }
    const idx = Math.max(0, Math.round((t - BRICK_BASE_TS) / 60));
    if (idx >= barsRawRef.length) return 'B' + idx;
    const rt = Number(barsRawRef[idx].time);
    if (!Number.isFinite(rt)) return 'B' + idx;
    return fmtUTC(new Date(rt * 1000));
  }

  function fmtUTC(d) {
    const yy = d.getUTCFullYear();
    const mo = String(d.getUTCMonth() + 1).padStart(2, '0');
    const dd = String(d.getUTCDate()).padStart(2, '0');
    const hh = String(d.getUTCHours()).padStart(2, '0');
    const mm = String(d.getUTCMinutes()).padStart(2, '0');
    return `${yy}-${mo}-${dd} ${hh}:${mm}`;
  }

  function tickMarkFormat(time) {
    const t = Number(time);
    if (!Number.isFinite(t)) return '';
    const idx = Math.max(0, Math.round((t - BRICK_BASE_TS) / 60));
    if (!Array.isArray(barsRawRef) || idx >= barsRawRef.length) return 'B' + idx;
    const rt = Number(barsRawRef[idx].time);
    if (!Number.isFinite(rt)) return 'B' + idx;
    const d = new Date(rt * 1000);
    return String(d.getUTCDate()).padStart(2, '0') + '.' +
           String(d.getUTCMonth() + 1).padStart(2, '0') + ' ' +
           String(d.getUTCHours()).padStart(2, '0') + ':' +
           String(d.getUTCMinutes()).padStart(2, '0');
  }

  /**
   * Expose the chart's timeScale for RegimeBand synchronization.
   * @returns {import('lightweight-charts').ITimeScaleApi | null}
   */
  export function getTimeScale() {
    return chart ? chart.timeScale() : null;
  }

  /** @returns {import('lightweight-charts').IChartApi | null} */
  export function getChart() {
    return chart;
  }

  onMount(() => {
    chart = createChart(chartEl, {
      layout: {
        background: { color: palette.bg },
        textColor: palette.textDim,
        fontFamily: 'system-ui, -apple-system, sans-serif',
        fontSize: 11,
      },
      rightPriceScale: {
        borderColor: palette.border,
        scaleMargins: { top: 0.08, bottom: 0.08 },
      },
      grid: {
        vertLines: { color: palette.grid, style: 1 },
        horzLines: { color: palette.grid, style: 1 },
      },
      crosshair: {
        mode: CrosshairMode.Magnet,
        vertLine: {
          color: 'rgba(136, 146, 166, 0.3)',
          width: 1,
          style: 2,
          labelBackgroundColor: palette.bgPanel,
        },
        horzLine: {
          color: 'rgba(136, 146, 166, 0.3)',
          width: 1,
          style: 2,
          labelBackgroundColor: palette.bgPanel,
        },
      },
      timeScale: {
        borderColor: palette.border,
        timeVisible: false,
        secondsVisible: false,
        tickMarkFormatter: tickMarkFormat,
      },
      localization: {
        timeFormatter: formatBrickTime,
      },
    });

    /* ── Candlestick series ─────────────────────── */
    candle = chart.addCandlestickSeries({
      upColor: palette.green,
      downColor: palette.red,
      borderDownColor: palette.red,
      borderUpColor: palette.green,
      wickDownColor: 'rgba(248, 113, 113, 0.6)',
      wickUpColor: 'rgba(52, 211, 153, 0.6)',
    });

    /* ── Level lines: SL, TTP, Entry, TP1, TP2 ── */
    slSeries = chart.addLineSeries({
      color: palette.red,
      lineWidth: 2,
      lineStyle: 0,
      title: 'SL',
      lastValueVisible: true,
      priceLineVisible: false,
    });

    ttpSeries = chart.addLineSeries({
      color: palette.amber,
      lineWidth: 2,
      lineStyle: 2,  /* dashed */
      lineType: 1,
      title: 'TTP',
      lastValueVisible: true,
      priceLineVisible: false,
    });

    entrySeries = chart.addLineSeries({
      color: 'rgba(209, 213, 224, 0.5)',
      lineWidth: 1,
      lineStyle: 2,  /* dashed */
      title: 'Entry',
      lastValueVisible: true,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
    });

    tp1Series = chart.addLineSeries({
      color: palette.blue,
      lineWidth: 1,
      lineStyle: 2,
      title: 'TP1',
      lastValueVisible: true,
      priceLineVisible: false,
    });

    tp2Series = chart.addLineSeries({
      color: palette.purple,
      lineWidth: 1,
      lineStyle: 3,  /* large dashed */
      title: 'TP2',
      lastValueVisible: true,
      priceLineVisible: false,
    });

    /* ── Fibonacci levels ── */
    fibLongSeries = chart.addLineSeries({
      color: 'rgba(52, 211, 153, 0.35)',
      lineWidth: 1,
      lineStyle: 2,
      lastValueVisible: false,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
    });
    fibMidSeries = chart.addLineSeries({
      color: 'rgba(209, 213, 224, 0.2)',
      lineWidth: 1,
      lineStyle: 3,
      lastValueVisible: false,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
    });
    fibShortSeries = chart.addLineSeries({
      color: 'rgba(248, 113, 113, 0.35)',
      lineWidth: 1,
      lineStyle: 2,
      lastValueVisible: false,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
    });

    /* ── Current price line ── */
    priceLineSeries = chart.addLineSeries({
      color: 'rgba(154, 165, 177, 0.5)',
      lineWidth: 1,
      title: 'Last',
      lineStyle: 3,
      lastValueVisible: false,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
    });

    resizeObs = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (chart && width > 0 && height > 0) {
          chart.resize(width, height);
        }
      }
    });
    resizeObs.observe(containerEl);
  });

  onDestroy(() => {
    resizeObs?.disconnect();
    chart?.remove();
    chart = null;
  });

  const unsubChart = chartStore.subscribe((data) => {
    if (!data || !chart) return;

    const prevLogicalRange = hasFittedOnce
      ? chart.timeScale().getVisibleLogicalRange()
      : null;

    const barsRaw = data.bars || [];
    const { map: timeMap, timeAxis } = buildTimeMapFromBars(barsRaw);
    const bars = mapBarsForChart(barsRaw);
    barsRawRef = barsRaw;

    candle.setData(bars);
    candle.setMarkers(mapMarkersForChart(data.markers || [], timeAxis));

    const levels = data.levels || {};
    slSeries.setData(levelLineFromEntry(bars, levels.sl, levels, timeMap, barsRaw));

    const ttpTrailData = buildTTPTrail(bars, levels, data.ttp_trail_pct, timeMap, barsRaw);
    ttpSeries.setData(
      ttpTrailData.length > 0
        ? ttpTrailData
        : levelLineFromEntry(bars, levels.ttp, levels, timeMap, barsRaw)
    );
    entrySeries.setData(levelLineFromEntry(bars, levels.entry_px, levels, timeMap, barsRaw));
    tp1Series.setData(levelLineData(bars, levels.tp1));
    tp2Series.setData(levelLineData(bars, levels.tp2));

    const fibo = data.fibo || {};
    fibLongSeries.setData(mapLineForChart(fibo.long || [], timeMap, barsRaw));
    fibMidSeries.setData(mapLineForChart(fibo.mid || [], timeMap, barsRaw));
    fibShortSeries.setData(mapLineForChart(fibo.short || [], timeMap, barsRaw));

    const lastBar = bars.length ? bars[bars.length - 1] : null;
    if (lastBar) {
      let livePx = lastBar.close;
      const st = statusStoreVal;
      if (st?.status?.ticker?.mid != null) {
        const m = Number(st.status.ticker.mid);
        if (Number.isFinite(m)) livePx = m;
      }
      priceLineSeries.setData([
        { time: bars[0].time, value: livePx },
        { time: lastBar.time, value: livePx },
      ]);
    } else {
      priceLineSeries.setData([]);
    }

    const segments = data.segments || [];
    const segSig = JSON.stringify(
      segments.map((s) => [s.from_time, s.to_time, s.from_price, s.to_price, s.color, !!s.positive])
    );
    if (segSig !== lastSegmentsSig) {
      for (const s of tradeSegmentSeries) chart.removeSeries(s);
      tradeSegmentSeries.length = 0;
      for (const seg of segments) {
        const ls = chart.addLineSeries({
          color: seg.color || palette.textDim,
          lineWidth: 2,
          title: seg.positive ? 'Trade +' : 'Trade -',
        });
        ls.setData(mapSegmentForChart(seg, timeMap, barsRaw));
        tradeSegmentSeries.push(ls);
      }
      lastSegmentsSig = segSig;
    }

    if (!hasFittedOnce) {
      chart.timeScale().fitContent();
      hasFittedOnce = true;
    } else if (prevLogicalRange) {
      chart.timeScale().setVisibleLogicalRange(prevLogicalRange);
    }
  });

  let statusStoreVal = null;
  const unsubStatus = statusStore.subscribe((v) => {
    statusStoreVal = v;
  });

  onDestroy(() => {
    unsubChart();
    unsubStatus();
  });
</script>

<div class="chart-container" bind:this={containerEl}>
  <div class="chart" bind:this={chartEl}></div>
</div>

<style>
  .chart-container {
    flex: 1;
    position: relative;
    min-height: 0;
    border-radius: 4px;
    overflow: hidden;
  }
  .chart {
    position: absolute;
    inset: 0;
  }
</style>
