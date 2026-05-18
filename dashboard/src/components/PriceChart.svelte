<script>
  import { onMount, onDestroy } from 'svelte';
  import { createChart, CrosshairMode } from 'lightweight-charts';
  import {
    BRICK_BASE_TS,
    buildTimeMapFromBars,
    mapBarsForChart,
    mapMarkersForChart,
    mapLineForChart,
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
      layout: { background: { color: '#000000' }, textColor: '#e0e0e8' },
      rightPriceScale: { borderColor: '#111118' },
      grid: {
        vertLines: { color: '#111118' },
        horzLines: { color: '#111118' },
      },
      crosshair: { mode: CrosshairMode.Magnet },
      timeScale: {
        borderColor: '#111118',
        timeVisible: false,
        secondsVisible: false,
        tickMarkFormatter: tickMarkFormat,
      },
      localization: {
        timeFormatter: formatBrickTime,
      },
    });

    candle = chart.addCandlestickSeries({
      upColor: '#2ecc71',
      downColor: '#f7768e',
      borderDownColor: '#f7768e',
      borderUpColor: '#2ecc71',
      wickDownColor: '#f7768e',
      wickUpColor: '#2ecc71',
    });

    slSeries = chart.addLineSeries({
      color: '#f7768e', lineWidth: 2, title: 'SL',
      lastValueVisible: true, priceLineVisible: false,
    });
    ttpSeries = chart.addLineSeries({
      color: '#e0af68', lineWidth: 2, lineStyle: 1, lineType: 1,
      title: 'TTP', lastValueVisible: true, priceLineVisible: false,
    });
    entrySeries = chart.addLineSeries({
      color: '#ffffff', lineWidth: 1, lineStyle: 0, title: 'Entry',
      lastValueVisible: true, priceLineVisible: false, crosshairMarkerVisible: false,
    });
    tp1Series = chart.addLineSeries({ color: '#7aa2f7', lineWidth: 2, title: 'TP1' });
    tp2Series = chart.addLineSeries({ color: '#bb9af7', lineWidth: 2, title: 'TP2' });
    fibLongSeries = chart.addLineSeries({
      color: '#2ecc71', lineWidth: 2, lineStyle: 0,
      lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false,
    });
    fibMidSeries = chart.addLineSeries({
      color: '#ffffff', lineWidth: 1, lineStyle: 2,
      lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false,
    });
    fibShortSeries = chart.addLineSeries({
      color: '#f7768e', lineWidth: 2, lineStyle: 0,
      lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false,
    });
    priceLineSeries = chart.addLineSeries({
      color: '#9aa5b1', lineWidth: 1, title: 'Last', lineStyle: 2,
      lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false,
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
  <!-- TODO: brick fill overlay canvas (animated last-brick filling) -->
</div>

<style>
  .chart-container {
    flex: 1;
    position: relative;
    min-height: 0;
  }
  .chart {
    position: absolute;
    inset: 0;
  }
</style>
