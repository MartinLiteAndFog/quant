import { useEffect, useRef } from "react";
import {
  createChart,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
  LineStyle,
  ColorType,
} from "lightweight-charts";
import type {
  ChartBar,
  ChartMarker,
  ChartSegment,
  ChartLevels,
} from "../../types/chart";

interface PriceChartProps {
  bars: ChartBar[];
  markers?: ChartMarker[];
  segments?: ChartSegment[];
  levels?: ChartLevels | Record<string, never>;
  ttpTrailPct?: number;
}

const CHART_BG = "#09090b";
const CHART_TEXT = "#a1a1aa";
const CHART_GRID = "#27272a";
const CANDLE_UP = "#22c55e";
const CANDLE_DOWN = "#ef4444";

type LinePoint = { time: UTCTimestamp; value: number };

function buildTTPTrail(
  bars: ChartBar[],
  levels: ChartLevels,
  trailPct: number,
): LinePoint[] {
  const entryPx = levels.entry_px;
  const side = String(levels.side ?? "").toLowerCase();
  if (!entryPx || !side) return [];

  const isLong = side === "long" || side === "l" || side === "1";
  const trail = trailPct > 0 ? trailPct : 0.012;

  const entryT = levels.entry_bar_ts;
  if (entryT == null) return [];

  let startIdx = bars.length - 1;
  for (let i = 0; i < bars.length; i++) {
    if (bars[i].time >= entryT) {
      startIdx = i;
      break;
    }
  }

  let bestFav = entryPx;
  const points: LinePoint[] = [];
  for (let i = startIdx; i < bars.length; i++) {
    const h = bars[i].high ?? bars[i].close;
    const l = bars[i].low ?? bars[i].close;
    if (isLong) {
      bestFav = Math.max(bestFav, h);
      points.push({
        time: bars[i].time as UTCTimestamp,
        value: bestFav * (1 - trail),
      });
    } else {
      bestFav = Math.min(bestFav, l);
      points.push({
        time: bars[i].time as UTCTimestamp,
        value: bestFav * (1 + trail),
      });
    }
  }
  return points;
}

function levelLineFromEntry(
  bars: ChartBar[],
  level: number | undefined,
  entryTs: number | undefined,
): LinePoint[] {
  if (!bars.length || level == null || !isFinite(level)) return [];
  let first = bars[0].time;
  if (entryTs != null) first = entryTs;
  const last = bars[bars.length - 1].time;
  return [
    { time: first as UTCTimestamp, value: level },
    { time: last as UTCTimestamp, value: level },
  ];
}

export default function PriceChart({
  bars,
  markers = [],
  segments = [],
  levels = {},
  ttpTrailPct = 0.012,
}: PriceChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const overlaySeriesRef = useRef<ISeriesApi<"Line">[]>([]);
  const fittedRef = useRef(false);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: CHART_BG },
        textColor: CHART_TEXT,
      },
      grid: {
        vertLines: { color: CHART_GRID },
        horzLines: { color: CHART_GRID },
      },
      autoSize: true,
      timeScale: { timeVisible: true },
    });

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: CANDLE_UP,
      downColor: CANDLE_DOWN,
      borderUpColor: CANDLE_UP,
      borderDownColor: CANDLE_DOWN,
      wickUpColor: CANDLE_UP,
      wickDownColor: CANDLE_DOWN,
    });

    chartRef.current = chart;
    candlestickRef.current = candlestickSeries;

    return () => {
      chart.remove();
      chartRef.current = null;
      candlestickRef.current = null;
      overlaySeriesRef.current = [];
      fittedRef.current = false;
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    const candlestickSeries = candlestickRef.current;
    if (!chart || !candlestickSeries) return;

    const candlestickData = bars.map((b) => ({
      time: b.time as UTCTimestamp,
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
    }));
    candlestickSeries.setData(candlestickData);

    if (candlestickData.length > 0 && !fittedRef.current) {
      chart.timeScale().fitContent();
      fittedRef.current = true;
    }

    const lwcMarkers = markers.map((m) => ({
      time: m.time as UTCTimestamp,
      position: m.position as "aboveBar" | "belowBar" | "inBar",
      shape: m.shape as "arrowUp" | "arrowDown" | "circle" | "square",
      color: m.color,
      text: m.text,
    }));
    candlestickSeries.setMarkers(lwcMarkers);

    // Remove old overlay series (levels + segments)
    for (const s of overlaySeriesRef.current) {
      chart.removeSeries(s);
    }
    overlaySeriesRef.current = [];

    const entryTs = (levels as ChartLevels).entry_bar_ts;
    const lvl = levels as ChartLevels;

    // SL — red line from entry
    const slData = levelLineFromEntry(bars, lvl.sl, entryTs);
    if (slData.length) {
      const sl = chart.addLineSeries({
        color: "#f7768e",
        lineWidth: 2,
        title: "SL",
        lastValueVisible: true,
        priceLineVisible: false,
      });
      sl.setData(slData);
      overlaySeriesRef.current.push(sl);
    }

    // TTP — trailing line following price
    const ttpData = buildTTPTrail(bars, lvl, ttpTrailPct);
    if (ttpData.length) {
      const ttp = chart.addLineSeries({
        color: "#e0af68",
        lineWidth: 2,
        lineStyle: LineStyle.Dashed,
        title: "TTP",
        lastValueVisible: true,
        priceLineVisible: false,
      });
      ttp.setData(ttpData);
      overlaySeriesRef.current.push(ttp);
    } else {
      // Fallback: static TTP level if no trail can be computed
      const ttpStatic = levelLineFromEntry(bars, lvl.ttp, entryTs);
      if (ttpStatic.length) {
        const ttp = chart.addLineSeries({
          color: "#e0af68",
          lineWidth: 2,
          lineStyle: LineStyle.Dashed,
          title: "TTP",
          lastValueVisible: true,
          priceLineVisible: false,
        });
        ttp.setData(ttpStatic);
        overlaySeriesRef.current.push(ttp);
      }
    }

    // Entry — white line from entry
    const entryData = levelLineFromEntry(bars, lvl.entry_px, entryTs);
    if (entryData.length) {
      const entry = chart.addLineSeries({
        color: "#ffffff",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        title: "Entry",
        lastValueVisible: true,
        priceLineVisible: false,
      });
      entry.setData(entryData);
      overlaySeriesRef.current.push(entry);
    }

    // TP1 — blue line
    const tp1Data = levelLineFromEntry(bars, lvl.tp1, entryTs);
    if (tp1Data.length) {
      const tp1 = chart.addLineSeries({
        color: "#3b82f6",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        title: "TP1",
        lastValueVisible: true,
        priceLineVisible: false,
      });
      tp1.setData(tp1Data);
      overlaySeriesRef.current.push(tp1);
    }

    // TP2 — purple line
    const tp2Data = levelLineFromEntry(bars, lvl.tp2, entryTs);
    if (tp2Data.length) {
      const tp2 = chart.addLineSeries({
        color: "#a855f7",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        title: "TP2",
        lastValueVisible: true,
        priceLineVisible: false,
      });
      tp2.setData(tp2Data);
      overlaySeriesRef.current.push(tp2);
    }

    // Trade segments
    for (const seg of segments) {
      const lineSeries = chart.addLineSeries({
        color: seg.color,
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
      });
      lineSeries.setData([
        { time: seg.from_time as UTCTimestamp, value: seg.from_price },
        { time: seg.to_time as UTCTimestamp, value: seg.to_price },
      ]);
      overlaySeriesRef.current.push(lineSeries);
    }
  }, [bars, markers, segments, levels, ttpTrailPct]);

  return <div ref={containerRef} className="h-[500px] w-full" />;
}
