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
  ChartFibo,
  FiboLine,
} from "../../types/chart";

interface PriceChartProps {
  bars: ChartBar[];
  markers?: ChartMarker[];
  segments?: ChartSegment[];
  levels?: ChartLevels;
  ttpTrailPct?: number;
  fibo?: ChartFibo;
  livePrice?: number | null;
}

const CHART_BG = "#09090b";
const CHART_TEXT = "#71717a";
const CHART_GRID = "rgba(63, 63, 70, 0.18)";
const CANDLE_UP = "#22c55e";
const CANDLE_DOWN = "#ef4444";

type LinePoint = { time: UTCTimestamp; value: number };

function parseTs(v: string | number | undefined | null): number | null {
  if (v == null) return null;
  if (typeof v === "number") return v;
  const d = new Date(v);
  if (isNaN(d.getTime())) return null;
  return Math.floor(d.getTime() / 1000);
}

function resolveLevels(raw: ChartLevels | undefined): {
  entry_px: number | undefined;
  entry_bar_ts: number | undefined;
  side: string;
  sl: number | undefined;
  ttp: number | undefined;
  tp1: number | undefined;
  tp2: number | undefined;
  mode: string;
} {
  const t = raw?.terminal;
  const entry_px = t?.entry_px ?? raw?.entry_px;
  const entry_bar_ts_raw = t?.entry_bar_ts ?? raw?.entry_bar_ts;
  const entry_bar_ts = parseTs(entry_bar_ts_raw) ?? undefined;
  const side = (t?.side ?? raw?.side ?? "").toLowerCase();
  const sl_raw = t?.sl ?? raw?.sl;
  const sl = sl_raw != null ? Number(sl_raw) : undefined;
  const ttp = t?.ttp ?? raw?.ttp;
  const tp1 = raw?.tp1;
  const tp2 = raw?.tp2;
  const mode = t?.mode ?? raw?.mode ?? "";

  return {
    entry_px,
    entry_bar_ts,
    side,
    sl: sl != null && isFinite(sl) ? sl : undefined,
    ttp,
    tp1,
    tp2,
    mode,
  };
}

function buildTTPTrail(
  bars: ChartBar[],
  entryPx: number,
  side: string,
  entryT: number,
  trailPct: number
): LinePoint[] {
  const isLong = side === "long" || side === "l" || side === "1";
  const trail = trailPct > 0 ? trailPct : 0.012;

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
  entryTs: number | undefined
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

function fiboToPoints(points: FiboLine[] | undefined): LinePoint[] {
  if (!points?.length) return [];

  const seen = new Set<number>();
  const out: LinePoint[] = [];

  for (const p of points) {
    if (p.time == null || p.value == null || !isFinite(p.value)) continue;
    const t = Number(p.time);
    if (seen.has(t)) continue;
    seen.add(t);
    out.push({ time: t as UTCTimestamp, value: p.value });
  }

  return out;
}

export default function PriceChart({
  bars,
  markers = [],
  segments = [],
  levels,
  ttpTrailPct = 0.012,
  fibo,
  livePrice,
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
        vertLines: { color: CHART_GRID, visible: true },
        horzLines: { color: CHART_GRID, visible: true },
      },
      autoSize: true,
      timeScale: {
        timeVisible: true,
        borderVisible: false,
        ticksVisible: false,
      },
      rightPriceScale: {
        borderVisible: false,
        scaleMargins: { top: 0.08, bottom: 0.08 },
      },
      leftPriceScale: {
        visible: false,
      },
      crosshair: {
        vertLine: { visible: false },
        horzLine: { visible: false },
      },
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

    candlestickSeries.setMarkers(
      markers.map((m) => ({
        time: m.time as UTCTimestamp,
        position: m.position as "aboveBar" | "belowBar" | "inBar",
        shape: m.shape as "arrowUp" | "arrowDown" | "circle" | "square",
        color: m.color,
        text: m.text,
      }))
    );

    for (const s of overlaySeriesRef.current) {
      try {
        chart.removeSeries(s);
      } catch {
        // already removed
      }
    }
    overlaySeriesRef.current = [];

    const lvl = resolveLevels(levels);

    const slData = levelLineFromEntry(bars, lvl.sl, lvl.entry_bar_ts);
    if (slData.length) {
      const s = chart.addLineSeries({
        color: "#f7768e",
        lineWidth: 2,
        title: "SL",
        lastValueVisible: true,
        priceLineVisible: false,
      });
      s.setData(slData);
      overlaySeriesRef.current.push(s);
    }

    if (lvl.entry_px && lvl.side && lvl.entry_bar_ts) {
      const ttpData = buildTTPTrail(
        bars,
        lvl.entry_px,
        lvl.side,
        lvl.entry_bar_ts,
        ttpTrailPct
      );
      if (ttpData.length) {
        const s = chart.addLineSeries({
          color: "#e0af68",
          lineWidth: 2,
          lineStyle: LineStyle.Dashed,
          title: "TTP",
          lastValueVisible: true,
          priceLineVisible: false,
        });
        s.setData(ttpData);
        overlaySeriesRef.current.push(s);
      }
    } else if (lvl.ttp != null) {
      const ttpStatic = levelLineFromEntry(bars, lvl.ttp, lvl.entry_bar_ts);
      if (ttpStatic.length) {
        const s = chart.addLineSeries({
          color: "#e0af68",
          lineWidth: 2,
          lineStyle: LineStyle.Dashed,
          title: "TTP",
          lastValueVisible: true,
          priceLineVisible: false,
        });
        s.setData(ttpStatic);
        overlaySeriesRef.current.push(s);
      }
    }

    const entryData = levelLineFromEntry(bars, lvl.entry_px, lvl.entry_bar_ts);
    if (entryData.length) {
      const s = chart.addLineSeries({
        color: "#ffffff",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        title: "Entry",
        lastValueVisible: true,
        priceLineVisible: false,
      });
      s.setData(entryData);
      overlaySeriesRef.current.push(s);
    }

    const tp1Data = levelLineFromEntry(bars, lvl.tp1, lvl.entry_bar_ts);
    if (tp1Data.length) {
      const s = chart.addLineSeries({
        color: "#7aa2f7",
        lineWidth: 2,
        title: "TP1",
        lastValueVisible: true,
        priceLineVisible: false,
      });
      s.setData(tp1Data);
      overlaySeriesRef.current.push(s);
    }

    const tp2Data = levelLineFromEntry(bars, lvl.tp2, lvl.entry_bar_ts);
    if (tp2Data.length) {
      const s = chart.addLineSeries({
        color: "#bb9af7",
        lineWidth: 2,
        title: "TP2",
        lastValueVisible: true,
        priceLineVisible: false,
      });
      s.setData(tp2Data);
      overlaySeriesRef.current.push(s);
    }

    if (fibo) {
      const configs: {
        key: "long" | "mid" | "short";
        color: string;
        width: number;
        style?: number;
      }[] = [
        { key: "long", color: "#2ecc71", width: 2 },
        { key: "mid", color: "#ffffff", width: 1, style: LineStyle.Dashed },
        { key: "short", color: "#f7768e", width: 2 },
      ];

      for (const cfg of configs) {
        const pts = fiboToPoints(fibo[cfg.key]);
        if (pts.length >= 2) {
          const s = chart.addLineSeries({
            color: cfg.color,
            lineWidth: cfg.width as 1 | 2,
            lineStyle: cfg.style,
            lastValueVisible: false,
            priceLineVisible: false,
            crosshairMarkerVisible: false,
          });
          s.setData(pts);
          overlaySeriesRef.current.push(s);
        }
      }
    }

    if (livePrice != null && isFinite(livePrice) && bars.length >= 2) {
      const s = chart.addLineSeries({
        color: "#9aa5b1",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        lastValueVisible: false,
        priceLineVisible: false,
        crosshairMarkerVisible: false,
      });
      s.setData([
        { time: bars[0].time as UTCTimestamp, value: livePrice },
        { time: bars[bars.length - 1].time as UTCTimestamp, value: livePrice },
      ]);
      overlaySeriesRef.current.push(s);
    }

    for (const seg of segments) {
      const s = chart.addLineSeries({
        color: seg.color,
        lineWidth: 2,
        title: seg.positive ? "Trade +" : "Trade -",
        priceLineVisible: false,
        lastValueVisible: false,
      });
      s.setData([
        { time: seg.from_time as UTCTimestamp, value: seg.from_price },
        { time: seg.to_time as UTCTimestamp, value: seg.to_price },
      ]);
      overlaySeriesRef.current.push(s);
    }
  }, [bars, markers, segments, levels, ttpTrailPct, fibo, livePrice]);

  return <div ref={containerRef} className="h-full w-full" />;
}