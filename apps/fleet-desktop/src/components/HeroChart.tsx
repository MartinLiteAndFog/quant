import { useEffect, useRef, useState } from "react";
import {
  ColorType,
  createChart,
  type IChartApi,
  type ISeriesApi,
  type LineData,
} from "lightweight-charts";
import type { BotSeries, ChartMode } from "../types";

interface Props {
  series: BotSeries[];
  visibleIds: Set<string>;
  mode: ChartMode;
  isolatedId: string | null;
  showMaxDd: boolean;
}

function toLineData(
  points: Array<{ t: number; value: number }>,
): LineData[] {
  // LWC requires strictly ascending unique times.
  const byT = new Map<number, number>();
  for (const p of points) {
    if (!Number.isFinite(p.t) || !Number.isFinite(p.value)) continue;
    byT.set(Math.floor(p.t), p.value);
  }
  return [...byT.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([t, value]) => ({ time: t as LineData["time"], value }));
}

export function HeroChart({ series, visibleIds, mode, isolatedId, showMaxDd }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const linesRef = useRef<Map<string, ISeriesApi<"Line">>>(new Map());
  const [empty, setEmpty] = useState(false);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#8a8680",
        fontFamily: '"IBM Plex Sans", sans-serif',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.04)" },
        horzLines: { color: "rgba(255,255,255,0.04)" },
      },
      rightPriceScale: {
        borderVisible: false,
        scaleMargins: { top: 0.12, bottom: 0.08 },
      },
      timeScale: {
        borderVisible: false,
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: {
        mode: 1,
        vertLine: { color: "rgba(232,228,220,0.25)", labelBackgroundColor: "#1a1c22" },
        horzLine: { color: "rgba(232,228,220,0.18)", labelBackgroundColor: "#1a1c22" },
      },
      handleScroll: { mouseWheel: true, pressedMouseMove: true },
      handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
    });
    chartRef.current = chart;

    const ro = new ResizeObserver(() => {
      if (!containerRef.current) return;
      chart.applyOptions({
        width: containerRef.current.clientWidth,
        height: containerRef.current.clientHeight,
      });
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      linesRef.current.clear();
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    for (const [id, line] of linesRef.current) {
      chart.removeSeries(line);
      linesRef.current.delete(id);
    }

    const active = series.filter((s) => {
      if (isolatedId) return s.id === isolatedId;
      return visibleIds.has(s.id);
    });

    let plotted = 0;
    for (const bot of active) {
      let raw: Array<{ t: number; value: number }> = [];
      if (mode === "account_abs") {
        raw = (bot.account_curve_abs || []).map((p) => ({ t: p.t, value: p.equity }));
      } else {
        const curve = mode === "trade" ? bot.trade_curve : bot.account_curve;
        raw = curve.map((p) => ({ t: p.t, value: p.equity_pct }));
      }
      const data = toLineData(raw);
      if (data.length < 1) continue;
      plotted += 1;

      // Single snapshot still needs two times for LWC to stroke a segment.
      const lineData =
        data.length === 1
          ? [
              data[0],
              { ...data[0], time: ((data[0].time as number) + 60) as LineData["time"] },
            ]
          : data;

      const line = chart.addLineSeries({
        color: bot.color || "#c4a35a",
        lineWidth: 2,
        lineType: 0, // simple continuous stroke (not stepped)
        lineVisible: true,
        pointMarkersVisible: false,
        priceLineVisible: false,
        lastValueVisible: true,
        title: bot.display_name,
        crosshairMarkerVisible: true,
        crosshairMarkerRadius: 4,
        priceFormat: { type: "price", precision: 2, minMove: 0.01 },
      });
      line.setData(lineData);
      linesRef.current.set(bot.id, line);

      if (showMaxDd && mode === "trade" && bot.stats.max_drawdown_pct > 0 && data.length > 1) {
        let peak = -Infinity;
        let peakIdx = 0;
        let troughIdx = 0;
        let maxDd = 0;
        const vals = data.map((d) => d.value);
        for (let i = 0; i < vals.length; i++) {
          if (vals[i] > peak) {
            peak = vals[i];
            peakIdx = i;
          }
          const dd = peak - vals[i];
          if (dd > maxDd) {
            maxDd = dd;
            troughIdx = i;
          }
        }
        if (maxDd > 0) {
          line.setMarkers([
            {
              time: data[peakIdx].time,
              position: "aboveBar",
              color: bot.color || "#c4a35a",
              shape: "arrowDown",
              text: "peak",
            },
            {
              time: data[troughIdx].time,
              position: "belowBar",
              color: "#a85a4a",
              shape: "arrowUp",
              text: `DD ${bot.stats.max_drawdown_pct.toFixed(1)}%`,
            },
          ]);
        }
      }
    }

    setEmpty(plotted === 0);
    if (plotted > 0) chart.timeScale().fitContent();
  }, [series, visibleIds, mode, isolatedId, showMaxDd]);

  return (
    <div className="relative h-full w-full min-h-[420px]">
      <div ref={containerRef} className="h-full w-full min-h-[420px]" />
      {empty && (
        <p className="pointer-events-none absolute inset-0 flex items-center justify-center px-6 text-center text-[12px] text-[var(--muted)]">
          No equity line in this window. Use Equity $ + ALL, then Refresh.
        </p>
      )}
    </div>
  );
}
