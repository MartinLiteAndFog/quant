import { useEffect, useRef } from "react";
import {
  ColorType,
  createChart,
  type IChartApi,
  type ISeriesApi,
  type LineData,
} from "lightweight-charts";
import type { BotSeries } from "../types";

interface Props {
  series: BotSeries[];
  visibleIds: Set<string>;
  mode: "trade" | "account";
  isolatedId: string | null;
  showMaxDd: boolean;
}

export function HeroChart({ series, visibleIds, mode, isolatedId, showMaxDd }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const linesRef = useRef<Map<string, ISeriesApi<"Line">>>(new Map());

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

    for (const bot of active) {
      const curve = mode === "trade" ? bot.trade_curve : bot.account_curve;
      const data: LineData[] = curve
        .filter((p) => Number.isFinite(p.t) && Number.isFinite(p.equity_pct))
        .map((p) => ({ time: p.t as LineData["time"], value: p.equity_pct }));
      if (!data.length) continue;

      const line = chart.addLineSeries({
        color: bot.color || "#c4a35a",
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: true,
        title: bot.display_name,
        crosshairMarkerRadius: 3,
      });
      line.setData(data);
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

    chart.timeScale().fitContent();
  }, [series, visibleIds, mode, isolatedId, showMaxDd]);

  return <div ref={containerRef} className="h-full w-full min-h-[420px]" />;
}
