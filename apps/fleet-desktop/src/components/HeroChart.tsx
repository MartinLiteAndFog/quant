import { useEffect, useRef, useState } from "react";
import {
  ColorType,
  createChart,
  type IChartApi,
  type ISeriesApi,
  type LineData,
} from "lightweight-charts";
import type { BotSeries, ChartMode, PortfolioSeries } from "../types";

interface Props {
  series: BotSeries[];
  portfolio: PortfolioSeries | null;
  visibleIds: Set<string>;
  mode: ChartMode;
  isolatedId: string | null;
  showMaxDd: boolean;
  showPortfolio: boolean;
}

const PORTFOLIO_ID = "__portfolio__";

function toLineData(points: Array<{ t: number; value: number }>): LineData[] {
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

export function HeroChart({
  series,
  portfolio,
  visibleIds,
  mode,
  isolatedId,
  showMaxDd,
  showPortfolio,
}: Props) {
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
        vertLines: { color: "rgba(255,255,255,0.035)" },
        horzLines: { color: "rgba(255,255,255,0.035)" },
      },
      rightPriceScale: {
        borderVisible: false,
        scaleMargins: { top: 0.1, bottom: 0.08 },
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

    const plotLine = (
      id: string,
      title: string,
      color: string,
      raw: Array<{ t: number; value: number }>,
      opts?: { width?: number; lastValueVisible?: boolean },
    ) => {
      const data = toLineData(raw);
      if (data.length < 1) return;
      plotted += 1;
      const line = chart.addLineSeries({
        color,
        lineWidth: (opts?.width ?? 2) as 1 | 2 | 3 | 4,
        lineVisible: true,
        pointMarkersVisible: false,
        priceLineVisible: false,
        lastValueVisible: opts?.lastValueVisible ?? true,
        title,
        crosshairMarkerVisible: true,
        crosshairMarkerRadius: 4,
        priceFormat: { type: "price", precision: 2, minMove: 0.01 },
      });
      line.setData(data);
      linesRef.current.set(id, line);
      return { line, data };
    };

    for (const bot of active) {
      let raw: Array<{ t: number; value: number }> = [];
      if (mode === "account_abs") {
        raw = (bot.account_curve_abs || []).map((p) => ({ t: p.t, value: p.equity }));
      } else {
        const curve = mode === "trade" ? bot.trade_curve : bot.account_curve;
        raw = curve.map((p) => ({ t: p.t, value: p.equity_pct }));
      }
      const plottedBot = plotLine(bot.id, bot.display_name, bot.color || "#c4a35a", raw);
      if (
        plottedBot &&
        showMaxDd &&
        mode === "trade" &&
        bot.stats.max_drawdown_pct > 0 &&
        plottedBot.data.length > 1
      ) {
        let peak = -Infinity;
        let peakIdx = 0;
        let troughIdx = 0;
        let maxDd = 0;
        const vals = plottedBot.data.map((d) => d.value);
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
          plottedBot.line.setMarkers([
            {
              time: plottedBot.data[peakIdx].time,
              position: "aboveBar",
              color: bot.color || "#c4a35a",
              shape: "arrowDown",
              text: "peak",
            },
            {
              time: plottedBot.data[troughIdx].time,
              position: "belowBar",
              color: "#a85a4a",
              shape: "arrowUp",
              text: `DD ${bot.stats.max_drawdown_pct.toFixed(1)}%`,
            },
          ]);
        }
      }
    }

    // White portfolio aggregate — equity modes only (trade % does not sum honestly).
    if (
      showPortfolio &&
      !isolatedId &&
      portfolio &&
      (mode === "account_abs" || mode === "account")
    ) {
      let raw: Array<{ t: number; value: number }> = [];
      if (mode === "account_abs") {
        raw = (portfolio.account_curve_abs || []).map((p) => ({
          t: p.t,
          value: p.equity,
        }));
      } else {
        raw = (portfolio.account_curve || []).map((p) => ({
          t: p.t,
          value: p.equity_pct,
        }));
      }
      plotLine(PORTFOLIO_ID, "Portfolio", "#ffffff", raw, {
        width: 3,
        lastValueVisible: true,
      });
    }

    setEmpty(plotted === 0);
    if (plotted > 0) chart.timeScale().fitContent();
  }, [series, portfolio, visibleIds, mode, isolatedId, showMaxDd, showPortfolio]);

  return (
    <div className="relative h-full w-full min-h-0">
      <div ref={containerRef} className="h-full w-full min-h-0" />
      {empty && (
        <p className="pointer-events-none absolute inset-0 flex items-center justify-center px-6 text-center text-[12px] text-[var(--muted)]">
          No equity line in this window. Use Equity $ + ALL, then Refresh.
        </p>
      )}
    </div>
  );
}
