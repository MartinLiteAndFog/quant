import { useEffect, useRef, useState } from "react";
import {
  ColorType,
  createChart,
  type IChartApi,
  type ISeriesApi,
  type LineData,
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";
import type { BotSeries, ChartMode, PortfolioSeries, RangeKey } from "../types";

interface Props {
  series: BotSeries[];
  portfolio: PortfolioSeries | null;
  visibleIds: Set<string>;
  mode: ChartMode;
  rangeKey: RangeKey;
  isolatedId: string | null;
  showMaxDd: boolean;
  showPortfolio: boolean;
  /** Optional shared clock from API (unix seconds). */
  clock?: { t0?: number; t1?: number; interval_sec?: number } | null;
}

const PORTFOLIO_ID = "__portfolio__";

function asUtc(t: number): UTCTimestamp {
  // Guard ms accidentally leaking through — LWC wants seconds.
  const sec = t > 1e12 ? Math.floor(t / 1000) : Math.floor(t);
  return sec as UTCTimestamp;
}

function toLineData(points: Array<{ t: number; value: number }>): LineData[] {
  // LWC requires strictly ascending unique UTCTimestamps (true time, not category).
  const byT = new Map<number, number>();
  for (const p of points) {
    if (!Number.isFinite(p.t) || !Number.isFinite(p.value)) continue;
    byT.set(asUtc(p.t) as number, p.value);
  }
  const sorted = [...byT.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([t, value]) => ({ time: t as Time, value }));

  // Single snapshot → short horizontal stub so the series stays visible.
  if (sorted.length === 1) {
    const only = sorted[0];
    const t0 = only.time as number;
    return [
      { time: t0 as Time, value: only.value },
      { time: (t0 + 120) as Time, value: only.value },
    ];
  }
  return sorted;
}

function curveForBot(
  bot: BotSeries,
  mode: ChartMode,
): Array<{ t: number; value: number }> {
  if (mode === "account_abs") {
    return (bot.account_curve_abs || []).map((p) => ({ t: p.t, value: p.equity }));
  }
  if (mode === "corrected") {
    return (bot.corrected_curve || []).map((p) => ({ t: p.t, value: p.equity_pct }));
  }
  const curve = mode === "trade" ? bot.trade_curve : bot.account_curve;
  return (curve || []).map((p) => ({ t: p.t, value: p.equity_pct }));
}

function curveForPortfolio(
  portfolio: PortfolioSeries,
  mode: ChartMode,
): Array<{ t: number; value: number }> {
  if (mode === "account_abs") {
    return (portfolio.account_curve_abs || []).map((p) => ({
      t: p.t,
      value: p.equity,
    }));
  }
  if (mode === "corrected") {
    return (portfolio.corrected_curve || []).map((p) => ({
      t: p.t,
      value: p.equity_pct,
    }));
  }
  return (portfolio.account_curve || []).map((p) => ({
    t: p.t,
    value: p.equity_pct,
  }));
}

function sharedDomain(
  series: BotSeries[],
  portfolio: PortfolioSeries | null,
  mode: ChartMode,
  clock?: Props["clock"],
): { from: UTCTimestamp; to: UTCTimestamp } | null {
  if (clock?.t0 != null && clock?.t1 != null && clock.t1 > clock.t0) {
    return { from: asUtc(clock.t0), to: asUtc(clock.t1) };
  }
  let lo = Infinity;
  let hi = -Infinity;
  const consider = (pts: Array<{ t: number }>) => {
    for (const p of pts) {
      if (!Number.isFinite(p.t)) continue;
      const t = asUtc(p.t) as number;
      if (t < lo) lo = t;
      if (t > hi) hi = t;
    }
  };
  for (const bot of series) {
    consider(curveForBot(bot, mode));
  }
  if (portfolio && (mode === "account_abs" || mode === "account" || mode === "corrected")) {
    consider(curveForPortfolio(portfolio, mode));
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return null;
  return { from: lo as UTCTimestamp, to: hi as UTCTimestamp };
}

export function HeroChart({
  series,
  portfolio,
  visibleIds,
  mode,
  rangeKey,
  isolatedId,
  showMaxDd,
  showPortfolio,
  clock,
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const linesRef = useRef<Map<string, ISeriesApi<"Line">>>(new Map());
  const fittedKeyRef = useRef<string | null>(null);
  const [empty, setEmpty] = useState(false);
  const [pointCount, setPointCount] = useState(0);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#9a958c",
        fontFamily: '"IBM Plex Mono", monospace',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "rgba(232,228,220,0.045)" },
        horzLines: { color: "rgba(232,228,220,0.055)" },
      },
      rightPriceScale: {
        borderVisible: false,
        scaleMargins: { top: 0.12, bottom: 0.1 },
      },
      timeScale: {
        borderVisible: false,
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 4,
        // Keep pan/zoom stable when live polls append/replace points.
        shiftVisibleRangeOnNewBar: false,
        lockVisibleTimeRangeOnResize: true,
      },
      crosshair: {
        mode: 1,
        vertLine: {
          color: "rgba(232,228,220,0.28)",
          labelBackgroundColor: "#1a1c22",
        },
        horzLine: {
          color: "rgba(232,228,220,0.2)",
          labelBackgroundColor: "#1a1c22",
        },
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

  // Range / mode / isolate changes warrant a fresh fit — not live polls.
  useEffect(() => {
    fittedKeyRef.current = null;
  }, [rangeKey, mode, isolatedId]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    // Preserve the user's view across series rebuilds. Use the LOGICAL
    // (bar-index) range, not the time range: with rightOffset whitespace the
    // visible range's `to` is a timestamp past the last point, so
    // setVisibleRange() throws (no bar there) and the pan/zoom silently
    // snapped back on every poll. Logical range handles whitespace and
    // right-edge appends correctly.
    const prevLogical = chart.timeScale().getVisibleLogicalRange();

    // Never drop series just because visibility hasn't hydrated yet.
    const active = series.filter((s) => {
      if (isolatedId) return s.id === isolatedId;
      if (visibleIds.size === 0) return true;
      return visibleIds.has(s.id);
    });

    let plotted = 0;
    let totalPts = 0;
    const wanted = new Set<string>();

    // Reuse existing line series (setData) instead of remove+add on every
    // poll — rebuilding reset the time scale and "snapped back" the user's
    // pan/zoom every refresh.
    const plotLine = (
      id: string,
      title: string,
      color: string,
      raw: Array<{ t: number; value: number }>,
      opts?: { width?: number; lastValueVisible?: boolean },
    ) => {
      const data = toLineData(raw);
      if (data.length < 1) return null;
      plotted += 1;
      totalPts += data.length;
      wanted.add(id);
      const options = {
        color,
        lineWidth: (opts?.width ?? 2) as 1 | 2 | 3 | 4,
        lineVisible: true,
        pointMarkersVisible: data.length <= 3,
        priceLineVisible: false,
        lastValueVisible: opts?.lastValueVisible ?? true,
        title,
        crosshairMarkerVisible: true,
        crosshairMarkerRadius: 4,
        priceFormat: { type: "price" as const, precision: 2, minMove: 0.01 },
      };
      let line = linesRef.current.get(id);
      if (line) {
        line.applyOptions(options);
      } else {
        line = chart.addLineSeries(options);
        linesRef.current.set(id, line);
      }
      line.setData(data);
      line.setMarkers([]);
      return { line, data };
    };

    for (const bot of active) {
      const raw = curveForBot(bot, mode);
      const plottedBot = plotLine(bot.id, bot.display_name, bot.color || "#c9a65a", raw);
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
              color: bot.color || "#c9a65a",
              shape: "arrowDown",
              text: "peak",
            },
            {
              time: plottedBot.data[troughIdx].time,
              position: "belowBar",
              color: "#b86a58",
              shape: "arrowUp",
              text: `DD ${bot.stats.max_drawdown_pct.toFixed(1)}%`,
            },
          ]);
        }
      }
    }

    // White portfolio aggregate — equity / corrected modes only (trade % does not sum honestly).
    if (
      showPortfolio &&
      !isolatedId &&
      portfolio &&
      (mode === "account_abs" || mode === "account" || mode === "corrected")
    ) {
      plotLine(PORTFOLIO_ID, "Portfolio", "#ffffff", curveForPortfolio(portfolio, mode), {
        width: 3,
        lastValueVisible: true,
      });
    }

    // Drop lines whose bot disappeared (toggle/isolate) — keep the rest.
    for (const [id, line] of linesRef.current) {
      if (!wanted.has(id)) {
        chart.removeSeries(line);
        linesRef.current.delete(id);
      }
    }

    setEmpty(plotted === 0);
    setPointCount(totalPts);

    if (plotted > 0) {
      const fitKey = `${rangeKey}|${mode}|${isolatedId ?? ""}`;
      const shouldFit = fittedKeyRef.current !== fitKey;
      if (shouldFit) {
        // Initial view / range-chip change only — never on curve poll refresh.
        const domain = sharedDomain(active, showPortfolio ? portfolio : null, mode, clock);
        if (domain) {
          try {
            chart.timeScale().setVisibleRange(domain);
          } catch {
            chart.timeScale().fitContent();
          }
        } else {
          chart.timeScale().fitContent();
        }
        fittedKeyRef.current = fitKey;
      } else if (prevLogical) {
        // Preserve pan/zoom across live data refreshes (bar-index space,
        // robust to right-edge whitespace and newly appended points).
        try {
          chart.timeScale().setVisibleLogicalRange(prevLogical);
        } catch {
          /* range may briefly be invalid while series swap */
        }
      }
    }
  }, [
    series,
    portfolio,
    visibleIds,
    mode,
    rangeKey,
    isolatedId,
    showMaxDd,
    showPortfolio,
    clock,
  ]);

  return (
    <div className="relative h-full w-full min-h-0">
      <div ref={containerRef} className="h-full w-full min-h-0" />
      {empty && (
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-2 px-8 text-center">
          <p className="text-[13px] font-medium text-[var(--text)]">No equity series in this window</p>
          <p className="max-w-sm text-[12px] leading-relaxed text-[var(--muted)]">
            Switch to Equity $ and ALL, then Refresh. Pilots without account history stay flat until
            they trade or report live equity.
          </p>
        </div>
      )}
      {!empty && pointCount > 0 && pointCount < 8 && (
        <p className="pointer-events-none absolute bottom-3 left-3 text-[10px] text-[var(--muted)]">
          Sparse history — bots snapshot equity every 15 min; curves fill in as history accrues
        </p>
      )}
    </div>
  );
}
