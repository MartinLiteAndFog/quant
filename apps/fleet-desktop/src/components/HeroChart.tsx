import { useCallback, useEffect, useRef, useState } from "react";
import {
  ColorType,
  createChart,
  type IChartApi,
  type ISeriesApi,
  type LineData,
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";
import { rangeWindowUnix } from "../lib/chartTimeDomain";
import type { BotSeries, ChartMode, PortfolioSeries, RangeKey } from "../types";
import { correctedCurveOrJumpTwr } from "../lib/performanceMetrics";
import { RANGE_HOURS } from "../types";

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

type AnySeries = ISeriesApi<"Line"> | ISeriesApi<"Area">;

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
    return correctedCurveOrJumpTwr(
      bot.corrected_curve,
      bot.account_curve_abs,
      10,
      true,
      bot.account_curve,
    ).map((p) => ({
      t: p.t,
      value: p.equity_pct,
    }));
  }
  if (mode === "trade") {
    const explicit = bot.price_move_curve_bps;
    if (explicit?.length) {
      return explicit.map((p) => ({ t: p.t, value: p.equity_pct }));
    }
    // Backward compatibility with pre-BPS APIs.
    return (bot.trade_curve || []).map((p) => ({
      t: p.t,
      value: p.equity_pct * 100,
    }));
  }
  if (mode === "strategy") {
    return (bot.strategy_curve || []).map((p) => ({
      t: p.t,
      value: p.equity_pct,
    }));
  }
  const curve = bot.account_curve;
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
    return correctedCurveOrJumpTwr(
      portfolio.corrected_curve,
      portfolio.account_curve_abs,
      10,
      true,
      portfolio.account_curve,
    ).map((p) => ({
      t: p.t,
      value: p.equity_pct,
    }));
  }
  return (portfolio.account_curve || []).map((p) => ({
    t: p.t,
    value: p.equity_pct,
  }));
}

function dataExtentDomain(
  series: BotSeries[],
  portfolio: PortfolioSeries | null,
  mode: ChartMode,
): { from: UTCTimestamp; to: UTCTimestamp } | null {
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
  if (
    portfolio &&
    (mode === "account_abs" || mode === "account" || mode === "corrected")
  ) {
    consider(curveForPortfolio(portfolio, mode));
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return null;
  return { from: lo as UTCTimestamp, to: hi as UTCTimestamp };
}

/**
 * Visible time domain for the selected range chip.
 * Fixed ranges (24h/7d/30d) always start at now−N — never from first data
 * point — so a stale "all" payload during fetch cannot expand the viewport.
 */
function sharedDomain(
  series: BotSeries[],
  portfolio: PortfolioSeries | null,
  mode: ChartMode,
  rangeKey: RangeKey,
  clock?: Props["clock"],
): { from: UTCTimestamp; to: UTCTimestamp } | null {
  const fixed = rangeWindowUnix(
    RANGE_HOURS[rangeKey],
    Date.now() / 1000,
    clock?.t1 ?? null,
  );
  if (fixed) {
    return { from: asUtc(fixed.t0), to: asUtc(fixed.t1) };
  }
  if (clock?.t0 != null && clock?.t1 != null && clock.t1 > clock.t0) {
    return { from: asUtc(clock.t0), to: asUtc(clock.t1) };
  }
  return dataExtentDomain(series, portfolio, mode);
}

/** Hex/rgb → rgba with alpha (glass stack bands on dark UI). */
function solidFill(color: string, alpha = 1): string {
  const c = (color || "#c9a65a").trim();
  if (c.startsWith("rgba(")) {
    const parts = c.slice(5, -1).split(",").map((x) => x.trim());
    if (parts.length >= 3) return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${alpha})`;
  }
  if (c.startsWith("rgb(")) {
    const parts = c.slice(4, -1).split(",").map((x) => x.trim());
    if (parts.length >= 3) return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${alpha})`;
  }
  const hex = c.replace("#", "");
  if (hex.length === 3 || hex.length === 6) {
    const full =
      hex.length === 3
        ? hex
            .split("")
            .map((ch) => ch + ch)
            .join("")
        : hex;
    const r = parseInt(full.slice(0, 2), 16);
    const g = parseInt(full.slice(2, 4), 16);
    const b = parseInt(full.slice(4, 6), 16);
    if ([r, g, b].every((n) => Number.isFinite(n))) {
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
  }
  return c;
}

function stackLayerPalette(layer: StackLayer): { line: string; fill: string } {
  const key = `${layer.id} ${layer.title}`.toLowerCase();
  if (key.includes("kraken")) return { line: "#ef4444", fill: "#f59e0b" };
  if (key.includes("counter-sl-reverse")) return { line: "#a78bfa", fill: "#7c3aed" };
  if (key.includes("countervariante")) return { line: "#38bdf8", fill: "#0284c7" };
  if (key.includes("pure-imbatp")) return { line: "#34d399", fill: "#059669" };
  if (key.includes("imba-runner")) return { line: "#fbbf24", fill: "#d97706" };
  if (key.includes("quant-main")) return { line: "#8b5cf6", fill: "#6d28d9" };
  return { line: layer.color, fill: layer.color };
}

const CHART_SURFACE = "#0e1014";

/** ~20% empty space to the right of last data (padding = 25% of data span). */
const RIGHT_PAD_FRAC = 0.25;

function padTimeDomain(from: UTCTimestamp, to: UTCTimestamp): {
  from: UTCTimestamp;
  to: UTCTimestamp;
} {
  const a = from as number;
  const b = to as number;
  const span = Math.max(b - a, 1);
  return { from, to: (b + span * RIGHT_PAD_FRAC) as UTCTimestamp };
}

function applyRightPadding(chart: IChartApi) {
  try {
    const logical = chart.timeScale().getVisibleLogicalRange();
    if (logical && Number.isFinite(logical.from) && Number.isFinite(logical.to)) {
      const span = Math.max(logical.to - logical.from, 1);
      chart.timeScale().setVisibleLogicalRange({
        from: logical.from,
        to: logical.to + span * RIGHT_PAD_FRAC,
      });
      return;
    }
  } catch {
    /* fall through */
  }
  try {
    const range = chart.timeScale().getVisibleRange();
    if (range) {
      chart.timeScale().setVisibleRange(padTimeDomain(range.from as UTCTimestamp, range.to as UTCTimestamp));
    }
  } catch {
    /* ignore */
  }
}

/** Force absolute-equity stack Y-axis to include 0 as baseline. */
function zeroBaselineAutoscale(
  original: () => { priceRange: { minValue: number; maxValue: number } | null } | null,
) {
  const res = original();
  if (!res?.priceRange) return res;
  return {
    ...res,
    priceRange: {
      minValue: 0,
      maxValue: Math.max(res.priceRange.maxValue, 0.01),
    },
  };
}

type StackLayer = {
  id: string;
  title: string;
  color: string;
  /** Cumulative top edge for this layer (bottom → this series). */
  data: LineData[];
  lowerData: LineData[];
};

/**
 * Build stacked abs-equity layers on a shared time grid.
 * Bottom layer = largest latest equity (OWID-style foundation).
 * LWC has no native stack — translucent area series with cumulative values,
 * drawn top→bottom so lower bands sit under upper glass fills.
 */
function buildStackLayers(bots: BotSeries[]): StackLayer[] {
  const maps = bots.map((bot) => {
    const pts = toLineData(curveForBot(bot, "account_abs"));
    const map = new Map<number, number>();
    for (const p of pts) map.set(p.time as number, p.value);
    return { bot, map };
  });

  const times = new Set<number>();
  for (const { map } of maps) {
    for (const t of map.keys()) times.add(t);
  }
  const sortedTimes = [...times].sort((a, b) => a - b);
  if (!sortedTimes.length) return [];

  const filled = maps.map(({ bot, map }) => {
    let last: number | null = null;
    const values: number[] = [];
    for (const t of sortedTimes) {
      if (map.has(t)) {
        const v = map.get(t)!;
        if (Number.isFinite(v) && v >= 0) last = v;
      }
      // Before first observation: 0 so stack doesn't invent cash.
      values.push(last ?? 0);
    }
    const lastVal = values.length ? values[values.length - 1] : 0;
    return { bot, values, lastVal };
  });

  // Largest latest equity at the bottom of the stack.
  filled.sort((a, b) => b.lastVal - a.lastVal);

  return filled.map((row, idx) => {
    const currentEquity = row.bot.account_curve_abs?.length
      ? row.lastVal
      : row.bot.live_equity;
    const data: LineData[] = sortedTimes.map((t, ti) => {
      let cum = 0;
      for (let j = 0; j <= idx; j++) cum += filled[j].values[ti];
      return { time: t as Time, value: cum };
    });
    const lowerData: LineData[] = sortedTimes.map((t, ti) => {
      let cum = 0;
      for (let j = 0; j < idx; j++) cum += filled[j].values[ti];
      return { time: t as Time, value: cum };
    });
    return {
      id: row.bot.id,
      title: `${row.bot.display_name}${
        currentEquity != null && Number.isFinite(currentEquity)
          ? ` ($${currentEquity.toFixed(2)})`
          : ""
      }`,
      color: row.bot.color || "#c9a65a",
      data,
      lowerData,
    };
  });
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
  const seriesRef = useRef<Map<string, AnySeries>>(new Map());
  const seriesKindRef = useRef<"line" | "baseline" | null>(null);
  const fittedKeyRef = useRef<string | null>(null);
  const [empty, setEmpty] = useState(false);
  const [pointCount, setPointCount] = useState(0);

  const fitView = useCallback((active: BotSeries[], useClock: boolean) => {
    const chart = chartRef.current;
    if (!chart) return;
    const stack = mode === "account_abs";
    try {
      chart.priceScale("right").applyOptions({
        autoScale: true,
        // Equity $ stack: pin visual baseline near 0 (no floating mid-scale).
        scaleMargins: stack
          ? { top: 0.1, bottom: 0.02 }
          : { top: 0.08, bottom: 0.06 },
      });
    } catch {
      /* ignore */
    }
    const domain = sharedDomain(
      active,
      showPortfolio ? portfolio : null,
      mode,
      rangeKey,
      useClock ? clock : null,
    );
    if (domain) {
      try {
        chart.timeScale().setVisibleRange(padTimeDomain(domain.from, domain.to));
        return;
      } catch {
        /* fall through */
      }
    }
    // "all" without clock/extent only — never use fitContent for fixed ranges
    // (fitContent expands to every plotted point and ignores the chip).
    chart.timeScale().fitContent();
    applyRightPadding(chart);
  }, [clock, mode, portfolio, rangeKey, showPortfolio]);

  const onAutoscaleClick = useCallback(() => {
    const chart = chartRef.current;
    if (!chart) return;
    const stack = mode === "account_abs";
    try {
      chart.priceScale("right").applyOptions({
        autoScale: true,
        scaleMargins: stack
          ? { top: 0.1, bottom: 0.02 }
          : { top: 0.08, bottom: 0.06 },
      });
    } catch {
      /* ignore */
    }
    const active = series.filter((s) => {
      if (isolatedId) return s.id === isolatedId;
      if (visibleIds.size === 0) return true;
      return visibleIds.has(s.id);
    });
    const domain = sharedDomain(
      active,
      showPortfolio ? portfolio : null,
      mode,
      rangeKey,
      clock,
    );
    if (domain) {
      try {
        chart.timeScale().setVisibleRange(padTimeDomain(domain.from, domain.to));
        return;
      } catch {
        /* fall through */
      }
    }
    chart.timeScale().fitContent();
    applyRightPadding(chart);
  }, [clock, isolatedId, mode, portfolio, rangeKey, series, showPortfolio, visibleIds]);

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
        vertLines: { visible: false },
        horzLines: { color: "rgba(232,228,220,0.07)", style: 2 },
      },
      rightPriceScale: {
        borderVisible: false,
        scaleMargins: { top: 0.08, bottom: 0.06 },
      },
      timeScale: {
        borderVisible: false,
        timeVisible: true,
        secondsVisible: false,
        // Extra bars past last point; intentional Autoscale also pads ~20%.
        rightOffset: 12,
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
      seriesRef.current.clear();
      seriesKindRef.current = null;
    };
  }, []);

  // Range / mode / isolate changes warrant a fresh fit — not live polls.
  useEffect(() => {
    fittedKeyRef.current = null;
  }, [rangeKey, mode, isolatedId]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    const prevLogical = chart.timeScale().getVisibleLogicalRange();
    const stackMode = mode === "account_abs";
    const wantKind: "line" | "baseline" = stackMode ? "baseline" : "line";

    // Never drop series just because visibility hasn't hydrated yet.
    const active = series.filter((s) => {
      if (isolatedId) return s.id === isolatedId;
      if (visibleIds.size === 0) return true;
      return visibleIds.has(s.id);
    });

    // Mode switch line↔area: wipe all series (incompatible APIs).
    if (seriesKindRef.current && seriesKindRef.current !== wantKind) {
      for (const [, s] of seriesRef.current) {
        chart.removeSeries(s);
      }
      seriesRef.current.clear();
    }
    seriesKindRef.current = wantKind;

    let plotted = 0;
    let totalPts = 0;
    const wanted = new Set<string>();

    const dropUnwanted = () => {
      for (const [id, s] of seriesRef.current) {
        if (!wanted.has(id)) {
          chart.removeSeries(s);
          seriesRef.current.delete(id);
        }
      }
    };

    if (stackMode) {
      try {
        chart.priceScale("right").applyOptions({
          scaleMargins: { top: 0.1, bottom: 0.02 },
        });
      } catch {
        /* ignore */
      }
      const layers = buildStackLayers(active);
      // Draw top-of-stack first so lower translucent bands composite correctly.
      for (let i = layers.length - 1; i >= 0; i--) {
        const layer = layers[i];
        if (layer.data.length < 1) continue;
        plotted += 1;
        totalPts += layer.data.length;
        wanted.add(layer.id);
        const palette = stackLayerPalette(layer);
        const options = {
          // The white fade is visible only inside each band; the mask below
          // the lower edge prevents transparent colors from mixing globally.
          topColor: "rgba(255, 255, 255, 0.28)",
          bottomColor: solidFill(palette.fill, 0.72),
          lineColor: solidFill(palette.line, 0.95),
          lineWidth: 1 as const,
          lineVisible: true,
          crosshairMarkerVisible: false,
          priceLineVisible: false,
          lastValueVisible: false,
          title: layer.title,
          priceFormat: { type: "price" as const, precision: 2, minMove: 0.01 },
        };
        let area = seriesRef.current.get(layer.id) as ISeriesApi<"Area"> | undefined;
        if (area) {
          area.applyOptions(options);
        } else {
          area = chart.addAreaSeries(options);
          seriesRef.current.set(layer.id, area);
        }
        area.setData(layer.data);

        const maskId = `${layer.id}__mask`;
        wanted.add(maskId);
        const maskOptions = {
          topColor: CHART_SURFACE,
          bottomColor: CHART_SURFACE,
          lineColor: CHART_SURFACE,
          lineWidth: 1 as const,
          lineVisible: false,
          crosshairMarkerVisible: false,
          priceLineVisible: false,
          lastValueVisible: false,
          title: "",
          priceFormat: { type: "price" as const, precision: 2, minMove: 0.01 },
        };
        let mask = seriesRef.current.get(maskId) as ISeriesApi<"Area"> | undefined;
        if (mask) {
          mask.applyOptions(maskOptions);
        } else {
          mask = chart.addAreaSeries(maskOptions);
          seriesRef.current.set(maskId, mask);
        }
        mask.setData(layer.lowerData);
      }

      // White portfolio top edge + end label (full fleet cash).
      const topLayer = layers.length ? layers[layers.length - 1] : null;
      if (topLayer && topLayer.data.length > 0 && !isolatedId) {
        const last = topLayer.data[topLayer.data.length - 1];
        wanted.add(PORTFOLIO_ID);
        plotted += 1;
        const lineOpts = {
          color: "#ffffff",
          lineWidth: 2 as const,
          lineVisible: true,
          priceLineVisible: false,
          lastValueVisible: true,
          title: "Portfolio",
          crosshairMarkerVisible: true,
          crosshairMarkerRadius: 5,
          priceFormat: { type: "price" as const, precision: 2, minMove: 0.01 },
          autoscaleInfoProvider: zeroBaselineAutoscale,
        };
        let line = seriesRef.current.get(PORTFOLIO_ID) as ISeriesApi<"Line"> | undefined;
        // Portfolio highlight is always a Line; recreate if a prior stack series leaked under this id.
        if (line && typeof (line as ISeriesApi<"Line">).setMarkers !== "function") {
          chart.removeSeries(line);
          seriesRef.current.delete(PORTFOLIO_ID);
          line = undefined;
        }
        if (line) {
          line.applyOptions(lineOpts);
        } else {
          line = chart.addLineSeries(lineOpts);
          seriesRef.current.set(PORTFOLIO_ID, line);
        }
        line.setData(topLayer.data);
        line.setMarkers([
          {
            time: last.time,
            position: "aboveBar",
            color: "#ffffff",
            shape: "circle",
            text: `$${last.value.toFixed(2)}`,
          },
        ]);
      }

      dropUnwanted();
    } else {
      try {
        chart.priceScale("right").applyOptions({
          scaleMargins: { top: 0.08, bottom: 0.06 },
        });
      } catch {
        /* ignore */
      }
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
        const priceFormat = mode === "trade"
          ? {
              type: "custom" as const,
              precision: 1,
              minMove: 0.1,
              formatter: (price: number) => `${price.toFixed(1)} bps`,
            }
          : { type: "price" as const, precision: 2, minMove: 0.01 };
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
          priceFormat,
        };
        let line = seriesRef.current.get(id) as ISeriesApi<"Line"> | undefined;
        if (line) {
          line.applyOptions(options);
        } else {
          line = chart.addLineSeries(options);
          seriesRef.current.set(id, line);
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
                text: `DD ${(bot.stats.max_drawdown_pct * 100).toFixed(0)} bps`,
              },
            ]);
          }
        }
      }

      // White portfolio aggregate — percent modes only (abs uses stack top as total).
      if (
        showPortfolio &&
        !isolatedId &&
        portfolio &&
        (mode === "account" || mode === "corrected")
      ) {
        plotLine(PORTFOLIO_ID, "Portfolio", "#ffffff", curveForPortfolio(portfolio, mode), {
          width: 3,
          lastValueVisible: true,
        });
      }

      dropUnwanted();
    }

    setEmpty(plotted === 0);
    setPointCount(totalPts);

    if (plotted > 0) {
      const fitKey = `${rangeKey}|${mode}|${isolatedId ?? ""}`;
      const shouldFit = fittedKeyRef.current !== fitKey;
      if (shouldFit) {
        fitView(active, true);
        fittedKeyRef.current = fitKey;
      } else if (prevLogical) {
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
    fitView,
  ]);

  return (
    <div className="relative h-full w-full min-h-0">
      <div ref={containerRef} className="h-full w-full min-h-0" />
      {!empty && (
        <button
          type="button"
          className="chart-fit-btn"
          onClick={onAutoscaleClick}
          title="Autoscale price; keep the selected time range (~20% empty on the right)"
          aria-label="Autoscale chart"
        >
          Autoscale
        </button>
      )}
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
      {!empty && mode === "account_abs" && (
        <p className="pointer-events-none absolute bottom-3 right-14 text-[10px] text-[var(--muted)]">
          Stacked cash · top edge = fleet total
        </p>
      )}
    </div>
  );
}
