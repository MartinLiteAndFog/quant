import { useMemo, useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  type TooltipProps,
} from "recharts";
import type {
  EquityComponent,
  TradeEquityPoint,
  DashboardEquityMode,
} from "../../types/chart";

interface EquityCurveProps {
  mode?: DashboardEquityMode;
  components?: EquityComponent[];
  totalEquity?: { time: number; equity: number }[];
  tradeEquity?: TradeEquityPoint[];
}

type TimeRange = "24h" | "7d" | "30d" | "all";

const RANGE_SEC: Record<TimeRange, number | null> = {
  "24h": 24 * 3600,
  "7d": 7 * 24 * 3600,
  "30d": 30 * 24 * 3600,
  all: null,
};

const COMPONENT_COLORS: Record<string, { fill: string; stroke: string }> = {
  kucoin: { fill: "#3b82f6", stroke: "#3b82f6" },
  kraken: { fill: "#f59e0b", stroke: "#f59e0b" },
};

const DEFAULT_COLORS = ["#3b82f6", "#f59e0b", "#22c55e", "#a855f7"];

function getColorForKey(key: string, index: number) {
  return COMPONENT_COLORS[key] ?? {
    fill: DEFAULT_COLORS[index % DEFAULT_COLORS.length],
    stroke: DEFAULT_COLORS[index % DEFAULT_COLORS.length],
  };
}

function formatUsd(value: number): string {
  if (value >= 1_000_000) return `$${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `$${(value / 1_000).toFixed(1)}k`;
  return `$${value.toFixed(0)}`;
}

function formatPct(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function mergeComponentsToData(
  components: EquityComponent[]
): Record<string, number | string>[] {
  if (!components.length) return [];

  const timeToRow = new Map<number, Record<string, number>>();

  for (const comp of components) {
    const key = comp.key;
    for (const pt of comp.points) {
      let row = timeToRow.get(pt.time);
      if (!row) {
        row = { time: pt.time };
        timeToRow.set(pt.time, row);
      }
      (row as Record<string, number>)[key] = pt.equity;
    }
  }

  const sorted = Array.from(timeToRow.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([, row]) => row);

  const keys = components.map((c) => c.key);
  let prev: Record<string, number> = {};

  for (const row of sorted) {
    for (const key of keys) {
      if (row[key] === undefined) {
        (row as Record<string, number>)[key] = prev[key] ?? 0;
      } else {
        prev[key] = row[key] as number;
      }
    }
  }

  return sorted;
}

function totalEquityToData(
  totalEquity: { time: number; equity: number }[]
): Record<string, number | string>[] {
  return totalEquity.map((pt) => ({ time: pt.time, total: pt.equity }));
}

function tradeEquityToData(
  tradeEquity: TradeEquityPoint[]
): Record<string, number | string>[] {
  return tradeEquity
    .filter(
      (pt) =>
        Number.isFinite(Number(pt.time)) && Number.isFinite(Number(pt.cum_pct))
    )
    .map((pt) => ({
      time: pt.time,
      cum_pct: Number(pt.cum_pct),
      pnl_pct: Number(pt.pnl_pct ?? 0),
    }));
}

function CustomTooltip({
  active,
  payload,
  label,
  mode,
}: TooltipProps<number, string> & { mode: DashboardEquityMode }) {
  if (!active || !payload?.length) return null;

  const dateStr =
    label != null
      ? new Date((label as number) * 1000).toLocaleDateString()
      : "";

  if (mode === "trade") {
    const point = payload[0]?.payload as Record<string, unknown> | undefined;
    const cumPct = Number(point?.cum_pct ?? payload[0]?.value ?? 0);
    const pnlPct = Number(point?.pnl_pct ?? 0);

    return (
      <div className="rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 shadow-lg">
        <div className="mb-1 text-xs text-zinc-400">{dateStr}</div>
        <div className="flex justify-between gap-4 text-sm">
          <span className="text-zinc-300">Trade:</span>
          <span className="font-medium text-zinc-100">
            {formatPct(pnlPct)}
          </span>
        </div>
        <div className="mt-1 flex justify-between gap-4 border-t border-zinc-700 pt-1 text-sm font-medium">
          <span className="text-zinc-300">Cumulative:</span>
          <span className="text-zinc-100">{formatPct(cumPct)}</span>
        </div>
      </div>
    );
  }

  const total = payload.reduce((sum, p) => sum + (p.value ?? 0), 0);

  return (
    <div className="rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 shadow-lg">
      <div className="mb-1 text-xs text-zinc-400">{dateStr}</div>
      {payload.map((p) => (
        <div key={String(p.dataKey)} className="flex justify-between gap-4 text-sm">
          <span className="text-zinc-300">{String(p.name)}:</span>
          <span className="font-medium text-zinc-100">
            {formatUsd(p.value ?? 0)}
          </span>
        </div>
      ))}
      <div className="mt-1 flex justify-between gap-4 border-t border-zinc-700 pt-1 text-sm font-medium">
        <span className="text-zinc-300">Total:</span>
        <span className="text-zinc-100">{formatUsd(total)}</span>
      </div>
    </div>
  );
}

export default function EquityCurve({
  mode = "account",
  components = [],
  totalEquity,
  tradeEquity = [],
}: EquityCurveProps) {
  const [range, setRange] = useState<TimeRange>("7d");

  const { chartData, keys } = useMemo(() => {
    if (mode === "trade") {
      return { chartData: tradeEquityToData(tradeEquity), keys: ["cum_pct"] };
    }
    if (components.length > 0) {
      return {
        chartData: mergeComponentsToData(components),
        keys: components.map((c) => c.key),
      };
    }
    if (totalEquity?.length) {
      return { chartData: totalEquityToData(totalEquity), keys: ["total"] };
    }
    return { chartData: [], keys: [] };
  }, [mode, components, totalEquity, tradeEquity]);

  const filteredData = useMemo(() => {
    if (!chartData.length) return [];
    const cutoff = RANGE_SEC[range];
    if (!cutoff) return chartData;
    const now = Math.max(...chartData.map((d) => Number(d.time)));
    const minTime = now - cutoff;
    return chartData.filter((d) => Number(d.time) >= minTime);
  }, [chartData, range]);

  const headlineValue = useMemo(() => {
    if (filteredData.length < 2) return null;

    if (mode === "trade") {
      return Number(filteredData[filteredData.length - 1]?.cum_pct ?? 0);
    }

    const first = filteredData[0];
    const last = filteredData[filteredData.length - 1];
    const firstVal = keys.reduce((s, k) => s + (Number(first[k]) || 0), 0);
    const lastVal = keys.reduce((s, k) => s + (Number(last[k]) || 0), 0);
    if (firstVal === 0) return null;
    return ((lastVal - firstVal) / firstVal) * 100;
  }, [filteredData, keys, mode]);

  const rangeButtons: { value: TimeRange; label: string }[] = [
    { value: "24h", label: "24h" },
    { value: "7d", label: "7d" },
    { value: "30d", label: "30d" },
    { value: "all", label: "All" },
  ];

  if (!chartData.length) {
    return (
      <div className="flex h-[180px] items-center justify-center text-zinc-500">
        No equity data
      </div>
    );
  }

  const labelByKey: Record<string, string> = {};
  for (const c of components) {
    labelByKey[c.key] = c.label;
  }

  return (
    <>
      <div className="mb-2 flex items-center justify-between">
        <div className="flex gap-1">
          {rangeButtons.map(({ value, label }) => (
            <button
              key={value}
              type="button"
              onClick={() => setRange(value)}
              className={`rounded px-2 py-1 text-xs font-medium transition-colors ${
                range === value
                  ? "bg-zinc-700 text-zinc-100"
                  : "bg-zinc-800 text-zinc-400 hover:text-zinc-300"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        {headlineValue != null && (
          <span
            className={`text-sm font-medium ${
              headlineValue >= 0 ? "text-emerald-500" : "text-red-500"
            }`}
          >
            {formatPct(headlineValue)}
          </span>
        )}
      </div>

      <ResponsiveContainer width="100%" height={180}>
        <AreaChart
          data={filteredData}
          margin={{ top: 4, right: 4, bottom: 4, left: 4 }}
        >
          <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
          <XAxis
            dataKey="time"
            tick={{ fill: "#71717a", fontSize: 10 }}
            tickFormatter={(t) =>
              new Date(Number(t) * 1000).toLocaleDateString()
            }
          />
          <YAxis
            tick={{ fill: "#71717a", fontSize: 10 }}
            tickFormatter={(v) =>
              mode === "trade" ? formatPct(Number(v)) : formatUsd(Number(v))
            }
          />
          <Tooltip content={<CustomTooltip mode={mode} />} />

          {mode === "trade" ? (
            <Area
              type="stepAfter"
              dataKey="cum_pct"
              name="Trade equity"
              fill="#22c55e"
              fillOpacity={0.28}
              stroke="#22c55e"
            />
          ) : (
            keys.map((key, i) => {
              const { fill, stroke } = getColorForKey(key, i);
              return (
                <Area
                  key={key}
                  type="stepAfter"
                  dataKey={key}
                  name={labelByKey[key] ?? key}
                  stackId="equity"
                  fill={fill}
                  fillOpacity={0.4}
                  stroke={stroke}
                />
              );
            })
          )}
        </AreaChart>
      </ResponsiveContainer>
    </>
  );
}