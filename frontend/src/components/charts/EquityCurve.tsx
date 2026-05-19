import { useMemo } from "react";
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
  TimeRange,
} from "../../types/chart";

interface EquityCurveProps {
  mode?: DashboardEquityMode;
  components?: EquityComponent[];
  totalEquity?: { time: number; equity: number }[];
  tradeEquity?: TradeEquityPoint[];
  range: TimeRange;
  onRangeChange: (range: TimeRange) => void;
}

const RANGE_SEC: Record<TimeRange, number | null> = {
  "24h": 24 * 3600,
  "7d": 7 * 24 * 3600,
  "30d": 30 * 24 * 3600,
  all: null,
};

const COMPONENT_COLORS: Record<string, { fill: string; stroke: string }> = {
  kucoin: { fill: "#3b82f6", stroke: "#3b82f6" },
};

const DEFAULT_COLORS = ["#3b82f6", "#22c55e", "#a855f7"];

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

function formatPrice(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  const num = Number(value);
  if (Math.abs(num) >= 1000) return num.toFixed(2);
  if (Math.abs(num) >= 1) return num.toFixed(4);
  return num.toFixed(6);
}

// 2000-01-01T00:00:00Z. Anything older is treated as a missing timestamp so
// the tooltip never prints "1/1/1970" when a legacy NaT->0 sentinel slips
// through from postgres.
const MIN_VALID_EPOCH_SEC = 946_684_800;

function formatDateTime(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(Number(seconds))) return "—";
  const num = Number(seconds);
  if (num < MIN_VALID_EPOCH_SEC) return "—";
  const ms = num * 1000;
  if (!Number.isFinite(ms)) return "—";
  return new Date(ms).toLocaleString();
}

function formatDuration(start: number | null | undefined, end: number | null | undefined): string | null {
  if (start == null || end == null) return null;
  const startNum = Number(start);
  const endNum = Number(end);
  if (!Number.isFinite(startNum) || !Number.isFinite(endNum)) return null;
  const seconds = Math.max(0, Math.round(endNum - startNum));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    const rem = seconds % 60;
    return rem === 0 ? `${minutes}m` : `${minutes}m ${rem}s`;
  }
  const hours = Math.floor(minutes / 60);
  const remMin = minutes % 60;
  if (hours < 24) return remMin === 0 ? `${hours}h` : `${hours}h ${remMin}m`;
  const days = Math.floor(hours / 24);
  const remHr = hours % 24;
  return remHr === 0 ? `${days}d` : `${days}d ${remHr}h`;
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

function _numOr(v: unknown, fallback?: number): number | undefined {
  if (v == null) return fallback;
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function tradeEquityToData(
  tradeEquity: TradeEquityPoint[]
): Record<string, number | string | boolean>[] {
  return tradeEquity
    .filter(
      (pt) =>
        Number.isFinite(Number(pt.time)) && Number.isFinite(Number(pt.cum_pct))
    )
    .map((pt) => {
      const row: Record<string, number | string | boolean> = {
        time: Number(pt.time),
        cum_pct: Number(pt.cum_pct),
        pnl_pct: Number(pt.pnl_pct ?? 0),
      };
      // Reject the 1970 sentinel here so even mis-typed payloads still
      // render "—" in the tooltip rather than the unix epoch.
      const entryTime = _numOr(pt.entry_time);
      if (entryTime !== undefined && entryTime >= MIN_VALID_EPOCH_SEC) {
        row.entry_time = entryTime;
      }
      const exitTime = _numOr(pt.exit_time, Number(pt.time));
      if (exitTime !== undefined && exitTime >= MIN_VALID_EPOCH_SEC) {
        row.exit_time = exitTime;
      }
      const entryPrice = _numOr(pt.entry_price);
      const exitPrice = _numOr(pt.exit_price);
      if (entryPrice !== undefined) row.entry_price = entryPrice;
      if (exitPrice !== undefined) row.exit_price = exitPrice;
      if (pt.side) row.side = String(pt.side);
      if (pt.open) row.open = true;
      if (pt.decision_id) row.decision_id = String(pt.decision_id);
      return row;
    });
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
    const isOpen = Boolean(point?.open);
    const rawPnl = point?.pnl_pct;
    const pnlPct =
      rawPnl != null && Number.isFinite(Number(rawPnl))
        ? Number(rawPnl)
        : null;
    const sideRaw = point?.side != null ? String(point.side).toLowerCase() : "";
    const sideLabel = sideRaw ? sideRaw.toUpperCase() : null;
    const sideColor =
      sideRaw === "long"
        ? "text-emerald-400"
        : sideRaw === "short"
          ? "text-red-400"
          : "text-zinc-300";
    const entryTime =
      point?.entry_time != null &&
      Number.isFinite(Number(point.entry_time)) &&
      Number(point.entry_time) >= MIN_VALID_EPOCH_SEC
        ? Number(point.entry_time)
        : null;
    const exitTimeCandidate =
      point?.exit_time != null && Number.isFinite(Number(point.exit_time))
        ? Number(point.exit_time)
        : label != null
          ? Number(label)
          : null;
    const exitTime =
      exitTimeCandidate != null && exitTimeCandidate >= MIN_VALID_EPOCH_SEC
        ? exitTimeCandidate
        : null;
    const entryPrice =
      point?.entry_price != null && Number.isFinite(Number(point.entry_price))
        ? Number(point.entry_price)
        : null;
    const exitPrice =
      point?.exit_price != null && Number.isFinite(Number(point.exit_price))
        ? Number(point.exit_price)
        : null;
    const duration = isOpen ? null : formatDuration(entryTime, exitTime);
    const pnlColor =
      pnlPct == null
        ? "text-zinc-400"
        : pnlPct >= 0
          ? "text-emerald-400"
          : "text-red-400";

    return (
      <div className="min-w-[220px] rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 shadow-lg">
        <div className="mb-1 flex items-center justify-between gap-3 text-xs">
          <span className="text-zinc-400">{dateStr}</span>
          <div className="flex items-center gap-2">
            {isOpen && (
              <span className="rounded bg-amber-700/30 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-amber-300">
                Open
              </span>
            )}
            {sideLabel && (
              <span className={`font-semibold ${sideColor}`}>{sideLabel}</span>
            )}
          </div>
        </div>
        <div className="space-y-0.5 border-t border-zinc-700 pt-1 text-xs">
          {(entryTime != null || entryPrice != null) && (
            <div className="flex justify-between gap-4">
              <span className="text-zinc-400">Entry</span>
              <span className="font-mono text-zinc-100">
                {formatDateTime(entryTime)}
                {entryPrice != null && (
                  <span className="ml-2 text-zinc-400">
                    @ {formatPrice(entryPrice)}
                  </span>
                )}
              </span>
            </div>
          )}
          {!isOpen && (exitTime != null || exitPrice != null) && (
            <div className="flex justify-between gap-4">
              <span className="text-zinc-400">Exit</span>
              <span className="font-mono text-zinc-100">
                {formatDateTime(exitTime)}
                {exitPrice != null && (
                  <span className="ml-2 text-zinc-400">
                    @ {formatPrice(exitPrice)}
                  </span>
                )}
              </span>
            </div>
          )}
          {duration && (
            <div className="flex justify-between gap-4">
              <span className="text-zinc-400">Duration</span>
              <span className="font-mono text-zinc-300">{duration}</span>
            </div>
          )}
        </div>
        <div className="mt-1 flex justify-between gap-4 border-t border-zinc-700 pt-1 text-sm">
          <span className="text-zinc-300">Trade</span>
          <span className={`font-medium ${pnlColor}`}>
            {pnlPct == null ? "—" : formatPct(pnlPct)}
          </span>
        </div>
        <div className="mt-0.5 flex justify-between gap-4 text-sm font-medium">
          <span className="text-zinc-300">Cumulative</span>
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
  range,
  onRangeChange,
}: EquityCurveProps) {

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
              onClick={() => onRangeChange(value)}
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
              type="linear"
              dataKey="cum_pct"
              name="Trade equity"
              fill="#22c55e"
              fillOpacity={0.28}
              stroke="#22c55e"
              isAnimationActive={false}
              dot={false}
              activeDot={{ r: 4, strokeWidth: 0, fill: "#22c55e" }}
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