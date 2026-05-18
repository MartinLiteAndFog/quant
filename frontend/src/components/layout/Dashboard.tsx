import { useMemo, useState } from "react";
import {
  useChartData,
  useStatus,
  usePosition,
  useDashboardStrategy,
  useDashboardPerformance,
} from "../../api/hooks";
import PriceChart from "../charts/PriceChart";
import EquityCurve from "../charts/EquityCurve";
import { Sidebar } from "./Sidebar";
import type { DashboardEquityMode, TimeRange } from "../../types/chart";

function ChartSkeleton() {
  return (
    <div className="flex h-[500px] items-center justify-center">
      <div className="flex items-center gap-2 text-sm text-zinc-500">
        <span className="h-2 w-2 animate-pulse rounded-full bg-amber-400" />
        Loading chart…
      </div>
    </div>
  );
}

function EquitySkeleton() {
  return (
    <div className="flex h-[200px] items-center justify-center">
      <div className="flex items-center gap-2 text-sm text-zinc-500">
        <span className="h-2 w-2 animate-pulse rounded-full bg-amber-400" />
        Loading equity…
      </div>
    </div>
  );
}

const RANGE_PARAMS: Record<TimeRange, { hours: number; maxPoints: number }> = {
  "24h": { hours: 24, maxPoints: 2000 },
  "7d": { hours: 24 * 7, maxPoints: 3000 },
  "30d": { hours: 24 * 30, maxPoints: 5000 },
  all: { hours: 24 * 120, maxPoints: 10000 },
};

export default function Dashboard() {
  const [equityMode, setEquityMode] = useState<DashboardEquityMode>("account");
  const [equityRange, setEquityRange] = useState<TimeRange>("7d");

  const symbol = "SOL-USDT";
  const { hours, maxPoints } = RANGE_PARAMS[equityRange];

  const chartQuery = useChartData(symbol, hours, maxPoints);
  const statusQuery = useStatus();
  const positionQuery = usePosition();
  const strategyQuery = useDashboardStrategy(symbol);
  const performanceQuery = useDashboardPerformance(symbol);

  const chartData = chartQuery.data;
  const status = statusQuery.data ?? null;
  const position = positionQuery.data ?? null;
  const strategy = strategyQuery.data ?? null;
  const performance = performanceQuery.data ?? null;

  const chartLevels = chartData?.levels;
  const equityComponents = chartData?.equity_components;
  const equityTotal = chartData?.equity_total;
  const equityCurve = chartData?.equity_curve;

  // Key memos on the leaf arrays (which only change when the actual equity
  // payload changes) instead of the whole `chartData` object — that way
  // refetch ticks that only update bars/markers/levels won't cause downstream
  // re-renders of the equity chart.
  const accountEquity = useMemo(
    () => ({
      components: equityComponents ?? [],
      totalEquity: equityTotal ?? [],
    }),
    [equityComponents, equityTotal]
  );

  const tradeEquity = useMemo(() => equityCurve ?? [], [equityCurve]);

  return (
    <div className="relative min-h-screen bg-black text-zinc-100">
      <div className="grid grid-cols-1 gap-3 p-3 lg:grid-cols-[1fr_20rem]">
        <div className="flex flex-col gap-3">
          <div className="p-3">
            {chartQuery.isLoading ? (
              <ChartSkeleton />
            ) : (
              <div className="relative mx-auto aspect-square w-full max-w-[760px]">
                <div className="absolute inset-0 overflow-hidden rounded-full">
                  <PriceChart
                    bars={chartData?.bars ?? []}
                    markers={chartData?.markers}
                    levels={chartData?.levels}
                    ttpTrailPct={chartData?.ttp_trail_pct}
                    fibo={chartData?.fibo}
                    livePrice={status?.ticker?.last ?? status?.ticker?.mid}
                  />
                </div>

                <div className="pointer-events-none absolute inset-0 rounded-full dashboard-chart-fade" />
                <div className="pointer-events-none absolute inset-0 rounded-full dashboard-chart-axis-overlay" />
              </div>
            )}
          </div>

          <div className="p-3">
            <div className="mb-2 flex items-center justify-between">
              <div className="text-xs font-medium uppercase tracking-wider text-zinc-400">
                Equity
              </div>
              <div className="inline-flex rounded-md border border-zinc-800 p-0.5">
                <button
                  type="button"
                  onClick={() => setEquityMode("account")}
                  className={`rounded px-2 py-1 text-xs font-medium transition-colors ${
                    equityMode === "account"
                      ? "border border-zinc-700 text-zinc-100"
                      : "text-zinc-400 hover:text-zinc-300"
                  }`}
                >
                  Account
                </button>
                <button
                  type="button"
                  onClick={() => setEquityMode("trade")}
                  className={`rounded px-2 py-1 text-xs font-medium transition-colors ${
                    equityMode === "trade"
                      ? "border border-zinc-700 text-zinc-100"
                      : "text-zinc-400 hover:text-zinc-300"
                  }`}
                >
                  Trade
                </button>
              </div>
            </div>

            {chartQuery.isLoading ? (
              <EquitySkeleton />
            ) : (
              <EquityCurve
                mode={equityMode}
                components={
                  equityMode === "account" ? accountEquity.components : []
                }
                totalEquity={
                  equityMode === "account" ? accountEquity.totalEquity : []
                }
                tradeEquity={equityMode === "trade" ? tradeEquity : []}
                range={equityRange}
                onRangeChange={setEquityRange}
              />
            )}
          </div>
        </div>

        <div className="order-first lg:order-last">
          <Sidebar
            status={status}
            position={position}
            strategy={strategy}
            performance={performance}
            chartLevels={chartLevels}
          />
        </div>
      </div>
    </div>
  );
}