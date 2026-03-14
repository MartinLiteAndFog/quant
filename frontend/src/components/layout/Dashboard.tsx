import { useChartData, useStatus, usePosition, useFills } from "../../api/hooks";
import PriceChart from "../charts/PriceChart";
import EquityCurve from "../charts/EquityCurve";
import { Sidebar } from "./Sidebar";

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
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
      <div className="flex h-[200px] items-center justify-center">
        <div className="flex items-center gap-2 text-sm text-zinc-500">
          <span className="h-2 w-2 animate-pulse rounded-full bg-amber-400" />
          Loading equity…
        </div>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const chartQuery = useChartData("SOL-USDT", 168);
  const statusQuery = useStatus();
  const positionQuery = usePosition();
  const fillsQuery = useFills();

  const chartData = chartQuery.data;
  const status = statusQuery.data ?? null;
  const position = positionQuery.data ?? null;
  const fills = fillsQuery.data ?? null;

  const chartLevels = chartData?.levels ?? undefined;
  const regimeState = chartData?.regime_state ?? null;
  const gateOn = chartData?.gate_on ?? null;

  return (
    <div className="relative min-h-screen bg-zinc-950 text-zinc-100">
      <div className="grid grid-cols-1 gap-3 p-3 lg:grid-cols-[1fr_20rem]">
        <div className="flex flex-col gap-3">
          <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-1">
            {chartQuery.isLoading ? (
              <ChartSkeleton />
            ) : (
              <PriceChart
                bars={chartData?.bars ?? []}
                markers={chartData?.markers}
                segments={chartData?.segments}
                levels={chartData?.levels}
                ttpTrailPct={chartData?.ttp_trail_pct}
                fibo={chartData?.fibo}
                livePrice={status?.ticker?.last ?? status?.ticker?.mid}
              />
            )}
          </div>
          {chartQuery.isLoading ? (
            <EquitySkeleton />
          ) : (
            <EquityCurve
              components={chartData?.equity_components}
              totalEquity={chartData?.equity_total}
            />
          )}
        </div>

        <div className="order-first lg:order-last">
          <Sidebar
            status={status}
            position={position}
            fills={fills}
            chartLevels={chartLevels}
            regimeState={regimeState}
            gateOn={gateOn}
            krakenMetrics={chartData?.kraken_metrics}
          />
        </div>
      </div>
    </div>
  );
}
