import { useChartData, useStatus, usePosition, useFills } from "../../api/hooks";
import PriceChart from "../charts/PriceChart";
import EquityCurve from "../charts/EquityCurve";
import { Sidebar } from "./Sidebar";

function LoadingIndicator() {
  return (
    <div className="absolute right-3 top-3 z-10 flex items-center gap-2 rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-xs text-zinc-400">
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-400" />
      Loading…
    </div>
  );
}

export default function Dashboard() {
  const chartQuery = useChartData("SOL-USDT", 168);
  const statusQuery = useStatus();
  const positionQuery = usePosition();
  const fillsQuery = useFills();

  const isLoading =
    chartQuery.isLoading ||
    statusQuery.isLoading ||
    positionQuery.isLoading ||
    fillsQuery.isLoading;

  const chartData = chartQuery.data;
  const status = statusQuery.data ?? null;
  const position = positionQuery.data ?? null;
  const fills = fillsQuery.data ?? null;

  const chartLevels = chartData?.levels ?? undefined;
  const regimeState = chartData?.regime_state ?? null;
  const gateOn = chartData?.gate_on ?? null;

  return (
    <div className="relative min-h-screen bg-zinc-950 text-zinc-100">
      {isLoading && <LoadingIndicator />}

      <div className="grid grid-cols-1 gap-3 p-3 lg:grid-cols-[1fr_20rem]">
        <div className="flex flex-col gap-3">
          <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-1">
            <PriceChart
              bars={chartData?.bars ?? []}
              markers={chartData?.markers}
              segments={chartData?.segments}
              levels={chartData?.levels}
              ttpTrailPct={chartData?.ttp_trail_pct}
              fibo={chartData?.fibo}
              livePrice={status?.ticker?.last ?? status?.ticker?.mid}
            />
          </div>
          <EquityCurve
            components={chartData?.equity_components}
            totalEquity={chartData?.equity_total}
          />
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
