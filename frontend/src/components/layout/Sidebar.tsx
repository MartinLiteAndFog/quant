import { memo } from "react";
import type { ReactNode } from "react";
import type {
  StatusResponse,
  PositionResponse,
  ChartLevels,
} from "../../types/chart";
import type {
  DashboardStrategyResponse,
  DashboardPerformanceResponse,
} from "../../api/hooks";

export interface SidebarProps {
  status: StatusResponse | null;
  position: PositionResponse | null;
  strategy: DashboardStrategyResponse | null;
  performance: DashboardPerformanceResponse | null;
  chartLevels?: ChartLevels;
}

function SectionCard({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <div className="rounded-lg border border-zinc-800 p-3">
      <h3 className="mb-2 text-xs font-medium uppercase tracking-wider text-zinc-400">
        {title}
      </h3>
      {children}
    </div>
  );
}

function VenueHeader({ name, accent }: { name: string; accent: string }) {
  return (
    <div className="mb-1.5 flex items-center gap-1.5">
      <span className={`h-1.5 w-1.5 rounded-full ${accent}`} />
      <span className="text-[10px] font-semibold uppercase tracking-widest text-zinc-500">
        {name}
      </span>
    </div>
  );
}

function fmt(v: number | null | undefined, digits = 2): string {
  if (v == null || !isFinite(v)) return "—";
  return v.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function fmtPct(v: number | null | undefined, digits = 2): string {
  if (v == null || !isFinite(v)) return "—";
  return `${fmt(v, digits)}%`;
}

function fmtInt(v: number | null | undefined): string {
  if (v == null || !isFinite(v)) return "—";
  return Math.round(v).toLocaleString();
}

function sideColor(side: string | null | undefined): string {
  if (!side) return "text-zinc-500";
  const s = side.toLowerCase();
  if (s === "long" || s === "buy" || s === "1") return "text-emerald-400";
  if (s === "short" || s === "sell" || s === "-1") return "text-red-400";
  return "text-zinc-400";
}

function sideLabel(side: number | string | null | undefined): string {
  if (side == null) return "Flat";
  if (typeof side === "number") {
    if (side > 0) return "Long";
    if (side < 0) return "Short";
    return "Flat";
  }
  const s = side.toLowerCase();
  if (s === "long" || s === "buy" || s === "1") return "Long";
  if (s === "short" || s === "sell" || s === "-1") return "Short";
  if (s === "flat" || s === "0" || s === "") return "Flat";
  return side;
}

function SidebarBase({
  status,
  position,
  strategy,
  performance,
  chartLevels,
}: SidebarProps) {
  const kucoinPrice = status?.ticker?.last ?? status?.ticker?.mid ?? null;
  const kucoinEquity = status?.balance?.equity ?? null;

  const kucoinSide = chartLevels?.side ?? position?.side ?? null;
  const kucoinSizeNum = Number(
    chartLevels?.live_pos ?? position?.position ?? NaN
  );
  const kucoinHasPos = Number.isFinite(kucoinSizeNum) && kucoinSizeNum !== 0;
  const kucoinDisplaySize = Number.isFinite(kucoinSizeNum)
    ? Math.abs(kucoinSizeNum) / 10
    : 0;

  const strategyLabel = strategy?.strategy_label ?? "—";

  // Main performance metrics are closed_trades aggregates. The separate
  // trade_decision_count field is diagnostic only and is not rendered here.
  const tradeCount = performance?.trade_count ?? null;

  return (
    <aside className="flex w-80 flex-col gap-3 overflow-y-auto">
      <SectionCard title="Status">
        <div>
          <VenueHeader name="KuCoin" accent="bg-blue-500" />
          <div className="space-y-0.5 font-mono text-xs text-zinc-100">
            <div>
              Price: <span className="text-zinc-300">{fmt(kucoinPrice, 4)}</span>
            </div>
            <div>
              Regime: <span className="text-zinc-300">{strategyLabel}</span>
            </div>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Position">
        <div>
          <VenueHeader name="KuCoin" accent="bg-blue-500" />
          <div className="font-mono text-sm">
            {kucoinHasPos ? (
              <span className={sideColor(kucoinSide)}>
                {sideLabel(kucoinSide)} {kucoinDisplaySize.toFixed(1)}
              </span>
            ) : (
              <span className="text-zinc-500">Flat</span>
            )}
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Capital">
        <div>
          <VenueHeader name="KuCoin" accent="bg-blue-500" />
          <div className="font-mono text-sm">
            {kucoinEquity != null ? (
              <span
                className={
                  kucoinEquity >= 0 ? "text-emerald-400" : "text-red-400"
                }
              >
                ${fmt(kucoinEquity)}
              </span>
            ) : (
              <span className="text-zinc-500">—</span>
            )}
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Performance">
        <div className="space-y-1 font-mono text-sm text-zinc-100">
          <div className="flex items-center justify-between gap-3">
            <span className="text-zinc-400">PnL %</span>
            <span>{fmtPct(performance?.pnl_pct)}</span>
          </div>
          <div className="flex items-center justify-between gap-3">
            <span className="text-zinc-400">Winrate</span>
            <span>{fmtPct(performance?.winrate)}</span>
          </div>
          <div className="flex items-center justify-between gap-3">
            <span className="text-zinc-400">Monthly growth</span>
            <span>{fmtPct(performance?.monthly_growth)}</span>
          </div>
          <div className="flex items-center justify-between gap-3">
            <span className="text-zinc-400">Average trade</span>
            <span>{fmtPct(performance?.average_gain)}</span>
          </div>
          <div className="mt-2 border-t border-zinc-800 pt-2" />
          <div className="flex items-center justify-between gap-3">
            <span className="text-zinc-400">Trades</span>
            <span>{fmtInt(tradeCount)}</span>
          </div>
          <div className="flex items-center justify-between gap-3">
            <span className="text-zinc-400">Wins</span>
            <span className="text-emerald-400">
              {fmtInt(performance?.winning_trade_count)}
            </span>
          </div>
          <div className="flex items-center justify-between gap-3">
            <span className="text-zinc-400">Losses</span>
            <span className="text-red-400">
              {fmtInt(performance?.losing_trade_count)}
            </span>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Levels">
        {(() => {
          const t = chartLevels?.terminal;
          const entryPx = t?.entry_px ?? chartLevels?.entry_px;
          const sl = t?.sl ?? chartLevels?.sl;
          const ttp = t?.ttp ?? chartLevels?.ttp;
          const tp1 = chartLevels?.tp1;
          const tp2 = chartLevels?.tp2;
          const mode = t?.mode ?? chartLevels?.mode;
          const side = t?.side ?? chartLevels?.side;
          const hasAny = entryPx != null || sl != null || ttp != null;

          if (!hasAny) {
            return <div className="font-mono text-sm text-zinc-500">—</div>;
          }

          return (
            <div className="space-y-1 font-mono text-sm text-zinc-100">
              {side && (
                <div
                  className={
                    side === "long" ? "text-emerald-400" : "text-red-400"
                  }
                >
                  Side: {sideLabel(side)}
                </div>
              )}
              {mode && (
                <div className="flex items-center justify-between gap-3">
                  <span className="text-zinc-400">Mode</span>
                  <span>{mode}</span>
                </div>
              )}
              {entryPx != null && (
                <div className="flex items-center justify-between gap-3">
                  <span className="text-zinc-400">Entry</span>
                  <span>{fmt(entryPx, 4)}</span>
                </div>
              )}
              {sl != null && (
                <div className="flex items-center justify-between gap-3">
                  <span className="text-zinc-400">SL</span>
                  <span>{fmt(sl, 4)}</span>
                </div>
              )}
              {ttp != null && (
                <div className="flex items-center justify-between gap-3">
                  <span className="text-zinc-400">TTP</span>
                  <span>{fmt(ttp, 4)}</span>
                </div>
              )}
              {tp1 != null && (
                <div className="flex items-center justify-between gap-3">
                  <span className="text-zinc-400">TP1</span>
                  <span>{fmt(tp1, 4)}</span>
                </div>
              )}
              {tp2 != null && (
                <div className="flex items-center justify-between gap-3">
                  <span className="text-zinc-400">TP2</span>
                  <span>{fmt(tp2, 4)}</span>
                </div>
              )}
            </div>
          );
        })()}
      </SectionCard>
    </aside>
  );
}

// Memoize so chart-data refreshes that don't affect the sidebar's props don't
// trigger a sidebar re-render.
export const Sidebar = memo(SidebarBase);
