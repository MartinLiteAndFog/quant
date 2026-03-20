import type { ReactNode } from "react";
import type {
  StatusResponse,
  PositionResponse,
  ChartLevels,
  KrakenMetrics,
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
  krakenMetrics?: KrakenMetrics;
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

function DualColumn({ children }: { children: ReactNode }) {
  return <div className="grid grid-cols-2 gap-3">{children}</div>;
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

export function Sidebar({
  status,
  position,
  strategy,
  performance,
  chartLevels,
  krakenMetrics,
}: SidebarProps) {
  const kucoinPrice = status?.ticker?.last ?? status?.ticker?.mid ?? null;
  const kucoinEquity = status?.balance?.equity ?? null;

  const kucoinSide = chartLevels?.side ?? position?.side ?? null;
  const kucoinSizeNum = Number(chartLevels?.live_pos ?? position?.position ?? NaN);
  const kucoinHasPos = Number.isFinite(kucoinSizeNum) && kucoinSizeNum !== 0;

  const kr = krakenMetrics;
  const krakenEquity = kr?.equity_usd ?? null;
  const krakenSide = kr?.venue_pos_side ?? kr?.pos_side ?? null;
  const krakenSizeNum = Number(kr?.venue_pos_size ?? kr?.size_rem ?? NaN);
  const krakenMode = kr?.mode ?? null;
  const krakenMark = kr?.mark_price ?? null;
  const krakenHasPos = Number.isFinite(krakenSizeNum) && krakenSizeNum !== 0;

  const strategyLabel = strategy?.strategy_label ?? "—";

  return (
    <aside className="flex w-80 flex-col gap-3 overflow-y-auto">
      <SectionCard title="Status">
        <DualColumn>
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
          <div>
            <VenueHeader name="Kraken" accent="bg-amber-500" />
            <div className="space-y-0.5 font-mono text-xs text-zinc-100">
              <div>
                Price: <span className="text-zinc-300">{fmt(krakenMark, 4)}</span>
              </div>
              {krakenMode && (
                <div>
                  Mode: <span className="text-zinc-300">{krakenMode}</span>
                </div>
              )}
            </div>
          </div>
        </DualColumn>
      </SectionCard>

      <SectionCard title="Position">
        <DualColumn>
          <div>
            <VenueHeader name="KuCoin" accent="bg-blue-500" />
            <div className="font-mono text-sm">
              {kucoinHasPos ? (
                <span className={sideColor(kucoinSide)}>
                  {sideLabel(kucoinSide)} {Math.abs(kucoinSizeNum)}
                </span>
              ) : (
                <span className="text-zinc-500">Flat</span>
              )}
            </div>
          </div>
          <div>
            <VenueHeader name="Kraken" accent="bg-amber-500" />
            <div className="font-mono text-sm">
              {krakenHasPos ? (
                <span className={sideColor(String(krakenSide))}>
                  {sideLabel(krakenSide)} {Math.abs(krakenSizeNum).toFixed(4)}
                </span>
              ) : (
                <span className="text-zinc-500">Flat</span>
              )}
            </div>
          </div>
        </DualColumn>
      </SectionCard>

      <SectionCard title="Capital">
        <DualColumn>
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
          <div>
            <VenueHeader name="Kraken" accent="bg-amber-500" />
            <div className="font-mono text-sm">
              {krakenEquity != null ? (
                <span
                  className={
                    krakenEquity >= 0 ? "text-emerald-400" : "text-red-400"
                  }
                >
                  ${fmt(krakenEquity)}
                </span>
              ) : (
                <span className="text-zinc-500">—</span>
              )}
            </div>
          </div>
        </DualColumn>
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
            <span className="text-zinc-400">Average gain</span>
            <span>{fmtPct(performance?.average_gain)}</span>
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
                  {side.toUpperCase()}
                </div>
              )}
              {entryPx != null && <div>Entry: {entryPx.toFixed(2)}</div>}
              {sl != null && (
                <div className="text-red-400">SL: {Number(sl).toFixed(2)}</div>
              )}
              {ttp != null && (
                <div className="text-amber-400">TTP: {ttp.toFixed(2)}</div>
              )}
              {tp1 != null && (
                <div className="text-blue-400">TP1: {tp1.toFixed(2)}</div>
              )}
              {tp2 != null && (
                <div className="text-purple-400">TP2: {tp2.toFixed(2)}</div>
              )}
              {mode && <div className="text-zinc-400">Mode: {mode}</div>}
            </div>
          );
        })()}
      </SectionCard>
    </aside>
  );
}