import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchActivity,
  fetchBots,
  fetchCapitalization,
  fetchPerformance,
  fetchTrades,
  fetchTradesForBots,
  probeConnection,
  type ConnectionProbe,
} from "./api";
import { HeroChart } from "./components/HeroChart";
import { DrawerShell, type DrawerId } from "./components/DrawerShell";
import { StatusRail } from "./components/StatusRail";
import { ActivityDrawer } from "./components/drawers/ActivityDrawer";
import { CapitalizationDrawer } from "./components/drawers/CapitalizationDrawer";
import { SettingsDrawer } from "./components/drawers/SettingsDrawer";
import { StatsDrawer } from "./components/drawers/StatsDrawer";
import { TradesDrawer } from "./components/drawers/TradesDrawer";
import { downloadCsv } from "./lib/csv";
import {
  type ActivityEvent,
  type BotSeries,
  type CapitalAccount,
  type ChartMode,
  type ClosedTrade,
  type FleetBot,
  type FleetConfig,
  type PortfolioSeries,
  type RangeKey,
  loadConfig,
  saveConfig,
} from "./types";

const RANGES: RangeKey[] = ["24h", "7d", "30d", "all"];
const CHART_MODES: Array<{ id: ChartMode; label: string }> = [
  { id: "account_abs", label: "Equity $" },
  { id: "account", label: "Equity %" },
  { id: "trade", label: "Trade %" },
];

const DRAWER_TITLES: Record<Exclude<DrawerId, null>, string> = {
  activity: "Activity map",
  trades: "Trades",
  stats: "Stats compare",
  capital: "Capitalization & health",
  settings: "Settings",
};

/** Drawers that are side panels — performance board stays center. */
const PANEL_DRAWERS = [
  ["activity", "Activity"],
  ["trades", "Trades"],
  ["stats", "Stats"],
  ["capital", "Capital"],
  ["settings", "Settings"],
] as const;

function legendValue(s: BotSeries, mode: ChartMode): string {
  if (mode === "account_abs") {
    const abs = s.account_curve_abs || [];
    const last = abs.length ? abs[abs.length - 1].equity : s.live_equity;
    if (last == null || !Number.isFinite(last)) return "—";
    const ccy = s.currency || (s.venue === "kraken" ? "USD" : "USDT");
    return `${last.toFixed(2)} ${ccy}`;
  }
  if (mode === "account") {
    const curve = s.account_curve || [];
    const last = curve.length ? curve[curve.length - 1].equity_pct : 0;
    return `${last >= 0 ? "+" : ""}${last.toFixed(2)}%`;
  }
  const pct = s.stats?.return_pct ?? 0;
  return `${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`;
}

function portfolioLegendValue(p: PortfolioSeries | null, mode: ChartMode): string {
  if (!p) return "—";
  if (mode === "account_abs") {
    const abs = p.account_curve_abs || [];
    const last = abs.length ? abs[abs.length - 1].equity : p.live_equity;
    if (last == null || !Number.isFinite(last)) return "—";
    return `${last.toFixed(2)}`;
  }
  if (mode === "account") {
    const curve = p.account_curve || [];
    const last = curve.length ? curve[curve.length - 1].equity_pct : 0;
    return `${last >= 0 ? "+" : ""}${last.toFixed(2)}%`;
  }
  return "n/a";
}

export default function App() {
  const [config, setConfig] = useState<FleetConfig>(() => loadConfig());
  const [range, setRange] = useState<RangeKey>("all");
  const [bots, setBots] = useState<FleetBot[]>([]);
  const [series, setSeries] = useState<BotSeries[]>([]);
  const [portfolio, setPortfolio] = useState<PortfolioSeries | null>(null);
  const [visibleIds, setVisibleIds] = useState<Set<string>>(new Set());
  const [isolatedId, setIsolatedId] = useState<string | null>(null);
  const [showPortfolio, setShowPortfolio] = useState(true);
  const [drawer, setDrawer] = useState<DrawerId>(null);
  const [chartMode, setChartMode] = useState<ChartMode>("account_abs");
  const [showMaxDd, setShowMaxDd] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);
  const [activity, setActivity] = useState<ActivityEvent[]>([]);
  const [trades, setTrades] = useState<ClosedTrade[]>([]);
  const [tradeBotId, setTradeBotId] = useState<string | null>("__all__");
  const [accounts, setAccounts] = useState<CapitalAccount[]>([]);
  const [drawerLoading, setDrawerLoading] = useState(false);
  const [connection, setConnection] = useState<ConnectionProbe | null>(null);

  const enabledIds = useMemo(
    () => new Set(config.bots.filter((b) => b.enabled).map((b) => b.id)),
    [config.bots],
  );

  const persistConfig = useCallback((next: FleetConfig) => {
    setConfig(next);
    saveConfig(next);
  }, []);

  const refreshHealth = useCallback(async () => {
    try {
      const [remote, probe] = await Promise.all([
        fetchBots(config, true),
        probeConnection(config),
      ]);
      setConnection(probe);
      const byId = new Map(remote.map((b) => [b.id, b]));
      const merged: FleetBot[] = config.bots
        .filter((b) => b.enabled)
        .map((local) => {
          const r = byId.get(local.id);
          return {
            ...local,
            ...r,
            display_name: local.display_name,
            color: local.color || r?.color,
            health_url: local.health_url || r?.health_url,
          };
        });
      setBots(merged);
      setVisibleIds((prev) => {
        if (prev.size) return prev;
        return new Set(merged.map((b) => b.id));
      });
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, [config]);

  const refreshCurves = useCallback(async () => {
    try {
      const ids = [...enabledIds];
      const perf = await fetchPerformance(config, range, ids);
      setSeries(perf.series || []);
      setPortfolio(perf.portfolio || null);
      setUpdatedAt(perf.ts || new Date().toISOString());
      setError(perf.error || null);
    } catch (e) {
      setError(String(e));
    }
  }, [config, range, enabledIds]);

  useEffect(() => {
    void refreshHealth();
    void refreshCurves();
  }, [refreshHealth, refreshCurves]);

  useEffect(() => {
    const h = window.setInterval(() => void refreshHealth(), config.healthPollMs);
    const c = window.setInterval(() => void refreshCurves(), config.curvePollMs);
    return () => {
      window.clearInterval(h);
      window.clearInterval(c);
    };
  }, [config.healthPollMs, config.curvePollMs, refreshHealth, refreshCurves]);

  useEffect(() => {
    if (drawer !== "activity") return;
    setDrawerLoading(true);
    void fetchActivity(config, range)
      .then(setActivity)
      .catch((e) => setError(String(e)))
      .finally(() => setDrawerLoading(false));
  }, [drawer, config, range]);

  useEffect(() => {
    if (drawer !== "capital") return;
    setDrawerLoading(true);
    void fetchCapitalization(config)
      .then(setAccounts)
      .catch((e) => setError(String(e)))
      .finally(() => setDrawerLoading(false));
  }, [drawer, config]);

  useEffect(() => {
    if (drawer !== "trades") return;
    setDrawerLoading(true);
    const run = async () => {
      try {
        if (!tradeBotId || tradeBotId === "__all__") {
          const list = bots.length
            ? bots
            : config.bots.filter((b) => b.enabled).map((b) => ({
                id: b.id,
                strategy_instance: b.strategy_instance,
                display_name: b.display_name,
              }));
          setTrades(await fetchTradesForBots(config, list, range));
        } else {
          const bot =
            bots.find((b) => b.id === tradeBotId) ||
            series.find((s) => s.id === tradeBotId);
          const instance = bot?.strategy_instance || tradeBotId;
          const rows = await fetchTrades(config, instance, range);
          setTrades(
            rows.map((t) => ({
              ...t,
              bot_id: t.bot_id || tradeBotId,
              display_name: t.display_name || bot?.display_name,
            })),
          );
        }
      } catch (e) {
        setError(String(e));
      } finally {
        setDrawerLoading(false);
      }
    };
    void run();
  }, [drawer, tradeBotId, bots, series, config, range]);

  const openDrawer = (id: DrawerId) => {
    setDrawer((cur) => (cur === id ? null : id));
  };

  const setHeroMode = (mode: ChartMode) => {
    setChartMode(mode);
    setDrawer(null);
  };

  const exportCurves = () => {
    const rows: Array<Record<string, unknown>> = [];
    for (const s of series) {
      if (!visibleIds.has(s.id)) continue;
      for (const p of s.account_curve_abs || []) {
        rows.push({
          bot: s.display_name,
          instance: s.strategy_instance,
          t: p.t,
          equity: p.equity,
          mode: "account_abs",
        });
      }
      for (const p of s.account_curve) {
        rows.push({
          bot: s.display_name,
          instance: s.strategy_instance,
          t: p.t,
          equity_pct: p.equity_pct,
          mode: "account",
        });
      }
      for (const p of s.trade_curve) {
        rows.push({
          bot: s.display_name,
          instance: s.strategy_instance,
          t: p.t,
          equity_pct: p.equity_pct,
          mode: "trade",
        });
      }
    }
    if (portfolio) {
      for (const p of portfolio.account_curve_abs || []) {
        rows.push({
          bot: "Portfolio",
          instance: "portfolio",
          t: p.t,
          equity: p.equity,
          mode: "account_abs",
        });
      }
    }
    downloadCsv("fleet-equity-curves.csv", rows);
  };

  const legend = series.filter((s) => enabledIds.has(s.id));
  const portfolioOn =
    showPortfolio && !isolatedId && (chartMode === "account_abs" || chartMode === "account");

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="flex items-center justify-between gap-3 border-b border-[var(--line)] px-4 py-2.5">
        <div className="min-w-0">
          <h1 className="font-serif text-[22px] leading-none tracking-tight text-[var(--text)]">
            Fleet <span className="text-[var(--accent)]">Cockpit</span>
          </h1>
          {connection && (
            <p className="mt-1 text-[10px] tracking-wide text-[var(--muted)]">
              {connection.mode === "fleet_api"
                ? "live board"
                : connection.mode === "direct_health"
                  ? "health only — fleet API unreachable"
                  : "offline"}
              {updatedAt ? ` · ${updatedAt.replace("T", " ").slice(0, 19)} UTC` : ""}
            </p>
          )}
        </div>
        <div className="flex flex-wrap items-center justify-end gap-1.5">
          <div className="flex border border-[var(--line)]">
            {RANGES.map((r) => (
              <button
                key={r}
                onClick={() => setRange(r)}
                className="px-2.5 py-1 text-[10px] tracking-wide uppercase"
                style={{
                  background: range === r ? "rgba(196,163,90,0.18)" : "transparent",
                  color: range === r ? "var(--accent)" : "var(--muted)",
                }}
              >
                {r}
              </button>
            ))}
          </div>
          <div className="flex border border-[var(--line)]">
            {CHART_MODES.map((m) => (
              <button
                key={m.id}
                onClick={() => setHeroMode(m.id)}
                className="px-2.5 py-1 text-[10px] tracking-wide"
                style={{
                  background: chartMode === m.id ? "rgba(196,163,90,0.18)" : "transparent",
                  color: chartMode === m.id ? "var(--accent)" : "var(--muted)",
                }}
              >
                {m.label}
              </button>
            ))}
          </div>
          {PANEL_DRAWERS.map(([id, label]) => (
            <button
              key={id}
              onClick={() => {
                if (id === "trades" && !tradeBotId) setTradeBotId("__all__");
                openDrawer(id);
              }}
              className="px-2 py-1 text-[10px] tracking-wide text-[var(--muted)] hover:text-[var(--text)]"
              style={{ color: drawer === id ? "var(--accent)" : undefined }}
            >
              {label}
            </button>
          ))}
          <button
            onClick={() => {
              void refreshHealth();
              void refreshCurves();
            }}
            className="border border-[var(--line)] px-2.5 py-1 text-[10px] tracking-wide text-[var(--muted)] hover:text-[var(--text)]"
          >
            Refresh
          </button>
          <button
            onClick={exportCurves}
            className="border border-[var(--line)] px-2.5 py-1 text-[10px] tracking-wide text-[var(--muted)] hover:text-[var(--text)]"
          >
            CSV
          </button>
        </div>
      </header>

      <div className="relative flex min-h-0 flex-1">
        <StatusRail
          bots={bots}
          visibleIds={visibleIds}
          isolatedId={isolatedId}
          onToggle={(id) =>
            setVisibleIds((prev) => {
              const next = new Set(prev);
              if (next.has(id)) next.delete(id);
              else next.add(id);
              return next;
            })
          }
          onIsolate={setIsolatedId}
          onOpenBot={(id) => {
            setTradeBotId(id);
            openDrawer("trades");
          }}
        />

        <main className="relative flex min-w-0 flex-1 flex-col px-3 pb-3 pt-2">
          <div className="mb-2 flex min-h-[28px] flex-wrap items-center gap-x-3 gap-y-1.5">
            <button
              onClick={() => setShowPortfolio((v) => !v)}
              className="flex items-center gap-2 text-[11px]"
              style={{
                opacity: portfolioOn ? 1 : 0.35,
                display: chartMode === "trade" ? "none" : undefined,
              }}
              title="Combined portfolio (sum of strategy equities)"
            >
              <span className="h-1.5 w-5 bg-white" />
              <span className="font-medium text-[var(--text)]">Portfolio</span>
              <span className="text-[var(--muted)]">
                {portfolioLegendValue(portfolio, chartMode)}
              </span>
            </button>
            {legend.map((s) => {
              const on = visibleIds.has(s.id) || isolatedId === s.id;
              return (
                <button
                  key={s.id}
                  onClick={() =>
                    setVisibleIds((prev) => {
                      const next = new Set(prev);
                      if (next.has(s.id)) next.delete(s.id);
                      else next.add(s.id);
                      return next;
                    })
                  }
                  onDoubleClick={() => setIsolatedId(isolatedId === s.id ? null : s.id)}
                  className="flex items-center gap-2 text-[11px]"
                  style={{ opacity: on ? 1 : 0.35 }}
                >
                  <span className="h-1.5 w-4" style={{ background: s.color || "var(--accent)" }} />
                  <span>{s.display_name}</span>
                  <span className="text-[var(--muted)]">{legendValue(s, chartMode)}</span>
                </button>
              );
            })}
            <label className="ml-auto flex items-center gap-2 text-[10px] text-[var(--muted)]">
              <input
                type="checkbox"
                checked={showMaxDd}
                onChange={(e) => setShowMaxDd(e.target.checked)}
              />
              Max DD
            </label>
          </div>

          <div className="relative min-h-0 flex-1 border border-[var(--line)] bg-black/20">
            <HeroChart
              series={series}
              portfolio={portfolio}
              visibleIds={visibleIds}
              mode={chartMode}
              isolatedId={isolatedId}
              showMaxDd={showMaxDd}
              showPortfolio={showPortfolio}
            />
          </div>

          {error && (
            <p className="mt-2 text-[11px] text-[var(--down)]">
              {error}
              {/401/.test(error)
                ? " — Settings → Railway quant, clear token, Test connection, then Refresh."
                : " — check Settings → API base (quant dashboard host)."}
            </p>
          )}
        </main>

        <DrawerShell
          open={drawer}
          onClose={() => {
            openDrawer(null);
          }}
          title={drawer ? DRAWER_TITLES[drawer] : ""}
          widthClass={drawer === "stats" || drawer === "settings" ? "w-[480px]" : "w-[420px]"}
        >
          {drawer === "activity" && <ActivityDrawer events={activity} loading={drawerLoading} />}
          {drawer === "trades" && (
            <div className="space-y-3">
              <select
                className="w-full border border-[var(--line)] bg-black/30 px-2 py-2 text-[12px]"
                value={tradeBotId || "__all__"}
                onChange={(e) => setTradeBotId(e.target.value)}
              >
                <option value="__all__">All bots</option>
                {bots.map((b) => (
                  <option key={b.id} value={b.id}>
                    {b.display_name}
                  </option>
                ))}
              </select>
              <TradesDrawer
                trades={trades}
                botLabel={
                  tradeBotId && tradeBotId !== "__all__"
                    ? bots.find((b) => b.id === tradeBotId)?.display_name || "bot"
                    : "All bots"
                }
                loading={drawerLoading}
                showBotColumn={!tradeBotId || tradeBotId === "__all__"}
              />
            </div>
          )}
          {drawer === "stats" && <StatsDrawer series={series} />}
          {drawer === "capital" && (
            <CapitalizationDrawer accounts={accounts} loading={drawerLoading} />
          )}
          {drawer === "settings" && (
            <SettingsDrawer
              config={config}
              onChange={persistConfig}
              lastProbe={connection}
              onProbed={setConnection}
            />
          )}
        </DrawerShell>
      </div>
    </div>
  );
}
