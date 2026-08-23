import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type CSSProperties,
} from "react";
import {
  fetchActivityFeed,
  fetchBots,
  fetchCapitalization,
  fetchPerformance,
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
import fleetRobotLogo from "./assets/fleet-robot-logo-transparent-v2.png";
import {
  correctedCurveOrJumpTwr,
  rawCommonScopeReturnPct,
  returnPctForView,
} from "./lib/performanceMetrics";
import {
  type ActivityItem,
  type BotSeries,
  type CapitalAccount,
  type ChartMode,
  type FleetBot,
  type FleetClock,
  type FleetConfig,
  type PortfolioSeries,
  type RangeKey,
  loadConfig,
  saveConfig,
} from "./types";

const RANGES: RangeKey[] = ["24h", "7d", "30d", "all"];
const CHART_MODES: Array<{ id: ChartMode; label: string }> = [
  { id: "trade", label: "Performance" },
  { id: "account_abs", label: "Equity $" },
  { id: "account", label: "Equity %" },
  { id: "corrected", label: "Bereinigt %" },
  { id: "strategy", label: "Strategie %" },
];

const DRAWER_TITLES: Record<Exclude<DrawerId, null>, string> = {
  activity: "Activity",
  stats: "Stats compare",
  capital: "Capitalization & health",
  settings: "Settings",
};

/** Secondary surfaces only — never a second performance chart. */
const RIGHT_DRAWERS = [
  ["capital", "Capital"],
  ["stats", "Stats"],
  ["settings", "Settings"],
] as const;

function legendValue(s: BotSeries, mode: ChartMode): string {
  if (mode === "account_abs") {
    const abs = s.account_curve_abs || [];
    const last = abs.length ? abs[abs.length - 1].equity : s.live_equity;
    if (last == null || !Number.isFinite(last)) return "—";
    return `$${last.toFixed(2)}`;
  }
  if (mode === "account") {
    const curve = s.account_curve || [];
    const last = curve.length ? curve[curve.length - 1].equity_pct : 0;
    return `${last >= 0 ? "+" : ""}${last.toFixed(2)}%`;
  }
  if (mode === "corrected") {
    const curve = correctedCurveOrJumpTwr(
      s.corrected_curve,
      s.account_curve_abs,
      10,
      true,
      s.account_curve,
    );
    if (!curve.length) return "—";
    const last = curve[curve.length - 1].equity_pct;
    const method = s.corrected_meta?.method;
    const suffix =
      method === "ledger"
        ? " · Ledger geprüft"
        : method === "jump_twr" || !s.corrected_curve?.length
          ? " · Fallback"
          : "";
    return `${last >= 0 ? "+" : ""}${last.toFixed(2)}%${suffix}`;
  }
  if (mode === "strategy") {
    if (!s.strategy_meta?.available) return "nicht verfügbar";
    const curve = s.strategy_curve || [];
    const value = curve.length ? curve[curve.length - 1].equity_pct : s.strategy_meta.return_pct;
    if (value == null || !Number.isFinite(value)) return "nicht verfügbar";
    const leverage = s.strategy_meta.assumed_leverage;
    const fraction = s.strategy_meta.assumed_capital_fraction;
    const assumption = leverage && fraction
      ? ` · ${leverage.toFixed(0)}× · ${(fraction * 100).toFixed(0)}% Kapital`
      : "";
    return `${value >= 0 ? "+" : ""}${value.toFixed(2)}% brutto${assumption}`;
  }
  const bps = s.price_move_meta?.return_bps ?? (s.stats?.return_pct ?? 0) * 100;
  return `${bps >= 0 ? "+" : ""}${bps.toFixed(0)} bps`;
}

/** Hex → rgba for translucent legend swatches (stack mode). */
function solidLegendFill(color: string, alpha = 0.55): string {
  const hex = (color || "#f0b429").replace("#", "").trim();
  const full =
    hex.length === 3
      ? hex
          .split("")
          .map((ch) => ch + ch)
          .join("")
      : hex;
  if (full.length !== 6) return color;
  const r = parseInt(full.slice(0, 2), 16);
  const g = parseInt(full.slice(2, 4), 16);
  const b = parseInt(full.slice(4, 6), 16);
  if (![r, g, b].every((n) => Number.isFinite(n))) return color;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function portfolioLegendValue(p: PortfolioSeries | null, mode: ChartMode): string {
  if (!p) return "—";
  if (mode === "account_abs") {
    const abs = p.account_curve_abs || [];
    const last = abs.length ? abs[abs.length - 1].equity : p.live_equity;
    if (last == null || !Number.isFinite(last)) return "—";
    return `$${last.toFixed(2)}`;
  }
  if (mode === "account") {
    const curve = p.account_curve || [];
    const last = curve.length ? curve[curve.length - 1].equity_pct : 0;
    return `${last >= 0 ? "+" : ""}${last.toFixed(2)}%`;
  }
  if (mode === "corrected") {
    const curve = correctedCurveOrJumpTwr(
      p.corrected_curve,
      p.account_curve_abs,
      10,
      true,
      p.account_curve,
    );
    if (!curve.length) return "—";
    const last = curve[curve.length - 1].equity_pct;
    const suffix = p.corrected_meta?.method === "ledger" ? " · Ledger geprüft" : " · Fallback";
    return `${last >= 0 ? "+" : ""}${last.toFixed(2)}%${suffix}`;
  }
  return "n/a";
}

export default function App() {
  const [config, setConfig] = useState<FleetConfig>(() => loadConfig());
  // Prefer full history + absolute equity so sparse pilots still draw something useful.
  const [range, setRange] = useState<RangeKey>("all");
  const [bots, setBots] = useState<FleetBot[]>([]);
  const [series, setSeries] = useState<BotSeries[]>([]);
  const [portfolio, setPortfolio] = useState<PortfolioSeries | null>(null);
  const [clock, setClock] = useState<FleetClock | null>(null);
  const [visibleIds, setVisibleIds] = useState<Set<string>>(
    () => new Set(loadConfig().bots.filter((b) => b.enabled).map((b) => b.id)),
  );
  const [isolatedId, setIsolatedId] = useState<string | null>(null);
  const [showPortfolio, setShowPortfolio] = useState(true);
  const [drawer, setDrawer] = useState<DrawerId>(null);
  const [chartMode, setChartMode] = useState<ChartMode>("trade");
  const [showMaxDd, setShowMaxDd] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);
  const [activity, setActivity] = useState<ActivityItem[]>([]);
  const [tradeBotId, setTradeBotId] = useState<string | null>("__all__");
  const [accounts, setAccounts] = useState<CapitalAccount[]>([]);
  const [drawerLoading, setDrawerLoading] = useState(false);
  const [connection, setConnection] = useState<ConnectionProbe | null>(null);
  const [booting, setBooting] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  /** Bumped by Refresh so open drawers re-fetch without rewriting their SoT. */
  const [refreshNonce, setRefreshNonce] = useState(0);

  const enabledIds = useMemo(
    () => new Set(config.bots.filter((b) => b.enabled).map((b) => b.id)),
    [config.bots],
  );

  /** Stable bot list for activity SoT — must NOT track live `bots` (health poll). */
  const activityBots = useMemo(
    () =>
      config.bots
        .filter((b) => b.enabled)
        .map((b) => ({
          id: b.id,
          strategy_instance: b.strategy_instance,
          display_name: b.display_name,
        })),
    [config.bots],
  );

  const persistConfig = useCallback((next: FleetConfig) => {
    setConfig(next);
    saveConfig(next);
  }, []);

  const refreshHealth = useCallback(async (fresh = false) => {
    try {
      // probe=false: the client probes health URLs itself (probeConnection);
      // asking the server to fan out to the same 6 URLs every poll doubled
      // every health check and could take longer than the poll interval.
      const [remote, probe] = await Promise.all([
        fetchBots(config, false, { fresh }),
        probeConnection(config),
      ]);
      setConnection(probe);
      const byId = new Map(remote.map((b) => [b.id, b]));
      const hitById = new Map(probe.healthHits.map((h) => [h.id, h]));
      const merged: FleetBot[] = config.bots
        .filter((b) => b.enabled)
        .map((local) => {
          const r = byId.get(local.id);
          const hit = hitById.get(local.id);
          return {
            ...local,
            ...r,
            display_name: local.display_name,
            color: local.color || r?.color,
            health_url: local.health_url || r?.health_url,
            status: hit?.status ?? r?.status,
          };
        });
      setBots(merged);
      setVisibleIds((prev) => {
        const next = new Set(prev);
        for (const b of merged) {
          if (!prev.size) next.add(b.id);
        }
        if (!next.size) return new Set(merged.map((b) => b.id));
        return next;
      });
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, [config]);

  const refreshCurves = useCallback(async (fresh = false) => {
    try {
      const ids = [...enabledIds];
      const perf = await fetchPerformance(config, range, ids, { fresh });
      const colorById = new Map(config.bots.map((b) => [b.id, b.color]));
      const named = (perf.series || []).map((s) => ({
        ...s,
        color: colorById.get(s.id) || s.color,
        display_name:
          config.bots.find((b) => b.id === s.id)?.display_name || s.display_name,
      }));
      setSeries(named);
      setPortfolio(perf.portfolio || null);
      setClock(perf.clock || null);
      setUpdatedAt(perf.ts || new Date().toISOString());
      setError(perf.error || null);
    } catch (e) {
      setError(String(e));
    } finally {
      setBooting(false);
    }
  }, [config, range, enabledIds]);

  const handleRefresh = useCallback(async () => {
    if (refreshing) return;
    setRefreshing(true);
    try {
      await Promise.all([refreshHealth(true), refreshCurves(true)]);
      // Force visible board clock — server perf.ts often unchanged on re-fetch.
      setUpdatedAt(new Date().toISOString());
      setRefreshNonce((n) => n + 1);
    } finally {
      setRefreshing(false);
    }
  }, [refreshing, refreshHealth, refreshCurves]);

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

  // No auto-escalation: an empty window renders an honest empty state
  // (stylebook §10) instead of silently switching mode/range for the user.

  useEffect(() => {
    if (drawer !== "activity") return;
    let cancelled = false;
    // Always mark loading for the soft "refreshing…" hint, but ActivityDrawer
    // only shows the empty spinner when items are empty — never remounts on
    // health-poll (bots intentionally omitted from deps).
    setDrawerLoading(true);
    const run = async () => {
      try {
        const fresh = refreshNonce > 0;
        const feed = await fetchActivityFeed(config, range, activityBots, {
          fresh,
        });
        if (!cancelled) setActivity(feed);
      } catch (e) {
        if (!cancelled) setError(String(e));
      } finally {
        if (!cancelled) setDrawerLoading(false);
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [drawer, activityBots, config, range, refreshNonce]);

  useEffect(() => {
    if (drawer !== "capital") return;
    setDrawerLoading(true);
    void fetchCapitalization(config, { fresh: refreshNonce > 0 })
      .then(setAccounts)
      .catch((e) => setError(String(e)))
      .finally(() => setDrawerLoading(false));
  }, [drawer, config, refreshNonce]);

  const openDrawer = (id: DrawerId) => {
    setDrawer((cur) => (cur === id ? null : id));
  };

  /** Chart modes only change the hero series — they never open a side chart panel. */
  const setHeroMode = (mode: ChartMode) => {
    setChartMode(mode);
    setDrawer(null);
  };

  const pickRange = (r: RangeKey) => {
    setRange(r);
  };

  const legend = series.filter((s) => enabledIds.has(s.id));
  // Equity $ stack always shows total at the top edge; portfolio chip is readout-only.
  const portfolioOn =
    chartMode === "account_abs"
      ? !isolatedId
      : showPortfolio &&
        !isolatedId &&
        (chartMode === "account" || chartMode === "corrected");

  const currentPortfolioEquity = useMemo(() => {
    const curve = portfolio?.account_curve_abs || [];
    if (curve.length) return curve[curve.length - 1].equity;
    return portfolio?.live_equity != null && Number.isFinite(portfolio.live_equity)
      ? portfolio.live_equity
      : null;
  }, [portfolio]);
  const tradeReturnBps = useMemo(() => {
    const active = series.filter(
      (item) =>
        (isolatedId ? item.id === isolatedId : visibleIds.has(item.id)) &&
        Number.isFinite(item.price_move_meta?.return_bps ?? item.stats?.return_pct),
    );
    if (!active.length) return null;
    return (
      active.reduce(
        (total, item) =>
          total + (item.price_move_meta?.return_bps ?? item.stats.return_pct * 100),
        0,
      ) /
      active.length
    );
  }, [series, isolatedId, visibleIds]);
  const strategyKpi = useMemo(() => {
    const active = series.filter((item) =>
      isolatedId ? item.id === isolatedId : visibleIds.has(item.id),
    );
    const available = active.filter((item) => item.strategy_meta?.available);
    if (isolatedId && available.length === 1) {
      const value = available[0].strategy_meta?.return_pct;
      return value == null || !Number.isFinite(value)
        ? "nicht verfügbar"
        : `${value >= 0 ? "+" : ""}${value.toFixed(2)}% brutto`;
    }
    return available.length ? `${available.length} Strategien` : "nicht verfügbar";
  }, [isolatedId, series, visibleIds]);
  const rawAccountReturnPct = useMemo(
    () => rawCommonScopeReturnPct(series),
    [series],
  );
  const displayedReturnPct = useMemo(
    () => returnPctForView(chartMode, portfolio, null, rawAccountReturnPct),
    [chartMode, portfolio, rawAccountReturnPct],
  );
  const cashflowReturn = portfolio?.cashflow_return;
  const allocation = portfolio?.allocation;
  const returnLabel =
    chartMode === "account_abs"
      ? cashflowReturn?.available
        ? `Return · ${cashflowReturn.scope_label.replace(" futures accounts", " accts")}`
        : "Return · corrected"
      : chartMode === "account"
        ? "Equity %"
        : chartMode === "corrected"
          ? portfolio?.corrected_meta?.method === "ledger"
            ? "Bereinigt · Ledger"
            : "Bereinigt · Fallback"
          : chartMode === "strategy"
            ? "Strategie-Rendite"
            : "Performance";
  const excludedScopeNote = cashflowReturn?.excluded_bot_ids.includes("counter-sl-reverse")
    ? " Counter SL Reverse is excluded because its ledger is unavailable."
    : "";
  const returnTitle =
    chartMode === "trade"
      ? "Performance = reine unlevered Entry/Exit-Kursbewegung in Basispunkten; 1 % = 100 bps. Keine Equity, Positionsgröße, Leverage, Gebühren oder Funding."
      : chartMode === "strategy"
        ? "Brutto-Strategierendite mit je Bot konstant angenommenem aktuellem Railway-Hebel und aktueller Kapitalquote. Historische Gebühren, Funding und Slippage sind nicht vollständig zuordenbar und deshalb ausdrücklich nicht enthalten."
      : chartMode === "account"
        ? `Raw ${range.toUpperCase()} Equity % chart value; deposits and withdrawals remain visible`
        : chartMode === "corrected"
          ? `${range.toUpperCase()} cashflow-corrected return. Ledger is used only when its coverage spans the complete displayed curve; otherwise the conservative jump-TWR fallback is shown.`
        : cashflowReturn?.available
          ? `${range.toUpperCase()} cashflow-corrected return across ${cashflowReturn.scope_label}. ${cashflowReturn.boundary_note}. Ledger as of ${cashflowReturn.as_of || "latest sync"}.${excludedScopeNote}`
          : `${range.toUpperCase()} cashflow-neutralized return using the conservative jump-TWR compatibility curve until confirmed futures-account ledgers finish synchronizing`;
  const netFlowTitle = cashflowReturn?.available
    ? `${cashflowReturn.flow_count} confirmed successful ${cashflowReturn.flow_count === 1 ? "flow" : "flows"} across ${cashflowReturn.flow_scope_label || cashflowReturn.scope_label}. ${cashflowReturn.boundary_note}.`
    : cashflowReturn?.reason === "reporting_currency_conversion_unavailable"
      ? `Unavailable: event-time conversion is missing for ${(cashflowReturn.unsupported_currencies || []).join(", ")}`
      : "Awaiting authoritative futures-account ledger synchronization; no cashflow is inferred from equity jumps";
  const formatMoney = (value: number | null, signed = false): string => {
    if (value == null) return "—";
    const prefix = signed && value > 0 ? "+" : "";
    return `${prefix}$${value.toFixed(2)}`;
  };

  return (
    <div className="cockpit-shell relative flex h-full min-h-0 flex-col">
      <header className="cockpit-toolbar">
        <div className="cockpit-brand">
          <img
            className="cockpit-brand-logo"
            src={fleetRobotLogo}
            alt=""
            aria-hidden="true"
          />
          <h1 className="text-[16px] font-semibold leading-none tracking-[-0.02em] text-[var(--text)]">
            Fleet Cockpit
          </h1>
          {connection && (
            <span
              className="connection-pill"
              data-mode={connection.mode}
              title={updatedAt ? `Updated ${updatedAt.replace("T", " ").slice(0, 19)} UTC` : undefined}
            >
              {connection.mode === "fleet_api"
                ? "Live"
                : connection.mode === "direct_health"
                  ? "Health only"
                  : "Offline"}
            </span>
          )}
        </div>

        <div className="toolbar-kpis" role="status" aria-label="Portfolio telemetry">
          <div className="readout">
            <span className="k">Portfolio</span>
            <span className="v">{formatMoney(currentPortfolioEquity)}</span>
          </div>
          <div
            className="readout"
            title={returnTitle}
          >
            <span className="k">{returnLabel}</span>
            <span
              className="v"
              data-tone={
                displayedReturnPct == null
                  ? undefined
                  : displayedReturnPct >= 0
                    ? "up"
                    : "down"
              }
            >
              {chartMode === "trade"
                ? tradeReturnBps == null
                  ? "—"
                  : `${tradeReturnBps >= 0 ? "+" : ""}${tradeReturnBps.toFixed(0)} bps`
                : chartMode === "strategy"
                  ? strategyKpi
                : displayedReturnPct == null
                  ? "—"
                  : `${displayedReturnPct >= 0 ? "+" : ""}${displayedReturnPct.toFixed(2)}%`}
            </span>
          </div>
          <div
            className="readout"
            title={netFlowTitle}
          >
            <span className="k">Net flows</span>
            <span className={`v${cashflowReturn?.available ? "" : " v-unavailable"}`}>
              {cashflowReturn?.available
                ? formatMoney(cashflowReturn.net_cashflow, true)
                : "Ledger pending"}
            </span>
          </div>
          <div
            className="readout"
            title={allocation?.available
              ? "Reales bereinigtes Portfolio relativ zu einem gleichgewichteten Strategiebasket auf gemeinsamem Zeitfenster und gleicher realisierter Risikohöhe."
              : "Noch nicht vergleichbar: vollständige historische Strategie-Notional-, Leverage- und Kostendaten fehlen."}
          >
            <span className="k">Allokationsbeitrag</span>
            <span
              className="v"
              data-tone={allocation?.contribution_pct == null
                ? undefined
                : allocation.contribution_pct >= 0 ? "up" : "down"}
            >
              {allocation?.available && allocation.contribution_pct != null
                ? `${allocation.contribution_pct >= 0 ? "+" : ""}${allocation.contribution_pct.toFixed(2)}%`
                : "nicht verfügbar"}
            </span>
          </div>
        </div>

        <nav className="toolbar-controls" aria-label="Fleet cockpit controls">
          <div className="chip-group" role="group" aria-label="Time range">
            {RANGES.map((r) => (
              <button
                key={r}
                type="button"
                className="chip"
                data-active={range === r}
                onClick={() => pickRange(r)}
              >
                {r}
              </button>
            ))}
          </div>
          <div className="chip-group" role="group" aria-label="Chart mode">
            {CHART_MODES.map((m) => (
              <button
                key={m.id}
                type="button"
                className="chip"
                data-active={chartMode === m.id}
                onClick={() => setHeroMode(m.id)}
              >
                {m.label}
              </button>
            ))}
          </div>
          <button
            type="button"
            className="toolbar-button"
            data-active={drawer === "activity"}
            onClick={() => {
              if (!tradeBotId) setTradeBotId("__all__");
              openDrawer("activity");
            }}
          >
            Activity
          </button>
          <div className="toolbar-actions" role="group" aria-label="Panels">
            {RIGHT_DRAWERS.map(([id, label]) => (
              <button
                key={id}
                type="button"
                className="toolbar-button"
                data-active={drawer === id}
                onClick={() => openDrawer(id)}
              >
                <span aria-hidden="true">
                  {id === "capital" ? "◇" : id === "stats" ? "▥" : "⚙"}
                </span>
                {label}
              </button>
            ))}
          </div>
        </nav>
      </header>

      <div className="cockpit-stage relative flex min-h-0 flex-1">
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
            openDrawer("activity");
          }}
        />

        <main className="cockpit-main relative flex min-w-0 flex-1 flex-col px-3 pb-3 pt-2">
          <div className="legend-row mb-2 flex min-h-[28px] flex-wrap items-center gap-x-3 gap-y-1.5">
            <button
              type="button"
              onClick={() => {
                if (chartMode === "account_abs") return;
                setShowPortfolio((v) => !v);
              }}
              className="legend-item"
              data-off={!portfolioOn}
              title={
                  chartMode === "account_abs"
                  ? "Portfolio total = top edge of stacked cash (sum of bot equities)"
                  : chartMode === "corrected"
                    ? "Portfolio mean of the available deposit/withdrawal-corrected bot curves"
                    : "Selected Equity % series for the current range; not cashflow-adjusted"
              }
              style={
                {
                  "--i": 0,
                  display:
                    chartMode === "trade" || chartMode === "strategy"
                      ? "none"
                      : undefined,
                } as CSSProperties
              }
            >
              <span
                className="legend-swatch portfolio"
                data-stack={chartMode === "account_abs" ? "true" : undefined}
              />
              <span>
                Portfolio
                <span className="legend-val">
                  {" · "}
                  {portfolioLegendValue(portfolio, chartMode)}
                </span>
              </span>
            </button>
            {legend.map((s, i) => {
              const on = visibleIds.has(s.id) || isolatedId === s.id;
              const swatch =
                chartMode === "account_abs"
                  ? solidLegendFill(s.color || "#f0b429", 0.55)
                  : s.color || "var(--accent)";
              return (
                <button
                  key={s.id}
                  type="button"
                  onClick={() =>
                    setVisibleIds((prev) => {
                      const next = new Set(prev);
                      if (next.has(s.id)) next.delete(s.id);
                      else next.add(s.id);
                      return next;
                    })
                  }
                  onDoubleClick={() => setIsolatedId(isolatedId === s.id ? null : s.id)}
                  className="legend-item"
                  data-off={!on}
                  style={{ "--i": i + 1 } as CSSProperties}
                >
                  <span
                    className="legend-swatch"
                    data-stack={chartMode === "account_abs" ? "true" : undefined}
                    style={{ background: swatch }}
                  />
                  <span>
                    {s.display_name}
                    {chartMode === "account_abs" ? (
                      <span className="legend-val">{" ("}{legendValue(s, chartMode)}{ ")"}</span>
                    ) : (
                      <span className="legend-val">
                        {" · "}
                        {legendValue(s, chartMode)}
                      </span>
                    )}
                  </span>
                  {s.snapshot_age_sec != null && s.snapshot_age_sec > 3600 && (
                    <span className="stale-tag" title="Last equity snapshot older than 1h">
                      STALE
                    </span>
                  )}
                </button>
              );
            })}
            <label
              className="ml-auto flex items-center gap-2 text-[10px] font-medium tracking-wide text-[var(--muted)]"
                  style={{ display: chartMode === "trade" ? undefined : "none" }}
            >
              <input
                type="checkbox"
                checked={showMaxDd}
                onChange={(e) => setShowMaxDd(e.target.checked)}
              />
              Max DD markers
            </label>
          </div>

          {chartMode === "strategy" && (
            <p className="mb-2 border border-[var(--line)] bg-[rgba(10,11,15,0.7)] px-2 py-1 text-[9px] leading-relaxed text-[var(--muted)]">
              Historische Brutto-Annahme mit dem am 23.08.2026 produktiv gesetzten Railway-Hebel und der aktuellen Kapitalquote je Bot. Gebühren, Funding und Slippage sind nicht vollständig pro Trade zuordenbar und nicht enthalten.
            </p>
          )}

          {/* Single full-bleed hero — drawers overlay this; they never replace it with a second chart */}
          <div className="cockpit-chart relative min-h-0 flex-1 overflow-hidden border border-[var(--line)] bg-[var(--bg-chart)]">
            <HeroChart
              series={series}
              portfolio={portfolio}
              visibleIds={visibleIds}
              mode={chartMode}
              rangeKey={range}
              isolatedId={isolatedId}
              showMaxDd={showMaxDd}
              showPortfolio={showPortfolio}
              clock={clock}
            />
            {booting && (
              <div className="hero-empty">
                <span className="spinner" aria-label="Loading" />
                <p className="hint">Loading fleet data…</p>
              </div>
            )}
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
          onClose={() => openDrawer(null)}
          title={drawer ? DRAWER_TITLES[drawer] : ""}
          widthClass={
            drawer === "stats" || drawer === "settings" || drawer === "activity"
              ? "w-[480px]"
              : "w-[420px]"
          }
        >
          {drawer === "activity" && (
            <ActivityDrawer
              items={activity}
              bots={
                bots.length
                  ? bots
                  : config.bots.filter((b) => b.enabled).map((b) => ({
                      id: b.id,
                      display_name: b.display_name,
                    }))
              }
              botFilter={tradeBotId}
              onBotFilter={(id) => setTradeBotId(id || "__all__")}
              loading={drawerLoading}
            />
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
      <button
        type="button"
        className="refresh-fab"
        onClick={() => void handleRefresh()}
        disabled={refreshing}
        aria-label={refreshing ? "Refreshing Fleet data" : "Refresh Fleet data"}
        aria-busy={refreshing}
        title={refreshing ? "Refreshing…" : "Refresh Fleet data"}
      >
        <span aria-hidden="true">{refreshing ? "…" : "↻"}</span>
      </button>
    </div>
  );
}
