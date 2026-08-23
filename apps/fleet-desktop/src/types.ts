export type RangeKey = "24h" | "7d" | "30d" | "all";

export type BotStatus = "live" | "dry" | "up" | "down";

export interface FleetBot {
  id: string;
  display_name: string;
  strategy_instance: string;
  venue: string;
  symbol?: string;
  health_url?: string;
  color?: string;
  status?: BotStatus;
  executor_ready?: boolean;
  live_trading_enabled?: boolean | null;
  dry_run?: boolean | null;
  health?: Record<string, unknown>;
}

export interface CurvePoint {
  t: number;
  equity_pct: number;
}

export interface AbsCurvePoint {
  t: number;
  equity: number;
}

export type ChartMode = "trade" | "account" | "account_abs" | "corrected";

export interface CorrectedMeta {
  method: "ledger" | "jump_twr" | "unavailable" | string;
  available: boolean;
  reason: string | null;
  flow_count: number;
  net_cashflow: number;
  source?: string | null;
}

export interface CashflowPoint {
  t: number;
  direction?: string | null;
  reporting_amount?: number | null;
  currency?: string | null;
  flow_type?: string | null;
}

export interface BotStats {
  return_pct: number;
  max_drawdown_pct: number;
  trade_count: number;
  wins?: number;
  losses?: number;
  win_rate: number | null;
  profit_factor: number | null;
}

export interface BotSeries {
  id: string;
  display_name: string;
  strategy_instance: string;
  venue: string;
  symbol: string;
  color?: string;
  currency?: string | null;
  live_equity?: number | null;
  trade_curve: CurvePoint[];
  account_curve: CurvePoint[];
  account_curve_abs?: AbsCurvePoint[];
  corrected_curve?: CurvePoint[];
  corrected_meta?: CorrectedMeta;
  cashflows?: CashflowPoint[];
  stats: BotStats;
  needs_backfill?: boolean;
  /** Unix seconds of the newest persisted equity snapshot (null = none). */
  last_snapshot_ts?: number | null;
  /** Seconds since last snapshot at response time (null = no snapshots). */
  snapshot_age_sec?: number | null;
  cashflow_scope?: CashflowScope;
}

export interface CashflowReturnMetric {
  available: boolean;
  return_pct: number | null;
  net_cashflow: number | null;
  flow_count: number;
  scope_label: string;
  flow_scope_label?: string;
  boundary_note: string;
  method: "ledger_segmented_equity" | string;
  excluded_bot_ids: string[];
  unavailable_bot_ids: string[];
  reason: string | null;
  as_of: string | null;
  unsupported_currencies?: string[];
}

export interface CashflowScope {
  available: boolean;
  reason: string | null;
  boundary: "futures" | string;
}

export interface PortfolioSeries {
  id: "portfolio" | string;
  display_name: string;
  color?: string;
  currency?: string | null;
  live_equity?: number | null;
  account_curve: CurvePoint[];
  account_curve_abs?: AbsCurvePoint[];
  corrected_curve?: CurvePoint[];
  bot_count?: number;
  note?: string;
  cashflow_return?: CashflowReturnMetric;
}

export interface FleetClock {
  t0: number;
  t1: number;
  interval_sec: number;
  note?: string;
}

export interface FleetPerformance {
  ok: boolean;
  hours: number | null;
  since?: string | null;
  series: BotSeries[];
  portfolio?: PortfolioSeries | null;
  clock?: FleetClock | null;
  ts?: string;
  error?: string;
}

/** Unified Activity feed row (events + closed-trade fills). */
export type ActivityKind = "event" | "fill";

export interface ActivityItem {
  id: string;
  kind: ActivityKind;
  t: number | null;
  ts: string;
  venue?: string;
  symbol?: string;
  strategy_instance: string;
  bot_id?: string;
  display_name: string;
  /** Stage / exit_event / fill label (entry, market_fill, sl_exit, tp_exit, flip, …). */
  action: string;
  side?: string;
  qty?: number | null;
  price?: number | null;
  status?: string | null;
  pnl_pct?: number | null;
  color?: string;
  /** Present on fills (closed trades). */
  trade_id?: string;
  /** Legacy event aliases (older API / client merge). */
  stage?: string;
  event_id?: string;
}

/** @deprecated Prefer ActivityItem — kept for legacy event payloads. */
export type ActivityEvent = ActivityItem & {
  stage?: string;
  event_id?: string;
};

export interface ClosedTrade {
  trade_id: string;
  side?: string;
  qty?: number | null;
  entry_price?: number | null;
  exit_price?: number | null;
  pnl_pct?: number | null;
  entry_ts?: string;
  exit_ts?: string;
  exit_event?: string;
  strategy_instance?: string;
  venue?: string;
  symbol?: string;
  bot_id?: string;
  display_name?: string;
}

export interface CapitalAccount {
  id: string;
  display_name: string;
  strategy_instance: string;
  venue: string;
  status?: BotStatus;
  executor_ready?: boolean;
  live_trading_enabled?: boolean | null;
  dry_run?: boolean | null;
  equity?: number | null;
  available?: number | null;
  unrealised_pnl?: number | null;
  equity_ts?: number | null;
  currency?: string | null;
  equity_source?: string | null;
  health?: Record<string, unknown>;
}

export interface FleetConfig {
  apiBase: string;
  token: string;
  healthPollMs: number;
  curvePollMs: number;
  bots: Array<{
    id: string;
    display_name: string;
    strategy_instance: string;
    venue: string;
    symbol: string;
    health_url: string;
    color: string;
    enabled: boolean;
  }>;
}

export const RANGE_HOURS: Record<RangeKey, number> = {
  "24h": 24,
  "7d": 168,
  "30d": 720,
  all: 0,
};

/** Prior muddy defaults — upgraded once on load so translucent stack bands stay vivid. */
const LEGACY_BOT_COLORS: Record<string, string[]> = {
  "imba-runner": ["#c4a35a", "#c9a65a"],
  "pure-imbatp": ["#6b9e7a"],
  countervariante: ["#5b8fad"],
  "counter-sl-reverse": ["#8a7a9a"],
  "quant-main": ["#9a8f6a", "#fbbf24"],
  "kraken-legacy": ["#b07050"],
};

export const DEFAULT_BOTS: FleetConfig["bots"] = [
  {
    id: "imba-runner",
    display_name: "Imba Runner",
    strategy_instance: "sol-pilot-canonical",
    venue: "kucoin",
    symbol: "SOL-USDT",
    health_url: "https://sol-pilot-canonical-production.up.railway.app/health",
    // Amber — reads clean as glass fill on charcoal
    color: "#f0b429",
    enabled: true,
  },
  {
    id: "pure-imbatp",
    display_name: "Pure ImbaTP",
    strategy_instance: "sol-pilot-pc3axis",
    venue: "kucoin",
    symbol: "SOL-USDT",
    health_url: "https://sol-pilot-pc3axis-production.up.railway.app/health",
    color: "#34d399",
    enabled: true,
  },
  {
    id: "countervariante",
    display_name: "Countervariante",
    strategy_instance: "sol-pilot-countertrend",
    venue: "kucoin",
    symbol: "SOL-USDT",
    health_url: "https://sol-pilot-countertrend-production.up.railway.app/health",
    color: "#38bdf8",
    enabled: true,
  },
  {
    id: "counter-sl-reverse",
    display_name: "Counter SL Reverse",
    strategy_instance: "sol-pilot-countertrend-sl-reverse",
    venue: "kucoin",
    symbol: "SOL-USDT",
    health_url:
      "https://sol-pilot-countertrend-sl-reverse-production.up.railway.app/health",
    color: "#a78bfa",
    enabled: true,
  },
  {
    id: "quant-main",
    display_name: "Quant (KuCoin main)",
    strategy_instance: "quant",
    venue: "kucoin",
    symbol: "SOL-USDT",
    health_url: "https://quant-production-5533.up.railway.app/health",
    // Deep violet — deliberately distinct from Imba Runner's amber.
    color: "#8b5cf6",
    enabled: true,
  },
  {
    id: "kraken-legacy",
    display_name: "Kraken Legacy",
    strategy_instance: "kraken_bot",
    venue: "kraken",
    symbol: "SOL-USD",
    health_url: "https://kraken-production-cb57.up.railway.app/health",
    color: "#fb7185",
    enabled: true,
  },
];

export const DEFAULT_CONFIG: FleetConfig = {
  // Live dashboard host (fleet routes require a deploy that includes /api/fleet/*).
  // Health rail works via direct bot URLs even before that deploy.
  apiBase: "https://quant-production-5533.up.railway.app",
  token: "",
  healthPollMs: 10_000,
  curvePollMs: 45_000,
  bots: DEFAULT_BOTS,
};

const STORAGE_KEY = "fleet-cockpit-config-v4";

function mergeBots(
  saved: FleetConfig["bots"] | undefined,
): FleetConfig["bots"] {
  if (!Array.isArray(saved) || !saved.length) {
    return structuredClone(DEFAULT_BOTS);
  }
  const byId = new Map(saved.map((b) => [b.id, b]));
  const merged = DEFAULT_BOTS.map((def) => {
    const prev = byId.get(def.id);
    if (!prev) return { ...def };
    const next = { ...def, ...prev, id: def.id };
    const legacy = LEGACY_BOT_COLORS[def.id] || [];
    const prevColor = (prev.color || "").toLowerCase();
    if (!prevColor || legacy.some((c) => c.toLowerCase() === prevColor)) {
      next.color = def.color;
    }
    return next;
  });
  // Keep any user-added bots not in defaults.
  for (const b of saved) {
    if (!DEFAULT_BOTS.some((d) => d.id === b.id)) merged.push(b);
  }
  return merged;
}

export function loadConfig(): FleetConfig {
  try {
    const raw =
      localStorage.getItem(STORAGE_KEY) ||
      localStorage.getItem("fleet-cockpit-config-v3") ||
      localStorage.getItem("fleet-cockpit-config-v2") ||
      localStorage.getItem("fleet-cockpit-config-v1");
    if (!raw) return structuredClone(DEFAULT_CONFIG);
    const parsed = JSON.parse(raw) as Partial<FleetConfig>;
    return {
      ...DEFAULT_CONFIG,
      ...parsed,
      // Fleet GETs are public by default — drop stale tokens that caused 401 banners.
      token: "",
      apiBase: (parsed.apiBase && parsed.apiBase.trim()) || DEFAULT_CONFIG.apiBase,
      bots: mergeBots(parsed.bots as FleetConfig["bots"] | undefined),
    };
  } catch {
    return structuredClone(DEFAULT_CONFIG);
  }
}

export function saveConfig(cfg: FleetConfig): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(cfg));
}
