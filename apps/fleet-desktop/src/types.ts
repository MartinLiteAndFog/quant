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

export type ChartMode = "trade" | "account" | "account_abs";

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
  stats: BotStats;
  needs_backfill?: boolean;
}

export interface PortfolioSeries {
  id: "portfolio" | string;
  display_name: string;
  color?: string;
  currency?: string | null;
  live_equity?: number | null;
  account_curve: CurvePoint[];
  account_curve_abs?: AbsCurvePoint[];
  bot_count?: number;
  note?: string;
}

export interface FleetPerformance {
  ok: boolean;
  hours: number | null;
  since?: string | null;
  series: BotSeries[];
  portfolio?: PortfolioSeries | null;
  ts?: string;
  error?: string;
}

export interface ActivityEvent {
  t: number | null;
  ts: string;
  venue: string;
  symbol: string;
  strategy_instance: string;
  bot_id?: string;
  display_name: string;
  side?: string;
  qty?: number | null;
  price?: number | null;
  stage?: string;
  status?: string;
  event_id?: string;
  color?: string;
}

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

export const DEFAULT_BOTS: FleetConfig["bots"] = [
  {
    id: "imba-runner",
    display_name: "Imba Runner",
    strategy_instance: "sol-pilot-canonical",
    venue: "kucoin",
    symbol: "SOL-USDT",
    health_url: "https://sol-pilot-canonical-production.up.railway.app/health",
    color: "#c4a35a",
    enabled: true,
  },
  {
    id: "pure-imbatp",
    display_name: "Pure ImbaTP",
    strategy_instance: "sol-pilot-pc3axis",
    venue: "kucoin",
    symbol: "SOL-USDT",
    health_url: "https://sol-pilot-pc3axis-production.up.railway.app/health",
    color: "#6b9e7a",
    enabled: true,
  },
  {
    id: "countervariante",
    display_name: "Countervariante",
    strategy_instance: "sol-pilot-countertrend",
    venue: "kucoin",
    symbol: "SOL-USDT",
    health_url: "https://sol-pilot-countertrend-production.up.railway.app/health",
    color: "#5b8fad",
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
    color: "#8a7a9a",
    enabled: true,
  },
  {
    id: "quant-main",
    display_name: "Quant (KuCoin main)",
    strategy_instance: "quant",
    venue: "kucoin",
    symbol: "SOL-USDT",
    health_url: "https://quant-production-5533.up.railway.app/health",
    color: "#9a8f6a",
    enabled: true,
  },
  {
    id: "kraken-legacy",
    display_name: "Kraken Legacy",
    strategy_instance: "kraken_bot",
    venue: "kraken",
    symbol: "SOL-USD",
    health_url: "https://kraken-production-cb57.up.railway.app/health",
    color: "#b07050",
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
    return prev ? { ...def, ...prev, id: def.id } : { ...def };
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
