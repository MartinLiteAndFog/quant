import type {
  ActivityEvent,
  CapitalAccount,
  ClosedTrade,
  FleetBot,
  FleetConfig,
  FleetPerformance,
  RangeKey,
} from "./types";
import { RANGE_HOURS } from "./types";
import { fetchBotsDirect } from "./lib/connection";
import { fleetFetch } from "./lib/http";

export { probeConnection } from "./lib/connection";
export type { ConnectionProbe, ConnectionMode } from "./lib/connection";

function authHeaders(token: string): HeadersInit {
  if (!token) return {};
  return {
    Authorization: `Bearer ${token}`,
    "X-Webhook-Token": token,
  };
}

function looksLikeSpaHtml(text: string): boolean {
  const t = text.trimStart().toLowerCase();
  return t.startsWith("<!doctype") || t.startsWith("<html");
}

async function getJsonOnce<T>(
  path: string,
  cfg: FleetConfig,
  query: Record<string, string | number | undefined>,
  token: string,
): Promise<T> {
  const base = (cfg.apiBase || "").replace(/\/$/, "");
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(query)) {
    if (v === undefined || v === "") continue;
    qs.set(k, String(v));
  }
  if (token) qs.set("token", token);
  const url = `${base}${path}${qs.toString() ? `?${qs}` : ""}`;
  const res = await fleetFetch(url, { headers: { ...authHeaders(token) } });
  const text = await res.text();
  if (!res.ok) {
    const err = new Error(`${path} → ${res.status}`) as Error & { status?: number };
    err.status = res.status;
    throw err;
  }
  if (looksLikeSpaHtml(text)) {
    throw new Error(
      `${path} returned SPA HTML — set API base to the quant dashboard host`,
    );
  }
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new Error(`${path} → invalid JSON`);
  }
}

async function getJson<T>(
  path: string,
  cfg: FleetConfig,
  query: Record<string, string | number | undefined> = {},
): Promise<T> {
  try {
    return await getJsonOnce<T>(path, cfg, query, cfg.token || "");
  } catch (e) {
    const status = (e as { status?: number })?.status;
    // Stale/wrong read token should not block public fleet GETs.
    if (status === 401 && cfg.token) {
      return getJsonOnce<T>(path, cfg, query, "");
    }
    throw e;
  }
}

export async function fetchBots(cfg: FleetConfig, probe = true): Promise<FleetBot[]> {
  try {
    const data = await getJson<{ ok: boolean; bots: FleetBot[] }>("/api/fleet/bots", cfg, {
      probe: probe ? 1 : 0,
    });
    if (!data.ok || !Array.isArray(data.bots)) {
      throw new Error("fleet bots payload invalid");
    }
    return data.bots;
  } catch {
    // Desktop still connects: probe Railway health URLs directly.
    return fetchBotsDirect(cfg);
  }
}

export async function fetchPerformance(
  cfg: FleetConfig,
  range: RangeKey,
  instanceIds?: string[],
): Promise<FleetPerformance> {
  try {
    return await getJson<FleetPerformance>("/api/fleet/performance", cfg, {
      hours: RANGE_HOURS[range],
      instances: instanceIds?.join(","),
    });
  } catch (e) {
    return {
      ok: false,
      hours: RANGE_HOURS[range],
      series: [],
      error: String(e),
      ts: new Date().toISOString(),
    };
  }
}

export async function fetchActivity(
  cfg: FleetConfig,
  range: RangeKey,
): Promise<ActivityEvent[]> {
  try {
    const data = await getJson<{ events: ActivityEvent[] }>("/api/fleet/activity", cfg, {
      hours: RANGE_HOURS[range],
      limit: 500,
    });
    return data.events || [];
  } catch {
    return [];
  }
}

export async function fetchTrades(
  cfg: FleetConfig,
  instance: string,
  range: RangeKey,
): Promise<ClosedTrade[]> {
  try {
    const data = await getJson<{ trades: ClosedTrade[] }>("/api/fleet/trades", cfg, {
      instance,
      hours: RANGE_HOURS[range],
      limit: 300,
    });
    return data.trades || [];
  } catch {
    return [];
  }
}

/** Fetch closed trades for many bots and merge newest-first. */
export async function fetchTradesForBots(
  cfg: FleetConfig,
  bots: Array<{ id: string; strategy_instance: string; display_name: string }>,
  range: RangeKey,
): Promise<ClosedTrade[]> {
  const chunks = await Promise.all(
    bots.map(async (b) => {
      const trades = await fetchTrades(cfg, b.strategy_instance || b.id, range);
      return trades.map((t) => ({
        ...t,
        bot_id: t.bot_id || b.id,
        display_name: t.display_name || b.display_name,
      }));
    }),
  );
  return chunks
    .flat()
    .sort((a, b) => String(b.exit_ts || "").localeCompare(String(a.exit_ts || "")));
}

export async function fetchCapitalization(cfg: FleetConfig): Promise<CapitalAccount[]> {
  try {
    const data = await getJson<{ accounts: CapitalAccount[] }>(
      "/api/fleet/capitalization",
      cfg,
    );
    return data.accounts || [];
  } catch {
    const bots = await fetchBotsDirect(cfg);
    return bots.map((b) => ({
      id: b.id,
      display_name: b.display_name,
      strategy_instance: b.strategy_instance,
      venue: b.venue,
      status: b.status,
      executor_ready: b.executor_ready,
      live_trading_enabled: b.live_trading_enabled,
      dry_run: b.dry_run,
      equity: null,
      equity_ts: null,
      currency: null,
      health: b.health,
    }));
  }
}
