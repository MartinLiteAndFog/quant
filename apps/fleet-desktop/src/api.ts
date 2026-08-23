import type {
  ActivityItem,
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

/** Manual Refresh / force-reload: bust intermediary caches without changing poll URLs. */
export type FleetFetchOpts = { fresh?: boolean };

function authHeaders(token: string): HeadersInit {
  if (!token) return {};
  return {
    Authorization: `Bearer ${token}`,
    "X-Webhook-Token": token,
  };
}

function withFresh(
  query: Record<string, string | number | undefined>,
  opts?: FleetFetchOpts,
): Record<string, string | number | undefined> {
  if (!opts?.fresh) return query;
  return { ...query, _t: Date.now() };
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

export async function fetchBots(
  cfg: FleetConfig,
  probe = true,
  opts?: FleetFetchOpts,
): Promise<FleetBot[]> {
  try {
    const data = await getJson<{ ok: boolean; bots: FleetBot[] }>(
      "/api/fleet/bots",
      cfg,
      withFresh({ probe: probe ? 1 : 0 }, opts),
    );
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
  opts?: FleetFetchOpts,
): Promise<FleetPerformance> {
  try {
    return await getJson<FleetPerformance>(
      "/api/fleet/performance",
      cfg,
      withFresh(
        {
          hours: RANGE_HOURS[range],
          instances: instanceIds?.join(","),
        },
        opts,
      ),
    );
  } catch (e) {
    return {
      ok: false,
      hours: RANGE_HOURS[range],
      series: [],
      portfolio: null,
      error: String(e),
      ts: new Date().toISOString(),
    };
  }
}

function normalizeActivityItem(raw: Record<string, unknown>): ActivityItem {
  const kind: ActivityItem["kind"] =
    raw.kind === "fill" || raw.trade_id ? "fill" : "event";
  const action = String(
    raw.action || raw.stage || raw.exit_event || (kind === "fill" ? "fill" : "event"),
  );
  const id = String(
    raw.id ||
      (kind === "fill"
        ? `fill:${raw.trade_id || raw.ts}`
        : raw.event_id || `event:${raw.strategy_instance}:${raw.ts}:${action}`),
  );
  return {
    id,
    kind,
    t: (raw.t as number | null | undefined) ?? null,
    ts: String(raw.ts || raw.exit_ts || ""),
    venue: raw.venue as string | undefined,
    symbol: raw.symbol as string | undefined,
    strategy_instance: String(raw.strategy_instance || ""),
    bot_id: raw.bot_id as string | undefined,
    display_name: String(raw.display_name || raw.strategy_instance || "—"),
    action,
    side: raw.side as string | undefined,
    qty: (raw.qty as number | null | undefined) ?? null,
    price: (raw.price as number | null | undefined) ?? null,
    entry_price: (raw.entry_price as number | null | undefined) ?? null,
    exit_price: (raw.exit_price as number | null | undefined) ?? null,
    status: (raw.status as string | null | undefined) ?? null,
    pnl_pct: (raw.pnl_pct as number | null | undefined) ?? null,
    realized_pnl: (raw.realized_pnl as number | null | undefined) ?? null,
    fee: (raw.fee as number | null | undefined) ?? null,
    fee_currency: (raw.fee_currency as string | undefined) || null,
    realized_funding: (raw.realized_funding as number | null | undefined) ?? null,
    position_before: (raw.position_before as number | null | undefined) ?? null,
    position_after: (raw.position_after as number | null | undefined) ?? null,
    position_ref: (raw.position_ref as string | undefined) || null,
    execution_uid: (raw.execution_uid as string | undefined) || null,
    source: (raw.source as string | undefined) || null,
    color: raw.color as string | undefined,
    trade_id: raw.trade_id as string | undefined,
    stage: (raw.stage as string | undefined) || action,
    event_id: raw.event_id as string | undefined,
  };
}

function fillFromClosedTrade(t: ClosedTrade): ActivityItem {
  return normalizeActivityItem({
    kind: "fill",
    id: `fill:${t.trade_id}`,
    trade_id: t.trade_id,
    ts: t.exit_ts,
    t: t.exit_ts ? Math.floor(Date.parse(t.exit_ts) / 1000) : null,
    venue: t.venue,
    symbol: t.symbol,
    strategy_instance: t.strategy_instance,
    bot_id: t.bot_id,
    display_name: t.display_name || t.strategy_instance,
    action: t.exit_event || "fill",
    side: t.side,
    qty: t.qty,
    price: t.exit_price,
    status: "closed",
    pnl_pct: t.pnl_pct,
  });
}

/** Preferred SoT: `/api/fleet/activity` `items`. Falls back to events + per-bot trades. */
export async function fetchActivityFeed(
  cfg: FleetConfig,
  range: RangeKey,
  bots: Array<{ id: string; strategy_instance: string; display_name: string }>,
  opts?: FleetFetchOpts,
): Promise<ActivityItem[]> {
  try {
    const data = await getJson<{
      items?: Array<Record<string, unknown>>;
      events?: Array<Record<string, unknown>>;
    }>(
      "/api/fleet/activity",
      cfg,
      withFresh(
        {
          hours: RANGE_HOURS[range],
          limit: 2000,
        },
        opts,
      ),
    );
    if ("items" in data && Array.isArray(data.items)) {
      // New SoT: server already merged events + fills.
      return data.items.map((row) => normalizeActivityItem(row));
    }
    // Older deploy: events only — merge closed trades client-side.
    const events = (data.events || []).map((row) =>
      normalizeActivityItem({ ...row, kind: "event" }),
    );
    const trades = await fetchTradesForBots(cfg, bots, range, opts);
    const fills = trades.map(fillFromClosedTrade);
    return [...events, ...fills].sort(
      (a, b) => (b.t || 0) - (a.t || 0) || String(b.ts).localeCompare(String(a.ts)),
    );
  } catch {
    try {
      const trades = await fetchTradesForBots(cfg, bots, range, opts);
      return trades.map(fillFromClosedTrade);
    } catch {
      return [];
    }
  }
}

/** @deprecated Use fetchActivityFeed. */
export async function fetchActivity(
  cfg: FleetConfig,
  range: RangeKey,
  opts?: FleetFetchOpts,
): Promise<ActivityItem[]> {
  return fetchActivityFeed(cfg, range, [], opts);
}

export async function fetchTrades(
  cfg: FleetConfig,
  instance: string,
  range: RangeKey,
  opts?: FleetFetchOpts,
): Promise<ClosedTrade[]> {
  try {
    const data = await getJson<{ trades: ClosedTrade[] }>(
      "/api/fleet/trades",
      cfg,
      withFresh(
        {
          instance,
          hours: RANGE_HOURS[range],
          limit: 300,
        },
        opts,
      ),
    );
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
  opts?: FleetFetchOpts,
): Promise<ClosedTrade[]> {
  const chunks = await Promise.all(
    bots.map(async (b) => {
      const trades = await fetchTrades(cfg, b.strategy_instance || b.id, range, opts);
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

export async function fetchCapitalization(
  cfg: FleetConfig,
  opts?: FleetFetchOpts,
): Promise<CapitalAccount[]> {
  try {
    const data = await getJson<{ accounts: CapitalAccount[] }>(
      "/api/fleet/capitalization",
      cfg,
      withFresh({}, opts),
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
