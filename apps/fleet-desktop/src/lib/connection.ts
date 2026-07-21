import type { FleetBot, FleetConfig, BotStatus } from "../types";
import { fleetFetch } from "./http";

export type ConnectionMode = "fleet_api" | "direct_health" | "offline";

export interface ConnectionProbe {
  mode: ConnectionMode;
  apiBase: string;
  fleetApiOk: boolean;
  fleetApiStatus: number | null;
  fleetApiDetail: string;
  healthHits: Array<{
    id: string;
    display_name: string;
    ok: boolean;
    status: BotStatus;
    detail: string;
  }>;
  checkedAt: string;
}

function authHeaders(token: string): HeadersInit {
  if (!token) return {};
  return {
    Authorization: `Bearer ${token}`,
    "X-Webhook-Token": token,
  };
}

async function probeUrl(
  url: string,
  headers: HeadersInit = {},
): Promise<{ ok: boolean; status: number | null; body: string; json?: unknown }> {
  try {
    const res = await fleetFetch(url, { headers });
    const body = await res.text();
    let json: unknown;
    try {
      json = JSON.parse(body);
    } catch {
      json = undefined;
    }
    return { ok: res.ok, status: res.status, body: body.slice(0, 240), json };
  } catch (e) {
    return { ok: false, status: null, body: String(e) };
  }
}

function statusFromHealth(h: Record<string, unknown> | undefined): BotStatus {
  if (!h || h.ok === false) return "down";
  if (h.executor_ready && h.live_trading_enabled) return "live";
  if (h.dry_run) return "dry";
  if (h.ok) return "up";
  return "down";
}

/** Probe fleet API + each bot health URL (native HTTP in Tauri). */
export async function probeConnection(cfg: FleetConfig): Promise<ConnectionProbe> {
  const base = (cfg.apiBase || "").replace(/\/$/, "");
  const headers = authHeaders(cfg.token);
  let fleetApiOk = false;
  let fleetApiStatus: number | null = null;
  let fleetApiDetail = "no api base configured";

  if (base) {
    const qs = cfg.token ? `?probe=0&token=${encodeURIComponent(cfg.token)}` : "?probe=0";
    const hit = await probeUrl(`${base}/api/fleet/bots${qs}`, headers);
    fleetApiStatus = hit.status;
    const looksLikeHtml = hit.body.trimStart().startsWith("<!");
    const payload = hit.json as { ok?: boolean; bots?: unknown[] } | undefined;
    fleetApiOk = Boolean(hit.ok && payload && payload.ok && Array.isArray(payload.bots) && !looksLikeHtml);
    fleetApiDetail = fleetApiOk
      ? `fleet api ok (${payload?.bots?.length ?? 0} bots)`
      : looksLikeHtml
        ? "host reachable but /api/fleet/* not deployed (got SPA HTML)"
        : hit.ok
          ? `unexpected payload: ${hit.body.slice(0, 80)}`
          : `fleet api failed: HTTP ${hit.status ?? "—"} ${hit.body.slice(0, 80)}`;
  }

  const enabled = cfg.bots.filter((b) => b.enabled);
  const healthHits = await Promise.all(
    enabled.map(async (b) => {
      if (!b.health_url) {
        return {
          id: b.id,
          display_name: b.display_name,
          ok: false,
          status: "down" as BotStatus,
          detail: "no health_url",
        };
      }
      const hit = await probeUrl(b.health_url);
      const json = (hit.json && typeof hit.json === "object" ? hit.json : {}) as Record<
        string,
        unknown
      >;
      const status = hit.ok ? statusFromHealth({ ok: true, ...json }) : "down";
      return {
        id: b.id,
        display_name: b.display_name,
        ok: hit.ok,
        status,
        detail: hit.ok
          ? `instance=${String(json.instance || b.strategy_instance)} ready=${String(json.executor_ready ?? "—")}`
          : hit.body.slice(0, 100),
      };
    }),
  );

  const anyHealth = healthHits.some((h) => h.ok);
  const mode: ConnectionMode = fleetApiOk
    ? "fleet_api"
    : anyHealth
      ? "direct_health"
      : "offline";

  return {
    mode,
    apiBase: base,
    fleetApiOk,
    fleetApiStatus,
    fleetApiDetail,
    healthHits,
    checkedAt: new Date().toISOString(),
  };
}

/** Build FleetBot list by probing health URLs directly (no fleet API). */
export async function fetchBotsDirect(cfg: FleetConfig): Promise<FleetBot[]> {
  const probe = await probeConnection(cfg);
  return cfg.bots
    .filter((b) => b.enabled)
    .map((b) => {
      const hit = probe.healthHits.find((h) => h.id === b.id);
      return {
        id: b.id,
        display_name: b.display_name,
        strategy_instance: b.strategy_instance,
        venue: b.venue,
        symbol: b.symbol,
        health_url: b.health_url,
        color: b.color,
        status: hit?.status || "down",
        executor_ready: hit?.status === "live",
        live_trading_enabled: hit?.status === "live",
        dry_run: hit?.status === "dry",
        health: { ok: hit?.ok, detail: hit?.detail },
      };
    });
}
