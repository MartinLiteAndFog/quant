import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "./client";
import type {
  ChartResponse,
  StatusResponse,
  PositionResponse,
  FillsResponse,
  StateSpaceResponse,
  EquityEventsResponse,
} from "../types/chart";

export interface DashboardStrategyResponse {
  ok: boolean;
  symbol: string;
  strategy_label: "Countertrend" | "Trendfollower" | string | null;
  regime_state?: string | null;
  source?: string;
  error?: string;
  ts?: string;
}

export interface DashboardPerformanceResponse {
  ok: boolean;
  symbol: string;
  venue: string;
  as_of?: string;
  window?: string;
  // Every number here is derived from the SAME decision list that the
  // trade-mode equity chart plots. ``pnl_pct`` == the chart's final
  // ``cum_pct``; ``trade_count`` == ``points.length``; wins / losses count
  // only closed decisions whose realized pnl is > 0 / < 0. Open decisions
  // (still in-position) bump ``trade_count`` and ``open_decision_count``
  // but do NOT contribute to wins / losses / pnl / avg.
  pnl_pct: number | null;
  winrate: number | null; // 0..100
  monthly_growth: number | null;
  average_gain: number | null;
  trade_count?: number;
  closed_decision_count?: number;
  winning_trade_count?: number;
  losing_trade_count?: number;
  open_decision_count?: number;
  // Back-compat alias for the old field name. In the decision-based world
  // this is the same value as ``trade_count``. Will be retired once the UI
  // stops referencing it directly.
  trade_decision_count?: number | null;
  cum_pct?: number | null;
  // True when ``trade_decisions`` is empty but ``closed_trades`` has rows —
  // the operator should run the ``backfill=1`` job on Railway.
  needs_backfill?: boolean;
  source?: string;
  error?: string;
  ts?: string;
}

export const DEFAULT_DASHBOARD_VENUE = "kucoin" as const;

export function useChartData(symbol: string, hours: number, maxPoints = 4000) {
  return useQuery({
    queryKey: ["chart", symbol, hours, maxPoints],
    queryFn: () =>
      apiFetch<ChartResponse>("/api/dashboard/chart", {
        symbol,
        hours: String(hours),
        max_points: String(maxPoints),
      }),
    refetchInterval: 4000,
  });
}

export function useStatus() {
  return useQuery({
    queryKey: ["status"],
    queryFn: () => apiFetch<StatusResponse>("/api/status"),
    refetchInterval: 4000,
  });
}

export function usePosition() {
  return useQuery({
    queryKey: ["position"],
    queryFn: () => apiFetch<PositionResponse>("/api/position"),
    refetchInterval: 4000,
  });
}

export function useFills() {
  return useQuery({
    queryKey: ["fills"],
    queryFn: () => apiFetch<FillsResponse>("/api/dashboard/fills"),
    refetchInterval: 10000,
  });
}

export function useDashboardStrategy(symbol: string = "SOL-USDT") {
  return useQuery({
    queryKey: ["dashboardStrategy", symbol],
    queryFn: () =>
      apiFetch<DashboardStrategyResponse>("/api/dashboard/strategy", {
        symbol,
      }),
    // Strategy / regime label changes on day-gate boundaries; 10s is plenty.
    refetchInterval: 10000,
    staleTime: 10000,
  });
}

export function useDashboardPerformance(
  symbol: string = "SOL-USDT",
  venue: string = DEFAULT_DASHBOARD_VENUE
) {
  return useQuery({
    queryKey: ["dashboardPerformance", symbol, venue],
    queryFn: () =>
      apiFetch<DashboardPerformanceResponse>("/api/dashboard/performance", {
        symbol,
        venue,
      }),
    // Performance metrics aggregate hundreds of closed trades; they only
    // change once a trade closes. Refetch at a relaxed cadence aligned with
    // the backend cache TTL.
    refetchInterval: 30000,
    staleTime: 30000,
  });
}

export function useStateSpace(windowMinutes: number) {
  return useQuery({
    queryKey: ["statespace", windowMinutes],
    queryFn: () =>
      apiFetch<StateSpaceResponse>("/api/dashboard/statespace", {
        window_minutes: String(windowMinutes),
      }),
    refetchInterval: 15000,
  });
}

export function useEquityEvents(range: string) {
  return useQuery({
    queryKey: ["equityEvents", range],
    queryFn: () =>
      apiFetch<EquityEventsResponse>("/api/equity/events", {
        range,
        venue: DEFAULT_DASHBOARD_VENUE,
      }),
    refetchInterval: 30000,
  });
}