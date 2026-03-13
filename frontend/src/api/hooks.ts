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

export function useChartData(symbol: string, hours: number) {
  return useQuery({
    queryKey: ["chart", symbol, hours],
    queryFn: () =>
      apiFetch<ChartResponse>("/api/dashboard/chart", {
        symbol,
        hours: String(hours),
        max_points: "3000",
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
      apiFetch<EquityEventsResponse>("/api/equity/events", { range }),
    refetchInterval: 30000,
  });
}
