import { writable } from 'svelte/store';
import { fetchChart, fetchStatus, fetchPosition, fetchPerformance, fetchStatespace, fetchEquityEvents } from './api.js';

export const chartStore = writable(null);
export const statusStore = writable({ status: null, position: null, performance: null });
export const statespaceStore = writable(null);
export const equityEventsStore = writable(null);

// Hardcoded for v1. In production, inject via window.__DASHBOARD_CONFIG__
// or a /api/config endpoint matching spec Environment Variables.
const CHART_MS = 4000;
const STATUS_MS = 10000;
const SS_MS = 10000;
const EQUITY_MS = 30000;

let chartInFlight = false;
let statusInFlight = false;
let ssInFlight = false;

export async function refreshChart() {
  if (chartInFlight) return;
  chartInFlight = true;
  try {
    const data = await fetchChart();
    if (data.ok) chartStore.set(data);
  } catch (e) { /* silent */ }
  finally { chartInFlight = false; }
}

export async function refreshStatus() {
  if (statusInFlight) return;
  statusInFlight = true;
  try {
    const [s, p, perf] = await Promise.all([
      fetchStatus(), fetchPosition(), fetchPerformance()
    ]);
    statusStore.set({ status: s, position: p, performance: perf });
  } catch (e) { /* silent */ }
  finally { statusInFlight = false; }
}

export async function refreshStatespace() {
  if (ssInFlight) return;
  ssInFlight = true;
  try {
    const data = await fetchStatespace();
    if (data.ok) statespaceStore.set(data);
  } catch (e) { /* silent */ }
  finally { ssInFlight = false; }
}

export async function refreshEquityEvents(range = '7d') {
  try {
    const data = await fetchEquityEvents(range);
    if (data.ok) equityEventsStore.set(data);
  } catch (e) { /* silent */ }
}

export async function refreshAll() {
  await Promise.all([refreshChart(), refreshStatus(), refreshStatespace()]);
}

export function startPolling() {
  refreshAll();
  const t1 = setInterval(refreshChart, CHART_MS);
  const t2 = setInterval(refreshStatus, STATUS_MS);
  const t3 = setInterval(refreshStatespace, SS_MS);
  return () => { clearInterval(t1); clearInterval(t2); clearInterval(t3); };
}
