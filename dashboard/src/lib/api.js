const BASE = '';

export async function fetchJson(url) {
  const res = await fetch(BASE + url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export function fetchChart(hours = 24 * 14, maxPoints = 4000) {
  return fetchJson(`/api/dashboard/chart?hours=${hours}&max_points=${maxPoints}`);
}

export function fetchStatus() {
  return fetchJson('/api/status');
}

export function fetchPosition() {
  return fetchJson('/api/position');
}

export function fetchPerformance(venue = 'kucoin') {
  return fetchJson(`/api/dashboard/performance?venue=${venue}`);
}

export function fetchStatespace(windowHours = 8) {
  return fetchJson(`/api/dashboard/statespace?window_hours=${windowHours}`);
}

export function fetchEquityEvents(range = '7d') {
  return fetchJson(`/api/equity/events?range=${range}&venue=kucoin`);
}
