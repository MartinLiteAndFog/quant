/**
 * TypeScript types for the trading dashboard API responses.
 */

// ---------------------------------------------------------------------------
// Chart API (/api/dashboard/chart)
// ---------------------------------------------------------------------------

export interface ChartBar {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface ChartMarker {
  time: number;
  position: "aboveBar" | "belowBar";
  shape: "arrowUp" | "arrowDown" | "circle" | "square";
  color: string;
  text: string;
}

export interface ChartLevels {
  entry_bar_ts?: number;
  entry_px?: number;
  side?: "long" | "short";
  sl?: number;
  ttp?: number;
  tp1?: number;
  tp2?: number;
  ttp_trail_pct?: number;
  mode?: string;
}

export interface RegimeSpan {
  from?: number;
  to?: number;
  gate_on?: number;
}

export interface RegimePoint {
  time?: number;
  score?: number;
  [key: string]: unknown;
}

export interface RegimeLatest {
  [key: string]: unknown;
}

export interface ChartRegime {
  spans?: RegimeSpan[];
  points?: RegimePoint[];
  latest?: RegimeLatest;
}

export interface ChartSegment {
  from_time: number;
  to_time: number;
  from_price: number;
  to_price: number;
  color: string;
  positive: boolean;
}

export interface FiboLine {
  time?: number;
  value?: number;
  [key: string]: unknown;
}

export interface ChartFibo {
  lookback?: number;
  long?: FiboLine[];
  mid?: FiboLine[];
  short?: FiboLine[];
  latest?: Record<string, unknown>;
}

export interface RenkoHealth {
  ok: boolean;
  bars?: number;
  last_ts?: string;
  age_sec?: number;
}

export interface RegimeScorePoint {
  time: number;
  score: number;
}

export interface EquityCurvePoint {
  time: number;
  pnl_pct: number;
}

export interface EquityPoint {
  time: number;
  equity: number;
}

export interface EquityComponent {
  key: string;
  label: string;
  kind: string;
  points: EquityPoint[];
  source?: string;
}

export interface DiaryEntry {
  time: number;
  pnl_pct: number;
  cum_pct: number;
  side?: string;
  entry_price?: number;
  exit_price?: number;
  qty?: number;
  source?: string;
}

export interface OpenPosition {
  side: string;
  entry_time: number;
  entry_price: number;
  sl?: number;
  mode?: string;
  ttp?: number;
  tp1?: number;
  tp2?: number;
}

export interface KrakenMetrics {
  ts?: string;
  equity_usd?: number;
  wallet_usd?: number;
  upnl_usd?: number;
  mark_price?: number;
  target_size?: number;
  gate_on?: number;
  gate_source?: string;
  engine?: string;
  mode?: string;
  pos_side?: number;
  entry_px?: number;
  best_fav?: number;
  size_rem?: number;
  tp1_done?: boolean;
  signal?: number;
  signal_ts?: string;
  venue_pos_side?: number;
  venue_pos_size?: number;
  dry_run?: boolean;
}

export interface ChartResponse {
  ok: boolean;
  symbol: string;
  bars: ChartBar[];
  markers: ChartMarker[];
  levels: ChartLevels | Record<string, never>;
  ttp_trail_pct?: number;
  regime: ChartRegime;
  confidence?: number;
  gate_on?: number;
  regime_state?: string;
  gate_confidence?: number | null;
  gate_confidence_error?: string | null;
  segments: ChartSegment[];
  fibo: ChartFibo;
  renko_health?: RenkoHealth;
  regime_scores: RegimeScorePoint[];
  regime_forecast: RegimeScorePoint[];
  equity_curve: EquityCurvePoint[];
  equity_source?: string;
  equity_real?: EquityPoint[];
  equity_real_source?: string;
  equity_kraken?: EquityPoint[];
  equity_kraken_source?: string;
  equity_combined?: EquityPoint[];
  equity_combined_source?: string;
  equity_total?: EquityPoint[];
  equity_total_source?: string;
  equity_components?: EquityComponent[];
  equity_live?: EquityPoint[];
  equity_live_source?: string;
  equity_realized?: EquityPoint[];
  equity_realized_source?: string;
  kraken_metrics?: KrakenMetrics;
  diary_entries: DiaryEntry[];
  diary_source?: string;
  open_position?: OpenPosition | null;
  _debug?: Record<string, unknown>;
  ts: string;
}

// ---------------------------------------------------------------------------
// Status API (/api/status)
// ---------------------------------------------------------------------------

export interface StatusTicker {
  symbol?: string;
  last?: number;
  bid: number;
  ask: number;
  mid?: number;
  vol?: number;
}

export interface StatusBalance {
  equity: number;
  available?: number;
}

export interface StatusResponse {
  ok: boolean;
  ts?: string;
  api_configured?: boolean;
  kucoin_key_set?: boolean;
  symbol?: string;
  ticker: StatusTicker | null;
  ticker_error?: string;
  balance: StatusBalance | null;
  balance_error?: string;
  hint?: string;
}

// ---------------------------------------------------------------------------
// Position API (/api/position)
// ---------------------------------------------------------------------------

export interface PositionResponse {
  ok: boolean;
  position: number | null;
  side: string | null;
  leverage?: number | null;
  symbol?: string;
  contract_multiplier?: number;
  error?: string;
  hint?: string;
}

// ---------------------------------------------------------------------------
// Fills API (/api/dashboard/fills)
// ---------------------------------------------------------------------------

export interface FillRow {
  time: number;
  time_utc?: string;
  ts?: string;
  side: "buy" | "sell" | string;
  size: number;
  price: number;
  fee?: number;
  order_id?: string;
  client_oid?: string | null;
  reason?: string;
  reduce_only?: boolean | null;
}

export interface FillsResponse {
  ok: boolean;
  fills?: FillRow[];
  rows?: FillRow[];
  count?: number;
  error?: string;
  ts?: string;
}

// ---------------------------------------------------------------------------
// State Space API (/api/dashboard/statespace)
// ---------------------------------------------------------------------------

export interface StateSpaceTrajectoryPoint {
  ts: number;
  x: number;
  y: number;
  z: number;
}

export interface StateSpaceCurrent {
  x: number;
  y: number;
  z: number;
  conf_x?: number;
  conf_y?: number;
  conf_z?: number;
}

export interface StateSpaceRecentDensity {
  xy: [number, number, number][];
  xz: [number, number, number][];
  yz: [number, number, number][];
}

export interface StateSpaceDensityBg {
  xy: string | null;
  xz: string | null;
  yz: string | null;
}

export interface StateSpaceResponse {
  ok: boolean;
  trajectory: StateSpaceTrajectoryPoint[];
  current: StateSpaceCurrent | null;
  recent_density: StateSpaceRecentDensity;
  density_bg: StateSpaceDensityBg;
  window_hours?: number;
  error?: string;
}

// ---------------------------------------------------------------------------
// Equity Events API (to be created)
// ---------------------------------------------------------------------------

export interface EquityEvent {
  ts: string;
  venue: string;
  equity: number;
  event_type: string;
  side?: string;
}

export interface EquityEventsResponse {
  ok: boolean;
  events: EquityEvent[];
}
