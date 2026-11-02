/**
 * Deterministic intraday paper execution with synthetic defaults and an explicit
 * historical-source research opt-in. Neither mode verifies a live venue.
 *
 * Semantics mirror ECO-02: UTC open-time candles, first open strictly after
 * feature availability and receipt, bounded latency, adverse half-spread plus
 * slippage embedded in fill price, one fee debit per filled notional, and
 * separate signed funding cashflows over (open, close].
 */

const SCALE = 100_000_000n;
const BPS = 10_000n;
const MINUTE_MS = 60_000;
const DAY_MS = 86_400_000;

export type MarketProvenance = Readonly<{
  source: string;
  license: string;
  symbol: string;
  marketType: "spot" | "linear_futures";
  contractMultiplier: string;
  intervalSeconds: number;
  synthetic: boolean;
  timezone: "UTC";
  timestampSemantics: "open";
}>;

export type MarketBarInput = Readonly<{
  timestamp: string;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
}>;

export type MarketBar = MarketBarInput & Readonly<{ timestampMs: number }>;

export type MarketDataset = Readonly<{
  bars: readonly MarketBar[];
  provenance: MarketProvenance;
  report: Readonly<{
    schemaVersion: 1;
    barCount: number;
    start: string;
    endExclusive: string;
    gapCount: number;
    missingBars: number;
    synthetic: boolean;
    fingerprint: string;
    eligibility: "synthetic_demo_only" | "provenance_unverified";
  }>;
  fingerprint: string;
}>;

export type TargetPositionSignal = Readonly<{
  orderId: string;
  strategyId: string;
  targetQuantity: string;
  signalAt: string;
  availableAt: string;
  receivedAt: string;
}>;

export type FundingSettlement = Readonly<{
  settlementId: string;
  timestamp: string;
  markPrice: string;
  rate: string;
}>;

export type PaperCosts = Readonly<{
  feeBps: string;
  spreadBps: string;
  slippageBps: string;
  maxDelaySeconds: number;
}>;

export const DEFAULT_PAPER_COSTS: PaperCosts = {
  feeBps: "6",
  spreadBps: "2",
  slippageBps: "1",
  maxDelaySeconds: 120,
};

export const PAPER_REPLAY_NOTICE =
  "Generated synthetic execution fixture only; not a strategy test, market evidence, or a profit claim.";

export const HISTORICAL_PAPER_NOTICE =
  "Historical-source research only, not_comparable: unverified venue and provenance; modeled 1x linear long/short exposure with assumed fees, spread and slippage. Actual venue funding is unknown; no profit or live-execution claim.";

export type PaperFill = Readonly<{
  orderId: string;
  strategyId: string;
  side: "buy" | "sell";
  targetQuantity: string;
  filledQuantity: string;
  executedAt: string;
  referencePrice: string;
  price: string;
  notional: string;
  fee: string;
  /** Embedded in `price`; diagnostic only and must not be debited again. */
  executionCost: string;
  datasetFingerprint: string;
}>;

export type RejectedSignal = Readonly<{
  orderId: string;
  reason: "no_timely_bar" | "zero_volume" | "same_bar_conflict" | "allocation_limit";
}>;

export type FundingCashflow = Readonly<{
  settlementId: string;
  timestamp: string;
  amount: string;
}>;

export type DailyPaperMark = Readonly<{
  dateUtc: string;
  equity: string;
  openQuantity: string;
  fillCount: number;
}>;

export type PaperReplayReport = Readonly<{
  schemaVersion: 1;
  benchmark: "cash_no_trade";
  benchmarkConclusion: "not_comparable";
  claim: typeof PAPER_REPLAY_NOTICE | typeof HISTORICAL_PAPER_NOTICE;
  days: number;
  dataset: MarketDataset["report"];
  allocationLimit: string;
  startingCash: string;
  endingCash: string;
  realizedPricePnl: string;
  fees: string;
  funding: string;
  netPnl: string;
  maxMarginUsed: string;
  finalQuantity: string;
  endedFlat: boolean;
  riskBreach: boolean;
  fills: readonly PaperFill[];
  rejectedSignals: readonly RejectedSignal[];
  fundingCashflows: readonly FundingCashflow[];
  dailyMarks: readonly DailyPaperMark[];
}>;

function parseFixed(value: string, label: string): bigint {
  if (typeof value !== "string" || !/^-?(?:0|[1-9]\d*)(?:\.\d{1,8})?$/.test(value)) {
    throw new RangeError(`${label} must be a decimal string with at most 8 places.`);
  }
  const negative = value.startsWith("-");
  const unsigned = negative ? value.slice(1) : value;
  const [whole = "0", fraction = ""] = unsigned.split(".");
  const scaled = BigInt(whole) * SCALE + BigInt(fraction.padEnd(8, "0") || "0");
  return negative ? -scaled : scaled;
}

function fixed(value: bigint): string {
  const negative = value < 0n;
  const absolute = negative ? -value : value;
  const whole = absolute / SCALE;
  const fraction = (absolute % SCALE).toString().padStart(8, "0").replace(/0+$/, "");
  return `${negative ? "-" : ""}${whole}${fraction ? `.${fraction}` : ""}`;
}

function multiply(left: bigint, right: bigint): bigint {
  const product = left * right;
  const negative = product < 0n;
  const absolute = negative ? -product : product;
  const rounded = (absolute + SCALE / 2n) / SCALE;
  return negative ? -rounded : rounded;
}

function divideInteger(value: bigint, divisor: bigint): bigint {
  if (divisor <= 0n) throw new RangeError("Divisor must be positive.");
  const negative = value < 0n;
  const absolute = negative ? -value : value;
  const rounded = (absolute + divisor / 2n) / divisor;
  return negative ? -rounded : rounded;
}

function divideFixed(numerator: bigint, denominator: bigint): bigint {
  if (denominator === 0n) throw new RangeError("Cannot divide by zero.");
  const negative = sign(numerator) !== sign(denominator);
  const product = absolute(numerator) * SCALE;
  const divisor = absolute(denominator);
  const quotient = (product + divisor / 2n) / divisor;
  return negative ? -quotient : quotient;
}

function absolute(value: bigint): bigint {
  return value < 0n ? -value : value;
}

function sign(value: bigint): -1n | 0n | 1n {
  return value < 0n ? -1n : value > 0n ? 1n : 0n;
}

function parseUtc(value: string, label: string): number {
  if (typeof value !== "string" || !/(?:Z|[+-]\d{2}:\d{2})$/i.test(value)) {
    throw new RangeError(`${label} must include a timezone.`);
  }
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) throw new RangeError(`${label} is not a valid timestamp.`);
  return timestamp;
}

function isoUtc(timestampMs: number): string {
  return new Date(timestampMs).toISOString();
}

function validateProvenance(provenance: MarketProvenance): void {
  if (!provenance.source.trim() || !provenance.license.trim() || !provenance.symbol.trim()) {
    throw new RangeError("Source, license, and symbol are required.");
  }
  if (!Number.isInteger(provenance.intervalSeconds) || provenance.intervalSeconds <= 0) {
    throw new RangeError("A positive whole-second interval is required.");
  }
  const multiplier = parseFixed(provenance.contractMultiplier, "Contract multiplier");
  if (multiplier <= 0n) throw new RangeError("Contract multiplier must be positive.");
  if (provenance.timezone !== "UTC" || provenance.timestampSemantics !== "open") {
    throw new RangeError("Market bars must be normalized to UTC open timestamps.");
  }
}

async function sha256(value: string): Promise<string> {
  const bytes = new TextEncoder().encode(value);
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

/** Normalize bars without filling gaps; historical data requires explicit research opt-in. */
export async function normalizeMarketData(
  inputBars: readonly MarketBarInput[],
  provenance: MarketProvenance,
): Promise<MarketDataset> {
  validateProvenance(provenance);
  if (inputBars.length === 0) throw new RangeError("At least one market bar is required.");
  const intervalMs = provenance.intervalSeconds * 1_000;
  const seen = new Set<number>();
  const bars = inputBars.map((bar, index): MarketBar => {
    const timestampMs = parseUtc(bar.timestamp, `Bar ${index} timestamp`);
    if (timestampMs % intervalMs !== 0) throw new RangeError("Bar timestamps must align to the declared interval.");
    if (seen.has(timestampMs)) throw new RangeError("Duplicate candle timestamp.");
    seen.add(timestampMs);
    const open = parseFixed(bar.open, "Open");
    const high = parseFixed(bar.high, "High");
    const low = parseFixed(bar.low, "Low");
    const close = parseFixed(bar.close, "Close");
    const volume = parseFixed(bar.volume, "Volume");
    if (open <= 0n || high <= 0n || low <= 0n || close <= 0n || volume < 0n
      || high < open || high < low || high < close || low > open || low > close) {
      throw new RangeError("Invalid OHLCV values.");
    }
    return { ...bar, timestamp: isoUtc(timestampMs), timestampMs };
  }).sort((left, right) => left.timestampMs - right.timestampMs);

  const startMs = bars[0]?.timestampMs;
  const lastMs = bars[bars.length - 1]?.timestampMs;
  if (startMs === undefined || lastMs === undefined) throw new RangeError("Dataset is empty.");
  let missingBars = 0;
  for (let index = 1; index < bars.length; index += 1) {
    const previous = bars[index - 1];
    const current = bars[index];
    if (previous && current && current.timestampMs > previous.timestampMs + intervalMs) {
      missingBars += (current.timestampMs - previous.timestampMs) / intervalMs - 1;
    }
  }
  const endExclusiveMs = lastMs + intervalMs;
  const canonical = JSON.stringify({
    provenance,
    bars: bars.map(({ timestamp, open, high, low, close, volume }) => ({ timestamp, open, high, low, close, volume })),
  });
  const fingerprint = await sha256(canonical);
  const report = {
    schemaVersion: 1 as const,
    barCount: bars.length,
    start: isoUtc(startMs),
    endExclusive: isoUtc(endExclusiveMs),
    gapCount: inputBars.length ? bars.slice(1).reduce((count, bar, index) => {
      const previous = bars[index];
      return count + (previous && bar.timestampMs > previous.timestampMs + intervalMs ? 1 : 0);
    }, 0) : 0,
    missingBars,
    synthetic: provenance.synthetic,
    fingerprint,
    eligibility: provenance.synthetic ? "synthetic_demo_only" as const : "provenance_unverified" as const,
  };
  return { bars, provenance, report, fingerprint };
}

const demoProvenance: MarketProvenance = {
  source: "Cloudgarden synthetic_week v1",
  license: "CC0 synthetic fixture",
  symbol: "SOL-DEMO",
  marketType: "linear_futures",
  contractMultiplier: "1",
  intervalSeconds: 60,
  synthetic: true,
  timezone: "UTC",
  timestampSemantics: "open",
};

/** Generated fixture only; these prices do not represent historical SOL. */
export async function createSyntheticMarketWeek(start = "2024-01-01T00:00:00Z"): Promise<MarketDataset> {
  const startMs = parseUtc(start, "Synthetic fixture start");
  if (startMs % MINUTE_MS !== 0) throw new RangeError("Synthetic fixture start must align to a UTC minute.");
  const bars: MarketBarInput[] = Array.from({ length: 7 * 24 * 60 }, (_, index) => {
    const open = 100 + (index % 240) / 100;
    return {
      timestamp: isoUtc(startMs + index * MINUTE_MS),
      open: open.toFixed(2),
      high: (open + 1).toFixed(2),
      low: (open - 1).toFixed(2),
      close: (open + 0.1).toFixed(2),
      volume: "1000",
    };
  });
  return normalizeMarketData(bars, demoProvenance);
}

function stableHash(value: string): number {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

function seededUnit(value: string): number {
  return stableHash(value) / 0x1_0000_0000;
}

/**
 * Generated, deterministic synthetic fixture for bounded garden-cycle replays.
 * Every day has a seed- and cycle-specific regime; no historical prices are used.
 */
export async function createSyntheticMarketRange(
  days: number,
  start = "2024-01-01T00:00:00Z",
  gardenSeed = "cloudgarden-demo",
  weekIndex = 0,
): Promise<MarketDataset> {
  if (!Number.isInteger(days) || days < 1 || days > 90) {
    throw new RangeError("Synthetic range must contain between 1 and 90 whole days.");
  }
  if (typeof gardenSeed !== "string" || gardenSeed.length === 0) {
    throw new RangeError("A non-empty garden seed is required.");
  }
  if (!Number.isInteger(weekIndex) || weekIndex < 0) {
    throw new RangeError("Week index must be a non-negative whole number.");
  }
  const startMs = parseUtc(start, "Synthetic fixture start");
  if (startMs % MINUTE_MS !== 0) throw new RangeError("Synthetic fixture start must align to a UTC minute.");

  const cycleKey = `${gardenSeed}:${weekIndex}:${days}`;
  const fixtureProvenance: MarketProvenance = {
    ...demoProvenance,
    source: `Cloudgarden synthetic_range v1 seed-${stableHash(gardenSeed).toString(16)} week-${weekIndex} days-${days}`,
  };
  const bars: MarketBarInput[] = [];
  let previousClose = 100 + seededUnit(`${cycleKey}:initial`) * 4;
  const barCount = days * 24 * 60;

  for (let index = 0; index < barCount; index += 1) {
    const day = Math.floor(index / (24 * 60));
    const minuteOfDay = index % (24 * 60);
    const dayKey = `${cycleKey}:${day}`;
    const regime = (stableHash(`${dayKey}:regime`) % 3) - 1;
    const phase = seededUnit(`${dayKey}:phase`) * Math.PI * 2;
    const volatility = 0.00006 + seededUnit(`${dayKey}:volatility`) * 0.00008;
    const cycle = Math.sin((minuteOfDay / 240) * Math.PI * 2 + phase);
    const shorterCycle = Math.sin((minuteOfDay / 37) * Math.PI * 2 + phase / 2);
    const change = previousClose * (regime * 0.000015 + cycle * volatility + shorterCycle * 0.000025);
    const open = Number(previousClose.toFixed(2));
    const close = Math.max(1, Number((open + change).toFixed(2)));
    const wick = 0.03 + seededUnit(`${dayKey}:wick:${minuteOfDay}`) * 0.07;
    const timestamp = isoUtc(startMs + index * MINUTE_MS);
    bars.push({
      timestamp,
      open: open.toFixed(2),
      high: (Math.max(open, close) + wick).toFixed(2),
      low: Math.max(0.01, Math.min(open, close) - wick).toFixed(2),
      close: close.toFixed(2),
      volume: (50 + seededUnit(`${dayKey}:volume:${minuteOfDay}`) * 1_950).toFixed(2),
    });
    previousClose = close;
  }
  return normalizeMarketData(bars, fixtureProvenance);
}

function fixtureTime(start: string, minutesFromStart: number): string {
  return isoUtc(parseUtc(start, "Synthetic fixture start") + minutesFromStart * MINUTE_MS);
}

/** Fixed external position script for execution checks; it is not alpha logic. */
export function syntheticDemoSignals(start = "2024-01-01T00:00:00Z", strategyId = "parent-01"): readonly TargetPositionSignal[] {
  const openSignal = fixtureTime(start, 1);
  const closeSignal = fixtureTime(start, 5);
  return [
    { orderId: "demo-open-long", strategyId, targetQuantity: "0.1", signalAt: openSignal, availableAt: openSignal, receivedAt: openSignal },
    { orderId: "demo-close-long", strategyId, targetQuantity: "0", signalAt: closeSignal, availableAt: closeSignal, receivedAt: closeSignal },
  ];
}

function fillPrice(open: bigint, buy: boolean, spreadBps: bigint, slippageBps: bigint): { price: bigint; cost: bigint } {
  const halfSpread = divideInteger(spreadBps, 2n);
  const impactBps = halfSpread + slippageBps;
  if (impactBps >= BPS * SCALE) throw new RangeError("Spread and slippage must remain below 100% of price.");
  const cost = divideInteger(multiply(open, impactBps), BPS);
  return { price: open + (buy ? cost : -cost), cost };
}

function findFirstBarAfter(bars: readonly MarketBar[], timestampMs: number): number {
  let low = 0;
  let high = bars.length;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    const bar = bars[mid];
    if (bar && bar.timestampMs <= timestampMs) low = mid + 1;
    else high = mid;
  }
  return low;
}

function dayKey(timestampMs: number): string {
  return new Date(timestampMs).toISOString().slice(0, 10);
}

export type PaperReplayInput = Readonly<{
  dataset: MarketDataset;
  signals: readonly TargetPositionSignal[];
  fundingSettlements?: readonly FundingSettlement[];
  allocationLimit: string;
  startingCash: string;
  costs?: PaperCosts;
  /** Explicit research opt-in; does not verify archive provenance or enable live execution. */
  historicalResearch?: boolean;
}>;

/** Run one isolated modeled linear position; historical research is opt-in only. */
export function runPaperReplay(input: PaperReplayInput): PaperReplayReport {
  const { dataset } = input;
  const historicalResearch = input.historicalResearch === true && dataset.provenance.synthetic === false;
  if ((!dataset.provenance.synthetic || !dataset.report.synthetic) && !historicalResearch) {
    throw new RangeError("Empirical replay is disabled until the provenance and data-quality gate is implemented.");
  }
  validateProvenance(dataset.provenance);
  if (dataset.provenance.synthetic !== dataset.report.synthetic
    || dataset.report.eligibility !== (dataset.provenance.synthetic ? "synthetic_demo_only" : "provenance_unverified")
    || !/^[0-9a-f]{64}$/.test(dataset.fingerprint)
    || dataset.report.fingerprint !== dataset.fingerprint
    || dataset.report.barCount !== dataset.bars.length) {
    throw new RangeError("Dataset provenance, report, and fingerprint must agree.");
  }
  if (dataset.bars.length === 0 || dataset.provenance.marketType !== "linear_futures") {
    throw new RangeError("Paper replay requires non-empty linear-futures bars.");
  }
  if (historicalResearch) {
    const intervalMs = dataset.provenance.intervalSeconds * 1_000;
    for (const [index, bar] of dataset.bars.entries()) {
      const timestampMs = parseUtc(bar.timestamp, "Historical bar timestamp");
      const open = parseFixed(bar.open, "Open");
      const high = parseFixed(bar.high, "High");
      const low = parseFixed(bar.low, "Low");
      const close = parseFixed(bar.close, "Close");
      const volume = parseFixed(bar.volume, "Volume");
      if (bar.timestampMs !== timestampMs || timestampMs % intervalMs !== 0
        || (index > 0 && timestampMs <= dataset.bars[index - 1]!.timestampMs)
        || open <= 0n || high <= 0n || low <= 0n || close <= 0n || volume < 0n
        || high < open || high < low || high < close || low > open || low > close) {
        throw new RangeError("Historical bars must retain normalized UTC chronology and valid OHLCV.");
      }
    }
    const first = dataset.bars[0]!;
    const last = dataset.bars[dataset.bars.length - 1]!;
    if (dataset.report.start !== first.timestamp
      || dataset.report.endExclusive !== isoUtc(last.timestampMs + intervalMs)
      || parseFixed(dataset.provenance.contractMultiplier, "Historical contract multiplier") !== SCALE) {
      throw new RangeError("Historical research requires matching report dates and a 1x linear price model.");
    }
  }
  const allocationLimit = parseFixed(input.allocationLimit, "Allocation limit");
  const startingCash = parseFixed(input.startingCash, "Starting cash");
  const multiplier = parseFixed(dataset.provenance.contractMultiplier, "Contract multiplier");
  if (allocationLimit <= 0n || startingCash <= 0n || allocationLimit > startingCash) {
    throw new RangeError("Allocation must be positive and within available starting cash.");
  }
  const costs = input.costs ?? DEFAULT_PAPER_COSTS;
  const feeBps = parseFixed(costs.feeBps, "Fee bps");
  const spreadBps = parseFixed(costs.spreadBps, "Spread bps");
  const slippageBps = parseFixed(costs.slippageBps, "Slippage bps");
  if (feeBps < 0n || spreadBps < 0n || slippageBps < 0n || feeBps > BPS * SCALE
    || !Number.isInteger(costs.maxDelaySeconds) || costs.maxDelaySeconds <= 0) {
    throw new RangeError("Paper execution costs and delay must be non-negative and bounded.");
  }

  const signals = [...input.signals].sort((left, right) => {
    const leftAt = parseUtc(left.signalAt, "Signal time");
    const rightAt = parseUtc(right.signalAt, "Signal time");
    return leftAt - rightAt || left.orderId.localeCompare(right.orderId);
  });
  if (new Set(signals.map((signal) => signal.strategyId)).size > 1) {
    throw new RangeError("A v1 replay accepts one strategy position stream at a time.");
  }
  const seenOrders = new Set<string>();
  const claimedBars = new Set<number>();
  const scheduled = new Map<number, TargetPositionSignal>();
  const rejectedSignals: RejectedSignal[] = [];
  for (const signal of signals) {
    const signalAt = parseUtc(signal.signalAt, "Signal time");
    const availableAt = parseUtc(signal.availableAt, "Availability time");
    const receivedAt = parseUtc(signal.receivedAt, "Receipt time");
    parseFixed(signal.targetQuantity, "Target quantity");
    if (!signal.orderId.trim() || !signal.strategyId.trim() || seenOrders.has(signal.orderId)) {
      throw new RangeError("Signal IDs must be non-empty and unique.");
    }
    seenOrders.add(signal.orderId);
    if (availableAt < signalAt || receivedAt < signalAt) throw new RangeError("Invalid signal chronology.");
    const readyAt = Math.max(availableAt, receivedAt);
    const barIndex = findFirstBarAfter(dataset.bars, readyAt);
    const bar = dataset.bars[barIndex];
    if (!bar || (bar.timestampMs - availableAt) / 1_000 > costs.maxDelaySeconds) {
      rejectedSignals.push({ orderId: signal.orderId, reason: "no_timely_bar" });
      continue;
    }
    if (claimedBars.has(barIndex)) {
      rejectedSignals.push({ orderId: signal.orderId, reason: "same_bar_conflict" });
      continue;
    }
    claimedBars.add(barIndex);
    if (parseFixed(bar.volume, "Volume") <= 0n) {
      rejectedSignals.push({ orderId: signal.orderId, reason: "zero_volume" });
      continue;
    }
    // Target quantities are resolved during the chronological pass. A claimed
    // candle stays claimed even if the target later fails the margin check.
    scheduled.set(barIndex, signal);
  }

  const settlements = [...(input.fundingSettlements ?? [])].map((event) => ({
    ...event,
    timestampMs: parseUtc(event.timestamp, "Funding timestamp"),
    markPriceRaw: parseFixed(event.markPrice, "Funding mark price"),
    rateRaw: parseFixed(event.rate, "Funding rate"),
  })).sort((left, right) => left.timestampMs - right.timestampMs || left.settlementId.localeCompare(right.settlementId));
  const firstBar = dataset.bars[0];
  const lastBar = dataset.bars[dataset.bars.length - 1];
  if (!firstBar || !lastBar) throw new RangeError("Dataset contains no bars.");
  const dataEndExclusiveMs = lastBar.timestampMs + dataset.provenance.intervalSeconds * 1_000;
  const settlementIds = new Set<string>();
  for (const event of settlements) {
    if (!event.settlementId.trim() || settlementIds.has(event.settlementId)) {
      throw new RangeError("Funding settlement IDs must be non-empty and unique.");
    }
    if (event.markPriceRaw <= 0n) throw new RangeError("Funding mark price must be positive.");
    if (event.timestampMs < firstBar.timestampMs || event.timestampMs >= dataEndExclusiveMs) {
      throw new RangeError("Funding settlement timestamp falls outside the dataset window.");
    }
    settlementIds.add(event.settlementId);
  }

  const fills: PaperFill[] = [];
  const fundingCashflows: FundingCashflow[] = [];
  const dailyMarks: DailyPaperMark[] = [];
  let position = 0n;
  let averageEntry = 0n;
  let realizedPricePnl = 0n;
  let fees = 0n;
  let funding = 0n;
  let cash = startingCash;
  let maxMarginUsed = 0n;
  let riskBreach = false;
  let settlementIndex = 0;
  let dailyFillCount = 0;
  let currentDay = "";
  let completedDayEquity = startingCash;
  let completedDayQuantity = 0n;

  const applySettlementsThrough = (timestampMs: number) => {
    while (settlementIndex < settlements.length) {
      const settlement = settlements[settlementIndex];
      if (!settlement || settlement.timestampMs > timestampMs) break;
      settlementIndex += 1;
      // Positive rate transfers from longs to shorts. At a fill timestamp the
      // existing position settles before the order changes that position.
      const amount = position === 0n
        ? 0n
        : -multiply(multiply(multiply(position, multiplier), settlement.markPriceRaw), settlement.rateRaw);
      funding += amount;
      cash += amount;
      fundingCashflows.push({ settlementId: settlement.settlementId, timestamp: isoUtc(settlement.timestampMs), amount: fixed(amount) });
    }
  };

  for (let index = 0; index < dataset.bars.length; index += 1) {
    const bar = dataset.bars[index];
    if (!bar) continue;
    const key = dayKey(bar.timestampMs);
    if (currentDay && key !== currentDay) {
      dailyMarks.push({ dateUtc: currentDay, equity: fixed(completedDayEquity), openQuantity: fixed(completedDayQuantity), fillCount: dailyFillCount });
      dailyFillCount = 0;
    }
    applySettlementsThrough(bar.timestampMs);
    const signal = scheduled.get(index);
    if (signal) {
      const target = parseFixed(signal.targetQuantity, "Target quantity");
      const delta = target - position;
      if (delta !== 0n) {
        const buy = delta > 0n;
        const { price, cost } = fillPrice(parseFixed(bar.open, "Open"), buy, spreadBps, slippageBps);
        const targetMargin = multiply(multiply(absolute(target), multiplier), price);
        if (targetMargin > allocationLimit) {
          rejectedSignals.push({ orderId: signal.orderId, reason: "allocation_limit" });
        } else {
          const quantity = absolute(delta);
          const notional = multiply(multiply(quantity, multiplier), price);
          const fee = divideInteger(multiply(notional, feeBps), BPS);
          const previousPosition = position;
          const previousEntry = averageEntry;
          const closingQuantity = sign(previousPosition) === sign(delta)
            ? 0n
            : (absolute(previousPosition) < quantity ? absolute(previousPosition) : quantity);
          if (closingQuantity > 0n) {
            const priceChange = price - previousEntry;
            const closePnl = multiply(multiply(closingQuantity, multiplier), priceChange) * sign(previousPosition);
            realizedPricePnl += closePnl;
            cash += closePnl;
          }
          if (target === 0n) {
            averageEntry = 0n;
          } else if (previousPosition === 0n || sign(target) !== sign(previousPosition)) {
            averageEntry = price;
          } else if (sign(previousPosition) === sign(delta)) {
            const oldCost = multiply(multiply(absolute(previousPosition), multiplier), previousEntry);
            const addCost = multiply(multiply(quantity, multiplier), price);
            const newQuantity = absolute(target);
            averageEntry = divideFixed(oldCost + addCost, multiply(newQuantity, multiplier));
          }
          position = target;
          fees += fee;
          cash -= fee;
          const fill: PaperFill = {
            orderId: signal.orderId,
            strategyId: signal.strategyId,
            side: buy ? "buy" : "sell",
            targetQuantity: fixed(target),
            filledQuantity: fixed(quantity),
            executedAt: bar.timestamp,
            referencePrice: bar.open,
            price: fixed(price),
            notional: fixed(notional),
            fee: fixed(fee),
            executionCost: fixed(multiply(multiply(quantity, multiplier), cost)),
            datasetFingerprint: dataset.fingerprint,
          };
          fills.push(fill);
          dailyFillCount += 1;
        }
      }
    }

    const mark = parseFixed(bar.close, "Close");
    const margin = multiply(multiply(absolute(position), multiplier), mark);
    if (margin > maxMarginUsed) maxMarginUsed = margin;
    const unrealized = position === 0n
      ? 0n
      : multiply(multiply(position, multiplier), mark - averageEntry);
    const equity = cash + unrealized;
    if (margin > allocationLimit || margin > equity || equity <= 0n) riskBreach = true;
    currentDay = key;
    completedDayEquity = equity;
    completedDayQuantity = position;
    // `unrealized` is intentionally computed at every close even though only
    // day-end marks are retained; it makes the mark path explicit for risk.
    void unrealized;
  }

  const finalBar = lastBar;
  applySettlementsThrough(finalBar.timestampMs + dataset.provenance.intervalSeconds * 1_000);
  if (currentDay) {
    const mark = parseFixed(finalBar.close, "Close");
    const unrealized = position === 0n ? 0n : multiply(multiply(position, multiplier), mark - averageEntry);
    dailyMarks.push({ dateUtc: currentDay, equity: fixed(cash + unrealized), openQuantity: fixed(position), fillCount: dailyFillCount });
  }
  if (position !== 0n) throw new RangeError("Paper replay must close every position before booking its outcome.");
  if (cash <= 0n) throw new RangeError("Paper replay exhausted its isolated plant allocation.");

  const intervalMs = dataset.provenance.intervalSeconds * 1_000;
  const spanMs = finalBar.timestampMs + intervalMs - (dataset.bars[0]?.timestampMs ?? finalBar.timestampMs);
  const days = Math.ceil(spanMs / DAY_MS);
  const netPnl = realizedPricePnl - fees + funding;
  return {
    schemaVersion: 1,
    benchmark: "cash_no_trade",
    benchmarkConclusion: "not_comparable",
    claim: historicalResearch ? HISTORICAL_PAPER_NOTICE : PAPER_REPLAY_NOTICE,
    days,
    dataset: dataset.report,
    allocationLimit: fixed(allocationLimit),
    startingCash: fixed(startingCash),
    endingCash: fixed(cash),
    realizedPricePnl: fixed(realizedPricePnl),
    fees: fixed(fees),
    funding: fixed(funding),
    netPnl: fixed(netPnl),
    maxMarginUsed: fixed(maxMarginUsed),
    finalQuantity: fixed(position),
    endedFlat: true,
    riskBreach,
    fills,
    rejectedSignals,
    fundingCashflows,
    dailyMarks,
  };
}

export async function runSyntheticPaperDemo(
  allocationLimit: string,
  startingCash: string,
  strategyId = "parent-01",
): Promise<PaperReplayReport> {
  const dataset = await createSyntheticMarketWeek();
  return runPaperReplay({
    dataset,
    signals: syntheticDemoSignals(undefined, strategyId),
    allocationLimit,
    startingCash,
  });
}
