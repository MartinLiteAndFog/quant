// SPDX-License-Identifier: MPL-2.0
// EW pivot/retracement/impulse calculations derived from EW_v3_2_MTF_AUDIT.pine.
// © Elliott Wave Advanced Multi-Degree Analyzer — v3.2 MTF Edition.
// IMBA channel derives from tradingview/sol_imba_strategy.pine and imba_renko_signal.pine.
// Module-derived candle research variants; no full Renko, HTF or Pine parity claim.
import type { MarketDataset, TargetPositionSignal } from "./paperReplay.ts";
import type { BeeMode } from "./simulation.ts";

export type SourceGenome = Readonly<{
  schemaVersion: 1;
  family: "imba_trend" | "imba_countertrend" | "imba_ew";
  channelLookback: number;
  pivotLength: number;
  atrLength: number;
  ewMinMoveAtr: number;
  ewRetraceMin: number;
  ewRetraceMax: number;
  ewFilter: boolean;
  ewExit: boolean;
  stopLossBps: number;
  sourceId?: string;
  name?: string;
}>;

export const SOURCE_PARENT_GENOMES: readonly SourceGenome[] = [
  { schemaVersion: 1, family: "imba_trend", channelLookback: 50, pivotLength: 13,
    atrLength: 14, ewMinMoveAtr: 3, ewRetraceMin: .382, ewRetraceMax: .786,
    ewFilter: false, ewExit: false, stopLossBps: 250,
    sourceId: "tradingview/imba_renko_signal.pine", name: "IMBA Trend · candle research" },
  { schemaVersion: 1, family: "imba_countertrend", channelLookback: 200, pivotLength: 13,
    atrLength: 14, ewMinMoveAtr: 3, ewRetraceMin: .382, ewRetraceMax: .786,
    ewFilter: false, ewExit: true, stopLossBps: 250,
    sourceId: "tradingview/sol_imba_strategy.pine", name: "IMBA Countertrend · candle research" },
  { schemaVersion: 1, family: "imba_ew", channelLookback: 50, pivotLength: 13,
    atrLength: 14, ewMinMoveAtr: 3, ewRetraceMin: .382, ewRetraceMax: .786,
    ewFilter: true, ewExit: true, stopLossBps: 250,
    sourceId: "EW_v3_2_MTF_AUDIT.pine + tradingview/imba_renko_signal.pine",
    name: "IMBA × EW · confirmed swing research" },
];

type Pivot = { index: number; price: number; high: boolean };
const FAMILIES = ["imba_trend", "imba_countertrend", "imba_ew"] as const;

export function validateSourceGenome(genome: SourceGenome): void {
  if (!genome || genome.schemaVersion !== 1 || !FAMILIES.includes(genome.family)) {
    throw new RangeError("A supported version 1 source genome is required.");
  }
  for (const value of [genome.sourceId, genome.name]) {
    if (value !== undefined && (typeof value !== "string" || value.length > 500)) {
      throw new RangeError("Optional source labels must be strings of at most 500 characters.");
    }
  }
  for (const [value, min, max] of [
    [genome.channelLookback, 20, 500], [genome.pivotLength, 2, 20], [genome.atrLength, 5, 50],
  ]) {
    if (!Number.isInteger(value) || value! < min! || value! > max!) throw new RangeError("Source lookbacks are out of bounds.");
  }
  if (!Number.isFinite(genome.ewMinMoveAtr) || genome.ewMinMoveAtr < .25 || genome.ewMinMoveAtr > 20
    || !Number.isFinite(genome.ewRetraceMin) || genome.ewRetraceMin < .1 || genome.ewRetraceMin > .8
    || !Number.isFinite(genome.ewRetraceMax) || genome.ewRetraceMax < .2 || genome.ewRetraceMax > .95
    || genome.ewRetraceMin >= genome.ewRetraceMax
    || typeof genome.ewFilter !== "boolean" || typeof genome.ewExit !== "boolean"
    || !Number.isFinite(genome.stopLossBps) || genome.stopLossBps < 50 || genome.stopLossBps > 500) {
    throw new RangeError("EW genes and mandatory close stop must be finite and bounded.");
  }
}

/** Decisions consume completed candles only. Stop is a close trigger, not a guaranteed fill price. */
export function generateSourceSignals(
  dataset: MarketDataset, genome: SourceGenome, strategyId: string, targetQuantity: string,
  startMs?: number, endExclusiveMs?: number,
): { signals: TargetPositionSignal[]; observations: number; opportunities: number; generatedSignals: number; status: "trade" | "no_trade" } {
  validateSourceGenome(genome);
  if (!strategyId?.trim() || !/^(?:0\.[0-9]{1,8}|[1-9][0-9]*(?:\.[0-9]{1,8})?)$/.test(targetQuantity)
    || !Number.isFinite(Number(targetQuantity)) || Number(targetQuantity) <= 0) throw new RangeError("A strategy ID and positive bounded decimal quantity are required.");
  const bars = dataset?.bars;
  const interval = dataset?.provenance?.intervalSeconds * 1_000;
  if (!bars?.length || !Number.isInteger(interval) || interval <= 0) throw new RangeError("A nonempty regular candle dataset is required.");
  const prices = bars.map((bar, i) => {
    const high = Number(bar.high), low = Number(bar.low), close = Number(bar.close), open = Number(bar.open);
    if (![high, low, close, open].every((value) => Number.isFinite(value) && value > 0)
      || high < Math.max(open, close) || low > Math.min(open, close) || high < low
      || !Number.isFinite(bar.timestampMs) || (i > 0 && bar.timestampMs !== bars[i - 1]!.timestampMs + interval)) {
      throw new RangeError("Source signals require positive, continuous, ordered OHLC candles.");
    }
    return { high, low, close };
  });
  const start = startMs ?? bars[0]!.timestampMs;
  const end = endExclusiveMs ?? bars[bars.length - 1]!.timestampMs + interval;
  if (!Number.isFinite(start) || !Number.isFinite(end) || start >= end) throw new RangeError("Invalid source signal session.");
  const session = bars.map((bar, i) => ({ bar, i })).filter(({ bar }) => bar.timestampMs >= start && bar.timestampMs < end);
  if (session.length < 3) throw new RangeError("A session requires three bars for a strictly later flat fill.");
  // At n-3 close, availability is n-2 open; strict next-open execution is n-1.
  const flatIndex = session[session.length - 3]!.i;
  const pivots: Pivot[] = [];
  const signals: TargetPositionSignal[] = [];
  let atr = 0, trSum = 0, trend = 0, target = 0, entryReference = 0;
  let waveBias = 0, observations = 0, opportunities = 0;
  const emit = (next: number, i: number) => {
    if (next === target) return;
    const at = new Date(bars[i]!.timestampMs + interval).toISOString();
    signals.push({ orderId: `${strategyId}-source-${signals.length + 1}`, strategyId,
      targetQuantity: next === 0 ? "0" : next > 0 ? targetQuantity : `-${targetQuantity}`,
      signalAt: at, availableAt: at, receivedAt: at });
    target = next;
    entryReference = next ? prices[i]!.close : 0;
  };
  for (let i = 0; i <= flatIndex; i += 1) {
    const { high, low, close } = prices[i]!;
    const prevClose = prices[i - 1]?.close ?? close;
    const tr = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
    trSum += tr;
    if (i === genome.atrLength - 1) atr = trSum / genome.atrLength;
    else if (i >= genome.atrLength) atr = (atr * (genome.atrLength - 1) + tr) / genome.atrLength;
    let pushed = false;
    if (i >= Math.max(2 * genome.pivotLength, genome.atrLength - 1)) {
      const center = i - genome.pivotLength;
      for (const isHigh of [true, false]) {
        const price = isHigh ? prices[center]!.high : prices[center]!.low;
        let confirmed = true;
        for (let k = center - genome.pivotLength; k <= i; k += 1) {
          if (k === center) continue;
          const other = isHigh ? prices[k]!.high : prices[k]!.low;
          // Pine pivots allow equal older values, reject equal newer values.
          if (isHigh ? (k < center ? other > price : other >= price) : (k < center ? other < price : other <= price)) confirmed = false;
        }
        if (!confirmed) continue;
        const last = pivots[pivots.length - 1];
        if (!last || (last.high !== isHigh && Math.abs(price - last.price) >= atr * genome.ewMinMoveAtr)) {
          pivots.push({ index: center, price, high: isHigh }); pushed = true;
          if (pivots.length > 100) pivots.shift();
        } else if (last.high === isHigh && (isHigh ? price > last.price : price < last.price)) {
          pivots[pivots.length - 1] = { index: center, price, high: isHigh };
        }
      }
    }
    let waveExit = 0;
    if (pushed && pivots.length >= 3) {
      const [a, b, c] = pivots.slice(-3) as [Pivot, Pivot, Pivot];
      const bull = a.price < b.price;
      const retrace = Math.abs(c.price - b.price) / Math.abs(b.price - a.price);
      if (retrace >= genome.ewRetraceMin && retrace <= genome.ewRetraceMax
        && (bull ? c.price > a.price : c.price < a.price)) waveBias = bull ? 1 : -1;
      if (pivots.length >= 6) {
        const [p0, p1, p2, p3, p4, p5] = pivots.slice(-6) as [Pivot, Pivot, Pivot, Pivot, Pivot, Pivot];
        const impulseBull = p0.price < p1.price;
        const w1 = Math.abs(p1.price - p0.price), w3 = Math.abs(p3.price - p2.price), w5 = Math.abs(p5.price - p4.price);
        const holds = impulseBull ? p2.price > p0.price && p4.price > p1.price : p2.price < p0.price && p4.price < p1.price;
        if (holds && !(w3 < w1 && w3 < w5)) waveExit = impulseBull ? 1 : -1;
      }
    }
    if (i >= genome.channelLookback - 1) {
      let channelHigh = -Infinity, channelLow = Infinity;
      for (let j = i - genome.channelLookback + 1; j <= i; j += 1) {
        channelHigh = Math.max(channelHigh, prices[j]!.high); channelLow = Math.min(channelLow, prices[j]!.low);
      }
      const range = channelHigh - channelLow;
      if (range > 0) {
        if (close >= channelHigh - range * .236) trend = 1;
        else if (close <= channelHigh - range * .786) trend = -1;
      }
    }
    if (bars[i]!.timestampMs < start) continue;
    observations += 1;
    if (i === flatIndex) { emit(0, i); break; }
    const stopped = target !== 0 && (close / entryReference - 1) * target * 10_000 <= -genome.stopLossBps;
    if (stopped || (genome.ewExit && target !== 0 && waveExit === target)) { emit(0, i); continue; }
    const desired = genome.family === "imba_countertrend" ? -trend : trend;
    if (desired !== 0 && (!genome.ewFilter || desired === waveBias)) {
      opportunities += 1;
      emit(desired, i);
    }
  }
  return { signals, observations, opportunities, generatedSignals: signals.length,
    status: signals.some((signal) => signal.targetQuantity !== "0") ? "trade" : "no_trade" };
}

/** Deterministic bounded inheritance; every offspring retains a compulsory stop. */
export function inheritSourceGenome(mode: BeeMode, parents: readonly SourceGenome[], seed: string, index: number): SourceGenome {
  if (!["random", "local", "hive", "hybrid"].includes(mode) || !parents.length || typeof seed !== "string"
    || !Number.isInteger(index) || index < 0) throw new RangeError("Invalid source inheritance request.");
  parents.forEach(validateSourceGenome);
  let state = 2166136261;
  for (const char of `${seed}:${mode}:${index}`) state = Math.imul(state ^ char.charCodeAt(0), 16777619) >>> 0;
  const random = () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296; };
  const donors = mode === "local" ? parents : mode === "hybrid" ? [...parents, ...SOURCE_PARENT_GENOMES] : SOURCE_PARENT_GENOMES;
  const base = donors[Math.floor(random() * donors.length)]!;
  const donor = () => donors[Math.floor(random() * donors.length)]!;
  const bounds = (key: "channelLookback" | "pivotLength" | "atrLength" | "ewMinMoveAtr" | "ewRetraceMin" | "ewRetraceMax" | "stopLossBps", min: number, max: number, integer = false) => {
    const localMean = parents.reduce((sum, parent) => sum + parent[key], 0) / parents.length;
    const centralMean = SOURCE_PARENT_GENOMES.reduce((sum, parent) => sum + parent[key], 0) / SOURCE_PARENT_GENOMES.length;
    const value = mode === "random" ? min + random() * (max - min)
      : (mode === "hive" ? centralMean : mode === "hybrid" ? (localMean + centralMean) / 2 : localMean) * (1 + (random() - .5) * .2);
    const result = Math.min(max, Math.max(min, value));
    return integer ? Math.round(result) : Math.round(result * 10_000) / 10_000;
  };
  const minRetrace = bounds("ewRetraceMin", .1, .8);
  const child: SourceGenome = { ...base,
    channelLookback: bounds("channelLookback", 20, 500, true), pivotLength: bounds("pivotLength", 2, 20, true),
    atrLength: bounds("atrLength", 5, 50, true), ewMinMoveAtr: bounds("ewMinMoveAtr", .25, 20),
    ewRetraceMin: minRetrace, ewRetraceMax: Math.max(minRetrace + .01, bounds("ewRetraceMax", .2, .95)),
    ewFilter: mode === "random" ? random() < .5 : donor().ewFilter,
    ewExit: mode === "random" ? random() < .5 : donor().ewExit,
    stopLossBps: bounds("stopLossBps", 50, 500, true),
    sourceId: "tradingview/imba_renko_signal.pine + tradingview/sol_imba_strategy.pine + EW_v3_2_MTF_AUDIT.pine (module recombination)",
  };
  validateSourceGenome(child);
  return child;
}
