import assert from "node:assert/strict";
import test from "node:test";
import { normalizeMarketData, type MarketProvenance } from "./paperReplay.ts";
import { generateSourceSignals, inheritSourceGenome, SOURCE_PARENT_GENOMES, validateSourceGenome, type SourceGenome } from "./sourceStrategies.ts";

const origin = Date.parse("2025-01-01T00:00:00Z");
const provenance: MarketProvenance = { source: "source strategy test candles", license: "CC0 test fixture",
  symbol: "SOLUSDT", marketType: "spot", contractMultiplier: "1", intervalSeconds: 60,
  synthetic: false, timezone: "UTC", timestampSemantics: "open" };
const genome = (changes: Partial<SourceGenome> = {}): SourceGenome => ({ ...SOURCE_PARENT_GENOMES[0]!, channelLookback: 20, ...changes });
async function fixture(closes: number[]) {
  return normalizeMarketData(closes.map((close, i) => ({
    timestamp: new Date(origin + i * 60_000).toISOString(), open: close.toFixed(6), high: (close + .05).toFixed(6),
    low: (close - .05).toFixed(6), close: close.toFixed(6), volume: "1000",
  })), provenance);
}

test("source parents identify real IMBA and EW source modules and retain mandatory stops", () => {
  assert.deepEqual(SOURCE_PARENT_GENOMES.map((parent) => parent.family), ["imba_trend", "imba_countertrend", "imba_ew"]);
  assert.ok(SOURCE_PARENT_GENOMES[0]!.sourceId!.includes("imba_renko_signal.pine"));
  assert.ok(SOURCE_PARENT_GENOMES[2]!.sourceId!.includes("EW_v3_2_MTF_AUDIT.pine"));
  SOURCE_PARENT_GENOMES.forEach(validateSourceGenome);
});

test("IMBA trend and countertrend produce opposite causal candle positions and close before session end", async () => {
  const data = await fixture(Array.from({ length: 70 }, (_, i) => 100 + i * .5));
  const trend = generateSourceSignals(data, genome(), "trend", "0.1");
  const counter = generateSourceSignals(data, genome({ family: "imba_countertrend" }), "counter", "0.1");
  assert.equal(trend.status, "trade");
  assert.equal(trend.signals[0]!.targetQuantity, "0.1");
  assert.equal(counter.signals[0]!.targetQuantity, "-0.1");
  assert.equal(trend.signals.at(-1)!.targetQuantity, "0");
  assert.equal(trend.signals.at(-1)!.availableAt, new Date(origin + 68 * 60_000).toISOString());
  assert.ok(Date.parse(trend.signals.at(-1)!.availableAt) < data.bars.at(-1)!.timestampMs);
  assert.equal(trend.generatedSignals, trend.signals.length);
});

test("source decisions are prefix invariant before the prefix forced-flat boundary", async () => {
  const values = Array.from({ length: 120 }, (_, i) => 100 + Math.sin(i / 7) * 8 + i * .02);
  const full = generateSourceSignals(await fixture(values), genome({ ewExit: true }), "same", "0.2");
  const prefix = generateSourceSignals(await fixture(values.slice(0, 80)), genome({ ewExit: true }), "same", "0.2");
  const beforeForcedFlat = origin + 78 * 60_000;
  assert.deepEqual(prefix.signals.filter((signal) => Date.parse(signal.availableAt) < beforeForcedFlat),
    full.signals.filter((signal) => Date.parse(signal.availableAt) < beforeForcedFlat));
});

test("EW filtering waits until a retracement pivot has its full right-hand confirmation", async () => {
  const values = [...Array<number>(25).fill(100),
    101, 104, 108, 112, 116, 120, 118, 115, 112, 110, 111, 113, 116, 120, 124, 126, 128, 130, 132, 134, 136];
  const result = generateSourceSignals(await fixture(values), genome({ family: "imba_ew", ewFilter: true,
    pivotLength: 3, atrLength: 5, ewMinMoveAtr: .25 }), "ew", "0.1");
  assert.equal(result.status, "trade");
  // The retracement low occurs at index 34, is known at index 37 close.
  assert.equal(result.signals[0]!.availableAt, new Date(origin + 38 * 60_000).toISOString());
  const beforeConfirmation = await fixture(values.slice(0, 37));
  assert.equal(generateSourceSignals(beforeConfirmation, genome({ family: "imba_ew", ewFilter: true,
    pivotLength: 3, atrLength: 5, ewMinMoveAtr: .25 }), "ew", "0.1").status, "no_trade");
});

test("compulsory close stop emits flat before considering an opposite entry", async () => {
  const values = [...Array.from({ length: 24 }, (_, i) => 100 + i), 110, 109, 108, 107, 106, 105];
  const result = generateSourceSignals(await fixture(values), genome({ stopLossBps: 100 }), "stop", "0.1");
  assert.equal(result.signals[0]!.targetQuantity, "0.1");
  const stop = result.signals.find((signal) => signal.availableAt === new Date(origin + 25 * 60_000).toISOString());
  assert.equal(stop?.targetQuantity, "0");
  assert.equal(result.signals.filter((signal) => signal.availableAt === stop!.availableAt).length, 1);
});

test("warmup observations influence state but never emit orders before the session", async () => {
  const data = await fixture(Array.from({ length: 100 }, (_, i) => 100 + i * .5));
  const start = origin + 70 * 60_000, end = origin + 95 * 60_000;
  const result = generateSourceSignals(data, genome(), "session", "0.1", start, end);
  assert.equal(result.signals[0]!.availableAt, new Date(start + 60_000).toISOString());
  assert.ok(result.signals.every((signal) => Date.parse(signal.availableAt) > start && Date.parse(signal.availableAt) < end));
  assert.equal(result.observations, 23);
});

test("all inheritance modes are deterministic, varied and bounded with compulsory stops", () => {
  const results: SourceGenome[] = [];
  for (const mode of ["random", "local", "hive", "hybrid"] as const) {
    for (let i = 0; i < 100; i += 1) {
      const child = inheritSourceGenome(mode, SOURCE_PARENT_GENOMES, "garden", i);
      assert.deepEqual(child, inheritSourceGenome(mode, SOURCE_PARENT_GENOMES, "garden", i));
      validateSourceGenome(child);
      assert.ok(child.stopLossBps >= 50 && child.stopLossBps <= 500);
      results.push(child);
    }
  }
  assert.ok(new Set(results.map((child) => JSON.stringify(child))).size > 350);
  assert.notDeepEqual(inheritSourceGenome("local", SOURCE_PARENT_GENOMES, "garden", 0),
    inheritSourceGenome("hive", SOURCE_PARENT_GENOMES, "garden", 0));
});

test("malformed and stop-free genomes fail before emitting signals", async () => {
  for (const change of [{ stopLossBps: 0 }, { stopLossBps: Number.NaN }, { stopLossBps: 501 },
    { pivotLength: 2.5 }, { channelLookback: 19 }, { ewRetraceMin: .8, ewRetraceMax: .7 },
    { ewMinMoveAtr: Infinity }, { ewFilter: undefined }, { sourceId: {} }, { name: {} },
    { sourceId: null }, { name: 123 }, { sourceId: "x".repeat(501) }, { name: "x".repeat(501) }]) {
    assert.throws(() => validateSourceGenome(genome(change as Partial<SourceGenome>)), RangeError);
  }
  const data = await fixture(Array<number>(40).fill(100));
  assert.throws(() => generateSourceSignals(data, genome(), "invalid", "-1"), RangeError);
});

test("local inheritance blends both parents and can acquire EW donor traits", () => {
  const first = genome({ channelLookback: 50, ewFilter: false, ewExit: false });
  const second = genome({ channelLookback: 200, ewFilter: true, ewExit: true });
  const alteredSecond = { ...second, channelLookback: 400 };
  const inherited = inheritSourceGenome("local", [first, second], "local-pair", 0);
  const altered = inheritSourceGenome("local", [first, alteredSecond], "local-pair", 0);
  assert.ok(inherited.channelLookback >= 112 && inherited.channelLookback <= 138);
  assert.ok(altered.channelLookback >= 202 && altered.channelLookback <= 248);
  assert.notEqual(inherited.channelLookback, altered.channelLookback);
  const offspring = Array.from({ length: 40 }, (_, i) => inheritSourceGenome("local", [first, second], "local-pair", i));
  assert.ok(offspring.some((child) => child.ewFilter));
  assert.ok(offspring.some((child) => child.ewExit));
  assert.ok(offspring.some((child) => !child.ewFilter));
});

test("Hive uses the central source library independently of supplied local parents", () => {
  const localA = [genome({ channelLookback: 20, pivotLength: 2, stopLossBps: 50 })];
  const localB = [genome({ channelLookback: 500, pivotLength: 20, stopLossBps: 500 }), SOURCE_PARENT_GENOMES[2]!];
  for (let i = 0; i < 20; i += 1) {
    assert.deepEqual(inheritSourceGenome("hive", localA, "central", i), inheritSourceGenome("hive", localB, "central", i));
  }
  assert.notEqual(inheritSourceGenome("hybrid", localA, "central", 0).channelLookback,
    inheritSourceGenome("hybrid", localB, "central", 0).channelLookback);
});
