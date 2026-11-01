import assert from "node:assert/strict";
import test from "node:test";
import { createSyntheticMarketWeek, normalizeMarketData, runPaperReplay, type MarketBarInput, type MarketProvenance } from "./paperReplay.ts";
import { generateSyntheticMomentumSignals, type TradingSignalGenome } from "./strategySignals.ts";
import { advanceSimulationDays, bookClosedPaperReplay, createSimulationWorld } from "./simulation.ts";

const provenance: MarketProvenance = {
  source: "Cloudgarden test fixture",
  license: "CC0 synthetic fixture",
  symbol: "TEST-DEMO",
  marketType: "linear_futures",
  contractMultiplier: "1",
  intervalSeconds: 60,
  synthetic: true,
  timezone: "UTC",
  timestampSemantics: "open",
};

const genome = (overrides: Partial<TradingSignalGenome> = {}): TradingSignalGenome => ({
  schemaVersion: 1,
  strategyId: "synthetic-momentum-v1",
  lookbackBars: 1,
  entryThresholdBps: 100,
  exitThresholdBps: 20,
  targetQuantity: "0.1",
  ...overrides,
});

async function fixture(closes: readonly number[]) {
  const bars: MarketBarInput[] = closes.map((close, index) => {
    const value = close.toFixed(2);
    return {
      timestamp: new Date(Date.parse("2024-01-01T00:00:00Z") + index * 60_000).toISOString(),
      open: value,
      high: (close + 1).toFixed(2),
      low: (close - 1).toFixed(2),
      close: value,
      volume: "1000",
    };
  });
  return normalizeMarketData(bars, provenance);
}

test("synthetic causal momentum signal fixture is deterministic and binds its report to the dataset fingerprint", async () => {
    const dataset = await fixture([100, 102, 104, 103, 101, 99, 99, 99]);
    const first = generateSyntheticMomentumSignals(dataset, genome());
    const second = generateSyntheticMomentumSignals(dataset, genome());
    assert.deepEqual(second, first);
    assert.equal(first.report.datasetFingerprint, dataset.fingerprint);
    assert.equal(first.report.benchmark, "cash_no_trade");
    assert.equal(first.report.benchmarkConclusion, "not_comparable");
  });

test("timestamps each decision at the observed close and leaves a later open for execution", async () => {
    const dataset = await fixture([100, 102, 104, 103, 101, 99, 99, 99]);
    const result = generateSyntheticMomentumSignals(dataset, genome());
    assert.ok(result.signals.length > 0);
    for (const signal of result.signals) {
      assert.equal(signal.signalAt, signal.availableAt);
      assert.equal(signal.signalAt, signal.receivedAt);
      const decisionMs = Date.parse(signal.signalAt);
      assert.equal(dataset.bars.some((bar) => bar.timestampMs + 60_000 === decisionMs), true);
      assert.equal(dataset.bars.some((bar) => bar.timestampMs > decisionMs), true);
    }
    assert.equal(result.signals.at(-1)?.targetQuantity, "0");
    assert.equal(result.report.endedFlat, true);
  });

test("emits long, flat, and short target transitions from completed close-to-close returns", async () => {
    const dataset = await fixture([100, 102, 104, 103.9, 101.5, 99, 97, 97, 97]);
    const result = generateSyntheticMomentumSignals(dataset, genome());
    assert.ok(result.signals.some((signal) => signal.targetQuantity === "0.1"));
    assert.ok(result.signals.some((signal) => signal.targetQuantity === "-0.1"));
    assert.equal(result.signals.at(-1)?.targetQuantity, "0");
    assert.equal(result.report.status, "trade");
    assert.ok(result.report.observations > 0);
    assert.ok(result.report.opportunities > 0);
  });

test("changes signal activity when the entry threshold changes", async () => {
    const dataset = await fixture([100, 100.5, 101, 100.7, 100.2, 100.2, 100.2, 100.2]);
    const active = generateSyntheticMomentumSignals(dataset, genome({ entryThresholdBps: 40, exitThresholdBps: 10 }));
    const inactive = generateSyntheticMomentumSignals(dataset, genome({ entryThresholdBps: 500, exitThresholdBps: 10 }));
    assert.equal(active.report.status, "trade");
    assert.ok(active.report.opportunities > inactive.report.opportunities);
    assert.equal(inactive.report.status, "no_trade");
  });

test("labels a zero-opportunity week as no_trade without inventing a loss", async () => {
    const dataset = await fixture([100, 100, 100, 100, 100, 100, 100, 100]);
    const result = generateSyntheticMomentumSignals(dataset, genome());
    assert.ok(result.report.observations > 0);
    assert.equal(result.report.opportunities, 0);
    assert.equal(result.report.status, "no_trade");
    assert.deepEqual(result.signals, []);
    assert.equal("pnl" in result.report, false);
    assert.equal(result.report.benchmark, "cash_no_trade");
  });

test("does not count the forced-flat observation as an executable opportunity", async () => {
  const dataset = await fixture([100, 100, 100, 102, 102, 102]);
  const result = generateSyntheticMomentumSignals(dataset, genome());

  assert.equal(result.report.observations, 3);
  assert.equal(result.report.opportunities, 0);
  assert.equal(result.report.status, "no_trade");
  assert.deepEqual(result.signals, []);
});

test("fails closed for invalid genomes and mismatched or empirical provenance", async () => {
    const dataset = await fixture([100, 102, 104, 103, 101, 99, 99, 99]);
    assert.throws(() => generateSyntheticMomentumSignals(dataset, genome({ schemaVersion: 2 as 1 })));
    assert.throws(() => generateSyntheticMomentumSignals(dataset, genome({ exitThresholdBps: 101 })));
    assert.throws(() => generateSyntheticMomentumSignals(dataset, genome({ targetQuantity: "0" })));
    assert.throws(() => generateSyntheticMomentumSignals({
      ...dataset,
      report: { ...dataset.report, fingerprint: "0".repeat(64) },
    }, genome()), /fingerprint-matched/);
    assert.throws(() => generateSyntheticMomentumSignals({
      ...dataset,
      provenance: { ...dataset.provenance, synthetic: false },
    }, genome()), /synthetic-only/);
    assert.throws(() => generateSyntheticMomentumSignals(dataset, genome({ lookbackBars: dataset.bars.length })), /enough bars/);
});

test("generated momentum targets pass through the actual paper replay and finish flat", async () => {
  const dataset = await createSyntheticMarketWeek();
  const strategy = generateSyntheticMomentumSignals(dataset, genome({
    lookbackBars: 5,
    entryThresholdBps: 40,
    exitThresholdBps: 10,
  }));
  const execution = runPaperReplay({
    dataset,
    signals: strategy.signals,
    allocationLimit: "20",
    startingCash: "20",
  });

  assert.equal(strategy.report.status, "trade");
  assert.ok(strategy.report.opportunities > 0);
  assert.ok(execution.fills.length > 0);
  assert.equal(execution.endedFlat, true);
  assert.equal(execution.finalQuantity, "0");
  assert.equal(execution.riskBreach, false);
  assert.ok(Math.abs(Number(execution.endingCash) - (20 + Number(execution.netPnl))) < 1e-8);
});

test("a no-opportunity strategy run advances and records the week without changing NAV", async () => {
  const world = createSimulationWorld("signal-no-trade-integration");
  const dataset = await createSyntheticMarketWeek();
  const tradingGenome = genome({ lookbackBars: 5, entryThresholdBps: 3000, exitThresholdBps: 10 });
  const strategy = generateSyntheticMomentumSignals(dataset, tradingGenome);
  const execution = runPaperReplay({ dataset, signals: strategy.signals, allocationLimit: "20", startingCash: "20" });
  assert.equal(strategy.report.status, "no_trade");
  assert.equal(execution.fills.length, 0);

  const advanced = advanceSimulationDays(world, execution.days);
  const booked = bookClosedPaperReplay(advanced, "parent-01", {
    startDay: world.day,
    days: execution.days,
    datasetFingerprint: dataset.fingerprint,
    synthetic: dataset.report.synthetic,
    endedFlat: execution.endedFlat,
    riskBreach: execution.riskBreach,
    fillCount: execution.fills.length,
    signalStatus: strategy.report.status,
    strategyId: tradingGenome.strategyId,
    signalGenome: {
      schemaVersion: tradingGenome.schemaVersion,
      lookbackBars: tradingGenome.lookbackBars,
      entryThresholdBps: tradingGenome.entryThresholdBps,
      exitThresholdBps: tradingGenome.exitThresholdBps,
      targetQuantity: tradingGenome.targetQuantity,
    },
    observations: strategy.report.observations,
    opportunities: strategy.report.opportunities,
    generatedSignals: strategy.report.generatedSignals,
    allocationLimitUsdt: Number(execution.allocationLimit),
    startingCashUsdt: Number(execution.startingCash),
    endingCashUsdt: Number(execution.endingCash),
    maxMarginUsedUsdt: Number(execution.maxMarginUsed),
    realizedPricePnlUsdt: Number(execution.realizedPricePnl),
    feesUsdt: Number(execution.fees),
    fundingUsdt: Number(execution.funding),
    netPnlUsdt: Number(execution.netPnl),
  });

  assert.equal(booked.day, 7);
  assert.equal(booked.capital.paperReplayCount, 1);
  assert.equal(booked.capital.navUsdt, 100);
  assert.equal(booked.events.find((event) => event.type === "paper_replay")?.detail.signalStatus, "no_trade");
});
