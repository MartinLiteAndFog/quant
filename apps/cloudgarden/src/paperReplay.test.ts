import assert from "node:assert/strict";
import test from "node:test";
import {
  createSyntheticMarketRange,
  HISTORICAL_PAPER_NOTICE,
  createSyntheticMarketWeek,
  normalizeMarketData,
  runPaperReplay,
  runSyntheticPaperDemo,
  syntheticDemoSignals,
  type MarketBarInput,
  type MarketProvenance,
  type TargetPositionSignal,
} from "./paperReplay.ts";
import { advanceSimulationDays, bookClosedPaperReplay, createSimulationWorld } from "./simulation.ts";

const provenance: MarketProvenance = {
  source: "test synthetic fixture",
  license: "CC0 synthetic fixture",
  symbol: "SOL-DEMO",
  marketType: "linear_futures",
  contractMultiplier: "1",
  intervalSeconds: 60,
  synthetic: true,
  timezone: "UTC",
  timestampSemantics: "open",
};

function bar(minute: number, open = "100", volume = "100"): MarketBarInput {
  const timestamp = new Date(Date.parse("2024-01-01T00:00:00Z") + minute * 60_000).toISOString();
  return { timestamp, open, high: "101", low: "99", close: "100.5", volume };
}

function pricedBar(minute: number, open: number): MarketBarInput {
  const timestamp = new Date(Date.parse("2024-01-01T00:00:00Z") + minute * 60_000).toISOString();
  return {
    timestamp,
    open: open.toFixed(2),
    high: (open + 1).toFixed(2),
    low: (open - 1).toFixed(2),
    close: open.toFixed(2),
    volume: "100",
  };
}

function signal(orderId: string, targetQuantity: string, minute: number): TargetPositionSignal {
  const availableAt = new Date(Date.parse("2024-01-01T00:00:00Z") + minute * 60_000).toISOString();
  return { orderId, strategyId: "plant-a", targetQuantity, signalAt: availableAt, availableAt, receivedAt: availableAt };
}

test("synthetic week is reproducible, complete, and has a canonical SHA-256 fingerprint", async () => {
  const first = await createSyntheticMarketWeek();
  const second = await createSyntheticMarketWeek();
  assert.equal(first.bars.length, 10_080);
  assert.equal(first.report.missingBars, 0);
  assert.equal(first.report.gapCount, 0);
  assert.equal(first.report.eligibility, "synthetic_demo_only");
  assert.match(first.fingerprint, /^[a-f0-9]{64}$/);
  assert.equal(first.fingerprint, second.fingerprint);
});

test("synthetic ranges repeat for the same seed and cycle, and differ across seeds or weeks", async () => {
  const first = await createSyntheticMarketRange(2, "2024-01-01T00:00:00Z", "garden-a", 0);
  const repeat = await createSyntheticMarketRange(2, "2024-01-01T00:00:00Z", "garden-a", 0);
  const nextWeek = await createSyntheticMarketRange(2, "2024-01-01T00:00:00Z", "garden-a", 1);
  const otherSeed = await createSyntheticMarketRange(2, "2024-01-01T00:00:00Z", "garden-b", 0);

  assert.equal(first.fingerprint, repeat.fingerprint);
  assert.deepEqual(first.bars, repeat.bars);
  assert.notEqual(first.fingerprint, nextWeek.fingerprint);
  assert.notEqual(first.fingerprint, otherSeed.fingerprint);
  assert.notDeepEqual(first.bars, nextWeek.bars);
  assert.notDeepEqual(first.bars, otherSeed.bars);
  assert.match(first.provenance.source, /seed-[a-f0-9]+ week-0 days-2/);
});

test("30-day synthetic ranges are bounded, complete, varied, and valid OHLCV", async () => {
  const dataset = await createSyntheticMarketRange(30, "2024-01-01T00:00:00Z", "garden-30d", 3);

  assert.equal(dataset.bars.length, 30 * 24 * 60);
  assert.equal(dataset.report.missingBars, 0);
  assert.equal(dataset.report.gapCount, 0);
  assert.equal(dataset.report.eligibility, "synthetic_demo_only");
  assert.equal(dataset.provenance.synthetic, true);
  assert.equal(dataset.report.start, "2024-01-01T00:00:00.000Z");
  assert.equal(dataset.report.endExclusive, "2024-01-31T00:00:00.000Z");
  assert.ok(new Set(dataset.bars.map((entry) => entry.close)).size > 100);
  assert.ok(new Set(dataset.bars.map((entry) => entry.volume)).size > 1_000);
  for (const entry of dataset.bars) {
    const open = Number(entry.open);
    const high = Number(entry.high);
    const low = Number(entry.low);
    const close = Number(entry.close);
    const volume = Number(entry.volume);
    assert.ok(open > 0 && low > 0 && close > 0 && high >= Math.max(open, low, close));
    assert.ok(low <= Math.min(open, close));
    assert.ok(volume >= 0);
  }
  await assert.rejects(() => createSyntheticMarketRange(91), { name: "RangeError" });
  await assert.rejects(() => createSyntheticMarketRange(0), { name: "RangeError" });
});

test("90-day synthetic ranges cover three exact months without data gaps", async () => {
  const dataset = await createSyntheticMarketRange(90, "2024-01-01T00:00:00Z", "garden-quarter", 12);

  assert.equal(dataset.bars.length, 90 * 24 * 60);
  assert.equal(dataset.report.start, "2024-01-01T00:00:00.000Z");
  assert.equal(dataset.report.endExclusive, "2024-03-31T00:00:00.000Z");
  assert.equal(dataset.report.missingBars, 0);
  assert.equal(dataset.report.gapCount, 0);
  assert.equal(dataset.provenance.synthetic, true);
  assert.ok(new Set(dataset.bars.map((entry) => entry.close)).size > 100);
});

test("normalization sorts deterministically and rejects duplicate, naive, off-grid, and malformed bars", async () => {
  const sorted = await normalizeMarketData([bar(1), bar(0)], provenance);
  const ordered = await normalizeMarketData([bar(0), bar(1)], provenance);
  assert.equal(sorted.fingerprint, ordered.fingerprint);
  assert.equal(sorted.report.missingBars, 0);
  await assert.rejects(() => normalizeMarketData([bar(0), bar(0)], provenance), { name: "RangeError" });
  await assert.rejects(() => normalizeMarketData([{ ...bar(0), timestamp: "2024-01-01T00:00:00" }], provenance), { name: "RangeError" });
  await assert.rejects(() => normalizeMarketData([{ ...bar(0), timestamp: "2024-01-01T00:00:30Z" }], provenance), { name: "RangeError" });
  await assert.rejects(() => normalizeMarketData([{ ...bar(0), high: "98" }], provenance), { name: "RangeError" });
});

test("target signals fill at the first open strictly after availability and receipt", async () => {
  const dataset = await createSyntheticMarketWeek();
  const report = runPaperReplay({
    dataset,
    signals: syntheticDemoSignals(),
    allocationLimit: "20",
    startingCash: "20",
  });
  assert.deepEqual(report.fills.map((fill) => fill.executedAt), [
    "2024-01-01T00:02:00.000Z",
    "2024-01-01T00:06:00.000Z",
  ]);
  assert.deepEqual(report.fills.map((fill) => fill.referencePrice), ["100.02", "100.06"]);
  assert.equal(report.endedFlat, true);
  assert.equal(report.finalQuantity, "0");
});

test("execution costs are embedded in fill price and fees are charged once", async () => {
  const report = await runSyntheticPaperDemo("20", "20");
  assert.equal(report.fills.length, 2);
  assert.ok(Number(report.fills[0]?.price) > Number(report.fills[0]?.referencePrice));
  assert.ok(Number(report.fills[1]?.price) < Number(report.fills[1]?.referencePrice));
  assert.equal(report.fills[0]?.executionCost, "0.0020004");
  const expectedNet = Number(report.realizedPricePnl) - Number(report.fees) + Number(report.funding);
  assert.ok(Math.abs(Number(report.netPnl) - expectedNet) < 1e-8);
  assert.ok(Math.abs(Number(report.endingCash) - (20 + Number(report.netPnl))) < 1e-8);
  assert.ok(Number(report.fees) > 0);
  assert.equal(report.claim.includes("not a strategy test"), true);
});

test("a missing or stale next open rejects instead of filling at a fabricated price", async () => {
  const dataset = await normalizeMarketData([bar(0), bar(1), bar(3), bar(4), bar(5)], provenance);
  const withinBound = runPaperReplay({
    dataset,
    signals: [signal("open", "0.1", 1), signal("close", "0", 4)],
    allocationLimit: "20",
    startingCash: "20",
  });
  assert.equal(withinBound.fills[0]?.executedAt, "2024-01-01T00:03:00.000Z");
  assert.equal(dataset.report.missingBars, 1);

  const beyondBound = runPaperReplay({
    dataset,
    signals: [signal("late-open", "0.1", 1)],
    allocationLimit: "20",
    startingCash: "20",
    costs: { feeBps: "6", spreadBps: "2", slippageBps: "1", maxDelaySeconds: 60 },
  });
  assert.deepEqual(beyondBound.rejectedSignals, [{ orderId: "late-open", reason: "no_timely_bar" }]);
  assert.equal(beyondBound.fills.length, 0);
});

test("zero-volume fills reject and over-budget targets are never silently clipped", async () => {
  const zeroVolume = await normalizeMarketData([bar(0), bar(1, "100", "0"), bar(2)], provenance);
  const noFill = runPaperReplay({ dataset: zeroVolume, signals: [signal("zero", "0.1", 0)], allocationLimit: "20", startingCash: "20" });
  assert.deepEqual(noFill.rejectedSignals, [{ orderId: "zero", reason: "zero_volume" }]);

  const dataset = await createSyntheticMarketWeek();
  const tooLarge = runPaperReplay({ dataset, signals: [signal("too-large", "1", 1)], allocationLimit: "20", startingCash: "20" });
  assert.deepEqual(tooLarge.rejectedSignals, [{ orderId: "too-large", reason: "allocation_limit" }]);
  assert.equal(tooLarge.fills.length, 0);
});

test("funding is signed once per unique settlement and uses the (open, close] boundary", async () => {
  const dataset = await createSyntheticMarketWeek();
  const report = runPaperReplay({
    dataset,
    signals: [signal("open", "0.1", 1), signal("close", "0", 3)],
    fundingSettlements: [
      { settlementId: "at-open", timestamp: "2024-01-01T00:02:00Z", markPrice: "100", rate: "0.001" },
      { settlementId: "inside", timestamp: "2024-01-01T00:03:00Z", markPrice: "100", rate: "0.001" },
      { settlementId: "at-close", timestamp: "2024-01-01T00:04:00Z", markPrice: "100", rate: "0.001" },
      { settlementId: "after-close", timestamp: "2024-01-01T00:05:00Z", markPrice: "100", rate: "0.001" },
    ],
    allocationLimit: "20",
    startingCash: "20",
  });
  assert.equal(report.fundingCashflows.find((entry) => entry.settlementId === "at-open")?.amount, "0");
  assert.equal(report.fundingCashflows.find((entry) => entry.settlementId === "inside")?.amount, "-0.01");
  assert.equal(report.fundingCashflows.find((entry) => entry.settlementId === "at-close")?.amount, "-0.01");
  assert.equal(report.fundingCashflows.find((entry) => entry.settlementId === "after-close")?.amount, "0");
  assert.equal(report.funding, "-0.02");
  assert.throws(() => runPaperReplay({
    dataset,
    signals: [signal("open", "0.1", 1), signal("close", "0", 3)],
    fundingSettlements: [
      { settlementId: "duplicate", timestamp: "2024-01-01T00:03:00Z", markPrice: "100", rate: "0.001" },
      { settlementId: "duplicate", timestamp: "2024-01-01T00:04:00Z", markPrice: "100", rate: "0.001" },
    ],
    allocationLimit: "20",
    startingCash: "20",
  }), /unique/);
});

test("adds, partial closes, reversals, short PnL, fees, and short funding reconcile", async () => {
  const dataset = await normalizeMarketData(Array.from({ length: 9 }, (_, minute) => pricedBar(minute, 100 + minute)), provenance);
  const report = runPaperReplay({
    dataset,
    signals: [
      signal("open-long", "0.1", 0),
      signal("add-long", "0.2", 1),
      signal("partial-close", "0.1", 2),
      signal("reverse-short", "-0.1", 3),
      signal("close-short", "0", 4),
    ],
    fundingSettlements: [
      { settlementId: "short-pays", timestamp: "2024-01-01T00:05:00Z", markPrice: "105", rate: "-0.001" },
    ],
    allocationLimit: "30",
    startingCash: "30",
    costs: { feeBps: "100", spreadBps: "0", slippageBps: "0", maxDelaySeconds: 120 },
  });

  assert.deepEqual(report.fills.map((fill) => fill.filledQuantity), ["0.1", "0.1", "0.1", "0.2", "0.1"]);
  assert.deepEqual(report.fills.map((fill) => fill.price), ["101", "102", "103", "104", "105"]);
  assert.equal(report.fills[3]?.fee, "0.208");
  assert.equal(report.realizedPricePnl, "0.3");
  assert.equal(report.fees, "0.619");
  assert.equal(report.funding, "-0.0105");
  assert.equal(report.netPnl, "-0.3295");
  assert.equal(report.endingCash, "29.6705");
  assert.equal(report.endedFlat, true);
  assert.equal(report.riskBreach, false);
});

test("funding settlements outside the covered dataset window are rejected", async () => {
  const dataset = await normalizeMarketData([bar(0), bar(1), bar(2)], provenance);
  for (const timestamp of ["2023-12-31T23:59:00Z", "2024-01-01T00:03:00Z", "2024-01-01T00:04:00Z"]) {
    assert.throws(() => runPaperReplay({
      dataset,
      signals: [],
      fundingSettlements: [{ settlementId: "outside-window", timestamp, markPrice: "100", rate: "0.001" }],
      allocationLimit: "20",
      startingCash: "20",
    }), /outside the dataset window/);
  }
});

test("replay fails closed on open positions and empirical provenance", async () => {
  const dataset = await createSyntheticMarketWeek();
  assert.throws(() => runPaperReplay({ dataset, signals: [signal("open", "0.1", 1)], allocationLimit: "20", startingCash: "20" }), /close every position/);
  const unverified = await normalizeMarketData([bar(0), bar(1)], { ...provenance, synthetic: false });
  assert.equal(unverified.report.eligibility, "provenance_unverified");
  assert.throws(() => runPaperReplay({ dataset: unverified, signals: [], allocationLimit: "20", startingCash: "20" }), /provenance and data-quality gate/);
});

test("explicit historical research executes archive prices and books exact nonzero PnL in the same ledger", async () => {
  const dataset = await normalizeMarketData(Array.from({ length: 5 }, (_, minute) => pricedBar(minute, 100 + minute)), {
    ...provenance, synthetic: false, source: "test archive; modeled linear 1x, venue unverified", license: "research use unverified",
  });
  const report = runPaperReplay({
    dataset, historicalResearch: true,
    signals: [signal("historical-open", "0.1", 0), signal("historical-close", "0", 3)],
    allocationLimit: "20", startingCash: "20",
    costs: { feeBps: "6", spreadBps: "0", slippageBps: "0", maxDelaySeconds: 120 },
  });
  assert.equal(report.dataset.synthetic, false);
  assert.equal(report.dataset.eligibility, "provenance_unverified");
  assert.equal(report.claim, HISTORICAL_PAPER_NOTICE);
  assert.equal(report.benchmarkConclusion, "not_comparable");
  assert.deepEqual(report.fills.map((fill) => fill.price), ["101", "104"]);
  assert.equal(report.realizedPricePnl, "0.3");
  assert.equal(report.fees, "0.0123");
  assert.equal(report.netPnl, "0.2877");
  assert.equal(report.endingCash, "20.2877");
  const world = createSimulationWorld("historical-ledger");
  const booked = bookClosedPaperReplay(advanceSimulationDays(world, report.days), "parent-01", {
    startDay: 0, days: report.days, datasetFingerprint: report.dataset.fingerprint,
    synthetic: false, historicalResearch: true, marketSource: dataset.provenance.source,
    endedFlat: report.endedFlat, riskBreach: report.riskBreach, fillCount: report.fills.length,
    allocationLimitUsdt: Number(report.allocationLimit), startingCashUsdt: Number(report.startingCash),
    endingCashUsdt: Number(report.endingCash), maxMarginUsedUsdt: Number(report.maxMarginUsed),
    realizedPricePnlUsdt: Number(report.realizedPricePnl), feesUsdt: Number(report.fees),
    fundingUsdt: Number(report.funding), netPnlUsdt: Number(report.netPnl),
  });
  assert.equal(booked.capital.navUsdt, 100.2877);
  assert.equal(booked.capital.protectedReserveUsdt, 10.0288);
  assert.equal(booked.capital.allocationCeilingUsdt, 90.2589);
  assert.equal(booked.events.at(-1)?.detail.synthetic, false);
  assert.equal(booked.events.at(-1)?.detail.marketSource, dataset.provenance.source);
  assert.equal(booked.events.at(-1)?.detail.historicalResearch, true);
});

test("historical opt-in still rejects inconsistent provenance, malformed bars, missing metadata and oversized positions", async () => {
  const dataset = await normalizeMarketData([pricedBar(0, 100), pricedBar(1, 101), pricedBar(2, 102)], {
    ...provenance, synthetic: false, source: "historical research archive",
  });
  const input = { dataset, historicalResearch: true, signals: [], allocationLimit: "20", startingCash: "20" };
  for (const corrupted of [
    { ...dataset, provenance: { ...dataset.provenance, source: " " } },
    { ...dataset, provenance: { ...dataset.provenance, license: " " } },
    { ...dataset, report: { ...dataset.report, synthetic: true } },
    { ...dataset, report: { ...dataset.report, fingerprint: "f".repeat(64) } },
    { ...dataset, bars: dataset.bars.map((entry, index) => index === 1 ? { ...entry, high: "1" } : entry) },
  ]) assert.throws(() => runPaperReplay({ ...input, dataset: corrupted }), RangeError);
  const oversized = runPaperReplay({ ...input, signals: [signal("oversized-historical", "1", 0)] });
  assert.deepEqual(oversized.rejectedSignals, [{ orderId: "oversized-historical", reason: "allocation_limit" }]);
  assert.equal(oversized.fills.length, 0);
});

test("the replay exposes daily marks across the seven-day window and is not comparable", async () => {
  const report = await runSyntheticPaperDemo("20", "20");
  assert.equal(report.days, 7);
  assert.equal(report.dailyMarks.length, 7);
  assert.deepEqual(report.dailyMarks.map((mark) => mark.dateUtc), [
    "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07",
  ]);
  assert.ok(Number(report.maxMarginUsed) > 0);
  assert.equal(report.benchmark, "cash_no_trade");
  assert.equal(report.benchmarkConclusion, "not_comparable");
});

test("the actual synthetic replay advances ecology and books once into the single garden ledger", async () => {
  const world = createSimulationWorld("integrated-paper-fixture");
  const report = await runSyntheticPaperDemo("20", "20", "parent-01");
  const advanced = advanceSimulationDays(world, report.days);
  const booked = bookClosedPaperReplay(advanced, "parent-01", {
    startDay: world.day,
    days: report.days,
    datasetFingerprint: report.dataset.fingerprint,
    synthetic: report.dataset.synthetic,
    endedFlat: report.endedFlat,
    riskBreach: report.riskBreach,
    fillCount: report.fills.length,
    allocationLimitUsdt: Number(report.allocationLimit),
    startingCashUsdt: Number(report.startingCash),
    endingCashUsdt: Number(report.endingCash),
    maxMarginUsedUsdt: Number(report.maxMarginUsed),
    realizedPricePnlUsdt: Number(report.realizedPricePnl),
    feesUsdt: Number(report.fees),
    fundingUsdt: Number(report.funding),
    netPnlUsdt: Number(report.netPnl),
  });
  assert.equal(booked.day, 7);
  assert.equal(booked.capital.paperReplayCount, 1);
  assert.ok(Math.abs(booked.capital.navUsdt - (100 + Number(report.netPnl))) < 0.0001);
  assert.equal(booked.capital.protectedReserveUsdt, Math.round(booked.capital.navUsdt * 0.1 * 10_000) / 10_000);
  assert.equal(booked.capital.allocationCeilingUsdt, Math.round(booked.capital.navUsdt * 0.9 * 10_000) / 10_000);
  assert.equal(booked.capital.paperFeesUsdt, Math.round(Number(report.fees) * 10_000) / 10_000);
});

test("a 30-day synthetic replay books the exact horizon and keeps the shared ledger reconciled", async () => {
  const world = createSimulationWorld("integrated-month-fixture");
  const start = "2024-01-01T00:00:00Z";
  const dataset = await createSyntheticMarketRange(30, start, world.seed, 0);
  const report = runPaperReplay({
    dataset,
    signals: syntheticDemoSignals(start, "parent-01"),
    allocationLimit: "20",
    startingCash: "20",
  });
  assert.equal(report.days, 30);
  assert.equal(report.dailyMarks.length, 30);
  assert.equal(report.endedFlat, true);
  assert.equal(report.benchmarkConclusion, "not_comparable");

  const booked = bookClosedPaperReplay(advanceSimulationDays(world, report.days), "parent-01", {
    startDay: world.day,
    days: report.days,
    datasetFingerprint: report.dataset.fingerprint,
    synthetic: report.dataset.synthetic,
    endedFlat: report.endedFlat,
    riskBreach: report.riskBreach,
    fillCount: report.fills.length,
    allocationLimitUsdt: Number(report.allocationLimit),
    startingCashUsdt: Number(report.startingCash),
    endingCashUsdt: Number(report.endingCash),
    maxMarginUsedUsdt: Number(report.maxMarginUsed),
    realizedPricePnlUsdt: Number(report.realizedPricePnl),
    feesUsdt: Number(report.fees),
    fundingUsdt: Number(report.funding),
    netPnlUsdt: Number(report.netPnl),
  });

  assert.equal(booked.day, 30);
  assert.equal(booked.capital.paperReplayCount, 1);
  assert.ok(Math.abs(booked.capital.navUsdt - (100 + Number(report.netPnl))) < 0.0001);
  assert.equal(booked.capital.paperFeesUsdt, Math.round(Number(report.fees) * 10_000) / 10_000);
});
