import assert from "node:assert/strict";
import test from "node:test";
import { createSyntheticMarketWeek, runPaperReplay } from "./paperReplay.ts";
import { buildSyntheticReplayEvidence } from "./replayEvidence.ts";
import { generateSyntheticMomentumSignals } from "./strategySignals.ts";

async function noTradeRun() {
  const dataset = await createSyntheticMarketWeek();
  const strategy = generateSyntheticMomentumSignals(dataset, {
    schemaVersion: 1,
    strategyId: "synthetic-evidence-test",
    lookbackBars: 5,
    entryThresholdBps: 3_000,
    exitThresholdBps: 10,
    targetQuantity: "0.1",
  });
  const execution = runPaperReplay({ dataset, signals: strategy.signals, allocationLimit: "20", startingCash: "20" });
  return { strategy: strategy.report, execution };
}

async function tradeRun() {
  const dataset = await createSyntheticMarketWeek();
  const strategy = generateSyntheticMomentumSignals(dataset, {
    schemaVersion: 1,
    strategyId: "synthetic-evidence-trade-test",
    lookbackBars: 5,
    entryThresholdBps: 40,
    exitThresholdBps: 10,
    targetQuantity: "0.1",
  });
  const execution = runPaperReplay({ dataset, signals: strategy.signals, allocationLimit: "20", startingCash: "20" });
  assert.equal(strategy.report.status, "trade");
  assert.ok(execution.fills.length > 0);
  return { strategy: strategy.report, execution };
}

test("evidence reconciles costs and labels the short synthetic sample unavailable for inference", async () => {
  const { strategy, execution } = await noTradeRun();
  const evidence = buildSyntheticReplayEvidence(strategy, execution);

  assert.equal(evidence.claim, "synthetic_descriptive_only");
  assert.equal(evidence.benchmarkConclusion, "not_comparable");
  assert.equal(evidence.statisticalStatus, "insufficient_sample");
  assert.equal(evidence.independentOpportunityCount, null);
  assert.equal(evidence.strategyStatus, "no_trade");
  assert.equal(evidence.executableOpportunityObservations, 0);
  assert.equal(evidence.generatedTargets, 0);
  assert.equal(evidence.fills, 0);
  assert.equal(evidence.netPnlUsdt, 0);
  assert.equal(evidence.netPnlResidualUsdt, 0);
  assert.equal(evidence.cashResidualUsdt, 0);
  assert.equal(evidence.maxDailyDrawdownUsdt, 0);
});

test("daily drawdown starts from initial cash and tracks the largest marked decline", async () => {
  const { strategy, execution } = await tradeRun();
  const startingCash = Number(execution.startingCash);
  const endingCash = Number(execution.endingCash);
  const first = execution.dailyMarks[0];
  const second = execution.dailyMarks[1];
  const last = execution.dailyMarks.at(-1);
  assert.ok(first && second && last);
  const evidence = buildSyntheticReplayEvidence(strategy, {
    ...execution,
    dailyMarks: [
      { ...first, equity: startingCash.toString() },
      { ...second, equity: (startingCash - 3).toString(), openQuantity: "0.1", fillCount: Math.max(1, second.fillCount) },
      { ...last, equity: endingCash.toString(), openQuantity: "0" },
    ],
  });

  assert.equal(evidence.dailyMarkCount, 3);
  assert.equal(evidence.maxDailyDrawdownUsdt, 3);
  assert.equal(evidence.maxDailyDrawdownPct, 3 / startingCash);
});

test("rejects mismatched datasets, unreconciled PnL, and unordered marks", async () => {
  const { strategy, execution } = await noTradeRun();
  assert.throws(() => buildSyntheticReplayEvidence({ ...strategy, datasetFingerprint: "0".repeat(64) }, execution), /fingerprint-matched/);
  assert.throws(() => buildSyntheticReplayEvidence({ ...strategy, opportunities: 1 }, execution), /status does not reconcile/);
  assert.throws(() => buildSyntheticReplayEvidence({ ...strategy, generatedSignals: 1 }, execution), /status does not reconcile/);
  assert.throws(() => buildSyntheticReplayEvidence(strategy, {
    ...execution,
    rejectedSignals: [{ orderId: "unexpected", reason: "zero_volume" }],
  }), /no fills, costs, PnL, or margin use/);
  const noTradeMark = execution.dailyMarks[0];
  assert.ok(noTradeMark);
  assert.throws(() => buildSyntheticReplayEvidence(strategy, {
    ...execution,
    dailyMarks: [{ ...noTradeMark, equity: "21" }],
  }), /No-trade daily marks/);
  assert.throws(() => buildSyntheticReplayEvidence(strategy, { ...execution, netPnl: "0.01" }), /do not reconcile/);
  assert.throws(() => buildSyntheticReplayEvidence(strategy, { ...execution, finalQuantity: "0.1" }), /flat execution/);
  assert.throws(() => buildSyntheticReplayEvidence(strategy, { ...execution, riskBreach: true }), /without a risk breach/);
  assert.throws(() => buildSyntheticReplayEvidence(strategy, {
    ...execution,
    dailyMarks: [
      { dateUtc: "2024-01-02", equity: "20", openQuantity: "0", fillCount: 0 },
      { dateUtc: "2024-01-01", equity: "20", openQuantity: "0", fillCount: 0 },
    ],
  }), /strictly increasing/);
});
