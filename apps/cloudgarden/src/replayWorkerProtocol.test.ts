import assert from "node:assert/strict";
import test from "node:test";
import { runSyntheticReplayCycle } from "./replayCycle.ts";
import { readSyntheticReplayWorkerResult } from "./replayWorkerProtocol.ts";

test("worker protocol accepts only the active, matching, reconciled run", async () => {
  const result = await runSyntheticReplayCycle({
    days: 1,
    start: "2024-01-01T00:00:00Z",
    gardenSeed: "worker-protocol-test",
    weekIndex: 0,
    plantId: "plant-protocol",
    allocationUsdt: 20,
    tradingTraits: { lookbackBars: 5, entryThresholdBps: 40, exitThresholdBps: 10 },
  });
  const response = { id: "active-run", result };

  const expectedStart = "2024-01-01T00:00:00.000Z";
  assert.equal(readSyntheticReplayWorkerResult(response, "active-run", 1, expectedStart), result);
  assert.throws(() => readSyntheticReplayWorkerResult(response, "stale-run", 1, expectedStart), /ID does not match/);
  assert.throws(() => readSyntheticReplayWorkerResult(response, "active-run", 7, expectedStart), /does not reconcile/);
  assert.throws(() => readSyntheticReplayWorkerResult({ id: "active-run", error: "fixture failed" }, "active-run", 1, expectedStart), /fixture failed/);
  assert.throws(() => readSyntheticReplayWorkerResult({ id: "active-run", error: "failed", result }, "active-run", 1, expectedStart), /exactly one/);

  const mismatchedEvidence = { ...result, evidence: { ...result.evidence, cashResidualUsdt: 0.1 } };
  assert.throws(() => readSyntheticReplayWorkerResult({ id: "active-run", result: mismatchedEvidence }, "active-run", 1, expectedStart), /does not reconcile/);

  const fabricatedPnl = Number(result.execution.realizedPricePnl) + 0.01;
  const fabricatedNet = Number(result.execution.netPnl) + 0.01;
  const fabricatedCash = Number(result.execution.endingCash) + 0.01;
  const coherentButUnbackedRollup = {
    ...result,
    execution: {
      ...result.execution,
      realizedPricePnl: fabricatedPnl.toFixed(8),
      netPnl: fabricatedNet.toFixed(8),
      endingCash: fabricatedCash.toFixed(8),
      dailyMarks: result.execution.dailyMarks.map((mark, index) => index === result.execution.dailyMarks.length - 1
        ? { ...mark, equity: fabricatedCash.toFixed(8) }
        : mark),
    },
    evidence: {
      ...result.evidence,
      realizedPricePnlUsdt: fabricatedPnl,
      netPnlUsdt: fabricatedNet,
    },
  };
  assert.throws(() => readSyntheticReplayWorkerResult({ id: "active-run", result: coherentButUnbackedRollup }, "active-run", 1, expectedStart), /does not reconcile/);

  const firstFill = result.execution.fills[0];
  if (firstFill) {
    const unchargedFee = Number(firstFill.fee);
    const feeTampered = {
      ...result,
      execution: {
        ...result.execution,
        fills: result.execution.fills.map((fill, index) => index === 0 ? { ...fill, fee: "0" } : fill),
        fees: (Number(result.execution.fees) - unchargedFee).toFixed(8),
        netPnl: (Number(result.execution.netPnl) + unchargedFee).toFixed(8),
        endingCash: (Number(result.execution.endingCash) + unchargedFee).toFixed(8),
        dailyMarks: result.execution.dailyMarks.map((mark, index) => index === result.execution.dailyMarks.length - 1
          ? { ...mark, equity: (Number(mark.equity) + unchargedFee).toFixed(8) }
          : mark),
      },
    };
    assert.throws(() => readSyntheticReplayWorkerResult({ id: "active-run", result: feeTampered }, "active-run", 1, expectedStart), /does not reconcile/);
  }

  const inventedFunding = {
    ...result,
    execution: {
      ...result.execution,
      funding: "0.1",
      netPnl: (Number(result.execution.netPnl) + 0.1).toFixed(8),
      endingCash: (Number(result.execution.endingCash) + 0.1).toFixed(8),
      fundingCashflows: [{ settlementId: "invented", timestamp: "2030-01-01T00:00:00Z", amount: "0.1" }],
      dailyMarks: result.execution.dailyMarks.map((mark, index) => index === result.execution.dailyMarks.length - 1
        ? { ...mark, equity: (Number(mark.equity) + 0.1).toFixed(8) }
        : mark),
    },
    evidence: { ...result.evidence, fundingUsdt: 0.1, netPnlUsdt: Number(result.evidence.netPnlUsdt) + 0.1 },
  };
  assert.throws(() => readSyntheticReplayWorkerResult({ id: "active-run", result: inventedFunding }, "active-run", 1, expectedStart), /does not reconcile/);

  if (result.execution.fills.length > 1) {
    const [first, second, ...rest] = result.execution.fills;
    const reorderedFills = {
      ...result,
      execution: {
        ...result.execution,
        fills: [
          { ...first!, executedAt: second!.executedAt },
          { ...second!, executedAt: first!.executedAt },
          ...rest,
        ],
      },
    };
    assert.throws(() => readSyntheticReplayWorkerResult({ id: "active-run", result: reorderedFills }, "active-run", 1, expectedStart), /does not reconcile/);
  }

  const mismatchedWindow = { ...result, execution: { ...result.execution, dataset: { ...result.execution.dataset, start: "2024-01-02T00:00:00.000Z" } } };
  assert.throws(() => readSyntheticReplayWorkerResult({ id: "active-run", result: mismatchedWindow }, "active-run", 1, expectedStart), /does not reconcile/);

  const mismatchedFill = {
    ...result,
    execution: {
      ...result.execution,
      fills: result.execution.fills.map((fill, index) => index === 0
        ? { ...fill, datasetFingerprint: "0".repeat(64) }
        : fill),
    },
  };
  if (result.execution.fills.length > 0) {
    assert.throws(() => readSyntheticReplayWorkerResult({ id: "active-run", result: mismatchedFill }, "active-run", 1, expectedStart), /does not reconcile/);
  }

  const invalidDate = {
    ...result,
    execution: {
      ...result.execution,
      dailyMarks: result.execution.dailyMarks.map((mark, index) => index === 0 ? { ...mark, dateUtc: "2024-02-31" } : mark),
    },
  };
  assert.throws(() => readSyntheticReplayWorkerResult({ id: "active-run", result: invalidDate }, "active-run", 1, expectedStart), /does not reconcile/);
});

test("worker protocol fails closed on null and incomplete messages", () => {
  const expectedStart = "2024-01-01T00:00:00.000Z";
  assert.throws(() => readSyntheticReplayWorkerResult(null, "active-run", 1, expectedStart), /ID does not match/);
  assert.throws(() => readSyntheticReplayWorkerResult({ id: "active-run", result: null }, "active-run", 1, expectedStart), /missing required reports/);
  assert.throws(() => readSyntheticReplayWorkerResult({ id: "active-run" }, "active-run", 1, expectedStart), /exactly one/);
});
