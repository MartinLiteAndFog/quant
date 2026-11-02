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
    controls: { lookbackBars: 5, entryThresholdBps: 40, exitThresholdBps: 10 },
  });
  const response = { id: "active-run", result };

  assert.equal(readSyntheticReplayWorkerResult(response, "active-run", 1), result);
  assert.throws(() => readSyntheticReplayWorkerResult(response, "stale-run", 1), /ID does not match/);
  assert.throws(() => readSyntheticReplayWorkerResult(response, "active-run", 7), /does not reconcile/);
  assert.throws(() => readSyntheticReplayWorkerResult({ id: "active-run", error: "fixture failed" }, "active-run", 1), /fixture failed/);
  assert.throws(() => readSyntheticReplayWorkerResult({ id: "active-run", error: "failed", result }, "active-run", 1), /exactly one/);

  const mismatchedEvidence = { ...result, evidence: { ...result.evidence, cashResidualUsdt: 0.1 } };
  assert.throws(() => readSyntheticReplayWorkerResult({ id: "active-run", result: mismatchedEvidence }, "active-run", 1), /does not reconcile/);
});

test("worker protocol fails closed on null and incomplete messages", () => {
  assert.throws(() => readSyntheticReplayWorkerResult(null, "active-run", 1), /ID does not match/);
  assert.throws(() => readSyntheticReplayWorkerResult({ id: "active-run", result: null }, "active-run", 1), /missing required reports/);
  assert.throws(() => readSyntheticReplayWorkerResult({ id: "active-run" }, "active-run", 1), /exactly one/);
});
