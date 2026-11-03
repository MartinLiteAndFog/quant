import assert from "node:assert/strict";
import test from "node:test";
import { handleSyntheticReplayWorkerMessage, runSyntheticReplayCycle, type SyntheticReplayCycleInput } from "./replayCycle.ts";
import { advanceSimulationDays, createSimulationWorld } from "./simulation.ts";

const cycleInput = (overrides: Partial<SyntheticReplayCycleInput> = {}): SyntheticReplayCycleInput => ({
  days: 90,
  start: "2024-01-01T00:00:00Z",
  gardenSeed: "cloudgarden-cycle-worker-test",
  weekIndex: 2,
  plantId: "plant-90d",
  allocationUsdt: 20,
  tradingTraits: { lookbackBars: 30, entryThresholdBps: 40, exitThresholdBps: 10 },
  ...overrides,
});

test("a 90-day replay cycle returns flat, reconciled, synthetic-only evidence", async () => {
  const result = await runSyntheticReplayCycle(cycleInput());

  assert.equal(result.execution.days, 90);
  assert.equal(result.execution.dailyMarks.length, 90);
  assert.equal(result.execution.endedFlat, true);
  assert.equal(result.execution.finalQuantity, "0");
  assert.ok(Math.abs(Number(result.execution.startingCash) + Number(result.execution.netPnl) - Number(result.execution.endingCash)) < 0.0001);
  assert.equal(result.genome.strategyId, "plant-90d:synthetic-momentum-v1");
  assert.equal(result.genome.targetQuantity, "0.1");
  assert.equal(result.evidence.statisticalStatus, "insufficient_sample");
  assert.equal(result.evidence.benchmarkConclusion, "not_comparable");
  assert.equal(result.evidence.independentOpportunityCount, null);
  assert.ok(Math.abs(result.evidence.cashResidualUsdt) < 0.0001);
  assert.ok(Math.abs(result.evidence.netPnlResidualUsdt) < 0.0001);
});

test("the replay genome is taken from the selected plant's inherited trading traits", async () => {
  const world = advanceSimulationDays(createSimulationWorld("replay-inherited", { conceptionPolicy: "hive" }), 7);
  const plant = world.plants.find((candidate) => candidate.id === "seedling-01");
  assert.ok(plant);
  const result = await runSyntheticReplayCycle(cycleInput({
    days: 7,
    plantId: plant.id,
    allocationUsdt: plant.allocationUsdt,
    tradingTraits: plant.tradingTraits,
  }));

  assert.equal(result.genome.strategyId, `${plant.id}:synthetic-momentum-v1`);
  assert.equal(result.genome.lookbackBars, plant.tradingTraits.lookbackBars);
  assert.equal(result.genome.entryThresholdBps, plant.tradingTraits.entryThresholdBps);
  assert.equal(result.genome.exitThresholdBps, plant.tradingTraits.exitThresholdBps);
});

test("worker success response preserves the request id and serializes", async () => {
  const response = await handleSyntheticReplayWorkerMessage({
    id: "cycle-ok",
    input: cycleInput({ days: 1, tradingTraits: { lookbackBars: 30, entryThresholdBps: 40, exitThresholdBps: 10 } }),
  });

  assert.equal(response.id, "cycle-ok");
  assert.ok("result" in response);
  assert.doesNotThrow(() => JSON.stringify(response));
});

test("worker errors use a serializable error envelope", async () => {
  const response = await handleSyntheticReplayWorkerMessage({ id: "cycle-bad", input: cycleInput({ days: 91 }) });

  assert.deepEqual(response, {
    id: "cycle-bad",
    error: "Synthetic range must contain between 1 and 90 whole days.",
  });
  assert.doesNotThrow(() => JSON.stringify(response));
});
