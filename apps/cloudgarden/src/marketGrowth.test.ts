import assert from "node:assert/strict";
import test from "node:test";
import {
  advanceSimulationDay,
  assertSimulationInvariants,
  createSimulationWorld,
  type TradingDayOutcomes,
} from "./simulation.ts";

const round4 = (value: number) => Math.round(value * 10_000) / 10_000;
const bounded = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

test("each plant compounds its own booked return without voting on a winner", () => {
  const start = createSimulationWorld("paper-ecology-g1");
  const outcomes: TradingDayOutcomes = {
    "parent-01": { netPnlUsdt: 2, startingAllocationUsdt: 20 },
    "parent-02": { netPnlUsdt: -1, startingAllocationUsdt: 20 },
    "parent-03": { netPnlUsdt: 0, startingAllocationUsdt: 20 },
  };
  const before = JSON.stringify(start);
  const next = advanceSimulationDay(start, outcomes);
  assert.equal(JSON.stringify(start), before, "economic feedback must never mutate a historical snapshot");
  for (const plant of start.plants) {
    const after = next.plants.find((p) => p.id === plant.id)!;
    const outcome = outcomes[plant.id]!;
    const factor = 1 + outcome.netPnlUsdt / outcome.startingAllocationUsdt;
    assert.equal(after.biomass, round4(bounded(plant.biomass * factor, 5, 100)));
    assert.equal(after.vigor, round4(bounded(plant.vigor * factor, 0, 100)));
    assert.equal(after.fitness, plant.fitness, "synthetic habitat fit is not a trading statistic");
    assert.equal(after.evidence, plant.evidence, "synthetic evidence cannot be promoted by trading");
    assert.equal(after.evidenceDays, plant.evidenceDays + 1);
  }
  assert.equal(next.events.find((event) => event.type === "daily_tick")?.detail.marketDrivenBiomass, true);
  assertSimulationInvariants(next);
  assert.deepEqual(advanceSimulationDay(createSimulationWorld("paper-ecology-g1"), outcomes), next);
});

test("a no-trade historical day preserves economic body and vigor even if synthetic climate changes", () => {
  const start = createSimulationWorld("no-trade-is-not-a-loss");
  const next = advanceSimulationDay(start, {});
  assert.deepEqual(next.plants.map((plant) => [plant.biomass, plant.vigor, plant.fitness, plant.evidence]),
    start.plants.map((plant) => [plant.biomass, plant.vigor, plant.fitness, plant.evidence]));
  assert.equal(next.day, 1);
  assert.notDeepEqual(advanceSimulationDay(start).plants, next.plants,
    "legacy synthetic game rules remain a separate explicit simulation mode");
});

test("invalid market outcomes fail before any ecology tick or resource transfer", () => {
  const start = createSimulationWorld("market-outcome-validation");
  const invalid = [
    { "parent-unknown": { netPnlUsdt: 1, startingAllocationUsdt: 20 } },
    { "parent-01": { netPnlUsdt: Number.NaN, startingAllocationUsdt: 20 } },
    { "parent-01": { netPnlUsdt: 1, startingAllocationUsdt: 0 } },
    { "parent-01": { netPnlUsdt: Number.POSITIVE_INFINITY, startingAllocationUsdt: 20 } },
    { "parent-01": { netPnlUsdt: -21, startingAllocationUsdt: 20 } },
  ] as const;
  const before = JSON.stringify(start);
  for (const outcomes of invalid) assert.throws(
    () => advanceSimulationDay(start, outcomes as unknown as TradingDayOutcomes), RangeError,
  );
  assert.equal(JSON.stringify(start), before);
  assertSimulationInvariants(start);
});
