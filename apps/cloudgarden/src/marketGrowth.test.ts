import assert from "node:assert/strict";
import test from "node:test";
import {
  advanceSimulationDay,
  BASE_IDLE_VIGOR_DECAY,
  INACTIVITY_GRACE_DAYS,
  assertSimulationInvariants,
  createSimulationWorld,
  type TradingDayOutcomes,
  type SimulationWorld,
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

test("14 idle days are free; from day 15 a tiny inherited patience-linked vigor decay starts", () => {
  const original = createSimulationWorld("genetic-patience-g2");
  const world = {
    ...original,
    plants: original.plants.map((p, index) => ({
      ...p, genes: { ...p.genes, patience: index === 0 ? 0.05 : index === 1 ? 0.95 : 0.5 },
    })),
  };
  let current = world;
  for (let day = 1; day <= INACTIVITY_GRACE_DAYS; day += 1) {
    current = advanceSimulationDay(current, {});
    for (let i = 0; i < 3; i += 1) {
      assert.equal(current.plants[i]!.vigor, world.plants[i]!.vigor);
      assert.equal(current.plants[i]!.biomass, world.plants[i]!.biomass);
      assert.equal(current.plants[i]!.inactiveDays, day);
    }
  }
  const fifteenth = advanceSimulationDay(current, {});
  const slow = fifteenth.plants[0]!;
  const patient = fifteenth.plants[1]!;
  const slowRate = BASE_IDLE_VIGOR_DECAY * (1.2 - 0.05);
  const patientRate = BASE_IDLE_VIGOR_DECAY * (1.2 - 0.95);
  assert.equal(slow.vigor, round4(world.plants[0]!.vigor * (1 - slowRate)));
  assert.equal(patient.vigor, round4(world.plants[1]!.vigor * (1 - patientRate)));
  assert.ok((world.plants[0]!.vigor - slow.vigor) / world.plants[0]!.vigor
    > (world.plants[1]!.vigor - patient.vigor) / world.plants[1]!.vigor);
  assert.equal(slow.inactiveDays, 15);
  assert.equal(patient.inactiveDays, 15);
  assert.equal(fifteenth.capital.navUsdt, 100, "opportunity cost is purely vitality, never invented NAV loss");
  assertSimulationInvariants(fifteenth);
});

test("an executed paper order resets inactivity even when its net PnL is exactly zero", () => {
  let world = createSimulationWorld("trade-breaks-inactivity");
  for (let day = 0; day < 25; day += 1) world = advanceSimulationDay(world, {});
  const before = world.plants[0]!;
  const after = advanceSimulationDay(world, {
    "parent-01": { netPnlUsdt: 0, startingAllocationUsdt: 20, executedFills: 1 },
  });
  assert.equal(after.plants[0]!.inactiveDays, 0);
  assert.equal(after.plants[0]!.vigor, before.vigor);
  assert.equal(after.plants[0]!.biomass, before.biomass);
  assert.equal(after.plants[1]!.inactiveDays, 26);
  assert.equal(after.capital.navUsdt, world.capital.navUsdt);
  assertSimulationInvariants(after);
});

test("paper execution count—not signals, imagined profits or statistical badges—defines inactivity", () => {
  let world = createSimulationWorld("fills-only");
  for (let day = 0; day < 15; day += 1) {
    world = advanceSimulationDay(world, {
      "parent-01": { netPnlUsdt: 0, startingAllocationUsdt: 20, executedFills: 0 },
    });
  }
  assert.equal(world.plants[0]!.inactiveDays, 15);
  const bad = [
    { netPnlUsdt: 0, startingAllocationUsdt: 20, executedFills: -1 },
    { netPnlUsdt: 0, startingAllocationUsdt: 20, executedFills: 0.5 },
    { netPnlUsdt: 0, startingAllocationUsdt: 20, executedFills: NaN },
  ];
  for (const o of bad) assert.throws(() => advanceSimulationDay(world, { "parent-01": o }), RangeError);
  assert.throws(() => assertSimulationInvariants({
    ...world, plants: world.plants.map((p,i)=>i === 0
      ? {...p, genes:{...p.genes,patience:1.5}} : p),
  }), /invalid patience genetics/);
  assert.throws(() => assertSimulationInvariants({
    ...world, plants: world.plants.map((p,i)=>i === 0
      ? {...p,inactiveDays:-2} : p),
  }), /invalid inactivity streak/);
});
