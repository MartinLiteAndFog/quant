import assert from "node:assert/strict";
import test from "node:test";
import { advanceSimulationDay, createSimulationWorld, type SimulationWorld } from "./simulation.ts";
import { exportVitalitySnapshot } from "./vitalitySnapshot.ts";

function revisions(world: SimulationWorld): Readonly<Record<string, string>> {
  return Object.fromEntries(world.plants.map((p) => [p.id, `g2-source-${p.id}-v1`]));
}

test("one record per canonical plant with heritable patience and no tile state", () => {
  const world = createSimulationWorld("vitality-export-g2");
  const before = JSON.stringify(world);
  const exported = exportVitalitySnapshot(world, revisions(world));
  assert.equal(exported.plants.length, world.plants.length);
  assert.equal(exported.sceneRef.seed, world.seed);
  assert.equal(exported.sceneRef.dayIndex, 0);
  assert.equal(exported.policy.oneDailyUpdatePerCanonicalPlant, true);
  assert.equal(exported.policy.individualTileVitalityState, false);
  assert.equal(exported.policy.readOnly, true);
  assert.equal(exported.policy.liveTrading, false);
  for (const plant of world.plants) {
    const record = exported.plants.find((x) => x.plantId === plant.id)!;
    assert.equal(record.patience, plant.genes.patience);
    assert.equal(record.vigor, plant.vigor);
    assert.equal(record.inactiveDays, 0);
    assert.equal(record.revision, `g2-source-${plant.id}-v1`);
  }
  assert.equal(JSON.stringify(world), before);
  assert.ok(!JSON.stringify(exported).includes("navUsdt"));
  assert.ok(!JSON.stringify(exported).includes("allocationUsdt"));
  assert.deepEqual(exportVitalitySnapshot(world, revisions(world)), exported);
});

test("15 days without executions export a single eligible idle streak, not per-tile increments", () => {
  let world = createSimulationWorld("vitality-export-idle");
  for (let day = 0; day < 15; day += 1) world = advanceSimulationDay(world, {});
  const exportView = exportVitalitySnapshot(world, revisions(world));
  assert.equal(exportView.sceneRef.dayIndex, 15);
  assert.equal(exportView.plants.find((p) => p.plantId === "parent-01")!.inactiveDays, 15);
  assert.equal(exportView.plants.find((p) => p.plantId === "parent-01")!.vigor,
               world.plants.find((p) => p.id === "parent-01")!.vigor);
  assert.equal(exportView.plants.length, world.plants.length);
  assert.equal(world.capital.navUsdt, 100);
});

test("unknown/partial revisions and missing physiological genes fail closed", () => {
  const world = createSimulationWorld("vitality-export-guards");
  const rs = revisions(world);
  assert.throws(() => exportVitalitySnapshot(world, {}), /exactly one/);
  assert.throws(() => exportVitalitySnapshot(world, { ...rs, "unknown": "fake" }), /exactly one/);
  assert.throws(() => exportVitalitySnapshot(world, { ...rs, "parent-01": "" }), /Missing genome revision/);
  const bad = { ...world, plants: world.plants.map((p, i) =>
    i === 0 ? { ...p, genes: { ...p.genes, patience: undefined } } : p) };
  assert.throws(() => exportVitalitySnapshot(bad, revisions(bad)), /Missing genome revision/);
});
