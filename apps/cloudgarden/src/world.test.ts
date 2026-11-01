import assert from "node:assert/strict";
import test from "node:test";
import { advanceDay, advanceDays, createWorld, phenotype, worldFingerprint } from "./world";

test("the same seed produces the same seven-day garden", () => {
  const left = advanceDays(createWorld(), 7);
  const right = advanceDays(createWorld(), 7);
  assert.equal(worldFingerprint(left), worldFingerprint(right));
  assert.equal(left.day, 50);
});

test("advancing a day does not mutate the prior snapshot", () => {
  const before = createWorld();
  const fingerprint = worldFingerprint(before);
  const after = advanceDay(before);
  assert.equal(worldFingerprint(before), fingerprint);
  assert.equal(after.day, before.day + 1);
  assert.notEqual(worldFingerprint(after), fingerprint);
});

test("bounded genome inputs create the intended phenotype classes", () => {
  assert.equal(phenotype({ tempo: 0.5, canopy: 0.3, rootDepth: 0.8, rootSpread: 0.4 }), "Reed");
  assert.equal(phenotype({ tempo: 0.5, canopy: 0.8, rootDepth: 0.3, rootSpread: 0.4 }), "Fern");
  assert.equal(phenotype({ tempo: 0.5, canopy: 0.5, rootDepth: 0.3, rootSpread: 0.8 }), "Shrub");
  assert.equal(phenotype({ tempo: 0.5, canopy: 0.5, rootDepth: 0.3, rootSpread: 0.4 }), "Grass");
});

test("daily state stays inside its physical UI bounds", () => {
  let world = createWorld();
  for (let day = 0; day < 200; day += 1) world = advanceDay(world);
  assert.ok(world.waterline >= 0.36 && world.waterline <= 0.72);
  assert.ok(world.sunlight >= 0.25 && world.sunlight <= 0.9);
  for (const plant of world.plants) {
    assert.ok(plant.biomass >= 8 && plant.biomass <= 100);
    assert.ok(plant.vigor >= 18 && plant.vigor <= 98);
    assert.ok(plant.fitness >= 10 && plant.fitness <= 95);
    assert.equal(plant.evidenceDays, 200 + 42 + [0, 22, 9, -6, 39][world.plants.indexOf(plant)]);
  }
});

test("bulk time advancement rejects invalid or unbounded input", () => {
  assert.throws(() => advanceDays(createWorld(), -1), RangeError);
  assert.throws(() => advanceDays(createWorld(), 1.5), RangeError);
  assert.throws(() => advanceDays(createWorld(), 366), RangeError);
});
