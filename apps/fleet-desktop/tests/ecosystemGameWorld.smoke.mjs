import assert from "node:assert/strict";
import {
  DEMO_PLANTS, DEMO_UNDERWORLD, PARAMETER_MEADOWS, getUnassignedPlants,
  groundwaterSurface, missingAxisGenomes, normalizeAxis, selectMeadowPlants,
} from "../src/lib/ecosystemGameWorld.ts";

assert.equal(PARAMETER_MEADOWS.length, 2);
for (const meadow of PARAMETER_MEADOWS) {
  const plants = selectMeadowPlants(DEMO_PLANTS, meadow);
  assert.ok(plants.length > 0, `${meadow.id} has a fixture population`);
  assert.ok(plants.every((plant) => plant.strategy === meadow.strategy));
  assert.equal(missingAxisGenomes(plants, meadow.xAxis).length, 0);
  assert.equal(missingAxisGenomes(plants, meadow.yAxis).length, 0);
  for (const plant of plants) {
    assert.ok(normalizeAxis(plant.genome[meadow.xAxis.id], meadow.xAxis) !== null);
    assert.ok(normalizeAxis(plant.genome[meadow.yAxis.id], meadow.yAxis) !== null);
  }
}

const imba = PARAMETER_MEADOWS[0];
assert.equal(normalizeAxis(undefined, imba.xAxis), null);
assert.equal(normalizeAxis(Number.NaN, imba.xAxis), null);
assert.equal(normalizeAxis(imba.xAxis.min, imba.xAxis), 0);
assert.equal(normalizeAxis(imba.xAxis.max, imba.xAxis), 1);
assert.ok(groundwaterSurface(0) >= 0.18);
assert.ok(groundwaterSurface(1) > groundwaterSurface(0.5));
assert.equal(groundwaterSurface(100), groundwaterSurface(1));
assert.ok(DEMO_UNDERWORLD.length > 0);
assert.ok(DEMO_UNDERWORLD.every((cell) => cell.count > 0));

const unassigned = getUnassignedPlants([
  DEMO_PLANTS[0],
  { ...DEMO_PLANTS[0], id: "unassigned-demo", strategy: "unlisted" },
]);
assert.deepEqual(unassigned.map((plant) => plant.id), ["unassigned-demo"]);
console.log("ECO-11 game-world fixtures: strategy categories, genome axes, groundwater and underworld clusters OK");
