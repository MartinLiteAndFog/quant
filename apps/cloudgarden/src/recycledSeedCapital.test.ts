import assert from "node:assert/strict";
import test from "node:test";
import {
  advanceSimulationDays,
  assertSimulationInvariants,
  createSimulationWorld,
  SEEDLING_ALLOCATION_USDT,
  SEEDLING_ENTITLEMENTS,
  type SimulationWorld,
} from "./simulation.ts";

const round4 = (x: number) => Math.round(x * 10_000) / 10_000;

/** Transfer the existing remaining free balance into a living adult for a
 * strictly funded zero-free-capital fixture; no new NAV is ever minted. */
function zeroFreeCapital(world: SimulationWorld): SimulationWorld {
  const donor = world.plants.find((plant) => plant.status === "active")!;
  const free = world.capital.freeCapitalUsdt;
  const clean: SimulationWorld = {
    ...world,
    plants: world.plants.map((plant) => plant.id === donor.id
      ? { ...plant, allocationUsdt: round4(plant.allocationUsdt + free) } : plant),
    capital: {
      ...world.capital,
      freeCapitalUsdt: 0,
      deployedCapitalUsdt: round4(world.capital.deployedCapitalUsdt + free),
    },
  };
  assertSimulationInvariants(clean);
  return clean;
}

function archiveDonor(world: SimulationWorld): SimulationWorld {
  const donor = world.plants.find((p) => p.status === "active" && p.role === "seedling")!;
  assert.ok(donor, "a living seedling must exist for a finite-capital archive fixture");
  const result: SimulationWorld = {
    ...world,
    plants: world.plants.map((plant) => plant.id === donor.id
      ? { ...plant, status: "archived" as const } : plant),
    capital: {
      ...world.capital,
      deployedCapitalUsdt: round4(world.capital.deployedCapitalUsdt - donor.allocationUsdt),
      freeCapitalUsdt: round4(world.capital.freeCapitalUsdt + donor.allocationUsdt),
    },
  };
  assertSimulationInvariants(result);
  return result;
}

test("a depleted seedbank alone does not mint a twenty-first funded seedling", () => {
  const after145 = advanceSimulationDays(createSimulationWorld("recycle-ledger"), 145);
  assert.equal(after145.capital.seedlingEntitlements.filter((s) => s.status === "used").length, 20);
  const zero = zeroFreeCapital(after145);
  const before = JSON.stringify(zero);
  const attempted = advanceSimulationDays(zero, 2); // next bee opportunity is day 147
  assert.equal(attempted.plants.filter((plant) => plant.role === "seedling").length, 20);
  assert.equal(attempted.capital.deployedCapitalUsdt, zero.capital.deployedCapitalUsdt);
  assert.equal(attempted.capital.freeCapitalUsdt, 0);
  assert.equal(JSON.stringify(zero), before);
  assertSimulationInvariants(attempted);
});

test("a released genuine allocation can fund the twenty-first child exactly once", () => {
  const after145 = advanceSimulationDays(createSimulationWorld("recycle-ledger"), 145);
  const zero = zeroFreeCapital(after145);
  const reclaimed = archiveDonor(zero);
  const released = reclaimed.capital.freeCapitalUsdt;
  assert.equal(released, SEEDLING_ALLOCATION_USDT);
  const first = advanceSimulationDays(reclaimed, 2); // day 147
  const newChild = first.plants.find((p) => p.id === "seedling-21")!;
  assert.ok(newChild);
  assert.equal(newChild.role, "seedling");
  assert.equal(newChild.status, "active");
  assert.equal(newChild.fundingSource, "recycled");
  assert.ok(newChild.parents && newChild.parents.length === 2);
  assert.equal(newChild.allocationUsdt, SEEDLING_ALLOCATION_USDT);
  assert.equal(first.capital.navUsdt, reclaimed.capital.navUsdt);
  assert.equal(first.capital.protectedReserveUsdt, reclaimed.capital.protectedReserveUsdt);
  assert.equal(first.capital.allocationCeilingUsdt, reclaimed.capital.allocationCeilingUsdt);
  assert.equal(first.capital.reservedMarginUsdt, reclaimed.capital.reservedMarginUsdt);
  assert.equal(first.capital.freeCapitalUsdt, 0);
  assert.equal(first.capital.deployedCapitalUsdt, round4(reclaimed.capital.deployedCapitalUsdt + released));
  assert.equal(first.capital.seedlingEntitlements.length, SEEDLING_ENTITLEMENTS);
  assert.equal(first.capital.seedlingEntitlements.filter((s) => s.status === "used").length, SEEDLING_ENTITLEMENTS);
  const conception = first.events.find((e) => e.type === "conception" && e.subject === newChild.id);
  assert.equal(conception?.detail.fundingSource, "recycled");
  assert.equal(conception?.detail.entitlementId, undefined, "recycled capital never claims a seedbank slot");
  assertSimulationInvariants(first);
  assert.deepEqual(advanceSimulationDays(reclaimed, 2), first);
});

test("recycled children may become ancestors; archived funds can finance a twenty-second birth", () => {
  const zero = zeroFreeCapital(advanceSimulationDays(createSimulationWorld("next-recycled-generation"), 145));
  const child21 = advanceSimulationDays(archiveDonor(zero), 2);
  assert.equal(child21.plants.at(-1)?.fundingSource, "recycled");
  const releasedAgain = archiveDonor(child21);
  const child22 = advanceSimulationDays(releasedAgain, 7);
  assert.equal(child22.plants.at(-1)?.id, "seedling-22");
  assert.equal(child22.plants.at(-1)?.fundingSource, "recycled");
  assert.equal(child22.capital.navUsdt, releasedAgain.capital.navUsdt);
  assert.equal(child22.capital.seedlingEntitlements.filter((s) => s.status === "used").length, 20);
  assertSimulationInvariants(child22);
});

test("forged or duplicated recycled funding cannot be mislabelled as an original entitlement", () => {
  const zero = zeroFreeCapital(advanceSimulationDays(createSimulationWorld("recycle-corruption"), 145));
  const funded = advanceSimulationDays(archiveDonor(zero), 2);
  const fake = {
    ...funded,
    plants: funded.plants.map((p) => p.id === "seedling-21"
      ? { ...p, fundingSource: "seedbank" as const } : p),
  };
  assert.throws(() => assertSimulationInvariants(fake), /Original seedling entitlements/);
  const duplicated = {
    ...funded,
    plants: funded.plants.map((p) => p.id === "seedling-21"
      ? { ...p, fundingSource: "wrong" as "recycled" } : p),
  };
  assert.throws(() => assertSimulationInvariants(duplicated), /invalid seed funding source/);
});
