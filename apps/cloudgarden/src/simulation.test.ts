import assert from "node:assert/strict";
import test from "node:test";
import {
  advanceSimulationDay,
  advanceSimulationDays,
  assertSimulationInvariants,
  bookClosedPaperReplay,
  createSimulationWorld,
  derivePhenotype,
  MAX_EVENT_HISTORY,
  MIN_OBSERVATION_DAYS,
  PREDATION_EVIDENCE_THRESHOLD,
  PREDATION_VIGOR_THRESHOLD,
  SEEDLING_ENTITLEMENTS,
  visiblePopulation,
  type GardenPlant,
  type SimulationWorld,
} from "./simulation.ts";

const fingerprint = (world: SimulationWorld) => JSON.stringify(world);

test("world construction funds three parents and twenty single-use seedling entitlements", () => {
  const world = createSimulationWorld();
  assert.equal(world.plants.filter((plant) => plant.role === "parent").length, 3);
  assert.equal(world.capital.navUsdt, 100);
  assert.equal(world.capital.protectedReserveUsdt, 10);
  assert.equal(world.capital.allocationCeilingUsdt, 90);
  assert.equal(world.capital.deployedCapitalUsdt, 60);
  assert.equal(world.capital.reservedMarginUsdt, 30);
  assert.equal(world.capital.freeCapitalUsdt, 0);
  assert.equal(world.capital.seedlingEntitlements.length, SEEDLING_ENTITLEMENTS);
  assert.equal(visiblePopulation(world).length, 3);
  const parent = world.plants[0];
  assert.ok(parent);
  assert.deepEqual([parent.fitness, parent.vigor, parent.evidence], [73, 82, 76]);
  assert.equal(new Set([parent.fitness, parent.vigor, parent.evidence]).size, 3);
  assertSimulationInvariants(world);
});

test("daily advancement is deterministic and does not mutate earlier snapshots", () => {
  const before = createSimulationWorld("same-seed");
  const beforeJson = fingerprint(before);
  const after = advanceSimulationDays(before, 21);
  const repeated = advanceSimulationDays(createSimulationWorld("same-seed"), 21);
  assert.equal(fingerprint(after), fingerprint(repeated));
  assert.equal(fingerprint(before), beforeJson);
  assert.equal(after.day, 21);
  assert.ok(after.events.length > 21); // tick, conception, and weather/predator audit trail
});

test("closed paper PnL updates the same NAV, reserve, and allocation ledger", () => {
  const initial = createSimulationWorld("paper-booking");
  const afterWeek = advanceSimulationDays(initial, 7);
  const booked = bookClosedPaperReplay(afterWeek, "parent-01", {
    startDay: 0,
    days: 7,
    datasetFingerprint: "a".repeat(64),
    synthetic: true,
    endedFlat: true,
    riskBreach: false,
    fillCount: 2,
    allocationLimitUsdt: 20,
    startingCashUsdt: 20,
    endingCashUsdt: 20.022,
    maxMarginUsedUsdt: 10,
    realizedPricePnlUsdt: 0.03,
    feesUsdt: 0.01,
    fundingUsdt: 0.002,
    netPnlUsdt: 0.022,
  });
  assert.equal(booked.capital.navUsdt, 100.022);
  assert.equal(booked.capital.protectedReserveUsdt, 10.0022);
  assert.equal(booked.capital.allocationCeilingUsdt, 90.0198);
  assert.equal(booked.capital.freeCapitalUsdt, 0.0198);
  assert.equal(booked.capital.realizedPricePnlUsdt, 0.03);
  assert.equal(booked.capital.paperFeesUsdt, 0.01);
  assert.equal(booked.capital.paperFundingUsdt, 0.002);
  assert.equal(booked.capital.paperReplayCount, 1);
  assert.equal(booked.events.filter((event) => event.type === "paper_replay").length, 1);
  assertSimulationInvariants(booked);
});

test("paper losses reduce the selected entitlement and a failed replay cannot book", () => {
  const afterWeek = advanceSimulationDays(createSimulationWorld("paper-loss"), 7);
  const booked = bookClosedPaperReplay(afterWeek, "parent-01", {
    startDay: 0,
    days: 7,
    datasetFingerprint: "b".repeat(64),
    synthetic: true,
    endedFlat: true,
    riskBreach: false,
    fillCount: 2,
    allocationLimitUsdt: 20,
    startingCashUsdt: 20,
    endingCashUsdt: 19.981,
    maxMarginUsedUsdt: 10,
    realizedPricePnlUsdt: 0.001,
    feesUsdt: 0.02,
    fundingUsdt: 0,
    netPnlUsdt: -0.019,
  });
  assert.equal(booked.capital.navUsdt, 99.981);
  assert.equal(booked.capital.protectedReserveUsdt, 9.9981);
  assert.equal(booked.capital.allocationCeilingUsdt, 89.9829);
  assert.equal(booked.capital.deployedCapitalUsdt, 61.4829);
  assert.equal(booked.plants.find((plant) => plant.id === "parent-01")?.allocationUsdt, 19.9829);
  assert.equal(booked.capital.freeCapitalUsdt, 0);
  assertSimulationInvariants(booked);

  assert.throws(() => bookClosedPaperReplay(afterWeek, "parent-01", {
    startDay: 0, days: 7, datasetFingerprint: "c".repeat(64), synthetic: true, endedFlat: false,
    riskBreach: false, fillCount: 2, allocationLimitUsdt: 20, startingCashUsdt: 20, endingCashUsdt: 20, maxMarginUsedUsdt: 0,
    realizedPricePnlUsdt: 0, feesUsdt: 0, fundingUsdt: 0, netPnlUsdt: 0,
  }), /flat synthetic replay/);
});

test("the three bee modes deterministically conceive funded local, hive, and hybrid seedlings", () => {
  const world = advanceSimulationDays(createSimulationWorld(), 21);
  const seedlings = world.plants.filter((plant) => plant.role === "seedling");
  assert.deepEqual(seedlings.map((plant) => plant.conceivedBy), ["local", "hive", "hybrid"]);
  assert.equal(new Set(seedlings.map((plant) => plant.id)).size, 3);
  assert.equal(world.capital.deployedCapitalUsdt, 64.5);
  assert.equal(world.capital.reservedMarginUsdt, 25.5);
  assert.equal(world.capital.seedlingEntitlements.filter((slot) => slot.status === "used").length, 3);
  assert.equal(world.events.filter((event) => event.type === "conception").length, 3);
  for (const event of world.events.filter((entry) => entry.type === "conception")) {
    assert.equal(event.detail.allocationUsdt, 1.5);
    assert.ok(event.actor?.endsWith("-bee"));
  }
  assertSimulationInvariants(world);
});

test("all 20 funded seedlings can be conceived without restoring archived entitlements", () => {
  const world = advanceSimulationDays(createSimulationWorld(), 140);
  assert.equal(world.plants.filter((plant) => plant.role === "seedling").length, 20);
  assert.equal(world.capital.seedlingEntitlements.filter((slot) => slot.status === "used").length, 20);
  assert.equal(world.capital.reservedMarginUsdt, 0);
  assert.equal(world.capital.deployedCapitalUsdt + world.capital.freeCapitalUsdt, 90);
  assertSimulationInvariants(world);
});

test("gene-derived phenotypes follow the Cloudgarden Grass, Bush, and Cactus rules", () => {
  assert.equal(derivePhenotype({ root: 0.4, leaf: 0.8, water: 0.5, sun: 0.5 }), "Bush");
  assert.equal(derivePhenotype({ root: 0.8, leaf: 0.4, water: 0.5, sun: 0.5 }), "Cactus");
  assert.equal(derivePhenotype({ root: 0.7, leaf: 0.7, water: 0.5, sun: 0.5 }), "Bush");
  assert.equal(derivePhenotype({ root: 0.5, leaf: 0.5, water: 0.5, sun: 0.5 }), "Grass");
});

test("root, leaf, water, and sun genes affect synthetic plant responses", () => {
  const baseline = createSimulationWorld();
  const plants: GardenPlant[] = baseline.plants.map((plant) => ({
    ...plant,
    genes: { root: 0.5, leaf: 0.5, water: 0.5, sun: 0.5 },
    phenotype: derivePhenotype({ root: 0.5, leaf: 0.5, water: 0.5, sun: 0.5 }),
  }));
  const variants = plants.map((plant, index) => {
    const trait = (["root", "leaf", "water", "sun"] as const)[index];
    const genes = { ...plant.genes, [trait]: 0.95 };
    return { ...plant, id: `variant-${trait}`, genes, phenotype: derivePhenotype(genes) };
  });
  const equalWorld: SimulationWorld = { ...baseline, plants };
  const variedWorld: SimulationWorld = { ...baseline, plants: variants };
  const equalAfter = advanceSimulationDay(equalWorld);
  const variedAfter = advanceSimulationDay(variedWorld);
  for (let index = 0; index < variants.length; index += 1) {
    assert.notEqual(variedAfter.plants[index].biomass, equalAfter.plants[index].biomass,
      `${variants[index].id} should change the plant's response`);
    assert.notEqual(variedAfter.plants[index].fitness, equalAfter.plants[index].fitness,
      `${variants[index].id} should change momentary environmental fitness`);
  }
});

test("fitness, vigor, and synthetic evidence remain distinct bounded plant measures", () => {
  let world = advanceSimulationDays(createSimulationWorld(), 4);
  for (const plant of world.plants) {
    assert.ok(plant.fitness >= 0 && plant.fitness <= 100);
    assert.ok(plant.vigor >= 0 && plant.vigor <= 100);
    assert.ok(plant.evidence >= 0 && plant.evidence <= 100);
  }
  const parent = world.plants.find((plant) => plant.id === "parent-01");
  assert.ok(parent);
  assert.equal(new Set([parent.fitness, parent.vigor, parent.evidence]).size, 3,
    "the same plant must hold separate instantaneous fit, accumulated physiology, and observation values");
  assert.throws(() => assertSimulationInvariants({
    ...world,
    plants: world.plants.map((plant) => plant.id === parent.id ? { ...plant, fitness: 101 } : plant),
  }), /out-of-range fitness, vigor, or evidence/);
});

test("high root and leaf allocation pays an explicit growth trade-off", () => {
  const baseline = createSimulationWorld();
  const traits = (root: number, leaf: number): GardenPlant[] => baseline.plants.map((plant) => {
    const genes = { ...plant.genes, root, leaf };
    return { ...plant, genes, phenotype: derivePhenotype(genes) };
  });
  const rootSpecialist = advanceSimulationDay({ ...baseline, plants: traits(0.9, 0.2) });
  const leafSpecialist = advanceSimulationDay({ ...baseline, plants: traits(0.2, 0.9) });
  const bothHigh = advanceSimulationDay({ ...baseline, plants: traits(0.9, 0.9) });
  assert.ok(bothHigh.plants[0].biomass < rootSpecialist.plants[0].biomass);
  assert.ok(bothHigh.plants[0].biomass < leafSpecialist.plants[0].biomass);
});

test("young plants are protected until minimum observation, and predation is threshold-gated", () => {
  let world = advanceSimulationDays(createSimulationWorld(), 7);
  const seedling = world.plants.find((plant) => plant.role === "seedling");
  assert.ok(seedling);
  const stressed: GardenPlant = { ...seedling, evidenceDays: MIN_OBSERVATION_DAYS - 1, vigor: 0, evidence: 0 };
  world = { ...world, day: 9, plants: world.plants.map((plant) => plant.id === seedling.id ? stressed : plant) };
  const after = advanceSimulationDay(world); // day 10 predator patrol, observation reaches 80
  const archived = after.plants.find((plant) => plant.id === seedling.id);
  assert.equal(archived?.evidenceDays, MIN_OBSERVATION_DAYS);
  assert.equal(archived?.status, "archived");
  assert.ok(!visiblePopulation(after).some((plant) => plant.id === seedling.id));
  assert.equal(after.capital.deployedCapitalUsdt, 60);
  assert.equal(after.capital.freeCapitalUsdt, 1.5);
  assert.equal(after.capital.seedlingEntitlements.filter((slot) => slot.status === "used").length, 1,
    "predation releases capital but does not restore the one-time entitlement");
  assert.equal(after.events.filter((event) => event.type === "predation").length, 1);
  assertSimulationInvariants(after);

  const nextCycle = advanceSimulationDays(after, 4); // day 14: next slot is used, old slot remains archived
  assert.equal(nextCycle.capital.seedlingEntitlements.filter((slot) => slot.status === "used").length, 2);
  assert.equal(nextCycle.plants.find((plant) => plant.id === seedling.id)?.status, "archived");
  assert.equal(nextCycle.plants.filter((plant) => plant.role === "seedling").length, 2);
  assert.equal(nextCycle.capital.freeCapitalUsdt, 1.5);
  assertSimulationInvariants(nextCycle);

  assert.ok(PREDATION_VIGOR_THRESHOLD === 70 && PREDATION_EVIDENCE_THRESHOLD === 80);
});

test("predation does not archive plants above either vulnerability threshold", () => {
  let world = advanceSimulationDays(createSimulationWorld(), 7);
  const seedling = world.plants.find((plant) => plant.role === "seedling");
  assert.ok(seedling);
  const protectedPlant: GardenPlant = {
    ...seedling,
    evidenceDays: MIN_OBSERVATION_DAYS,
    vigor: PREDATION_VIGOR_THRESHOLD + 10,
    evidence: PREDATION_EVIDENCE_THRESHOLD + 10,
  };
  world = { ...world, day: 9, plants: world.plants.map((plant) => plant.id === seedling.id ? protectedPlant : plant) };
  const after = advanceSimulationDay(world);
  assert.equal(after.plants.find((plant) => plant.id === seedling.id)?.status, "active");
  assert.equal(after.events.filter((event) => event.type === "predation").length, 0);
});

test("mature bark adds protection beyond the minimum observation period", () => {
  let world = advanceSimulationDays(createSimulationWorld(), 7);
  const seedling = world.plants.find((plant) => plant.role === "seedling");
  assert.ok(seedling);
  const barked: GardenPlant = { ...seedling, evidenceDays: 120, vigor: 65, evidence: 75 };
  world = { ...world, day: 9, plants: world.plants.map((plant) => plant.id === seedling.id ? barked : plant) };
  const after = advanceSimulationDay(world);
  assert.equal(after.plants.find((plant) => plant.id === seedling.id)?.status, "active");
  assert.equal(after.events.filter((event) => event.type === "predation").length, 0);
});

test("dormancy is reversible climate physiology that keeps plants visible and capitalized", () => {
  let world = createSimulationWorld("DORMANCY-0");
  const parent = world.plants[0];
  assert.ok(parent);
  world = { ...world, plants: world.plants.map((plant) => plant.id === parent.id ? { ...plant, vigor: 40 } : plant) };
  const startingCapital = world.capital;

  const dormant = advanceSimulationDay(world);
  const dormantParent = dormant.plants.find((plant) => plant.id === parent.id);
  assert.equal(dormantParent?.physiology, "dormant");
  assert.equal(dormantParent?.status, "active");
  assert.ok(visiblePopulation(dormant).some((plant) => plant.id === parent.id));
  assert.equal(dormant.capital.deployedCapitalUsdt, startingCapital.deployedCapitalUsdt);
  assert.equal(dormant.capital.freeCapitalUsdt, startingCapital.freeCapitalUsdt);
  assert.equal(dormant.capital.navUsdt, 100);
  assert.ok(dormant.events.some((event) => event.type === "dormancy" && event.subject === parent.id));
  assertSimulationInvariants(dormant);

  const beforeWake = advanceSimulationDays(dormant, 6); // day 7 uses one seed entitlement as usual
  const recovered = advanceSimulationDay(beforeWake); // day 8 has recovered water and sunlight
  const recoveredParent = recovered.plants.find((plant) => plant.id === parent.id);
  assert.equal(recoveredParent?.physiology, "awake");
  assert.equal(recoveredParent?.status, "active");
  assert.ok(recovered.events.some((event) => event.type === "awakening" && event.subject === parent.id));
  assert.equal(beforeWake.capital.deployedCapitalUsdt, startingCapital.deployedCapitalUsdt + 1.5);
  assert.equal(recovered.capital.deployedCapitalUsdt, beforeWake.capital.deployedCapitalUsdt);
  assert.equal(recovered.capital.freeCapitalUsdt, beforeWake.capital.freeCapitalUsdt);
  assert.equal(recovered.capital.reservedMarginUsdt, beforeWake.capital.reservedMarginUsdt);
  assert.equal(recovered.capital.navUsdt, 100);
  assertSimulationInvariants(recovered);
});

test("predators ignore dormant plants without archiving or releasing their capital", () => {
  let world = advanceSimulationDays(createSimulationWorld("DORMANCY-0"), 7);
  const seedling = world.plants.find((plant) => plant.role === "seedling");
  assert.ok(seedling);
  const dormant: GardenPlant = {
    ...seedling,
    physiology: "dormant",
    evidenceDays: 120,
    vigor: 0,
    evidence: 0,
  };
  world = { ...world, day: 9, plants: world.plants.map((plant) => plant.id === seedling.id ? dormant : plant) };
  const deployedBefore = world.capital.deployedCapitalUsdt;
  const freeBefore = world.capital.freeCapitalUsdt;
  const after = advanceSimulationDay(world); // day 10 is a predator patrol; sunlight is below the wake threshold
  assert.equal(after.plants.find((plant) => plant.id === seedling.id)?.physiology, "dormant");
  assert.equal(after.plants.find((plant) => plant.id === seedling.id)?.status, "active");
  assert.ok(visiblePopulation(after).some((plant) => plant.id === seedling.id));
  assert.equal(after.events.filter((event) => event.type === "predation").length, 0);
  assert.equal(after.capital.deployedCapitalUsdt, deployedBefore);
  assert.equal(after.capital.freeCapitalUsdt, freeBefore);
  assertSimulationInvariants(after);
});

test("autoplay event history is capped to the latest 2,000 deterministic events", () => {
  let left = createSimulationWorld("LONG-RUN");
  let right = createSimulationWorld("LONG-RUN");
  for (let block = 0; block < 7; block += 1) {
    left = advanceSimulationDays(left, 365);
    right = advanceSimulationDays(right, 365);
  }
  assert.equal(left.day, 2_555);
  assert.equal(left.events.length, MAX_EVENT_HISTORY);
  assert.equal(right.events.length, MAX_EVENT_HISTORY);
  assert.deepEqual(left.events, right.events);
  assert.ok(left.events[0].day > 0);
  assert.equal(left.events.at(-1)?.day, left.day);
  assert.equal(new Set(left.events.map((event) => event.id)).size, left.events.length);
});

test("bulk advancement rejects invalid or unbounded time steps", () => {
  const world = createSimulationWorld();
  assert.throws(() => advanceSimulationDays(world, -1), RangeError);
  assert.throws(() => advanceSimulationDays(world, 1.5), RangeError);
  assert.throws(() => advanceSimulationDays(world, 366), RangeError);
  assert.throws(() => createSimulationWorld("  "), RangeError);
});
