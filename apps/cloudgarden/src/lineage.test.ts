import assert from "node:assert/strict";
import test from "node:test";
import {
  advanceSimulationDays,
  assertSimulationInvariants,
  createSimulationWorld,
  type SimulationWorld,
} from "./simulation.ts";

test("an adult funded offspring becomes a real parent, without ranking or statistical gates", () => {
  const before = advanceSimulationDays(createSimulationWorld("multi-generation"), 21);
  const original = JSON.stringify(before);
  const after = advanceSimulationDays(before, 7);
  const fourth = after.plants.find((p) => p.id === "seedling-04")!;
  const mother = after.plants.find((p) => p.id === fourth.parents?.[0])!;
  assert.ok(fourth);
  assert.ok(fourth.parents);
  assert.equal(mother.id, "seedling-01");
  assert.equal(mother.role, "seedling");
  assert.equal(mother.status, "active");
  assert.ok(mother.evidenceDays >= 7);
  assert.notEqual(fourth.parents![0], fourth.parents![1]);
  assert.equal(fourth.conceivedBy, "local");
  assert.ok(fourth.genes.patience !== undefined && fourth.genes.patience >= 0.05 && fourth.genes.patience <= 0.95);
  assert.equal(JSON.stringify(before), original);
  assert.deepEqual(advanceSimulationDays(before, 7), after);
  assert.equal(after.capital.seedlingEntitlements.filter((x) => x.status === "used").length, 4);
  assertSimulationInvariants(after);
});

test("offspring patience is inherited with bounded mutation, not a statistical winner selection", () => {
  const base = advanceSimulationDays(createSimulationWorld("patience-inheritance"), 21);
  const setFirstbornPatience = (v: number): SimulationWorld => ({
    ...base, plants: base.plants.map((p) => p.id === "seedling-01"
      ? { ...p, genes: { ...p.genes, patience: v } } : p),
  });
  const low = advanceSimulationDays(setFirstbornPatience(0.05), 7);
  const high = advanceSimulationDays(setFirstbornPatience(0.95), 7);
  const babyLow = low.plants.find((p) => p.id === "seedling-04")!;
  const babyHigh = high.plants.find((p) => p.id === "seedling-04")!;
  assert.equal(babyLow.parents?.[0], "seedling-01");
  assert.equal(babyHigh.parents?.[0], "seedling-01");
  assert.ok((babyHigh.genes.patience ?? 0) > (babyLow.genes.patience ?? 1) + .3);
  assert.equal(babyLow.allocationUsdt, babyHigh.allocationUsdt);
  assert.equal(low.capital.navUsdt, high.capital.navUsdt);
  assertSimulationInvariants(low);
  assertSimulationInvariants(high);
});

test("archived progenitors remain in genealogy but never mate again", () => {
  const base = advanceSimulationDays(createSimulationWorld("archive-lineage"), 21);
  const target = base.plants.find((p) => p.id === "parent-01")!;
  const archived: SimulationWorld = {
    ...base,
    plants: base.plants.map((p) => p.id === target.id
      ? { ...p, status: "archived" as const } : p),
    capital: {
      ...base.capital,
      deployedCapitalUsdt: base.capital.deployedCapitalUsdt - target.allocationUsdt,
      freeCapitalUsdt: base.capital.freeCapitalUsdt + target.allocationUsdt,
    },
  };
  assertSimulationInvariants(archived);
  const next = advanceSimulationDays(archived, 7);
  const newborn = next.plants.find((p) => p.id === "seedling-04")!;
  assert.ok(newborn.parents);
  assert.ok(!newborn.parents!.includes(target.id));
  assert.equal(next.plants.find((p) => p.id === target.id)?.status, "archived");
  assert.ok(next.plants.some((p) => p.id === "seedling-01" && p.parents?.includes(target.id)),
    "archival must preserve the old ancestral lineage");
  assertSimulationInvariants(next);
});
