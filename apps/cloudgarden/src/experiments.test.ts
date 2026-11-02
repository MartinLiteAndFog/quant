import assert from "node:assert/strict";
import test from "node:test";
import {
  CONTROLLED_COHORT_MODES,
  runControlledExperiment,
  SYNTHETIC_ONLY_NOTICE,
} from "./experiments.ts";
import { advanceSimulationDays, createSimulationWorld } from "./simulation.ts";

test("controlled cohorts use the same seeds and horizon and disclose synthetic-only scope", () => {
  const report = runControlledExperiment({ seeds: ["MATCH-A", "MATCH-B"], days: 28 });
  assert.deepEqual(report.cohorts.map((cohort) => cohort.mode), CONTROLLED_COHORT_MODES);
  assert.deepEqual(report.inputSeeds, ["MATCH-A", "MATCH-B"]);
  assert.deepEqual(report.seeds, report.inputSeeds);
  assert.equal(report.days, 28);
  assert.equal(report.initialNavUsdt, 100);
  assert.equal(report.benchmark, "random");
  assert.equal(report.benchmarkConclusion, "not_comparable");
  assert.ok(report.interpretation.startsWith("not_comparable"));
  assert.equal(report.claim, SYNTHETIC_ONLY_NOTICE);
  for (const cohort of report.cohorts) {
    assert.equal(cohort.trials, 2);
    assert.equal(cohort.daysPerTrial, 28);
    assert.equal(cohort.policy, cohort.mode);
    assert.equal(cohort.capitalConserved, true);
    assert.deepEqual(cohort.outcomes.map((outcome) => outcome.seed), report.inputSeeds);
    assert.ok(cohort.outcomes.every((outcome) => outcome.days === report.days));
  }
});

test("each conception policy is honored, and random genes are independent of parent genes", () => {
  const firstSeedlings = CONTROLLED_COHORT_MODES.map((mode) => {
    const world = advanceSimulationDays(createSimulationWorld("POLICY-CHECK", { conceptionPolicy: mode }), 7);
    return world.plants.find((plant) => plant.role === "seedling");
  });
  assert.deepEqual(firstSeedlings.map((plant) => plant?.conceivedBy), CONTROLLED_COHORT_MODES);
  const randomGenes = JSON.stringify(firstSeedlings[0]?.genes);
  const localGenes = JSON.stringify(firstSeedlings[1]?.genes);
  assert.notEqual(randomGenes, localGenes, "random mode must generate its own bounded genome rather than cross local parents");
  for (const seedling of firstSeedlings) {
    assert.ok(seedling);
    for (const value of Object.values(seedling.genes)) assert.ok(value >= 0.05 && value <= 0.95);
  }
});

test("controlled experiments are reproducible and conserve each cohort's capital ledger", () => {
  const input = { seeds: ["BUDGET-1", "BUDGET-2"], days: 70 };
  const first = runControlledExperiment(input);
  const repeated = runControlledExperiment(input);
  assert.deepEqual(first, repeated);
  for (const cohort of first.cohorts) {
    for (const outcome of cohort.outcomes) {
      assert.equal(outcome.navUsdt, 100);
      assert.equal(outcome.protectedReserveUsdt, 10);
      assert.equal(
        outcome.protectedReserveUsdt + outcome.freeCapitalUsdt + outcome.reservedMarginUsdt + outcome.deployedCapitalUsdt,
        100,
      );
      assert.equal(outcome.seedlingsConceived, outcome.entitlementsUsed);
      assert.ok(outcome.entitlementsUsed <= 20);
      assert.ok(outcome.meanHabitatFit >= 0 && outcome.meanHabitatFit <= 100);
      const finalWorld = advanceSimulationDays(
        createSimulationWorld(outcome.seed, { conceptionPolicy: cohort.mode }),
        outcome.days,
      );
      const active = finalWorld.plants.filter((plant) => plant.status === "active");
      const expectedFit = active.length
        ? Math.round(active.reduce((sum, plant) => sum + plant.fitness, 0) / active.length * 10_000) / 10_000
        : 0;
      assert.equal(outcome.meanHabitatFit, expectedFit);
    }
    assert.ok(cohort.meanHabitatFit >= 0 && cohort.meanHabitatFit <= 100);
  }
});

test("matched cohorts summarize inherited trading traits descriptively and repeatably", () => {
  const input = { seeds: ["GENE-A", "GENE-B", "GENE-C"], days: 42 };
  const report = runControlledExperiment(input);
  assert.deepEqual(runControlledExperiment(input), report);

  const roundMean = (values: number[]) => values.length
    ? Math.round(values.reduce((sum, value) => sum + value, 0) / values.length * 10_000) / 10_000
    : 0;
  const fields = [
    ["meanTradingLookbackBars", "lookbackBars"],
    ["meanTradingEntryThresholdBps", "entryThresholdBps"],
    ["meanTradingExitThresholdBps", "exitThresholdBps"],
  ] as const;

  for (const cohort of report.cohorts) {
    for (const [outcomeIndex, outcome] of cohort.outcomes.entries()) {
      const finalWorld = advanceSimulationDays(
        createSimulationWorld(outcome.seed, { conceptionPolicy: cohort.mode }),
        outcome.days,
      );
      const active = finalWorld.plants.filter((plant) => plant.status === "active");
      for (const [outcomeField, traitField] of fields) {
        const expected = roundMean(active.map((plant) => plant.tradingTraits[traitField]));
        assert.equal(outcome[outcomeField], expected, `${cohort.mode}/${outcome.seed} ${traitField}`);
      }
      assert.equal(outcomeIndex, input.seeds.indexOf(outcome.seed));
    }

    for (const [cohortField, outcomeField] of fields.map(([cohortKey]) => [cohortKey, cohortKey] as const)) {
      assert.equal(
        cohort[cohortField],
        roundMean(cohort.outcomes.map((outcome) => outcome[outcomeField])),
        `${cohort.mode} equal-weight trial mean for ${cohortField}`,
      );
    }
  }

  for (const [cohortField] of fields) {
    const policyMeans = report.cohorts.map((cohort) => cohort[cohortField]);
    assert.ok(new Set(policyMeans).size > 1, `${cohortField} should reflect the distinct conception policies`);
  }
  assert.equal(report.benchmarkConclusion, "not_comparable");
  assert.equal(report.claim, SYNTHETIC_ONLY_NOTICE);
  assert.ok(report.interpretation.includes("do not compare against market evidence or trading performance"));
});

test("every fixed policy can use all twenty seedling slots without entitlement reuse", () => {
  for (const mode of CONTROLLED_COHORT_MODES) {
    const world = advanceSimulationDays(createSimulationWorld(`FULL-${mode}`, { conceptionPolicy: mode }), 140);
    const seedlings = world.plants.filter((plant) => plant.role === "seedling");
    assert.equal(seedlings.length, 20, `${mode} should fill the full funded seed bank`);
    assert.ok(seedlings.every((plant) => plant.conceivedBy === mode));
    assert.equal(world.capital.seedlingEntitlements.filter((slot) => slot.status === "used").length, 20);
    assert.equal(new Set(world.capital.seedlingEntitlements.map((slot) => slot.plantId).filter(Boolean)).size, 20);
    assert.equal(world.capital.navUsdt, 100);
    assert.equal(world.capital.protectedReserveUsdt, 10);
  }
});

test("the existing mixed default remains reproducible and rotates local, hive, hybrid", () => {
  const world = advanceSimulationDays(createSimulationWorld("DEFAULT-POLICY"), 21);
  assert.deepEqual(
    world.plants.filter((plant) => plant.role === "seedling").map((plant) => plant.conceivedBy),
    ["local", "hive", "hybrid"],
  );
});

test("invalid experiment inputs fail before running cohorts", () => {
  assert.throws(() => runControlledExperiment({ seeds: [], days: 1 }), RangeError);
  assert.throws(() => runControlledExperiment({ seeds: ["A", "A"], days: 1 }), RangeError);
  assert.throws(() => runControlledExperiment({ seeds: ["A"], days: 1.5 }), RangeError);
  assert.throws(() => runControlledExperiment({ seeds: ["A"], days: 366 }), RangeError);
});
