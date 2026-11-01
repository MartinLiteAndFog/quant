import {
  advanceSimulationDays,
  assertSimulationInvariants,
  createSimulationWorld,
  type BeeMode,
  type SimulationWorld,
} from "./simulation.ts";

export const CONTROLLED_COHORT_MODES = ["random", "local", "hive", "hybrid"] as const satisfies readonly BeeMode[];
export const SYNTHETIC_ONLY_NOTICE = "Synthetic garden outcomes only; these are not market evidence or a profit claim.";

export type ControlledExperimentInput = Readonly<{
  /** Every cohort receives this same ordered seed list. */
  seeds: readonly string[];
  /** Every cohort advances exactly this many whole simulation days. */
  days: number;
}>;

export type SeedOutcome = Readonly<{
  seed: string;
  days: number;
  activePlants: number;
  archivedPlants: number;
  seedlingsConceived: number;
  entitlementsUsed: number;
  navUsdt: number;
  protectedReserveUsdt: number;
  freeCapitalUsdt: number;
  reservedMarginUsdt: number;
  deployedCapitalUsdt: number;
  meanHabitatFit: number;
  meanVigor: number;
  meanSyntheticEvidence: number;
  meanBiomass: number;
}>;

export type CohortResult = Readonly<{
  mode: (typeof CONTROLLED_COHORT_MODES)[number];
  /** UI-friendly alias for mode; kept identical for a single source of truth. */
  policy: (typeof CONTROLLED_COHORT_MODES)[number];
  trials: number;
  daysPerTrial: number;
  outcomes: readonly SeedOutcome[];
  meanActivePlants: number;
  meanArchivedPlants: number;
  meanSeedlingsConceived: number;
  meanHabitatFit: number;
  meanVigor: number;
  meanSyntheticEvidence: number;
  meanBiomass: number;
  meanNavUsdt: number;
  meanProtectedReserveUsdt: number;
  meanDeployedCapitalUsdt: number;
  meanReservedMarginUsdt: number;
  capitalConserved: boolean;
}>;

export type ExperimentReport = Readonly<{
  experiment: "cloudgarden-controlled-cohorts-v1";
  benchmark: "random";
  benchmarkConclusion: "not_comparable";
  interpretation: string;
  claim: typeof SYNTHETIC_ONLY_NOTICE;
  inputSeeds: readonly string[];
  /** UI-friendly alias for inputSeeds. */
  seeds: readonly string[];
  days: number;
  initialNavUsdt: 100;
  cohorts: readonly CohortResult[];
}>;

function mean(values: readonly number[]): number {
  if (values.length === 0) return 0;
  return Math.round((values.reduce((sum, value) => sum + value, 0) / values.length) * 10_000) / 10_000;
}

function outcomeFor(world: SimulationWorld, seed: string, days: number): SeedOutcome {
  const visible = world.plants.filter((plant) => plant.status === "active");
  const archived = world.plants.filter((plant) => plant.status === "archived");
  const seedlings = world.plants.filter((plant) => plant.role === "seedling");
  const activeMetrics = (key: "fitness" | "vigor" | "evidence" | "biomass") => visible.map((plant) => plant[key]);
  return {
    seed,
    days,
    activePlants: visible.length,
    archivedPlants: archived.length,
    seedlingsConceived: seedlings.length,
    entitlementsUsed: world.capital.seedlingEntitlements.filter((slot) => slot.status === "used").length,
    navUsdt: world.capital.navUsdt,
    protectedReserveUsdt: world.capital.protectedReserveUsdt,
    freeCapitalUsdt: world.capital.freeCapitalUsdt,
    reservedMarginUsdt: world.capital.reservedMarginUsdt,
    deployedCapitalUsdt: world.capital.deployedCapitalUsdt,
    meanHabitatFit: mean(activeMetrics("fitness")),
    meanVigor: mean(activeMetrics("vigor")),
    meanSyntheticEvidence: mean(activeMetrics("evidence")),
    meanBiomass: mean(activeMetrics("biomass")),
  };
}

function summarize(mode: CohortResult["mode"], outcomes: readonly SeedOutcome[], days: number): CohortResult {
  const values = (key: keyof SeedOutcome) => outcomes.map((outcome) => outcome[key] as number);
  return {
    mode,
    trials: outcomes.length,
    daysPerTrial: days,
    policy: mode,
    outcomes,
    meanActivePlants: mean(values("activePlants")),
    meanArchivedPlants: mean(values("archivedPlants")),
    meanSeedlingsConceived: mean(values("seedlingsConceived")),
    meanHabitatFit: mean(values("meanHabitatFit")),
    meanVigor: mean(values("meanVigor")),
    meanSyntheticEvidence: mean(values("meanSyntheticEvidence")),
    meanBiomass: mean(values("meanBiomass")),
    meanNavUsdt: mean(values("navUsdt")),
    meanProtectedReserveUsdt: mean(values("protectedReserveUsdt")),
    meanDeployedCapitalUsdt: mean(values("deployedCapitalUsdt")),
    meanReservedMarginUsdt: mean(values("reservedMarginUsdt")),
    capitalConserved: outcomes.every((outcome) => Math.abs(
      outcome.navUsdt - outcome.protectedReserveUsdt - outcome.freeCapitalUsdt
      - outcome.reservedMarginUsdt - outcome.deployedCapitalUsdt,
    ) < 0.0001 && outcome.navUsdt === 100),
  };
}

/**
 * Run matched synthetic cohorts. Each mode receives identical starting NAV,
 * seed values, climate sequences, and day counts; only conception policy varies.
 */
export function runControlledExperiment(input: ControlledExperimentInput): ExperimentReport {
  if (!Array.isArray(input.seeds) || input.seeds.length === 0) {
    throw new RangeError("At least one fixed seed is required for a controlled experiment.");
  }
  if (input.seeds.some((seed) => typeof seed !== "string" || !seed.trim()) || new Set(input.seeds).size !== input.seeds.length) {
    throw new RangeError("Experiment seeds must be non-empty and unique.");
  }
  if (!Number.isInteger(input.days) || input.days < 0 || input.days > 365) {
    throw new RangeError("Controlled experiments advance in whole days, up to 365 per trial.");
  }
  const inputSeeds = [...input.seeds];
  const cohorts: CohortResult[] = CONTROLLED_COHORT_MODES.map((mode) => {
    const outcomes = inputSeeds.map((seed) => {
      const initial = createSimulationWorld(seed, { conceptionPolicy: mode });
      const finalWorld = advanceSimulationDays(initial, input.days);
      assertSimulationInvariants(finalWorld);
      return outcomeFor(finalWorld, seed, input.days);
    });
    return summarize(mode, outcomes, input.days);
  });
  return {
    experiment: "cloudgarden-controlled-cohorts-v1",
    benchmark: "random",
    benchmarkConclusion: "not_comparable",
    interpretation: "not_comparable — matched game-rule outcomes do not compare against market evidence or trading performance.",
    claim: SYNTHETIC_ONLY_NOTICE,
    inputSeeds,
    seeds: inputSeeds,
    days: input.days,
    initialNavUsdt: 100,
    cohorts,
  };
}
