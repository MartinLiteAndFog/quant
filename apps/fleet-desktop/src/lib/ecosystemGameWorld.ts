export type StrategyFamily = "IMBA" | "RSI";
export type Phenotype = "grass" | "flower" | "shrub" | "cactus" | "orchid";

export interface WorldPlant {
  id: string;
  familyId: string;
  strategy: StrategyFamily;
  phenotype: Phenotype;
  genome: Record<string, number>;
  profitPct: number;
  biomass: number;
  vitality: number;
  rootDepth: number;
  rootSpread: number;
  leafArea: number;
  evidenceDays: number;
  ageDays: number;
}

export interface GenomeAxis {
  id: string;
  label: string;
  min: number;
  max: number;
  step: number;
}

export interface ParameterMeadow {
  id: string;
  label: string;
  strategy: StrategyFamily;
  xAxis: GenomeAxis;
  yAxis: GenomeAxis;
}

export const PARAMETER_MEADOWS: ParameterMeadow[] = [
  {
    id: "imba-sensitivity-ttp", label: "IMBA · Sensitivität × TTP", strategy: "IMBA",
    xAxis: { id: "sensitivity", label: "IMBA-Sensitivität", min: 12, max: 25, step: 1 },
    yAxis: { id: "ttp_pct", label: "TTP %", min: 0.25, max: 1.65, step: 0.05 },
  },
  {
    id: "rsi-period-stop", label: "RSI · Periode × Stop-Abstand", strategy: "RSI",
    xAxis: { id: "rsi_period", label: "RSI-Periode", min: 5, max: 40, step: 1 },
    yAxis: { id: "stop_distance_pct", label: "Stop-Abstand %", min: 0.5, max: 8, step: 0.25 },
  },
];

// Explicit demo fixtures keep this screen useful before an empirical backend exists.
export const DEMO_PLANTS: WorldPlant[] = Array.from({ length: 36 }, (_, index) => {
  const strategy: StrategyFamily = index % 3 === 0 ? "RSI" : "IMBA";
  const n = index + 1;
  const genome: Record<string, number> = strategy === "IMBA"
    ? { sensitivity: 12 + (index * 7) % 14, ttp_pct: 0.25 + ((index * 11) % 29) * 0.05 }
    : { rsi_period: 5 + (index * 9) % 36, stop_distance_pct: 0.5 + ((index * 13) % 31) * 0.25 };
  return {
    id: `demo-${n.toString().padStart(3, "0")}`,
    familyId: `family-${1 + (index % 3)}`,
    strategy,
    phenotype: (["grass", "flower", "shrub", "cactus", "orchid"] as const)[index % 5],
    genome,
    profitPct: ((index * 17) % 41 - 14) / 10,
    biomass: 0.2 + ((index * 19) % 80) / 100,
    vitality: 0.38 + ((index * 23) % 58) / 100,
    rootDepth: 0.2 + ((index * 5) % 80) / 100,
    rootSpread: 0.25 + ((index * 11) % 75) / 100,
    leafArea: 0.2 + ((index * 17) % 70) / 100,
    evidenceDays: (index * 37) % 430,
    ageDays: 12 + (index * 29) % 900,
  };
});

export interface UnderworldCell {
  id: string;
  count: number;
  medianScore: number;
  evidenceDays: number;
}

export const DEMO_UNDERWORLD: UnderworldCell[] = Array.from({ length: 24 }, (_, index) => ({
  id: `research-cluster-${index + 1}`,
  count: 12_400 + (index * 7_913) % 88_000,
  medianScore: ((index * 19) % 61 - 22) / 10,
  evidenceDays: 20 + (index * 31) % 500,
}));

export function selectMeadowPlants(plants: WorldPlant[], meadow: ParameterMeadow): WorldPlant[] {
  return plants.filter((plant) => plant.strategy === meadow.strategy);
}

export function normalizeAxis(value: number | undefined, axis: GenomeAxis): number | null {
  if (value === undefined || !Number.isFinite(value) || axis.max <= axis.min) return null;
  return Math.max(0, Math.min(1, (value - axis.min) / (axis.max - axis.min)));
}

export function groundwaterSurface(depth: number, minDepth = 0.18): number {
  const safeDepth = Math.max(0, Math.min(1, depth));
  const safeMinimum = Math.max(0, Math.min(0.95, minDepth));
  return safeMinimum + (1 - safeMinimum) * (1 - Math.exp(-3.2 * safeDepth));
}

export function missingAxisGenomes(plants: WorldPlant[], axis: GenomeAxis): string[] {
  return plants.filter((plant) => normalizeAxis(plant.genome[axis.id], axis) === null).map((plant) => plant.id);
}

export function getUnassignedPlants(plants: WorldPlant[], meadows = PARAMETER_MEADOWS): WorldPlant[] {
  const assigned = new Set(meadows.map((meadow) => meadow.strategy));
  return plants.filter((plant) => !assigned.has(plant.strategy));
}
