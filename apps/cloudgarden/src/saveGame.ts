import { assertSimulationInvariants, type SimulationWorld } from "./simulation.ts";
import { readSyntheticReplayWorkerResult } from "./replayWorkerProtocol.ts";
import type { PaperReplayReport } from "./paperReplay.ts";
import type { SyntheticReplayEvidence } from "./replayEvidence.ts";
import type { StrategySignalReport, TradingSignalGenome } from "./strategySignals.ts";

export const CLOUDGARDEN_SAVE_KEY = "cloudgarden.save.v1";
export const CLOUDGARDEN_SAVE_SCHEMA_VERSION = 2 as const;

export type GeneAxis = "root" | "leaf" | "water" | "sun";
export type GardenView = "meadow" | "understory";
export type StrategyControls = Pick<TradingSignalGenome, "lookbackBars" | "entryThresholdBps" | "exitThresholdBps">;
export type SavedExperimentInput = Readonly<{ seeds: readonly string[]; days: number }>;
export type SavedPaperReplayConfig = Readonly<{
  plantId: string;
  days: 7 | 30 | 90;
  tradingTraits: StrategyControls;
}>;
export type SavedPaperReplay = Readonly<{
  execution: PaperReplayReport;
  strategy: StrategySignalReport;
  genome: TradingSignalGenome;
  evidence: SyntheticReplayEvidence;
}>;

export type CloudgardenSave = Readonly<{
  schemaVersion: typeof CLOUDGARDEN_SAVE_SCHEMA_VERSION;
  world: SimulationWorld;
  selectedId: string;
  view: GardenView;
  axisX: GeneAxis;
  axisY: GeneAxis;
  experimentInput: SavedExperimentInput | null;
  paperReplay: SavedPaperReplay | null;
  paperReplayConfig: SavedPaperReplayConfig | null;
}>;

export type StorageLike = Pick<Storage, "getItem" | "setItem">;

export type LoadSaveResult = Readonly<{
  save: CloudgardenSave | null;
  status: "empty" | "restored" | "invalid" | "unavailable";
}>;

/** Read and validate a versioned browser-local snapshot before restoring it. */
export function loadCloudgardenSave(storage: StorageLike): LoadSaveResult {
  let raw: string | null;
  try {
    raw = storage.getItem(CLOUDGARDEN_SAVE_KEY);
  } catch {
    return { save: null, status: "unavailable" };
  }
  if (raw === null) return { save: null, status: "empty" };
  try {
    const parsed: unknown = migrateCloudgardenSave(JSON.parse(raw));
    if (!isCloudgardenSave(parsed)) return { save: null, status: "invalid" };
    return { save: parsed, status: "restored" };
  } catch {
    return { save: null, status: "invalid" };
  }
}

/** Persist only the local synthetic world; no server or account is contacted. */
export function saveCloudgardenSave(storage: StorageLike, save: CloudgardenSave): void {
  if (!isCloudgardenSave(save)) throw new RangeError("Cloudgarden snapshot does not match the versioned save contract.");
  storage.setItem(CLOUDGARDEN_SAVE_KEY, JSON.stringify(save));
}

/** Upgrade the first playable save format without discarding a local garden or replay. */
function migrateCloudgardenSave(value: unknown): unknown {
  const save = record(value);
  if (!save || save.schemaVersion === CLOUDGARDEN_SAVE_SCHEMA_VERSION) return value;
  if (save.schemaVersion !== 1) return value;
  const world = record(save.world);
  if (!world || !Array.isArray(world.plants)) return value;
  const plants = world.plants.map((item) => {
    const plant = record(item);
    if (!plant) return item;
    const genes = record(plant.genes);
    const existing = record(plant.tradingTraits);
    if (existing) return item;
    if (!genes || ![genes.root, genes.leaf, genes.water, genes.sun].every((trait) => typeof trait === "number" && Number.isFinite(trait))) return item;
    const lookbackBars = clamp(Math.round(3 + Number(genes.root) * 12), 1, 120);
    const entryThresholdBps = clamp(Math.round(20 + Number(genes.sun) * 100), 1, 3_000);
    const exitThresholdBps = Math.min(entryThresholdBps, Math.round(entryThresholdBps * Number(genes.water) * 0.25));
    return { ...plant, tradingTraits: { lookbackBars, entryThresholdBps, exitThresholdBps } };
  });
  const legacyConfig = record(save.paperReplayConfig);
  const paperReplayConfig = legacyConfig
    ? { ...legacyConfig, tradingTraits: legacyConfig.tradingTraits ?? legacyConfig.controls }
    : save.paperReplayConfig;
  return {
    ...save,
    schemaVersion: CLOUDGARDEN_SAVE_SCHEMA_VERSION,
    world: { ...world, plants },
    paperReplayConfig,
  };
}

function isCloudgardenSave(value: unknown): value is CloudgardenSave {
  const save = record(value);
  if (!save || save.schemaVersion !== CLOUDGARDEN_SAVE_SCHEMA_VERSION || typeof save.selectedId !== "string"
    || (save.view !== "meadow" && save.view !== "understory")
    || !isGeneAxis(save.axisX) || !isGeneAxis(save.axisY)) return false;
  const world = record(save.world);
  try {
    if (!world) return false;
    assertSimulationInvariants(world as unknown as SimulationWorld);
  } catch {
    return false;
  }
  const simulation = world as unknown as SimulationWorld;
  if (!simulation.plants.some((plant) => plant.id === save.selectedId)
    || !validExperimentInput(save.experimentInput)
    || !validPaperReplay(save.paperReplay, save.paperReplayConfig, simulation)) return false;
  return true;
}

function validExperimentInput(value: unknown): value is SavedExperimentInput | null {
  if (value === null) return true;
  const input = record(value);
  return !!input && Array.isArray(input.seeds) && input.seeds.length > 0 && input.seeds.length <= 4
    && input.seeds.every((seed) => typeof seed === "string" && seed.trim().length > 0)
    && new Set(input.seeds).size === input.seeds.length
    && Number.isInteger(input.days) && Number(input.days) >= 0 && Number(input.days) <= 140;
}

function validPaperReplay(
  value: unknown,
  configValue: unknown,
  world: SimulationWorld,
): value is SavedPaperReplay | null {
  if (value === null || configValue === null) return value === null && configValue === null;
  const replay = record(value);
  const config = record(configValue);
  const tradingTraits = record(config?.tradingTraits);
  const execution = record(replay?.execution);
  const dataset = record(execution?.dataset);
  if (!replay || !config || !tradingTraits || !execution || !dataset
    || typeof config.plantId !== "string" || !world.plants.some((plant) => plant.id === config.plantId)
    || (config.days !== 7 && config.days !== 30 && config.days !== 90)
    || !Number.isInteger(tradingTraits.lookbackBars) || Number(tradingTraits.lookbackBars) < 1 || Number(tradingTraits.lookbackBars) > 120
    || !Number.isInteger(tradingTraits.entryThresholdBps) || Number(tradingTraits.entryThresholdBps) < 1 || Number(tradingTraits.entryThresholdBps) > 3_000
    || !Number.isInteger(tradingTraits.exitThresholdBps) || Number(tradingTraits.exitThresholdBps) < 0
    || Number(tradingTraits.exitThresholdBps) > Number(tradingTraits.entryThresholdBps)
    || typeof dataset.start !== "string" || execution.days !== config.days) return false;
  try {
    const checked = readSyntheticReplayWorkerResult(
      { id: "saved-cloudgarden-replay", result: value },
      "saved-cloudgarden-replay",
      config.days,
      dataset.start,
    );
    return checked.genome.strategyId === `${config.plantId}:synthetic-momentum-v1`
      && checked.genome.lookbackBars === tradingTraits.lookbackBars
      && checked.genome.entryThresholdBps === tradingTraits.entryThresholdBps
      && checked.genome.exitThresholdBps === tradingTraits.exitThresholdBps;
  } catch {
    return false;
  }
}

function isGeneAxis(value: unknown): value is GeneAxis {
  return value === "root" || value === "leaf" || value === "water" || value === "sun";
}

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
