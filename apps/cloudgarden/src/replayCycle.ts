import { createSyntheticMarketRange, runPaperReplay, type PaperReplayReport } from "./paperReplay.ts";
import { buildSyntheticReplayEvidence, type SyntheticReplayEvidence } from "./replayEvidence.ts";
import { generateSyntheticMomentumSignals, type StrategySignalReport, type TradingSignalGenome } from "./strategySignals.ts";
import type { TradingTraits } from "./simulation.ts";

export type SyntheticReplayCycleInput = Readonly<{
  days: number;
  start: string;
  gardenSeed: string;
  weekIndex: number;
  plantId: string;
  allocationUsdt: number;
  tradingTraits: TradingTraits;
}>;

export type SyntheticReplayCycleResult = Readonly<{
  execution: PaperReplayReport;
  strategy: StrategySignalReport;
  genome: TradingSignalGenome;
  evidence: SyntheticReplayEvidence;
}>;

export type SyntheticReplayWorkerResponse =
  | Readonly<{ id: string; result: SyntheticReplayCycleResult }>
  | Readonly<{ id: string; error: string }>;

function quantityForPlantAllocation(allocationUsdt: number): string {
  const quantity = Math.floor((allocationUsdt * 0.5 / 100) * 100_000_000) / 100_000_000;
  return quantity.toFixed(8).replace(/\.?0+$/, "");
}

/** Run the full deterministic, synthetic-only signal → execution → evidence cycle. */
export async function runSyntheticReplayCycle(input: SyntheticReplayCycleInput): Promise<SyntheticReplayCycleResult> {
  if (!input || typeof input.plantId !== "string" || !input.plantId.trim()) {
    throw new RangeError("A non-empty plant ID is required.");
  }
  if (!Number.isFinite(input.allocationUsdt) || input.allocationUsdt <= 0) {
    throw new RangeError("Plant allocation must be a positive finite USDT amount.");
  }

  const allocation = input.allocationUsdt.toFixed(4);
  const dataset = await createSyntheticMarketRange(input.days, input.start, input.gardenSeed, input.weekIndex);
  const genome: TradingSignalGenome = {
    schemaVersion: 1,
    strategyId: `${input.plantId}:synthetic-momentum-v1`,
    ...input.tradingTraits,
    targetQuantity: quantityForPlantAllocation(input.allocationUsdt),
  };
  const strategy = generateSyntheticMomentumSignals(dataset, genome);
  const execution = runPaperReplay({
    dataset,
    signals: strategy.signals,
    allocationLimit: allocation,
    startingCash: allocation,
  });
  const evidence = buildSyntheticReplayEvidence(strategy.report, execution);
  return { execution, strategy: strategy.report, genome, evidence };
}

/** Pure worker message handler, exported so success and error envelopes are testable. */
export async function handleSyntheticReplayWorkerMessage(message: unknown): Promise<SyntheticReplayWorkerResponse> {
  const record = typeof message === "object" && message !== null ? message as Record<string, unknown> : null;
  const id = typeof record?.id === "string" ? record.id : "";
  try {
    if (id.length === 0 || !record || typeof record.input !== "object" || record.input === null) {
      throw new RangeError("Worker message must include a non-empty id and replay input.");
    }
    const result = await runSyntheticReplayCycle(record.input as SyntheticReplayCycleInput);
    return { id, result };
  } catch (error) {
    return { id, error: error instanceof Error ? error.message : "Synthetic replay cycle failed." };
  }
}
