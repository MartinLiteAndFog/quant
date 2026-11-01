/** Versioned descriptions only. Execution is owned by the audited paper engine. */
import { DEFAULT_GENES, DEFAULT_COMPLEXITY_PENALTY_WEIGHT, normalizeGenome, policyFor, type GeneDefinition, type Genome } from "./geneLibrary";
export interface StrategyModule {
  id: "imba" | "ttp" | "protective-stop" | "rsi" | "adx";
  version: "1.0";
  inputs: readonly string[];
  allowedData: "closed_bars";
  implementation: "descriptor_only";
  riskContract: "global_caps_and_protective_stop";
  parity: "unverified";
}
export const STRATEGY_MODULES: readonly StrategyModule[] = [
  { id:"imba", inputs:["sensitivity"] },
  { id:"ttp", inputs:["ttp"] },
  { id:"protective-stop", inputs:["protective_stop", "stop_loss_pct"] },
  { id:"rsi", inputs:["rsi_length", "rsi_entry"] },
  { id:"adx", inputs:["adx_gate", "adx_length"] },
].map(module => ({...module, id:module.id as StrategyModule["id"], version:"1.0", allowedData:"closed_bars", implementation:"descriptor_only", riskContract:"global_caps_and_protective_stop", parity:"unverified"}));
export function resolveStrategyModule(id: string, version: string): StrategyModule {
  const module = STRATEGY_MODULES.find(m => m.id === id && m.version === version);
  if (!module) throw new Error("Unknown strategy module or version");
  return {...module, inputs:[...module.inputs]};
}
/** Optional tuning cannot remove the required protective stop. Global caps win. */
export function effectiveStopLossPct(genome: Genome, globalCap = 5): number {
  if (!Number.isFinite(globalCap) || globalCap <= 0) throw new Error("Invalid global stop cap");
  if (genome.protective_stop !== true) throw new Error("Protective stop required");
  const tuning = genome.stop_loss_pct ?? 1.8;
  if (typeof tuning !== "number" || !Number.isFinite(tuning) || tuning <= 0) throw new Error("Invalid stop tuning");
  return Math.min(tuning, globalCap);
}
/** Explicit demo migration to the ECO-01 wire contract; no executable source. */
export function toStrategyGenome(genomeId: string, genes: Genome, library: GeneDefinition[] = DEFAULT_GENES) {
  if (!genomeId.trim()) throw new Error("Genome ID required");
  return {
    genome_id:genomeId, version:2, strategy_code_version:"synthetic-demo-v1",
    genes:normalizeGenome(genes, library), policies:library.map(policyFor),
    parent_genome_ids:[] as string[], complexity_penalty_weight:DEFAULT_COMPLEXITY_PENALTY_WEIGHT,
  };
}
