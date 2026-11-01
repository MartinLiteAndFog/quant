/** Configurable, typed gene library and five bee crossover algorithms.
 * Pure SIMULATION: a gene is a schema/parameter, NOT executable Pine code.
 * Required genes (the two meadow axes) cannot be deleted.
 */
export const GENE_SCHEMA_VERSION = 2;
export const DEFAULT_COMPLEXITY_PENALTY_WEIGHT = 0;
export interface GenePolicy {
  gene_id: string;
  kind: "immutable_safety_policy" | "required_bounded_financial" | "optional_strategy_gene";
  minimum: number | null;
  maximum: number | null;
}
export type GeneKind = "number" | "switch";
export type GeneValue = number | boolean;
export type Genome = Record<string, GeneValue>;
export interface GeneDefinition {
  id: string;
  label: string;
  group: string;
  kind: GeneKind;
  min?: number;
  max?: number;
  step?: number;
  defaultValue: GeneValue;
  required: boolean;
  schemaVersion?: 2;
  policy?: GenePolicy;
}
export interface BeeBreed {
  id: "mixer" | "mosaic" | "explorer" | "collector" | "pruner";
  label: string;
  description: string;
}
export const BEE_BREEDS: BeeBreed[] = [
  { id: "mixer", label: "Mischerin", description: "Mittelt vorhandene Zahlen-Gene beider Eltern; übernimmt seltene Gene und verliert manchmal optionale." },
  { id: "mosaic", label: "Mosaikbiene", description: "Wählt für jedes Gen zufällig einen der beiden Elternwerte." },
  { id: "explorer", label: "Entdeckerin", description: "Kreuzung mit breiter Mutation für neue Parameterregionen." },
  { id: "collector", label: "Sammlerin", description: "Sammelt auch Gene, die nur bei einer Elternpflanze vorkommen." },
  { id: "pruner", label: "Gärtnerin", description: "Kreuzt erfolgreich, entfernt aber 10 % optionaler Gene und hält Genome schlank." },
];
export const DEFAULT_GENES: GeneDefinition[] = [
  { id:"protective_stop", label:"Schutz-Stop verpflichtend", group:"Sicherheit", kind:"switch", defaultValue:true, required:true, schemaVersion:2, policy:{gene_id:"protective_stop",kind:"immutable_safety_policy",minimum:null,maximum:null} },
  { id:"max_leverage", label:"Maximaler Hebel", group:"Finanzen", kind:"number", min:1,max:3,step:0.1,defaultValue:1,required:true,schemaVersion:2,policy:{gene_id:"max_leverage",kind:"required_bounded_financial",minimum:1,maximum:3} },
  { id:"max_margin_fraction", label:"Maximaler Marginanteil", group:"Finanzen", kind:"number", min:0.01,max:0.9,step:0.01,defaultValue:0.5,required:true,schemaVersion:2,policy:{gene_id:"max_margin_fraction",kind:"required_bounded_financial",minimum:0.01,maximum:0.9} },
  { id:"reproduction_budget", label:"Reproduktionsbudget", group:"Finanzen", kind:"number", min:0.08,max:0.15,step:0.01,defaultValue:0.1,required:true,schemaVersion:2,policy:{gene_id:"reproduction_budget",kind:"required_bounded_financial",minimum:0.08,maximum:0.15} },
  { id:"sensitivity", label:"IMBA Sensitivität", group:"Signal", kind:"number", min:12, max:25, step:0.1, defaultValue:17.2, required:true },
  { id:"ttp", label:"TTP Abstand (%)", group:"Exit", kind:"number", min:0.25, max:1.65, step:0.05, defaultValue:0.65, required:true },
  { id:"stop_loss_pct", label:"Optionales Stop-Tuning (%)", group:"Risiko", kind:"number", min:0.5, max:5, step:0.1, defaultValue:1.8, required:false },
  { id:"rsi_length", label:"RSI Periode", group:"Signal", kind:"number", min:5, max:50, step:1, defaultValue:14, required:false },
  { id:"rsi_entry", label:"RSI Einstiegsschwelle", group:"Signal", kind:"number", min:10, max:50, step:1, defaultValue:30, required:false },
  { id:"adx_gate", label:"ADX Gate", group:"Regime", kind:"switch", defaultValue:false, required:false },
  { id:"adx_length", label:"ADX Periode", group:"Regime", kind:"number", min:5, max:40, step:1, defaultValue:14, required:false },
];
/** Legacy required axes remain required for the existing meadow UI. */
export function policyFor(gene: GeneDefinition): GenePolicy {
  return gene.policy ?? {gene_id:gene.id, kind:gene.required ? "required_bounded_financial" : "optional_strategy_gene", minimum:gene.min ?? null, maximum:gene.max ?? null};
}
export function migrateLibrary(input: unknown): GeneDefinition[] {
  if (!Array.isArray(input)) throw new Error("Ungültige Bibliothek");
  const migrated = input.map(g => ({...g}));
  for (const gene of DEFAULT_GENES.filter(g => g.policy)) {
    if (!migrated.some(g => g.id === gene.id)) migrated.push({...gene});
  }
  return validateLibrary(migrated);
}
const ID_PATTERN = /^[a-z][a-z0-9_]{1,39}$/;
export const GENE_LIBRARY_KEY = "fleet-evolution-gene-library-v1";
export function validateLibrary(input: unknown): GeneDefinition[] {
  if (!Array.isArray(input) || input.length > 2000) throw new Error("Ungültige Bibliothek oder zu viele Gene");
  const seen = new Set<string>();
  const output: GeneDefinition[] = [];
  for (const raw of input) {
    if (!raw || typeof raw !== "object") throw new Error("Ungültiges Gen");
    const g = raw as GeneDefinition;
    if (!ID_PATTERN.test(g.id) || seen.has(g.id)) throw new Error("ID ungültig oder doppelt: " + g.id);
    if (typeof g.label !== "string" || !g.label.trim() || g.label.length > 100 ||
        typeof g.group !== "string" || g.group.length > 50 ||
        typeof g.required !== "boolean" || !["number", "switch"].includes(g.kind))
      throw new Error("Ungültige Genbeschreibung: " + g.id);
    if (g.kind === "number") {
      if (![g.min, g.max, g.step, g.defaultValue].every(v => typeof v === "number" && Number.isFinite(v)) ||
          g.min! >= g.max! || g.step! <= 0 ||
          (g.defaultValue as number) < g.min! || (g.defaultValue as number) > g.max!)
        throw new Error("Ungültiger Wertebereich für " + g.id);
    } else if (typeof g.defaultValue !== "boolean") throw new Error("Schaltgen muss Boolean sein: " + g.id);
    if (g.schemaVersion !== undefined && g.schemaVersion !== GENE_SCHEMA_VERSION) throw new Error("Unbekannte Genversion");
    const policy = policyFor(g);
    if (policy.gene_id !== g.id || !["immutable_safety_policy", "required_bounded_financial", "optional_strategy_gene"].includes(policy.kind) ||
        (policy.kind !== "optional_strategy_gene" && !g.required) ||
        (g.kind === "number" && (policy.minimum !== g.min || policy.maximum !== g.max)) ||
        (g.kind === "switch" && (policy.minimum !== null || policy.maximum !== null)))
      throw new Error("Ungültige Genpolicy: " + g.id);
    seen.add(g.id); output.push({...g});
  }
  for (const def of DEFAULT_GENES.filter(g => g.required)) {
    const actual = output.find(g => g.id === def.id);
    if (!actual || JSON.stringify(actual) !== JSON.stringify(def))
      throw new Error("Pflichtgen darf nicht gelöscht oder verändert werden: " + def.id);
  }
  return output;
}
export function loadGeneLibrary(): GeneDefinition[] {
  try {
    const text = localStorage.getItem(GENE_LIBRARY_KEY);
    if (!text) return DEFAULT_GENES.map(g => ({...g}));
    return migrateLibrary(JSON.parse(text));
  } catch {
    return DEFAULT_GENES.map(g => ({...g}));
  }
}
export function normalizeGene(g: GeneDefinition, value: GeneValue): GeneValue {
  if (policyFor(g).kind === "immutable_safety_policy" && value !== g.defaultValue)
    throw new Error("Sicherheitsgen darf nicht deaktiviert werden: " + g.id);
  if (g.kind === "switch") {
    if (typeof value !== "boolean") throw new Error("Ungültiger Genwert: " + g.id);
    return value;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error("Ungültiger Genwert: " + g.id);
  const x = value;
  const steps = Math.round((x - g.min!) / g.step!);
  const quantized = g.min! + steps * g.step!;
  return Number(Math.min(g.max!, Math.max(g.min!, quantized)).toFixed(8));
}
export function normalizeGenome(genome: Genome, library: GeneDefinition[]): Genome {
  validateLibrary(library);
  for (const key of Object.keys(genome)) if (!library.some(g => g.id === key)) throw new Error("Unbekanntes Gen: " + key);
  const out: Genome = {};
  for (const g of library) {
    if (g.required || Object.prototype.hasOwnProperty.call(genome, g.id))
      out[g.id] = normalizeGene(g, Object.hasOwn(genome, g.id) ? genome[g.id] : g.defaultValue);
  }
  return out;
}
export function defaultGenome(library: GeneDefinition[], sensitivity: number, ttp: number): Genome {
  return normalizeGenome({sensitivity, ttp, stop_loss_pct:1.8}, library);
}
export function geneCount(genome: Genome): number { return Object.keys(genome).length; }
export function addGene(library: GeneDefinition[], gene: GeneDefinition): GeneDefinition[] {
  return validateLibrary([...library, gene]);
}

/** Numeric gene mixing/creation/deletion; all algorithm choices are deterministic
 * given the provided RNG. No arbitrary executable code is ever interpreted.
 */
export function breedGenomes(
  leftRaw: Genome, rightRaw: Genome, library: GeneDefinition[],
  breed: BeeBreed["id"], rng: () => number,
): Genome {
  if (!BEE_BREEDS.some(b => b.id === breed)) throw new Error("Unbekannte Bienenart");
  const source = rng;
  rng = () => { const value = source(); if (!Number.isFinite(value) || value < 0 || value >= 1) throw new Error("Ungültiger RNG-Wert"); return value; };
  const left = normalizeGenome(leftRaw, library);
  const right = normalizeGenome(rightRaw, library);
  const result: Genome = {};
  for (const gene of library) {
    if (policyFor(gene).kind === "immutable_safety_policy") { result[gene.id] = gene.defaultValue; continue; }
    const hasL = Object.hasOwn(left, gene.id);
    const hasR = Object.hasOwn(right, gene.id);
    const a = left[gene.id], b = right[gene.id];
    let includes = gene.required || (hasL && hasR) ||
      ((hasL || hasR) && rng() < (breed === "collector" ? 0.95 : 0.65));
    // Missing on both parents: collectors can introduce genes from the library.
    if (!includes && !hasL && !hasR && breed === "collector" && rng() < 0.10)
      includes = true;
    if (!gene.required && includes && rng() < (breed === "pruner" ? 0.10 : 0.05))
      includes = false;
    if (!includes) continue;
    let value: GeneValue = hasL ? a : hasR ? b : gene.defaultValue;
    if (gene.kind === "number") {
      const av = Number(hasL ? a : hasR ? b : gene.defaultValue);
      const bv = Number(hasR ? b : hasL ? a : gene.defaultValue);
      if (breed === "mixer" || breed === "collector" || breed === "pruner")
        value = (av + bv) / 2;
      else if (breed === "mosaic")
        value = rng() < 0.5 ? av : bv;
      else {
        const span = gene.max! - gene.min!;
        value = (av + bv) / 2 + (rng() - 0.5) * span * 0.22;
      }
      if (breed !== "explorer" && rng() < 0.08)
        value = Number(value) + (rng() - 0.5) * (gene.max! - gene.min!) * 0.05;
    } else if (hasL && hasR) {
      value = rng() < 0.5 ? a : b;
      if (breed === "explorer" && rng() < 0.05) value = !Boolean(value);
    }
    result[gene.id] = normalizeGene(gene, value);
  }
  return normalizeGenome(result, library);
}

/** Profit modifies plant biomass gradually; never kills a plant on 1–2 losses.
 * Clip monthly growth to +36% and shrinkage to -25%, e.g. 36% net
 * monthly return -> x1.36 biomass (prior to the visual logarithmic scale).
 */
export function growBiomass(previous: number, netReturn: number): number {
  if (!Number.isFinite(previous) || previous < 0 || !Number.isFinite(netReturn))
    throw new Error("Ungültiges Biomassenwachstum");
  const monthly = Math.max(-0.25, Math.min(0.36, netReturn));
  return Math.min(1e9, Math.max(0.04, previous * (1 + monthly)));
}
export function visualPlantHeight(biomass: number): number {
  return Math.log1p(Math.max(0, biomass)) / Math.log(2);
}
