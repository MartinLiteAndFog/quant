/** Configurable, typed gene library and five bee crossover algorithms.
 * Pure SIMULATION: a gene is a schema/parameter, NOT executable Pine code.
 * Required genes (the two meadow axes) cannot be deleted.
 */
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
  { id:"sensitivity", label:"IMBA Sensitivität", group:"Signal", kind:"number", min:12, max:25, step:0.1, defaultValue:17.2, required:true },
  { id:"ttp", label:"TTP Abstand (%)", group:"Exit", kind:"number", min:0.25, max:1.65, step:0.05, defaultValue:0.65, required:true },
  { id:"stop_loss_pct", label:"Stop Loss (%)", group:"Risiko", kind:"number", min:0.5, max:5, step:0.1, defaultValue:1.8, required:false },
  { id:"rsi_length", label:"RSI Periode", group:"Signal", kind:"number", min:5, max:50, step:1, defaultValue:14, required:false },
  { id:"rsi_entry", label:"RSI Einstiegsschwelle", group:"Signal", kind:"number", min:10, max:50, step:1, defaultValue:30, required:false },
  { id:"adx_gate", label:"ADX Gate", group:"Regime", kind:"switch", defaultValue:false, required:false },
  { id:"adx_length", label:"ADX Periode", group:"Regime", kind:"number", min:5, max:40, step:1, defaultValue:14, required:false },
];
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
    return validateLibrary(JSON.parse(text));
  } catch {
    return DEFAULT_GENES.map(g => ({...g}));
  }
}
export function normalizeGene(g: GeneDefinition, value: GeneValue): GeneValue {
  if (g.kind === "switch") return Boolean(value);
  const x = typeof value === "number" && Number.isFinite(value) ? value : Number(g.defaultValue);
  const steps = Math.round((x - g.min!) / g.step!);
  const quantized = g.min! + steps * g.step!;
  return Number(Math.min(g.max!, Math.max(g.min!, quantized)).toFixed(8));
}
export function normalizeGenome(genome: Genome, library: GeneDefinition[]): Genome {
  const out: Genome = {};
  for (const g of library) {
    if (g.required || Object.prototype.hasOwnProperty.call(genome, g.id))
      out[g.id] = normalizeGene(g, genome[g.id] ?? g.defaultValue);
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
  const left = normalizeGenome(leftRaw, library);
  const right = normalizeGenome(rightRaw, library);
  const result: Genome = {};
  for (const gene of library) {
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
