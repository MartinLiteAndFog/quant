/** Genetics of appearance, separate from trading/financial/safety genes.
 * Physiological preview traits must eventually come from ECO-06, not this demo.
 * All inputs are seeded and reproducible, with no Math.random side effects.
 */
export type Morphotype = "grass" | "flower" | "shrub" | "cactus";
export interface VisualDNA {
  readonly version: 1;
  readonly hue: number;               // degrees; not a market signal
  readonly saturation: number;        // 0..1
  readonly petalCount: number;        // 3..12, used only when blossoms exist
  readonly petalRoundness: number;    // 0..1
  readonly bloomTwist: number;        // 0..360
  readonly stemCurve: number;         // -1..1
  readonly ribbing: number;           // 0..1
  readonly spineDensity: number;      // 0..1
  readonly leafShape: number;         // 0..1
}
export interface FunctionalMorphology {
  readonly branching: number;         // functional sun access in ECO-06
  readonly leaf_count: number;        // *always* integer >= 1
  readonly leaf_size: number;
  readonly photosynthetic_rind: boolean;
  readonly rind_efficiency: number;   // separate from bark protection
  readonly bark_strength: number;
}
export type Rand = () => number;
const clamp = (x:number, lo:number, hi:number) => Math.max(lo,Math.min(hi,x));
const u = (rand:Rand) => clamp(rand(),0,1);
const n = (rand:Rand,min:number,max:number) => min+u(rand)*(max-min);
export const VISUAL_DNA_VERSION = 1;
export const APPEARANCE_MUTATION_RATE = 0.12;
export const VISUAL_PRESETS: readonly Morphotype[] = ["grass","flower","shrub","cactus"];

export function validateVisualDNA(d: VisualDNA): VisualDNA {
  if (d.version !== 1 || !Number.isFinite(d.hue) || d.hue<0 || d.hue>=360 ||
      !Number.isInteger(d.petalCount) || d.petalCount < 3 || d.petalCount > 12 ||
      !Number.isFinite(d.bloomTwist) || d.bloomTwist<0 || d.bloomTwist>=360 ||
      !Number.isFinite(d.stemCurve) || d.stemCurve < -1 || d.stemCurve > 1 ||
      [d.saturation,d.petalRoundness,d.ribbing,d.spineDensity,d.leafShape]
        .some(x=>!Number.isFinite(x) || x<0 || x>1)) throw new Error("Ungültige optische DNA");
  return d;
}
export function validateFunctionalMorphology(d: FunctionalMorphology): FunctionalMorphology {
  if (!Number.isInteger(d.leaf_count) || d.leaf_count < 1 || d.leaf_count > 64 ||
      [d.branching,d.leaf_size,d.rind_efficiency,d.bark_strength]
      .some(x=>!Number.isFinite(x) || x<0 || x>1) ||
      typeof d.photosynthetic_rind !== "boolean") throw new Error("Ungültige Physiologie-Gene");
  return d;
}
export function randomVisualDNA(rand:Rand, preset:Morphotype="flower"): VisualDNA {
  const hue=n(rand,0,359.99);
  return validateVisualDNA({
    version:1, hue, saturation:n(rand,.40,.92),
    petalCount:3+Math.floor(u(rand)*10),
    petalRoundness:u(rand), bloomTwist:n(rand,0,359.99),
    stemCurve:n(rand,-1,1), ribbing:u(rand),
    spineDensity:preset==="cactus"?n(rand,.35,1):n(rand,0,.20),
    leafShape:u(rand),
  });
}
export function randomFunctionalMorphology(rand:Rand,preset:Morphotype="flower"):FunctionalMorphology {
  const cactus=preset==="cactus";
  return validateFunctionalMorphology({
    branching:cactus?n(rand,.02,.18):preset==="shrub"?n(rand,.65,.95):n(rand,.12,.52),
    leaf_count:cactus?1:preset==="shrub"?20+Math.floor(u(rand)*40)
      :preset==="grass"?2+Math.floor(u(rand)*5):6+Math.floor(u(rand)*20),
    leaf_size:cactus?n(rand,0,.18):n(rand,.30,.95),
    photosynthetic_rind:cactus,
    rind_efficiency:cactus?n(rand,.3,.85):n(rand,0,.3),
    bark_strength:cactus?n(rand,.6,.98):preset==="shrub"?n(rand,.25,.8):n(rand,0,.3),
  });
}
/** Morphotype emerges from *functional* phenotype, not an on/off species gene. */
export function classifyMorphotype(p:FunctionalMorphology):Morphotype {
  validateFunctionalMorphology(p);
  if (p.leaf_count===1 && p.photosynthetic_rind) return "cactus";
  if (p.branching>=.62 && p.leaf_count>=18) return "shrub";
  if (p.leaf_count<=5 && p.branching<=.35) return "grass";
  return "flower";
}
function wrapHue(h:number):number { return (h%360+360)%360; }
function select(a:number,b:number,rand:Rand,mutation:number,lo:number,hi:number):number {
  const parent=u(rand)<.5?a:b;
  return clamp(parent+(u(rand)<APPEARANCE_MUTATION_RATE?n(rand,-mutation,mutation):0),lo,hi);
}
export function breedVisualDNA(mother:VisualDNA,father:VisualDNA,rand:Rand):VisualDNA {
  validateVisualDNA(mother);validateVisualDNA(father);
  const hueParent=u(rand)<.5?mother.hue:father.hue;
  const hue=wrapHue(hueParent+(u(rand)<APPEARANCE_MUTATION_RATE?n(rand,-34,34):0));
  return validateVisualDNA({
    version:1, hue,
    saturation:select(mother.saturation,father.saturation,rand,.1,0,1),
    petalCount:Math.round(select(mother.petalCount,father.petalCount,rand,2,3,12)),
    petalRoundness:select(mother.petalRoundness,father.petalRoundness,rand,.15,0,1),
    bloomTwist:wrapHue(select(mother.bloomTwist,father.bloomTwist,rand,22,0,359.99)),
    stemCurve:select(mother.stemCurve,father.stemCurve,rand,.2,-1,1),
    ribbing:select(mother.ribbing,father.ribbing,rand,.2,0,1),
    spineDensity:select(mother.spineDensity,father.spineDensity,rand,.2,0,1),
    leafShape:select(mother.leafShape,father.leafShape,rand,.2,0,1),
  });
}
export function breedFunctionalMorphology(
  mother:FunctionalMorphology,father:FunctionalMorphology,rand:Rand,
):FunctionalMorphology {
  validateFunctionalMorphology(mother);validateFunctionalMorphology(father);
  return validateFunctionalMorphology({
    branching:select(mother.branching,father.branching,rand,.12,0,1),
    leaf_count:Math.round(select(mother.leaf_count,father.leaf_count,rand,3,1,64)),
    leaf_size:select(mother.leaf_size,father.leaf_size,rand,.10,0,1),
    photosynthetic_rind:u(rand)<.5?mother.photosynthetic_rind:father.photosynthetic_rind,
    rind_efficiency:select(mother.rind_efficiency,father.rind_efficiency,rand,.10,0,1),
    bark_strength:select(mother.bark_strength,father.bark_strength,rand,.10,0,1),
  });
}
export function visualColor(d:VisualDNA):string {
  validateVisualDNA(d);
  return `hsl(${d.hue.toFixed(1)} ${(d.saturation*100).toFixed(0)}% 66%)`;
}
/** Visual-only photosynthesis preview; game notional comes from ECO-06. */
export function potentialLightArea(p:FunctionalMorphology,biomass:number):number {
  validateFunctionalMorphology(p);
  if (!Number.isFinite(biomass) || biomass<=0) throw new Error("Ungültige Biomasse");
  const size=biomass**(2/3);
  const foliage=size*p.leaf_size*(1+2*p.branching)*p.leaf_count/64;
  const stem=p.photosynthetic_rind && p.leaf_count===1
    ? size*p.rind_efficiency*.22:0;
  return Math.max(foliage,stem);
}
