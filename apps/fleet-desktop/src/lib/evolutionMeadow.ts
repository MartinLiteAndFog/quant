/** Evolution meadow: deterministic SIMULATION, never exchange or live PnL data.
 *
 * X = sensitivity, Y = TTP %, flower height = net monthly return.
 * Flower size / seed count = risk- and sample-adjusted fitness.
 * Seeds: stratified uniform angular distribution + area-uniform sqrt(U) radii.
 */
import {
  BEE_BREEDS, DEFAULT_GENES, breedGenomes, defaultGenome,
  growBiomass, normalizeGenome, type BeeBreed, type GeneDefinition, type Genome,
} from "./geneLibrary";

export const BOUNDS = {
  sensitivity: [12, 25] as const,
  ttp: [0.25, 1.65] as const,
};

export interface MeadowFlower {
  id: string;
  parentId: string | null;
  rootId: string;
  generation: number;
  sensitivity: number;
  ttp: number;
  roi: number; // synthetic cashflow-adjusted net return fraction, 0.12 = 12%
  drawdown: number; // synthetic DD as fraction
  trades: number;
  score: number; // display-only risk/credibility-adjusted fitness
  genome: Genome; // complete parameter record, including hidden genes
  biomass: number; // persistent visual organism size (NOT account equity)
  beeBreed?: BeeBreed["id"];
  fatherId?: string;
}

const ROOTS = [
  { id: "blaupunkt", x: 16.6, y: 0.48 },
  { id: "blaupunkt-p02", x: 19.6, y: 1.10 },
  { id: "blaupunkt-p03", x: 22.1, y: 0.68 },
];

export const ROOT_COLORS: Record<string, string> = {
  blaupunkt: "#85c99f",
  "blaupunkt-p02": "#e0b870",
  "blaupunkt-p03": "#aab6ed",
};

function seeded(seed: number): () => number {
  let x = seed >>> 0;
  return () => {
    x += 0x6d2b79f5;
    let t = x;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n));
}

export function syntheticMonth(x: number, y: number, rand: () => number) {
  const peakA = 0.40 * Math.exp(-((x - 17.6) ** 2 / 5.5 + (y - 0.66) ** 2 / 0.067));
  const peakB = 0.35 * Math.exp(-((x - 21.25) ** 2 / 3.0 + (y - 1.11) ** 2 / 0.076));
  const roi = clamp(peakA + peakB - 0.105 + (rand() - 0.5) * 0.095, -0.3, 0.65);
  const drawdown = clamp(0.035 + (1 - Math.max(0, roi) / 0.5) * 0.07 + rand() * 0.07, 0.02, 0.25);
  return { roi, drawdown };
}

export function scoreFlower(roi: number, drawdown: number, trades: number): number {
  const credibility = trades / (trades + 20);
  return (100 * roi * credibility) - 35 * drawdown;
}

function makeFlower(
  id: string, parentId: string | null, rootId: string,
  generation: number, sensitivity: number, ttp: number,
  trades: number, rand: () => number,
  genome: Genome, biomass: number, beeBreed?: BeeBreed["id"], fatherId?: string,
): MeadowFlower {
  const { roi, drawdown } = syntheticMonth(sensitivity, ttp, rand);
  return {
    id, parentId, rootId, generation, sensitivity, ttp, roi,
    drawdown, trades, score: scoreFlower(roi, drawdown, trades),
    genome, biomass, beeBreed, fatherId,
  };
}

export function createInitialMeadow(library: GeneDefinition[] = DEFAULT_GENES): MeadowFlower[] {
  const rand = seeded(71337);
  const plants: MeadowFlower[] = [];
  for (const root of ROOTS) {
    const rootFlower = makeFlower(
      root.id, null, root.id, 0, root.x, root.y, 60, rand,
      defaultGenome(library, root.x, root.y), 1.0,
    );
    plants.push(rootFlower);
    // Three manually configured daughters are initial placeholders in this
    // visual DEMO. Not proof of live TradingView configuration or funding.
    for (let k = 0; k < 3; k++) {
      const angle = 2 * Math.PI * k / 3 + (rand() - 0.5) * 0.25;
      const radius = 0.065 + rand() * 0.032;
      const sensitivity = clamp(root.x + Math.cos(angle) * radius * 13, ...BOUNDS.sensitivity);
      const ttp = clamp(root.y + Math.sin(angle) * radius * 1.4, ...BOUNDS.ttp);
      const id = root.id === "blaupunkt"
        ? "blaupunkt-d0" + (k + 1)
        : root.id + "-d0" + (k + 1);
      plants.push(makeFlower(
        id, root.id, root.id, 1, sensitivity, ttp,
        7 + Math.floor(rand() * 9), rand,
        defaultGenome(library, sensitivity, ttp), 0.28,
      ));
    }
  }
  return plants;
}

export function seedCount(flower: MeadowFlower): number {
  if (flower.trades < 15 || flower.score <= 3 || flower.roi < 0.05) return 0;
  return clamp(Math.round(5 + 10 * (flower.score - 3) / 20), 5, 15);
}

export function mutateMeadow(
  previous: MeadowFlower[], generation: number,
  library: GeneDefinition[] = DEFAULT_GENES,
  beeBreed: BeeBreed["id"] = BEE_BREEDS[0].id,
): MeadowFlower[] {
  // Existing plants ALWAYS keep aging, even when the visualization is at
  // its rendering limit. Only NEW seed generation is capped below.
  const rand = seeded(71337 + 811 * generation);
  const updated = previous.map(f => {
    const trades = f.trades + 15 + Math.floor(rand() * 22);
    const m = syntheticMonth(f.sensitivity, f.ttp, rand);
    return {
      ...f, ...m, trades, score: scoreFlower(m.roi, m.drawdown, trades),
      biomass: growBiomass(f.biomass, m.roi),
    };
  });
  const parents = updated.filter(f => seedCount(f) > 0)
    .sort((a, b) => b.score - a.score).slice(0, 10); // visual density, not exchange cap
  const next = [...updated];
  for (const flower of parents) {
    const potentialMates = parents.filter(p => p.id !== flower.id && p.score > 3);
    const n = seedCount(flower);
    const spread = clamp(0.05 + Math.max(0, flower.score - 3) * 0.0038, 0.05, 0.16);
    for (let k = 0; k < n; k++) {
      if (next.length >= 450) break;
      // Sector stratification avoids angular clustering. sqrt(U) yields
      // uniform density over the disk, unlike radius = U.
      const angle = 2 * Math.PI * (k + 0.5 + (rand() - 0.5) * 0.60) / n;
      const radius = spread * Math.sqrt(rand());
      const mate = potentialMates.length
        ? potentialMates[Math.floor(rand() * potentialMates.length)]
        : flower;
      const cross = breedGenomes(flower.genome, mate.genome, library, beeBreed, rand);
      // Seeds explore a sector around the selected parent. Other hidden
      // genes are inherited/crossed; visible axes are further localized.
      const x = clamp(
        Number(cross.sensitivity) + Math.cos(angle) * radius * 13,
        ...BOUNDS.sensitivity,
      );
      const y = clamp(
        Number(cross.ttp) + Math.sin(angle) * radius * 1.4,
        ...BOUNDS.ttp,
      );
      const genome = normalizeGenome({ ...cross, sensitivity:x, ttp:y }, library);
      const sx = Number(genome.sensitivity), ty = Number(genome.ttp);
      next.push(makeFlower(
        flower.id + "-g" + generation + "s" + k, flower.id,
        flower.rootId, generation, sx, ty, 3, rand,
        genome, 0.10, beeBreed, mate.id,
      ));
    }
  }
  return next;
}


/** DBSCAN-like neighborhoods of independently credible profitable strategies.
 * Centers are weighted by fitness, not by which parent first occupied them.
 */
export interface MeadowCluster {
  id: number;
  sensitivity: number;
  ttp: number;
  members: number;
  averageRoi: number;
}

export function findFitnessClusters(flowers: readonly MeadowFlower[]): MeadowCluster[] {
  const winners = flowers.filter(f => f.trades >= 15 && f.roi > 0.05 && f.score > 3);
  const xSpan = BOUNDS.sensitivity[1] - BOUNDS.sensitivity[0];
  const ySpan = BOUNDS.ttp[1] - BOUNDS.ttp[0];
  const radiusSq = 0.18 ** 2;
  const near = (i: number): number[] => winners.flatMap((other, j) => {
    const dx = (other.sensitivity - winners[i].sensitivity) / xSpan;
    const dy = (other.ttp - winners[i].ttp) / ySpan;
    return dx * dx + dy * dy <= radiusSq ? [j] : [];
  });
  const visited = new Set<number>();
  const assigned = new Set<number>();
  const clusters: MeadowCluster[] = [];

  for (let i = 0; i < winners.length; i++) {
    if (visited.has(i)) continue;
    visited.add(i);
    const neighbors = near(i);
    if (neighbors.length < 3) continue;
    const members = new Set<number>();
    const queue = [...neighbors];
    while (queue.length) {
      const j = queue.shift()!;
      if (assigned.has(j)) continue;
      members.add(j);
      assigned.add(j);
      if (!visited.has(j)) {
        visited.add(j);
        const expanded = near(j);
        if (expanded.length >= 3) queue.push(...expanded);
      }
    }
    const selected = [...members].map(index => winners[index]);
    if (selected.length < 3) continue;
    const weight = (f: MeadowFlower) => Math.max(0.1, f.score);
    const sumW = selected.reduce((acc, f) => acc + weight(f), 0);
    clusters.push({
      id: clusters.length + 1,
      sensitivity: selected.reduce((acc, f) => acc + f.sensitivity * weight(f), 0) / sumW,
      ttp: selected.reduce((acc, f) => acc + f.ttp * weight(f), 0) / sumW,
      members: selected.length,
      averageRoi: selected.reduce((acc, f) => acc + f.roi, 0) / selected.length,
    });
  }
  return clusters.sort((a, b) => b.members - a.members).slice(0, 5);
}
