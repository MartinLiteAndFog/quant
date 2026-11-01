export type Genes = {
  tempo: number;
  canopy: number;
  rootDepth: number;
  rootSpread: number;
};

export type Plant = {
  id: string;
  name: string;
  family: string;
  x: number;
  genes: Genes;
  biomass: number;
  vigor: number;
  fitness: number;
  evidenceDays: number;
};

export type World = {
  seed: string;
  day: number;
  waterline: number;
  sunlight: number;
  plants: Plant[];
};

const clamp = (value: number, min: number, max: number) =>
  Math.max(min, Math.min(max, value));

const STARTING_PLANTS: Plant[] = [
  { id: "aster-04", name: "Aster 04", family: "Aster", x: 13, genes: { tempo: 0.72, canopy: 0.48, rootDepth: 0.34, rootSpread: 0.46 }, biomass: 42, vigor: 77, fitness: 62, evidenceDays: 42 },
  { id: "fern-12", name: "Cloud fern", family: "Fern", x: 30, genes: { tempo: 0.42, canopy: 0.78, rootDepth: 0.28, rootSpread: 0.58 }, biomass: 63, vigor: 84, fitness: 71, evidenceDays: 64 },
  { id: "reed-02", name: "Blue reed", family: "Reed", x: 48, genes: { tempo: 0.61, canopy: 0.34, rootDepth: 0.82, rootSpread: 0.43 }, biomass: 54, vigor: 69, fitness: 58, evidenceDays: 51 },
  { id: "clover-08", name: "Clover 08", family: "Clover", x: 66, genes: { tempo: 0.56, canopy: 0.67, rootDepth: 0.51, rootSpread: 0.73 }, biomass: 49, vigor: 73, fitness: 66, evidenceDays: 36 },
  { id: "sage-03", name: "Stone sage", family: "Sage", x: 84, genes: { tempo: 0.29, canopy: 0.32, rootDepth: 0.77, rootSpread: 0.31 }, biomass: 35, vigor: 81, fitness: 74, evidenceDays: 81 },
];

export function createWorld(): World {
  return {
    seed: "CLOUD-071",
    day: 43,
    waterline: 0.54,
    sunlight: 0.68,
    plants: STARTING_PLANTS.map((plant) => ({ ...plant, genes: { ...plant.genes } })),
  };
}

export function advanceDay(world: World): World {
  const nextDay = world.day + 1;
  const sunlight = clamp(0.58 + Math.sin(nextDay * 0.17) * 0.2, 0.25, 0.9);
  const waterline = clamp(0.53 + Math.sin(nextDay * 0.11 + 1.4) * 0.08, 0.36, 0.72);

  const plants = world.plants.map((plant, index) => {
    const shade = world.plants.reduce((sum, neighbor) => {
      if (neighbor.id === plant.id) return sum;
      const distance = Math.abs(neighbor.x - plant.x) / 100;
      return sum + Math.max(0, 0.16 - distance) * neighbor.genes.canopy;
    }, 0);
    const rootAccess = clamp(0.3 + plant.genes.rootDepth * waterline + plant.genes.rootSpread * 0.18, 0.2, 1);
    const lightAccess = clamp(sunlight - shade * 0.45 + plant.genes.canopy * 0.16, 0.16, 1);
    const balance = 1 - Math.abs(plant.genes.tempo - 0.55) * 0.35;
    const growth = (rootAccess * lightAccess * balance - 0.34) * 1.8;
    const vigorDelta = (rootAccess * 0.45 + lightAccess * 0.55 - 0.56) * 3.2;
    const seasonalPulse = Math.sin(nextDay * 0.13 + index) * 0.35;

    return {
      ...plant,
      biomass: Math.round(clamp(plant.biomass + growth + seasonalPulse, 8, 100) * 10) / 10,
      vigor: Math.round(clamp(plant.vigor + vigorDelta, 18, 98) * 10) / 10,
      fitness: Math.round(clamp(plant.fitness + growth * 0.55 + seasonalPulse * 0.25, 10, 95) * 10) / 10,
      evidenceDays: plant.evidenceDays + 1,
    };
  });

  return { ...world, day: nextDay, sunlight, waterline, plants };
}

export function advanceDays(world: World, count: number): World {
  if (!Number.isInteger(count) || count < 0 || count > 365) {
    throw new RangeError("Cloudgarden advances in whole days, up to 365 at a time.");
  }
  let current = world;
  for (let day = 0; day < count; day += 1) current = advanceDay(current);
  return current;
}

export function phenotype(genes: Genes): "Grass" | "Fern" | "Reed" | "Shrub" {
  if (genes.rootDepth >= 0.72 && genes.canopy <= 0.38) return "Reed";
  if (genes.canopy >= 0.72) return "Fern";
  if (genes.rootSpread >= 0.68) return "Shrub";
  return "Grass";
}

export function worldFingerprint(world: World): string {
  return JSON.stringify(world);
}
