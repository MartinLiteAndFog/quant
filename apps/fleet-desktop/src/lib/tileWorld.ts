/** ECO-14: pure, read-only multi-asset tile world. Never creates orders.
 * Each meadow tile is a true two-gene parameter plane, NOT an UMAP/PCA map.
 */
export type AssetClass = "crypto" | "fx" | "commodities" | "indices";
export type DataStatus = "synthetic" | "verified";
export type WorldEvidenceStatus = "synthetic" | "unverified" | "validated";
export interface WorldEvidence {
  readonly status: WorldEvidenceStatus;
  readonly validationIds?: readonly string[];
  readonly validatorVersion?: string;
  readonly validatedAt?: string;
}

export interface AssetUniverse {
  readonly id: string;
  readonly assetClass: AssetClass;
  readonly symbol: string;
  readonly label: string;
  readonly quoteCurrency: string;
  readonly dataSourceId: string;
  readonly dataStatus: DataStatus;
  readonly contractVersion: string;
}
export interface WorldRegion {
  readonly id: string;
  readonly assetClass: AssetClass;
  readonly label: string;
  readonly worldX: number;
  readonly worldY: number;
}
export interface GeneAxis {
  readonly geneId: string;
  readonly label: string;
  readonly unit: string;
  readonly min: number;
  readonly max: number;
  readonly step: number;
}
export interface MeadowTile {
  readonly id: string;
  readonly regionId: string;
  readonly assetId: string;
  readonly moduleId: string;
  readonly moduleVersion: string;
  readonly label: string;
  readonly axisX: GeneAxis;
  readonly axisY: GeneAxis;
  readonly gridX: number;
  readonly gridY: number;
  readonly version: number;
}
export interface StrategyPlacement {
  readonly strategyId: string;
  readonly assetId: string;
  readonly homeTileId: string | null;
  readonly moduleIds: readonly string[];
  readonly genome: Readonly<Record<string, number | boolean>>;
  readonly vitality: number;
  readonly synthetic: boolean;
}
export interface MyceliumKnowledgeEdge {
  readonly id: string;
  readonly sourceTileId: string;
  readonly targetTileId: string;
  readonly geneId: string;
  readonly researchId: string;
  readonly confidence: number;
  readonly outOfSampleVerified: boolean;
}
export interface TileWorld {
  readonly schemaVersion: "1.1";
  readonly synthetic: boolean;
  readonly evidence: WorldEvidence;
  readonly regions: readonly WorldRegion[];
  readonly assets: readonly AssetUniverse[];
  readonly tiles: readonly MeadowTile[];
  readonly plants: readonly StrategyPlacement[];
  readonly mycelium: readonly MyceliumKnowledgeEdge[];
}
export type ViewportFocus =
  | {readonly kind: "world"}
  | {readonly kind: "region"; readonly regionId: string}
  | {readonly kind: "tile"; readonly tileId: string};
export interface WorldViewportState {
  readonly focus: ViewportFocus;
  readonly zoom: number;
  readonly panX: number;
  readonly panY: number;
}
export interface TilePlantView {
  readonly strategyId: string;
  readonly x: number;
  readonly y: number;
  readonly home: boolean;
  readonly vitality: number;
  readonly synthetic: boolean;
}
export interface TileView {
  readonly tile: MeadowTile;
  readonly plants: readonly TilePlantView[];
  readonly homePopulation: number;
  readonly visiblePopulation: number;
}
export interface WorldStats {
  readonly population: number;
  readonly unassigned: readonly string[];
  readonly regionCount: number;
  readonly tileCount: number;
  readonly homePopulationByTile: Readonly<Record<string, number>>;
}
export interface KnowledgeProposal {
  readonly kind: "research-only";
  readonly sourceTileId: string;
  readonly targetTileId: string;
  readonly geneId: string;
  readonly researchId: string;
  readonly confidence: number;
  readonly moduleId: string;
  readonly moduleVersion: string;
  readonly parameter: {
    readonly geneId: string;
    readonly unit: string;
    readonly min: number;
    readonly max: number;
    readonly step: number;
  };
  readonly requiresOutOfSampleValidation: true;
  readonly fundingTransfer: false;
}

function must(condition: boolean, reason: string): asserts condition {
  if (!condition) throw new Error("Tile world: " + reason);
}
const unique = (items: readonly string[]) => new Set(items).size === items.length;
const finite = (x: number) => Number.isFinite(x);

function validateAxis(axis: GeneAxis): void {
  must(Boolean(axis.geneId && axis.label && axis.unit), "missing axis identity or unit");
  must([axis.min, axis.max, axis.step].every(finite), "non-finite axis");
  must(axis.min < axis.max && axis.step > 0, "invalid axis range");
}
function compatible(plant: StrategyPlacement, tile: MeadowTile): boolean {
  if (plant.assetId !== tile.assetId || !plant.moduleIds.includes(tile.moduleId)) return false;
  return [tile.axisX, tile.axisY].every(axis => {
    const value = plant.genome[axis.geneId];
    return typeof value === "number" && finite(value)
      && value >= axis.min - 1e-9 && value <= axis.max + 1e-9;
  });
}
export function validateTileWorld(world: TileWorld): TileWorld {
  must(world.schemaVersion === "1.1", "unsupported version");
  must(Boolean(world.evidence), "missing world evidence status");
  must(["synthetic", "unverified", "validated"].includes(world.evidence.status),
    "unknown world evidence status");
  must(world.synthetic === (world.evidence.status === "synthetic"),
    "synthetic flag and evidence status disagree");
  if (world.evidence.status === "validated") {
    must(!world.synthetic, "synthetic worlds cannot be marked validated");
    must(Boolean(world.evidence.validatorVersion && world.evidence.validatedAt),
      "validated worlds require validator version and timestamp");
    must((world.evidence.validationIds?.length ?? 0) > 0,
      "validated worlds require evidence IDs");
    must(unique(world.evidence.validationIds!), "duplicate world validation IDs");
    must(Number.isFinite(Date.parse(world.evidence.validatedAt!)),
      "validatedAt must be a valid timestamp");
  }
  must(unique(world.regions.map(x => x.id)), "duplicate region IDs");
  must(unique(world.assets.map(x => x.id)), "duplicate asset IDs");
  must(unique(world.tiles.map(x => x.id)), "duplicate tile IDs");
  must(unique(world.plants.map(x => x.strategyId)), "duplicate global strategy IDs");
  must(unique(world.mycelium.map(x => x.id)), "duplicate mycelium edges");
  const regions = new Map(world.regions.map(x => [x.id, x]));
  const assets = new Map(world.assets.map(x => [x.id, x]));
  const tiles = new Map(world.tiles.map(x => [x.id, x]));
  const grid = new Set<string>();
  const worldGrid = new Set<string>();
  for (const a of world.assets) {
    must(Boolean(a.id && a.symbol && a.quoteCurrency && a.contractVersion), "incomplete asset");
    must(a.dataStatus === "synthetic" || a.dataStatus === "verified", "unknown asset data status");
    must(Boolean(a.dataSourceId) || a.dataStatus === "synthetic", "missing data provenance");
  }
  for (const r of world.regions) {
    must([r.worldX, r.worldY].every(Number.isInteger), "invalid region position");
  }
  for (const t of world.tiles) {
    const region = regions.get(t.regionId), asset = assets.get(t.assetId);
    must(Boolean(region && asset), "tile refers to unknown asset/region");
    must(region!.assetClass === asset!.assetClass, "cross-class tile placement");
    must(Boolean(t.moduleId && t.moduleVersion), "tile is missing module version");
    must(t.version >= 1 && Number.isInteger(t.version), "invalid tile version");
    must(Number.isInteger(t.gridX) && Number.isInteger(t.gridY), "invalid tile grid position");
    must(t.axisX.geneId !== t.axisY.geneId, "two distinct real genes are required");
    validateAxis(t.axisX); validateAxis(t.axisY);
    const loc = t.regionId + ":" + t.gridX + ":" + t.gridY;
    must(!grid.has(loc), "overlapping home tiles");
    grid.add(loc);
    const worldPosition = `${region!.worldX * 4 + t.gridX}:${region!.worldY * 4 + t.gridY}`;
    must(!worldGrid.has(worldPosition), "overlapping world tile positions");
    worldGrid.add(worldPosition);
  }
  for (const p of world.plants) {
    must(Boolean(p.strategyId && p.assetId), "incomplete plant");
    must(assets.has(p.assetId), "unknown plant asset");
    must(finite(p.vitality) && p.vitality >= 0 && p.vitality <= 1, "invalid vitality");
    must(unique(p.moduleIds), "duplicate module IDs");
    if (p.homeTileId !== null) {
      const t = tiles.get(p.homeTileId);
      must(Boolean(t), "unknown home tile");
      must(compatible(p, t!), "plant's home tile lacks two compatible genes");
    }
  }
  for (const m of world.mycelium) {
    must(m.sourceTileId !== m.targetTileId, "self edge");
    must(Boolean(tiles.get(m.sourceTileId) && tiles.get(m.targetTileId)), "unknown mycelium tile");
    must(finite(m.confidence) && m.confidence >= 0 && m.confidence <= 1, "invalid mycelium confidence");
    must(Boolean(m.geneId && m.researchId), "missing research provenance");
    const from = tiles.get(m.sourceTileId)!, to = tiles.get(m.targetTileId)!;
    must(from.moduleId === to.moduleId && from.moduleVersion === to.moduleVersion,
      "cross-module/version knowledge not yet semantically validated");
    const fromGenes = [from.axisX.geneId, from.axisY.geneId];
    const toGenes = [to.axisX.geneId, to.axisY.geneId];
    must(fromGenes.includes(m.geneId) && toGenes.includes(m.geneId),
      "mycelium gene must have same semantic identity in both tiles");
    const fromAxis = [from.axisX, from.axisY].find(axis => axis.geneId === m.geneId)!;
    const toAxis = [to.axisX, to.axisY].find(axis => axis.geneId === m.geneId)!;
    must(fromAxis.unit === toAxis.unit
      && fromAxis.min === toAxis.min
      && fromAxis.max === toAxis.max
      && fromAxis.step === toAxis.step,
    "mycelium parameter range, step, and unit must match exactly");
  }
  return world;
}
export function listTiles(world: TileWorld, regionId?: string): readonly MeadowTile[] {
  return world.tiles.filter(t => !regionId || t.regionId === regionId);
}
export function getWorldStats(world: TileWorld): WorldStats {
  validateTileWorld(world);
  const counts: Record<string, number> = Object.fromEntries(world.tiles.map(t => [t.id, 0]));
  const unassigned: string[] = [];
  for (const p of world.plants) {
    if (p.homeTileId === null) unassigned.push(p.strategyId);
    else counts[p.homeTileId] += 1;
  }
  return {
    population: world.plants.length,
    unassigned,
    regionCount: world.regions.length,
    tileCount: world.tiles.length,
    homePopulationByTile: counts,
  };
}
export function focusTile(world: TileWorld, tileId: string): TileView {
  const t = world.tiles.find(t => t.id === tileId);
  must(Boolean(t), "unknown tile");
  const plants = world.plants.filter(p => compatible(p, t!)).map(p => ({
    strategyId: p.strategyId,
    x: (Number(p.genome[t!.axisX.geneId]) - t!.axisX.min) / (t!.axisX.max - t!.axisX.min),
    y: (Number(p.genome[t!.axisY.geneId]) - t!.axisY.min) / (t!.axisY.max - t!.axisY.min),
    home: p.homeTileId === tileId,
    vitality: p.vitality,
    synthetic: p.synthetic,
  }));
  return {
    tile: t!,
    plants,
    homePopulation: plants.filter(p => p.home).length,
    visiblePopulation: plants.length,
  };
}
export function proposeCrossTileKnowledge(world: TileWorld, edgeId: string): KnowledgeProposal {
  validateTileWorld(world);
  const edge = world.mycelium.find(e => e.id === edgeId);
  must(Boolean(edge), "unknown knowledge edge");
  const source = world.tiles.find(tile => tile.id === edge!.sourceTileId)!;
  const parameter = [source.axisX, source.axisY].find(axis => axis.geneId === edge!.geneId)!;
  return {
    kind: "research-only",
    sourceTileId: edge!.sourceTileId,
    targetTileId: edge!.targetTileId,
    geneId: edge!.geneId,
    researchId: edge!.researchId,
    confidence: edge!.confidence,
    moduleId: source.moduleId,
    moduleVersion: source.moduleVersion,
    parameter: {
      geneId: parameter.geneId,
      unit: parameter.unit,
      min: parameter.min,
      max: parameter.max,
      step: parameter.step,
    },
    requiresOutOfSampleValidation: true,
    fundingTransfer: false,
  };
}
// Layout coordinates are WORLD positions; seed mutations always operate in
// normalized gene-parameter coordinates within a single home tile.
export function tileWorldPosition(world: TileWorld, tileId: string): {x: number;y: number} {
  const tile = world.tiles.find(x => x.id === tileId);
  must(Boolean(tile), "unknown tile");
  const region = world.regions.find(r => r.id === tile!.regionId)!;
  return {x: region.worldX * 4 + tile!.gridX, y: region.worldY * 4 + tile!.gridY};
}

export function initialWorldViewport(): WorldViewportState {
  return {focus: {kind: "world"}, zoom: 1, panX: 0, panY: 0};
}

export function validateWorldViewport(
  world: TileWorld,
  viewport: WorldViewportState,
): WorldViewportState {
  must(Number.isFinite(viewport.zoom) && viewport.zoom >= 0.5 && viewport.zoom <= 4,
    "viewport zoom must be between 0.5 and 4");
  must(Number.isFinite(viewport.panX) && Number.isFinite(viewport.panY),
    "viewport pan must be finite");
  const focus = viewport.focus;
  if (focus.kind === "region") {
    must(world.regions.some(region => region.id === focus.regionId),
      "viewport refers to unknown region");
  }
  if (focus.kind === "tile") {
    must(world.tiles.some(tile => tile.id === focus.tileId),
      "viewport refers to unknown tile");
  }
  return viewport;
}

export function focusWorldViewport(
  world: TileWorld,
  viewport: WorldViewportState,
  focus: ViewportFocus,
): WorldViewportState {
  return validateWorldViewport(world, {...viewport, focus});
}

export function zoomWorldViewport(
  world: TileWorld,
  viewport: WorldViewportState,
  zoom: number,
): WorldViewportState {
  return validateWorldViewport(world, {...viewport, zoom});
}
