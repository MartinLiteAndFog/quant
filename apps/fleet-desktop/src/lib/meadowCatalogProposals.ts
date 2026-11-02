/** ECO-14b: suggest missing two-real-gene meadows without moving capital.
 *
 * A module maintainer must register a semantically valid pair of STRATEGY
 * genes. We never infer axes from arbitrary numeric/visual/financial fields.
 * This is a read-only view-model; accepting a suggestion requires separate
 * schema review and cannot migrate a living plant's home/capital.
 */
import {
  validateTileWorld,
  type GeneAxis,
  type MeadowTile,
  type StrategyPlacement,
  type TileWorld,
} from "./tileWorld.ts";

export interface GenePairCatalogEntry {
  readonly pairId: string;
  readonly moduleId: string;
  readonly moduleVersion: string;
  readonly axisX: GeneAxis;
  readonly axisY: GeneAxis;
  /** Distinct provenance for human-reviewable module axis semantics. */
  readonly approvedStrategySpecId: string;
}

export interface ExistingPlacementSuggestion {
  readonly strategyId: string;
  readonly existingTileIds: readonly string[];
  readonly effect: "suggest_existing_tile_only";
  readonly transfersCapital: false;
}
export interface NewMeadowProposal {
  readonly proposalId: string;
  readonly regionId: string;
  readonly assetId: string;
  readonly moduleId: string;
  readonly moduleVersion: string;
  readonly axisX: GeneAxis;
  readonly axisY: GeneAxis;
  readonly gridX: number;
  readonly gridY: number;
  readonly strategyIds: readonly string[];
  readonly strategySpecId: string;
  readonly proposalOnly: true;
  readonly createsFundedPlants: false;
  readonly transfersCapital: false;
}
export interface UnassignedPlacement {
  readonly strategyId: string;
  readonly reason: "no_registered_two_gene_pair" | "no_free_region_slot";
}
export interface MeadowProposalReport {
  readonly suggestions: readonly NewMeadowProposal[];
  readonly existingMatches: readonly ExistingPlacementSuggestion[];
  readonly unresolved: readonly UnassignedPlacement[];
  readonly originalPlantCount: number;
  readonly originalTileCount: number;
  readonly worldIsUnchanged: true;
}

function check(condition: boolean, reason: string): asserts condition {
  if (!condition) throw new Error("Meadow proposals: " + reason);
}
function validAxis(axis: GeneAxis): boolean {
  return Boolean(axis.geneId && axis.label && axis.unit)
    && [axis.min, axis.max, axis.step].every(Number.isFinite)
    && axis.min < axis.max && axis.step > 0;
}
function canPlace(plant: StrategyPlacement, pair: GenePairCatalogEntry): boolean {
  if (!plant.moduleIds.includes(pair.moduleId)) return false;
  return [pair.axisX, pair.axisY].every(axis => {
    const v = plant.genome[axis.geneId];
    return typeof v === "number" && Number.isFinite(v)
      && v >= axis.min - 1e-9 && v <= axis.max + 1e-9;
  });
}
function compatibleExisting(plant: StrategyPlacement, tile: MeadowTile): boolean {
  if (tile.assetId !== plant.assetId || !plant.moduleIds.includes(tile.moduleId)) return false;
  return [tile.axisX, tile.axisY].every(axis => {
    const v=plant.genome[axis.geneId];
    return typeof v === "number" && Number.isFinite(v)
      && v >= axis.min - 1e-9 && v <= axis.max + 1e-9;
  });
}
function matchKey(assetId: string, pair: GenePairCatalogEntry): string {
  return JSON.stringify([assetId, pair.moduleId, pair.moduleVersion, pair.pairId]);
}
function proposalId(assetId: string, pair: GenePairCatalogEntry): string {
  // Human-readable stable ID; never passed to a trading gateway.
  return "proposed:" + [assetId, pair.moduleId, pair.moduleVersion, pair.pairId]
    .map(item => encodeURIComponent(item)).join(":");
}

/** Deterministic suggestion-only coverage; input arrays never mutated. */
export function proposeMissingMeadows(
  world: TileWorld, catalog: readonly GenePairCatalogEntry[],
): MeadowProposalReport {
  validateTileWorld(world);
  const pairKeys = new Set<string>();
  const approved: GenePairCatalogEntry[] = [];
  for (const item of catalog) {
    check(Boolean(item.pairId && item.moduleId && item.moduleVersion
                  && item.approvedStrategySpecId), "catalog pair needs explicit strategy specification");
    check(item.axisX.geneId !== item.axisY.geneId, "exactly two DISTINCT real axis genes required");
    check(validAxis(item.axisX) && validAxis(item.axisY), "invalid strategy axis range");
    const k = JSON.stringify([item.moduleId, item.moduleVersion, item.pairId]);
    check(!pairKeys.has(k), "duplicate module/version/pair registration");
    pairKeys.add(k);
    approved.push(item);
  }
  approved.sort((a,b)=>matchKey("",a).localeCompare(matchKey("",b)));

  const assets = new Map(world.assets.map(a=>[a.id,a]));
  const regions=world.regions.slice().sort((a,b)=>a.id.localeCompare(b.id));
  const plants=world.plants.filter(p=>p.homeTileId===null)
    .slice().sort((a,b)=>a.strategyId.localeCompare(b.strategyId));

  const existingMatches: ExistingPlacementSuggestion[]=[];
  const unresolved: UnassignedPlacement[]=[];
  const candidateGroups = new Map<string, {pair:GenePairCatalogEntry; assetId:string; ids:string[]}>();
  for (const plant of plants) {
    // If an already-existing true two-gene category fits, never invent a tile.
    const existing = world.tiles.filter(t=>compatibleExisting(plant,t))
      .map(t=>t.id).sort();
    if (existing.length) {
      existingMatches.push({
        strategyId:plant.strategyId,existingTileIds:existing,
        effect:"suggest_existing_tile_only",transfersCapital:false,
      });
      continue;
    }
    const candidates=approved.filter(pair=>canPlace(plant,pair));
    if (!candidates.length) {
      unresolved.push({strategyId:plant.strategyId,
                       reason:"no_registered_two_gene_pair"});
      continue;
    }
    // One canonical home recommendation per strategy. The game may offer
    // other catalog-proven parameter views separately after schema review.
    const pair=candidates[0];
    const key=matchKey(plant.assetId,pair);
    const group=candidateGroups.get(key);
    if (group) group.ids.push(plant.strategyId);
    else candidateGroups.set(key,{pair,assetId:plant.assetId,ids:[plant.strategyId]});
  }

  // No placements outside a region's owned 4x4 tile grid, so one asset-class
  // region cannot silently overlap a neighbor's visualization.
  const occupied = new Set(world.tiles.map(t=>`${t.regionId}:${t.gridX}:${t.gridY}`));
  const suggestions:NewMeadowProposal[]=[];
  const groups=[...candidateGroups.entries()].sort(([a],[b])=>a.localeCompare(b));
  for (const [,group] of groups) {
    const asset=assets.get(group.assetId);
    check(Boolean(asset), "unknown asset for a proposed category");
    const region=regions.find(r=>r.assetClass===asset!.assetClass);
    if (!region) {
      group.ids.forEach(strategyId=>unresolved.push({
        strategyId,reason:"no_free_region_slot",
      }));
      continue;
    }
    let slot:{x:number;y:number}|null=null;
    for(let y=0;y<4&&!slot;y++){
      for(let x=0;x<4&&!slot;x++){
        if(!occupied.has(`${region.id}:${x}:${y}`)) slot={x,y};
      }
    }
    if (!slot) {
      group.ids.forEach(strategyId=>unresolved.push({
        strategyId,reason:"no_free_region_slot",
      }));
      continue;
    }
    occupied.add(`${region.id}:${slot.x}:${slot.y}`);
    suggestions.push({
      proposalId:proposalId(group.assetId,group.pair),
      regionId:region.id,assetId:group.assetId,
      moduleId:group.pair.moduleId,moduleVersion:group.pair.moduleVersion,
      axisX:group.pair.axisX,axisY:group.pair.axisY,
      gridX:slot.x,gridY:slot.y,
      strategyIds:group.ids.slice().sort(),
      strategySpecId:group.pair.approvedStrategySpecId,
      proposalOnly:true,createsFundedPlants:false,transfersCapital:false,
    });
  }
  return {
    suggestions,existingMatches,
    unresolved:unresolved.sort((a,b)=>a.strategyId.localeCompare(b.strategyId)),
    originalPlantCount:world.plants.length,
    originalTileCount:world.tiles.length,
    worldIsUnchanged:true,
  };
}
