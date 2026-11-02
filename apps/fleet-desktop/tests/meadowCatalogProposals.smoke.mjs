import assert from "node:assert/strict";
import {TILE_WORLD_DEMO} from "../src/lib/tileWorldDemo.ts";
import {proposeMissingMeadows} from "../src/lib/meadowCatalogProposals.ts";
import {getWorldStats, validateTileWorld} from "../src/lib/tileWorld.ts";

const pair = {
  pairId:"window-target-v1",
  moduleId:"new-module",
  moduleVersion:"1",
  approvedStrategySpecId:"synthetic-verified-axis-catalog-fixture",
  axisX:{geneId:"signal_window",label:"Window",unit:"bars",min:5,max:50,step:1},
  axisY:{geneId:"target_percent",label:"Target",unit:"percent",min:.1,max:5,step:.1},
};
const pending = [
  {strategyId:"new-seed-2",assetId:"sol",homeTileId:null,moduleIds:["new-module"],
   genome:{signal_window:16,target_percent:.7,visual_color:3},vitality:.6,synthetic:true},
  {strategyId:"new-seed-1",assetId:"sol",homeTileId:null,moduleIds:["new-module"],
   genome:{signal_window:12,target_percent:1.2},vitality:.4,synthetic:true},
  {strategyId:"existing-imba-but-unassigned",assetId:"sol",homeTileId:null,moduleIds:["imba"],
   genome:{sensitivity:17.2,ttp:.65},vitality:.6,synthetic:true},
];
const world = validateTileWorld({...TILE_WORLD_DEMO,
  plants:[...TILE_WORLD_DEMO.plants,...pending],
});
const snapshot = JSON.stringify(world);
const proposed=proposeMissingMeadows(world,[pair]);
assert.equal(proposed.worldIsUnchanged,true);
assert.equal(proposed.originalPlantCount,11);
assert.equal(proposed.originalTileCount,6);
assert.equal(proposed.suggestions.length,1);
assert.equal(proposed.existingMatches.length,1);
assert.deepEqual(proposed.existingMatches[0].existingTileIds,["sol-imba"]);
assert.equal(proposed.existingMatches[0].transfersCapital,false);
assert.deepEqual(proposed.suggestions[0].strategyIds,["new-seed-1","new-seed-2"]);
assert.equal(proposed.suggestions[0].axisX.geneId,"signal_window");
assert.equal(proposed.suggestions[0].axisY.geneId,"target_percent");
assert.equal(proposed.suggestions[0].regionId,"crypto");
assert.equal(proposed.suggestions[0].gridX,3);
assert.equal(proposed.suggestions[0].gridY,0);
assert.equal(proposed.suggestions[0].proposalOnly,true);
assert.equal(proposed.suggestions[0].createsFundedPlants,false);
assert.equal(proposed.suggestions[0].transfersCapital,false);
assert.equal(proposed.unresolved.length,1);
assert.equal(proposed.unresolved[0].strategyId,"unassigned-research");
assert.equal(proposed.unresolved[0].reason,"no_registered_two_gene_pair");
assert.equal(JSON.stringify(world),snapshot);
assert.equal(getWorldStats(world).population,11);
assert.equal(world.tiles.length,6);

const reverse=proposeMissingMeadows(
  {...world,plants:world.plants.slice().reverse()},[pair],
);
assert.deepEqual(reverse,proposed);  // no dependence on map/import order
const noCatalog=proposeMissingMeadows(world,[]);
assert.equal(noCatalog.suggestions.length,0);
assert.equal(noCatalog.existingMatches.length,1);
assert.equal(noCatalog.unresolved.length,3);
assert.throws(
  ()=>proposeMissingMeadows(world,[pair,pair]),
  /duplicate module\/version\/pair/,
);
assert.throws(
  ()=>proposeMissingMeadows(world,[{...pair,axisY:pair.axisX}]),
  /two DISTINCT real axis genes/,
);
assert.throws(
  ()=>proposeMissingMeadows(world,[{...pair,axisY:{...pair.axisY,geneId:"max_leverage"}}]),
  /financial, safety and optical/,
);
assert.throws(
  ()=>proposeMissingMeadows(world,[{...pair,axisY:{...pair.axisY,geneId:"visual_color"}}]),
  /financial, safety and optical/,
);

// Same strategic pair for the same module in another ASSET creates a separate
// *proposal*, never shared price/capital/region ownership.
const crossAsset=validateTileWorld({...world,plants:[
  ...world.plants,{strategyId:"wti-new",assetId:"wti",homeTileId:null,
    moduleIds:["new-module"],genome:{signal_window:10,target_percent:.5},
    vitality:.5,synthetic:true},
]});
const two=proposeMissingMeadows(crossAsset,[pair]);
assert.equal(two.suggestions.length,2);
assert.deepEqual(two.suggestions.map(p=>p.assetId),["sol","wti"]);
assert.equal(two.suggestions.find(x=>x.assetId==="wti").regionId,"commodities");
assert.equal(two.suggestions.find(x=>x.assetId==="wti").gridX,1);

// A full 4×4 region must fail closed instead of colliding with adjacent
// islands / inventing a floating meadow in another market's region.
const taken=new Set(TILE_WORLD_DEMO.tiles.filter(t=>t.regionId==="crypto")
  .map(t=>t.gridX+":"+t.gridY));
const fillers=[];
for(let y=0;y<4;y++){
  for(let x=0;x<4;x++){
    if(taken.has(x+":"+y)) continue;
    const template=TILE_WORLD_DEMO.tiles[0];
    fillers.push({...template,id:"filler-"+x+"-"+y,gridX:x,gridY:y});
  }
}
const fullRegion=validateTileWorld({...world,tiles:[...world.tiles,...fillers]});
const cannotPlace=proposeMissingMeadows(fullRegion,[pair]);
assert.equal(cannotPlace.suggestions.length,0);
assert.deepEqual(cannotPlace.unresolved.filter(x=>x.reason==="no_free_region_slot")
  .map(x=>x.strategyId),["new-seed-1","new-seed-2"]);
assert.equal(JSON.stringify(fullRegion.tiles),JSON.stringify([...world.tiles,...fillers]));

console.log("ECO-14b: category proposals are true two-gene, stable, cross-asset separated and never book funding.");
