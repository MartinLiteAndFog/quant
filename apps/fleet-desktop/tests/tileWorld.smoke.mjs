import assert from "node:assert/strict";
import {
  focusTile, getWorldStats, listTiles, proposeCrossTileKnowledge,
  focusWorldViewport, initialWorldViewport, tileWorldPosition,
  validateTileWorld, validateWorldViewport, zoomWorldViewport,
} from "../src/lib/tileWorld.ts";
import {TILE_WORLD_DEMO} from "../src/lib/tileWorldDemo.ts";

const world = TILE_WORLD_DEMO;
const stats = getWorldStats(world);
assert.equal(stats.regionCount, 4);
assert.equal(stats.tileCount, 6);
assert.equal(stats.population, 8);
assert.equal(stats.unassigned.length, 1);
assert.equal(Object.values(stats.homePopulationByTile).reduce((a,b)=>a+b,0), 7);

const imba = focusTile(world, "sol-imba");
const rsi = focusTile(world, "sol-rsi");
assert.equal(imba.homePopulation, 2);
assert.equal(rsi.homePopulation, 1);
assert.equal(rsi.visiblePopulation, 2); // blaupunkt: *same* strategy in secondary view
assert.equal(rsi.plants.find(p=>p.strategyId==="blaupunkt")?.home, false);
assert.equal(imba.plants.find(p=>p.strategyId==="blaupunkt")?.home, true);
assert.equal(focusTile(world,"btc-trend").visiblePopulation, 1);
assert.equal(listTiles(world,"crypto").length,3);
assert.notDeepEqual(tileWorldPosition(world,"sol-imba"),tileWorldPosition(world,"oil-rsi"));
assert.deepEqual(getWorldStats(world), stats); // navigation cannot mutate world

const insight = proposeCrossTileKnowledge(world,"rsi-knowledge");
assert.equal(insight.kind, "research-only");
assert.equal(insight.fundingTransfer, false);
assert.equal(insight.requiresOutOfSampleValidation, true);
assert.equal(insight.moduleVersion, "1");
assert.deepEqual(insight.parameter, {
  geneId: "rsi_length", unit: "bars", min: 5, max: 50, step: 1,
});
assert.throws(() => proposeCrossTileKnowledge(world,"unknown"),/unknown knowledge/);

const viewport = initialWorldViewport();
assert.deepEqual(viewport, {focus:{kind:"world"},zoom:1,panX:0,panY:0});
assert.deepEqual(focusWorldViewport(world,viewport,{kind:"region",regionId:"crypto"}),
  {...viewport,focus:{kind:"region",regionId:"crypto"}});
assert.deepEqual(focusWorldViewport(world,viewport,{kind:"tile",tileId:"sol-rsi"}),
  {...viewport,focus:{kind:"tile",tileId:"sol-rsi"}});
assert.equal(zoomWorldViewport(world,viewport,2).zoom,2);
assert.throws(() => zoomWorldViewport(world,viewport,4.1),/zoom/);
assert.throws(() => validateWorldViewport(world,{...viewport,panX:Infinity}),/pan/);
assert.throws(() => focusWorldViewport(world,viewport,{kind:"tile",tileId:"missing"}),/unknown tile/);
assert.deepEqual(getWorldStats(world), stats); // viewport transitions never mutate ecology state

assert.throws(
  () => validateTileWorld({...world, plants:[world.plants[0],world.plants[0]]}),
  /duplicate global strategy/,
);
assert.throws(
  () => validateTileWorld({...world,plants:world.plants.map((p,i)=>
    i===0 ? {...p,genome:{rsi_length:14,rsi_entry:30}} : p)}),
  /two compatible genes/,
);
assert.throws(
  () => validateTileWorld({...world, tiles:world.tiles.map((t,i)=>
    i===0 ? {...t,axisY:t.axisX} : t)}),
  /two distinct real genes/,
);
assert.throws(
  () => validateTileWorld({...world,mycelium:[{
    ...world.mycelium[0], sourceTileId:"btc-trend",targetTileId:"oil-rsi",
  }]}),
  /cross-module\/version/,
);
assert.throws(
  () => validateTileWorld({...world,tiles:world.tiles.map(t=>
    t.id==="oil-rsi" ? {...t,moduleVersion:"2"} : t)}),
  /module\/version/,
);
assert.throws(
  () => validateTileWorld({...world,tiles:world.tiles.map(t=>
    t.id==="oil-rsi" ? {...t,axisX:{...t.axisX,unit:"minutes"}} : t)}),
  /range, step, and unit/,
);
assert.throws(
  () => validateTileWorld({...world,tiles:world.tiles.map(t=>
    t.id==="oil-rsi" ? {...t,axisX:{...t.axisX,max:60}} : t)}),
  /range, step, and unit/,
);
assert.throws(
  () => validateTileWorld({...world,synthetic:false,evidence:{
    status:"validated",validatorVersion:"v1",validatedAt:"2026-11-01T00:00:00Z",
  }}),
  /evidence IDs/,
);
assert.throws(
  () => validateTileWorld({...world,evidence:{status:"unverified"}}),
  /synthetic flag and evidence status disagree/,
);
const attested = {...world,synthetic:false,evidence:{status:"validated",
  validationIds:["walk-forward-2026-01"],validatorVersion:"validator-1",
  validatedAt:"2026-11-01T00:00:00Z"}};
assert.equal(validateTileWorld(attested),attested);
assert.throws(
  () => validateTileWorld({...world,regions:world.regions.map(r=>
    r.id==="fx" ? {...r,worldX:0} : r)}),
  /overlapping world tile positions/,
);
console.log("ECO-14 tile world: valid 6 tiles, 4 regions, deduplicated views, safe mycelium.");
