import assert from "node:assert/strict";
import {
  focusTile, getWorldStats, listTiles, proposeCrossTileKnowledge,
  tileWorldPosition, validateTileWorld,
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
assert.throws(() => proposeCrossTileKnowledge(world,"unknown"),/unknown knowledge/);

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
  /cross-module knowledge/,
);
console.log("ECO-14 tile world: valid 6 tiles, 4 regions, deduplicated views, safe mycelium.");
