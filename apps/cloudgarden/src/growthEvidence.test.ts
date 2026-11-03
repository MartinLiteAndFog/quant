import test from'node:test';import assert from'node:assert/strict';
import{createSimulationWorld,advanceSimulationDay,fundSimulationWorld,assertSimulationInvariants}from'./simulation.ts';
test('published growth breakdown reconciles actual biomass and exposes the root/leaf resource cost',()=>{
 const before=createSimulationWorld('growth-evidence');const after=advanceSimulationDay(before,{water:.7,sunlight:.7,predatorPatrol:false});
 const a=before.plants[0]!,b=after.plants[0]!,g=b.growthBreakdown!;
 assert.ok(Object.values(g).every(Number.isFinite));assert.equal(g.rootLeafCost,Math.max(0,a.genes.root+a.genes.leaf-1)*1.25);
 assert.equal(b.biomass,Math.round(Math.max(5,Math.min(100,a.biomass+g.netGrowth))*10)/10);
});
test('the 50-strategy research limit remains hard through prolonged funded reproduction',()=>{
 let w=fundSimulationWorld(createSimulationWorld('cap50',{dynamicEcology:true,maxLivingPlants:50}),10000);
 for(let i=0;i<2000;i++){w=advanceSimulationDay(w,{water:.7,sunlight:.7,predatorPatrol:false});assert.ok(w.plants.filter(p=>p.status==='active').length<=50);}
 assert.equal(w.plants.filter(p=>p.status==='active').length,50);assert.ok(w.events.length<=2000);assertSimulationInvariants(w);
});
