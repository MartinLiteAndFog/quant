import test from'node:test';import assert from'node:assert/strict';
import{createFocusedHistoricalSession}from'./focusResearch.ts';
import{advanceSimulationDay,assertSimulationInvariants}from'./simulation.ts';
import type{SolArchive}from'./historicalGarden.ts';
test('single-plant research preserves capital and suppresses births through repeated weeks',()=>{
 const archive:SolArchive={schemaVersion:1,source:'fixture',symbol:'SOL-USDT',sourceSha256:'a'.repeat(64),venue:'unverified',firstDate:'2020-08-11',lastDate:'2020-08-11',totalBars:1,missingMinutes:0,duplicateCount:0,days:[{date:'2020-08-11',barCount:1,gapCount:0,missingMinutes:0,file:'2020-08-11.json',sha256:'b'.repeat(64)}]};
 let world=createFocusedHistoricalSession(archive).world;
 assert.equal(world.capital.navUsdt,100);assert.equal(world.capital.deployedCapitalUsdt,20);assert.equal(world.capital.freeCapitalUsdt,70);assert.equal(world.dynamicEcology?.maxLivingPlants,50);
 for(let day=0;day<100;day++)world=advanceSimulationDay(world,{water:.7,sunlight:.7,predatorPatrol:false});
 assert.equal(world.plants.filter(p=>p.status==='active').length,1);assert.equal(world.dynamicEcology?.birthCount,0);assertSimulationInvariants(world);
});
