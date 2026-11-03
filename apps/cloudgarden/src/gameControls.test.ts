import assert from "node:assert/strict";
import test from "node:test";
import { advanceSimulationDay, assertSimulationInvariants, createSimulationWorld,
  GENTLE_ECOLOGY, setEcologyControls, setEcologyEvolution, visiblePopulation } from "./simulation.ts";
import { observedStartDate, requireObservedDate } from "./gameStart.ts";
import { bookHistoricalMark } from "./simulation.ts";
import { DEFAULT_FUNDING_SETTINGS, topUpWorldAfterDay, validateFundingSettings } from "./fundingPolicy.ts";
import { createHistoricalSession, configureHistoricalFunding, fundingSettings, applyEndOfDayFunding, type SolArchive } from "./historicalGarden.ts";

const observed = Array.from({length: 100}, (_,i) => ({
  date: new Date(Date.UTC(2026,0,1+i)).toISOString().slice(0,10),
  barCount: 1440, gapCount: 0, missingMinutes: 0, file: "unused", sha256: "a".repeat(64),
}));
const archive = {
  schemaVersion: 1, source: "Fixture", sourceSha256: "a".repeat(64),
  symbol: "SOL-USDT", venue: "unverified", firstDate: observed[0]!.date,
  lastDate: observed.at(-1)!.date, totalBars: 144000, missingMinutes: 0,
  duplicateCount: 0, days: observed
} as const satisfies SolArchive;

test("observed-day presets start a new session at the final 30/90 actual days", () => {
  assert.equal(observedStartDate(archive,"last30"),observed[70]!.date);
  assert.equal(observedStartDate(archive,"last90"),observed[10]!.date);
  assert.equal(observedStartDate(archive,"last365"),observed[0]!.date);
  assert.equal(observedStartDate(archive,"first"),observed[0]!.date);
  assert.equal(requireObservedDate(archive,observed[99]!.date),observed[99]!.date);
  assert.throws(()=>requireObservedDate(archive,"2025-10-10"),/Datum/);
  assert.throws(()=>observedStartDate({days:[]},"last30"),/archive/);
});

test("historical starts have predators and negative vitality penalty OFF, without changing capital", () => {
  const session=createHistoricalSession(archive,observedStartDate(archive,"last30"));
  assert.equal(session.world.day,0);
  assert.deepEqual(session.world.ecologyControls,GENTLE_ECOLOGY);
  assert.equal(session.world.capital.navUsdt,100);
  assert.equal(session.world.capital.protectedReserveUsdt,10);
  assert.equal(session.world.capital.deployedCapitalUsdt,60);
  assertSimulationInvariants(session.world);
});

test("predators OFF never archives a stressed plant, while explicit ON preserves original predation", () => {
  const base = createSimulationWorld("predation-debug",{dynamicEcology:true});
  const stressed = {...base, plants:base.plants.map((p,i)=>i===0?
    {...p, vigor:1, evidence:1, evidenceDays:100}:p)};
  const badWeather={water:0.7,sunlight:0.7,predatorPatrol:true};
  const safe=advanceSimulationDay(setEcologyControls(stressed,GENTLE_ECOLOGY),badWeather);
  assert.equal(visiblePopulation(safe).length,3);
  assert.equal(safe.events.filter((event)=>event.type==="predation").length,0);
  const dangerous=advanceSimulationDay(setEcologyControls(stressed,{predatorsEnabled:true,vitalityMalus:1}),badWeather);
  assert.ok(dangerous.events.some((event)=>event.type==="predation"));
  assert.equal(visiblePopulation(dangerous).length,2);
  assert.equal(dangerous.capital.navUsdt,safe.capital.navUsdt);
  assertSimulationInvariants(safe);
  assertSimulationInvariants(dangerous);
});

test("0% vitality malus preserves stressed vigor and evolution can be independently paused", () => {
  const base=createSimulationWorld("no-vigor-penalty");
  const weather={water:0,sunlight:0,predatorPatrol:false};
  const normal=advanceSimulationDay(setEcologyControls(base,{predatorsEnabled:false,vitalityMalus:1}),weather);
  const gentle=advanceSimulationDay(setEcologyControls(base,GENTLE_ECOLOGY),weather);
  assert.ok(gentle.plants[0]!.vigor>=normal.plants[0]!.vigor);
  assert.equal(gentle.plants[0]!.vigor,base.plants[0]!.vigor);
  assert.throws(()=>setEcologyControls(base,{predatorsEnabled:false,vitalityMalus:-0.1}));
  assert.throws(()=>setEcologyControls(base,{predatorsEnabled:true,vitalityMalus:1.1}));
  const paused=setEcologyEvolution(setEcologyControls(base,GENTLE_ECOLOGY),false);
  let week=paused;
  for(let i=0;i<7;i++) week=advanceSimulationDay(week);
  assert.equal(week.plants.length,3);
  assert.equal(week.capital.navUsdt,100);
  assertSimulationInvariants(week);
});


test("new research worlds default to top-up 100 €, monthly contribution OFF", () => {
  const session=createHistoricalSession(archive,archive.firstDate);
  assert.equal(session.funding?.mode,"topup");
  assert.equal(session.funding?.topUpTargetEur,100);
  assert.equal(session.funding?.monthlyEur,100);
  assert.equal(session.funding?.totalUsdt,0);
  assert.equal(session.world.capital.externalDepositsUsdt??0,0);
  assert.equal(DEFAULT_FUNDING_SETTINGS.mode,"topup");
  assert.equal(createHistoricalSession(archive,archive.firstDate,{fundingMode:"off"}).funding?.mode,"off");
  assert.equal(createHistoricalSession(archive,archive.firstDate,{fundingMode:"monthly",monthlyEur:250}).funding?.monthlyEur,250);
  // An older v1 save with no explicit mode must not silently rewrite the historical monthly rule.
  const old={...session.funding!,mode:undefined};
  assert.equal(fundingSettings(old).mode,"monthly");
});

test("100 € top-up deposits only missing amount after negative marked return", () => {
  const base=createSimulationWorld("funding-topup",{dynamicEcology:true});
  const afterLoss=bookHistoricalMark(base,"parent-01",{
    datasetFingerprint:"b".repeat(64),marketSource:"observed sample",fillCount:2,strategyId:"research",
    netEquityDeltaUsdt:-7,realizedPnlUsdt:-7,unrealizedDeltaUsdt:0,feesUsdt:0,
  });
  assert.equal(afterLoss.capital.navUsdt,93);
  const {world,depositEur,depositUsdt}=topUpWorldAfterDay(afterLoss,DEFAULT_FUNDING_SETTINGS);
  assert.equal(world.capital.navUsdt,100);
  assert.equal(world.capital.externalDepositsUsdt,7);
  assert.equal(depositEur,7);assert.equal(depositUsdt,7);
  assert.equal(world.capital.realizedPricePnlUsdt,-7);
  assert.equal(world.capital.deployedCapitalUsdt,60);
  assert.equal(topUpWorldAfterDay(world,DEFAULT_FUNDING_SETTINGS).depositUsdt,0);
  assertSimulationInvariants(world);
});

test("full session only funds once per completed day, counted as contribution, never trading profit", () => {
  const base=createHistoricalSession(archive,archive.firstDate);
  const underwater=bookHistoricalMark(base.world,"parent-01",{
    datasetFingerprint:"d".repeat(64),marketSource:"observed sample",fillCount:2,strategyId:"research",
    netEquityDeltaUsdt:-7,realizedPnlUsdt:-7,unrealizedDeltaUsdt:0,feesUsdt:0,
  });
  const withDay={...base,world:{...underwater,day:1}};
  const funded=applyEndOfDayFunding(withDay);
  assert.equal(funded.world.capital.navUsdt,100);
  assert.equal(funded.funding!.totalUsdt,7);
  assert.equal(funded.funding!.totalEur,7);
  assert.equal(funded.funding!.lastTopUpDay,1);
  assert.equal(funded.world.capital.navUsdt - 100 - funded.world.capital.externalDepositsUsdt!,-7);
  assert.deepEqual(applyEndOfDayFunding(funded),funded);
  const off=configureHistoricalFunding(withDay,{...DEFAULT_FUNDING_SETTINGS,mode:"off"});
  assert.equal(applyEndOfDayFunding(off).world.capital.navUsdt,93);
});

test("switching from top-up to optional monthly subsidy never backfills skipped months", () => {
  const base=createHistoricalSession(archive,archive.firstDate);
  const later={...base,nextDate:observed[80]!.date,loop:{...base.loop,cycle:3,archiveIndex:80}};
  const changed=configureHistoricalFunding(later,{...DEFAULT_FUNDING_SETTINGS,mode:"monthly",monthlyEur:225});
  assert.equal(changed.funding!.mode,"monthly");
  assert.equal(changed.funding!.monthlyEur,225);
  assert.equal(changed.funding!.lastMonthKey,`3:${observed[80]!.date.slice(0,7)}`);
  assert.equal(changed.world.capital.navUsdt,100);
  assert.equal(changed.world.capital.externalDepositsUsdt??0,0);
  assert.throws(()=>validateFundingSettings({...DEFAULT_FUNDING_SETTINGS,topUpTargetEur:0}),/Kapitalregel/);
  assert.throws(()=>validateFundingSettings({...DEFAULT_FUNDING_SETTINGS,eurUsdt:Infinity}),/Kapitalregel/);
});
