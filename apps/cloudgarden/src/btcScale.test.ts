import test from 'node:test';
import assert from 'node:assert/strict';
import {createHistoricalSession,validateHistoricalSession,type SolArchive}from'./historicalGarden.ts';
const archive=(symbol:string):SolArchive=>({schemaVersion:1,source:'local source fixture',symbol,sourceSha256:'a'.repeat(64),venue:'unverified',firstDate:'2020-08-12',lastDate:'2020-08-12',totalBars:1,missingMinutes:0,duplicateCount:0,days:[{date:'2020-08-12',barCount:1,gapCount:0,missingMinutes:0,file:'2020-08-12.json',sha256:'b'.repeat(64)}]});
test('BTC brick uses the same relative threshold as SOL from already observed reference opens',()=>{
 const btc=createHistoricalSession(archive('BTC-USDT')),sol=createHistoricalSession(archive('SOL-USDT'));
 for(const profile of Object.values(btc.krakenProfiles!))assert.ok(Math.abs(profile.renkoBoxUsd/11392.09-.1/2.85)<1e-12);
 for(const profile of Object.values(sol.krakenProfiles!))assert.equal(profile.renkoBoxUsd,.1);
 assert.deepEqual(validateHistoricalSession(structuredClone(btc),archive('BTC-USDT')),btc);
});
