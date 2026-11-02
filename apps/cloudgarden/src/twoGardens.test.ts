import test from 'node:test';
import assert from 'node:assert/strict';
import {createHistoricalSession, loadSolArchive, normalizeArchiveBars, validateHistoricalSession, type SolArchive} from './historicalGarden.ts';
const manifest=(symbol:string,sha:string):SolArchive=>({schemaVersion:1,source:'local immutable fixture',symbol,sourceSha256:sha.repeat(64),venue:'unverified',firstDate:'2020-08-12',lastDate:'2020-08-12',totalBars:1,missingMinutes:0,duplicateCount:0,days:[{date:'2020-08-12',barCount:1,gapCount:0,missingMinutes:0,file:'2020-08-12.json',sha256:'c'.repeat(64)}]});
test('two gardens request distinct manifests and reject a cross-symbol response',async()=>{
 const before=globalThis.fetch;const paths:string[]=[];
 try{globalThis.fetch=async(input)=>{paths.push(String(input));return new Response(JSON.stringify(manifest(String(input).includes('market-btc')?'BTC-USDT':'SOL-USDT','a')));};
 assert.equal((await loadSolArchive('SOL-USDT')).symbol,'SOL-USDT');assert.equal((await loadSolArchive('BTC-USDT')).symbol,'BTC-USDT');assert.deepEqual(paths,['/market/manifest.json','/market-btc/manifest.json']);
 globalThis.fetch=async()=>new Response(JSON.stringify(manifest('SOL-USDT','a')));await assert.rejects(loadSolArchive('BTC-USDT'),/symbol/);
 }finally{globalThis.fetch=before;}
});
test('BTC and SOL accounts remain distinct and market provenance keeps the actual symbol',async()=>{
 const sol=manifest('SOL-USDT','a'),btc=manifest('BTC-USDT','b');const first=createHistoricalSession(sol),second=createHistoricalSession(btc);
 assert.notEqual(first.world.seed,second.world.seed);assert.notEqual(first.world,second.world);assert.notEqual(first.world.capital,second.world.capital);assert.throws(()=>validateHistoricalSession(first,btc));
 const data=await normalizeArchiveBars([{timestamp:'2020-08-12T00:00:00.000Z',open:'10000',high:'10001',low:'9999',close:'10000',volume:'1'}],btc);assert.equal(data.provenance.symbol,'BTC-USDT');assert.equal(first.funding?.totalUsdt,0);assert.equal(second.funding?.monthlyEur,100);
});
