import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {loadSolArchive,createHistoricalSession,advanceHistoricalSession,validateHistoricalSession} from '../src/historicalGarden.ts';
const days=Number(process.argv[2]??7);
if(!Number.isInteger(days)||days<1||days>5000)throw Error('Choose 1..5000 days');
globalThis.fetch=async(input)=>{
 const path=String(input);
 if(!/^\/market-btc\/(manifest|\d{4}-\d{2}-\d{2})\.json$/.test(path))throw Error('Cross-garden asset request');
 return new Response(await readFile(new URL('../public'+path,import.meta.url)));
};
const archive=await loadSolArchive('BTC-USDT');
let session=createHistoricalSession(archive);
for(let i=0;i<days;i++){session=await advanceHistoricalSession(session,archive);if((i+1)%100===0)console.log('BTC days',i+1);}
assert.equal(session.loop.totalDays,days);
assert.ok(session.loop.totalFills>0,'Real BTC execution required');
assert.deepEqual(validateHistoricalSession(structuredClone(session),archive),session);
const evidence={symbol:archive.symbol,sourceSha256:archive.sourceSha256,archiveBars:archive.totalBars,days,fills:session.loop.totalFills,nav:session.world.capital.navUsdt,deposits:session.funding?.totalUsdt??0,netTradingPnl:session.world.capital.navUsdt-100-(session.funding?.totalUsdt??0),verdict:'not_comparable',note:'SOL templates transferred unchanged to BTC, modeled costs; no independent profitability evidence.'};
await writeFile('/tmp/cloudgarden-btc-evidence.json',JSON.stringify(evidence,null,2));console.log(evidence);
