import test from "node:test";
import assert from "node:assert/strict";
import { getKrakenProfile } from "./krakenProfiles.ts";
import { inheritKrakenProfile, validateEvolvedKrakenProfile } from "./krakenEvolution.ts";
import { advanceKrakenPaper, createKrakenPaperState } from "./krakenPaper.ts";
import type { MarketBarInput } from "./paperReplay.ts";

test("frozen Kraken birth genetics are deterministic, bounded and policy-distinct with compulsory safety",()=>{
 const parents=[getKrakenProfile("kraken_legacy"),getKrakenProfile("kraken_17_2_close")];
 const results=["random","local","hive","hybrid"].map(mode=>inheritKrakenProfile(mode as "random"|"local"|"hive"|"hybrid",parents,"birth-seed","child1"));
 assert.deepEqual(results[1],inheritKrakenProfile("local",parents,"birth-seed","child1"));
 assert.ok(new Set(results.map(p=>JSON.stringify([p.channelLookback,p.ttpTrailPct,p.partialTpPct]))).size===4);
 for(let i=0;i<100;i++){const child=inheritKrakenProfile("hybrid",parents,"birth-seed",`child${i}`);validateEvolvedKrakenProfile(child);assert.equal(child.paperOverrides.maxNotionalMultiple,1);assert.ok(child.hardSlPct>0&&child.hardSlPct<=2.5);assert.equal(child.paperOverrides.mandatoryHardSlPct,2.5);assert.equal(child.channelLookback,Math.round(child.originalSensitivity*10));}
});
test("inherited TTP parameters change real next-open executions rather than labels alone",()=>{
 const parent=getKrakenProfile("kraken_legacy"),child=inheritKrakenProfile("local",[parent],"frozen-birth","child1");
 assert.notEqual(parent.ttpTrailPct,child.ttpTrailPct);
 const close=103*(1-(parent.ttpTrailPct+child.ttpTrailPct)/2/100);
 const data:MarketBarInput[]=Array.from({length:6},(_,i)=>({timestamp:new Date(Date.parse("2026-01-01T00:00:00Z")+i*60000).toISOString(),open:String(close),high:String(close),low:String(close),close:String(close),volume:"100"}));
 const run=(p:typeof parent)=>advanceKrakenPaper({...createKrakenPaperState(p.id,100),quantity:.1,entryPrice:100,lastMarkPrice:100,bestFavorable:103,renkoClose:100,imbaTrend:1,bricks:[{high:110,low:90,close:100}]},data,p,90);
 const a=run(parent),b=run(child);assert.notEqual(a.fills.some(f=>f.reason==="ttp_reverse"),b.fills.some(f=>f.reason==="ttp_reverse"));
});
test("birth inheritance rejects malformed request and unsafe mutated profile",()=>{
 assert.throws(()=>inheritKrakenProfile("local",[],"seed","child"),/Invalid/);
 const p=inheritKrakenProfile("local",[getKrakenProfile("kraken_17_2_close")],"seed","child");
 assert.throws(()=>validateEvolvedKrakenProfile({...p,hardSlPct:0}),/bounds/);
 assert.throws(()=>validateEvolvedKrakenProfile({...p,paperOverrides:{...p.paperOverrides,maxNotionalMultiple:10 as 1}}),/Unsafe/);
});
