import test from "node:test";
import assert from "node:assert/strict";
import { advanceKrakenEw, type KrakenEwState } from "./krakenEw.ts";
import { getKrakenProfile, type KrakenEwGenes } from "./krakenProfiles.ts";
import { advanceKrakenPaper, createKrakenPaperState } from "./krakenPaper.ts";
import { attachKrakenEwGenes, inheritKrakenProfile } from "./krakenEvolution.ts";
import { SOURCE_PARENT_GENOMES } from "./sourceStrategies.ts";
import type { MarketBarInput } from "./paperReplay.ts";
const genes:KrakenEwGenes={filter:true,exit:true,pivotLength:2,atrLength:5,minMoveAtr:.25,retraceMin:.382,retraceMax:.786};
test("EW retracement bias is unavailable until right-hand pivot confirmation, with exact prefix invariance",()=>{
 const prices=[100,99,98,99,100,101,102,101,100,101,102];let state:KrakenEwState|undefined;
 const observed=[];
 for(const close of prices){const r=advanceKrakenEw(state,{high:close+.1,low:close-.1,close},genes);state=r.state;observed.push(JSON.parse(JSON.stringify(state)));}
 assert.equal(observed[9].bias,0);assert.equal(observed[10].bias,1);assert.equal(observed[10].pivots.at(-1).index,8);
 for(const future of [[150,160],[50,40]]){let s:KrakenEwState|undefined;const prefix=[];for(const close of [...prices,...future]){s=advanceKrakenEw(s,{high:close+.1,low:close-.1,close},genes).state;if(prefix.length<prices.length)prefix.push(JSON.parse(JSON.stringify(s)));}assert.deepEqual(prefix,observed);}
});
test("EW source flags attach frozen gene values and actually filter unconfirmed IMBA entries",()=>{
 const source={...SOURCE_PARENT_GENOMES[2]!,pivotLength:2,atrLength:5,ewMinMoveAtr:.25};
 const profile=attachKrakenEwGenes(getKrakenProfile("kraken_17_2_close"),source);
 const child=inheritKrakenProfile("hybrid",[profile],"birth","child",source);assert.deepEqual(child.ew,profile.ew);
 const data:MarketBarInput[]=Array.from({length:36},(_,i)=>{const p=100+i*.02;return{timestamp:new Date(Date.parse("2026-01-01T00:00:00Z")+i*60000).toISOString(),open:String(p),high:String(p),low:String(p),close:String(p),volume:"100"};});
 const unfiltered=getKrakenProfile("kraken_17_2_close");const a=advanceKrakenPaper(createKrakenPaperState(unfiltered.id,100),data,unfiltered,90),b=advanceKrakenPaper(createKrakenPaperState(profile.id,100),data,profile,90);
 assert.ok(a.fills.some(f=>f.reason==="imba_entry"));assert.equal(b.fills.length,0);assert.equal(b.state.ew?.bias,0);assert.equal(b.state.ew?.samples,7);
});
test("confirmed sixth EW pivot emits an executable next-open impulse exit",()=>{
 const p={...getKrakenProfile("kraken_17_2_close"),ew:{...genes,filter:false}};
 const seed:KrakenEwState={samples:14,previousClose:112.8,atr:1,trSum:14,bias:1,window:[{index:12,high:112,low:111,close:111.5},{index:13,high:113,low:112,close:112.8}],pivots:[{index:0,price:90,high:false},{index:2,price:100,high:true},{index:4,price:95,high:false},{index:7,price:110,high:true},{index:10,price:103,high:false}]};
 const values=[114.5,113.5,112.5];const data:MarketBarInput[]=values.flatMap((close,bucket)=>Array.from({length:5},(_,i)=>({timestamp:new Date(Date.parse("2026-01-01T00:00:00Z")+(bucket*5+i)*60000).toISOString(),open:String(close),high:String(close+.5),low:String(close-.5),close:String(close),volume:"100"}))).concat([{timestamp:"2026-01-01T00:15:00.000Z",open:"112.5",high:"112.5",low:"112.5",close:"112.5",volume:"100"}]);
 const state={...createKrakenPaperState(p.id,100),quantity:.1,entryPrice:112,lastMarkPrice:112,bestFavorable:112,mode:"WAIT" as const,renkoClose:112,imbaTrend:1 as const,bricks:[{high:130,low:90,close:112}],ew:seed};
 const result=advanceKrakenPaper(state,data,p,90);assert.ok(result.fills.some(f=>f.reason==="ew_confirmed_impulse_exit"),JSON.stringify(result.fills));assert.equal(result.state.quantity,0);assert.equal(result.fills.at(-1)?.executedAt,"2026-01-01T00:15:00.000Z");
});
