import test from "node:test";
import assert from "node:assert/strict";
import type { MarketBarInput } from "./paperReplay.ts";
import { getKrakenProfile, type KrakenProfile } from "./krakenProfiles.ts";
import { advanceKrakenPaper, createKrakenPaperState, resumeKrakenPaperAfterGap, type KrakenPaperState } from "./krakenPaper.ts";
const START = Date.parse("2026-01-01T00:00:00Z");
function bars(prices: readonly number[], start = START, volume = "100"): MarketBarInput[] {
  return prices.map((price, i) => ({ timestamp: new Date(start + i * 60_000).toISOString(), open:String(price),high:String(price),low:String(price),close:String(price),volume }));
}
function positioned(p:KrakenProfile,side=1):KrakenPaperState {
 return {...createKrakenPaperState(p.id,100),quantity:side*.1,entryPrice:100,lastMarkPrice:100,renkoClose:100,imbaTrend:1,bestFavorable:100,bricks:[{high:110,low:90,close:100}]};
}
test("CT positive bootstrap enters long causally at actual next minute open with costs and cap",()=>{
 const p=getKrakenProfile("kraken_17_2_close");const r=advanceKrakenPaper(createKrakenPaperState(p.id,100),bars([...Array(5).fill(100),...Array(6).fill(101)]),p,90);
 assert.ok(r.fills.length>0);assert.equal(r.fills[0]!.side,"buy");assert.equal(r.fills[0]!.reason,"imba_entry");assert.equal(r.fills[0]!.executedAt,"2026-01-01T00:10:00.000Z");assert.ok(r.fills[0]!.quantity*r.fills[0]!.price<=90);assert.ok(r.fees>0);
});
test("persistent replay identical whole vs UTC day and arbitrary JSON chunk boundaries",()=>{
 const p=getKrakenProfile("kraken_17_2_reverse"),data=bars(Array.from({length:2880},(_,i)=>100+Math.sin(i/43)*3+i/10000));const whole=advanceKrakenPaper(createKrakenPaperState(p.id,100),data,p,90);
 let state=createKrakenPaperState(p.id,100),net=0,fees=0;const fills=[];
 for(const [a,b] of [[0,1439],[1439,1440],[1440,1603],[1603,2880]]){const part=advanceKrakenPaper(JSON.parse(JSON.stringify(state)),data.slice(a,b),p,90);state=part.state;net+=part.netPnl;fees+=part.fees;fills.push(...part.fills);}
 assert.deepEqual(state,whole.state);assert.deepEqual(fills,whole.fills);assert.ok(Math.abs(net-whole.netPnl)<1e-10);assert.ok(Math.abs(fees-whole.fees)<1e-10);assert.ok(whole.fills.length>5);
});
test("TTP reverses to WAIT; WAIT SL toggle controls close vs reversal",()=>{
 for(const id of ["kraken_17_2_close","kraken_17_2_reverse"]){const p=getKrakenProfile(id),s=positioned(p);s.bestFavorable=103;const r=advanceKrakenPaper(s,bars([...Array(5).fill(102),...Array(5).fill(102),...Array(6).fill(104.1)]),p,90);const trail=r.fills.filter(f=>f.reason==="ttp_reverse");assert.equal(trail.length,2);assert.ok(trail.every(f=>f.side==="sell"));assert.ok(r.fills.some(f=>f.reason==="wait_sl"),JSON.stringify(r.fills));assert.equal(r.state.quantity===0,!p.reverseWaitSl);if(p.reverseWaitSl)assert.equal(r.state.mode,"WAIT");}
});
test("independent CT TP1 and partial TP own flags and actual reducing fills",()=>{
 const p=getKrakenProfile("kraken_17_2_close"),r=advanceKrakenPaper(positioned(p),bars([...Array(5).fill(103.3),103.3]),p,90);assert.deepEqual(r.fills.map(f=>f.reason),["ct_tp1","ct_partial_tp"]);assert.ok(Math.abs(r.state.quantity-.0025)<1e-12);assert.equal(r.state.tp1Done,true);assert.equal(r.state.partialDone,true);assert.ok(r.fills.every(f=>f.side==="sell"&&f.realizedPnl>0&&f.fee>0));assert.ok(Math.abs(r.netPnl-(r.markedEquity-100))<1e-10);
});
test("ambiguous OHLC hard stop and takeprofit chooses protective close",()=>{
 const p=getKrakenProfile("kraken_17_2_close"),data=bars(Array(6).fill(100));data[2]={...data[2]!,high:"104",low:"97"};const r=advanceKrakenPaper(positioned(p),data,p,90);assert.deepEqual(r.fills.map(f=>f.reason),["hard_sl"]);assert.equal(r.state.quantity,0);
});
test("gap through armed stop loses actual opening gap",()=>{
 const p=getKrakenProfile("kraken_17_2_close"),s=positioned(p);s.lastTimestampMs=START;const r=advanceKrakenPaper(resumeKrakenPaperAfterGap(s),bars([90],START+3600000),p,90);assert.equal(r.fills[0]!.reason,"hard_sl_gap");assert.equal(r.fills[0]!.referencePrice,90);assert.ok(r.fills[0]!.price<90);assert.ok(r.netPnl<-1);assert.equal(r.state.quantity,0);
});
test("predeclared final exit horizon retries zero volume and never reopens",()=>{
 const p=getKrakenProfile("kraken_17_2_close"),a=advanceKrakenPaper(positioned(p),bars([100,101],START,"0"),p,90,{exitOnly:true});assert.equal(a.state.quantity,.1);assert.equal(a.fills.length,0);assert.equal(a.state.pending.length,1);const b=advanceKrakenPaper(a.state,bars(Array(30).fill(102),START+120000),p,90,{exitOnly:true});assert.equal(b.fills.length,1);assert.equal(b.fills[0]!.reason,"archive_exit");assert.equal(b.state.quantity,0);assert.equal(b.state.pending.length,0);
});
test("TF TP1 and BE state survives to later candles",()=>{
 const p=getKrakenProfile("imbatp"),s=positioned(p);s.mode="TF";const a=advanceKrakenPaper(s,bars([...Array(5).fill(103.5),103.5]),p,90);assert.equal(a.fills[0]!.reason,"tf_tp1");assert.equal(a.state.breakEvenArmed,true);const b=advanceKrakenPaper(a.state,bars(Array(10).fill(99.9),START+360000),p,90);assert.ok(b.fills.some(f=>f.reason==="tf_be"));assert.equal(b.state.quantity,0);
});
test("minute gaps and malformed saved indicator state fail closed",()=>{
 const p=getKrakenProfile("kraken_17_2_close");assert.throws(()=>advanceKrakenPaper(createKrakenPaperState(p.id,100),bars([100]).concat(bars([100],START+120000)),p,90),/continuous/);const s=positioned(p);s.bricks[0]!.close=NaN;assert.throws(()=>advanceKrakenPaper(s,[],p,90),/Renko/);
});

test("future candles cannot change prior fills or the pending five-minute decision",()=>{
 const p=getKrakenProfile("kraken_17_2_reverse"),past=bars(Array.from({length:65},(_,i)=>100+i*.02));
 const first=advanceKrakenPaper(createKrakenPaperState(p.id,100),past,p,90);
 const futureA=bars(Array(40).fill(105),START+65*60000),futureB=bars(Array(40).fill(90),START+65*60000);
 const a=advanceKrakenPaper(createKrakenPaperState(p.id,100),past.concat(futureA),p,90),b=advanceKrakenPaper(createKrakenPaperState(p.id,100),past.concat(futureB),p,90);
 const cutoff=START+65*60000;
 assert.deepEqual(a.fills.filter(f=>Date.parse(f.executedAt)<cutoff),first.fills);
 assert.deepEqual(b.fills.filter(f=>Date.parse(f.executedAt)<cutoff),first.fills);
 assert.deepEqual(first.state.pending,advanceKrakenPaper(createKrakenPaperState(p.id,100),past,p,90).state.pending);
 assert.ok(first.fills.length>0);assert.ok(first.state.pending.every(o=>o.availableAtMs>=cutoff));
});
