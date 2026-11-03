// SPDX-License-Identifier: MPL-2.0
// Confirmed pivot/retracement/impulse modules derived from EW_v3_2_MTF_AUDIT.pine.
// Elliott Wave Advanced Multi-Degree Analyzer — v3.2 MTF Edition.
// Selected causal single-timeframe modules; not a full MTF implementation.
import type { KrakenEwGenes } from "./krakenProfiles.ts";
type Sample = { index: number; high: number; low: number; close: number };
type Pivot = { index: number; price: number; high: boolean };
export type KrakenEwState = {
  samples: number; previousClose: number; atr: number; trSum: number;
  window: Sample[]; pivots: Pivot[]; bias: -1 | 0 | 1;
};
export function validateKrakenEw(genes: KrakenEwGenes, state?: KrakenEwState): void {
  if (!genes || typeof genes.filter !== "boolean" || typeof genes.exit !== "boolean"
    || !Number.isInteger(genes.pivotLength) || genes.pivotLength < 2 || genes.pivotLength > 20
    || !Number.isInteger(genes.atrLength) || genes.atrLength < 5 || genes.atrLength > 50
    || !Number.isFinite(genes.minMoveAtr) || genes.minMoveAtr < .25 || genes.minMoveAtr > 20
    || !Number.isFinite(genes.retraceMin) || !Number.isFinite(genes.retraceMax)
    || genes.retraceMin < .1 || genes.retraceMin > .8 || genes.retraceMax < .2 || genes.retraceMax > .95
    || genes.retraceMin >= genes.retraceMax) throw new RangeError("Invalid confirmed EW genes.");
  if (!state) return;
  if (!Number.isSafeInteger(state.samples) || state.samples < 0 || ![state.previousClose,state.atr,state.trSum].every(v=>Number.isFinite(v)&&v>=0)
    || !Array.isArray(state.window) || state.window.length > 41 || !Array.isArray(state.pivots) || state.pivots.length > 100
    || ![-1,0,1].includes(state.bias)) throw new RangeError("Invalid persistent EW state.");
  for (const sample of state.window) if (!Number.isSafeInteger(sample.index) || sample.index < 0 || sample.index >= state.samples
    || ![sample.high,sample.low,sample.close].every(v=>Number.isFinite(v)&&v>0) || sample.high < sample.low
    || sample.close > sample.high || sample.close < sample.low) throw new RangeError("Invalid EW sample.");
  for (const pivot of state.pivots) if (!Number.isSafeInteger(pivot.index) || pivot.index < 0 || pivot.index >= state.samples
    || !Number.isFinite(pivot.price) || pivot.price <= 0 || typeof pivot.high !== "boolean") throw new RangeError("Invalid EW pivot.");
}
/** A pivot at n-L only becomes available after candle n closes. ATR and all samples are observed at n. */
export function advanceKrakenEw(input: KrakenEwState | undefined, candle: {high:number;low:number;close:number}, genes: KrakenEwGenes): {state:KrakenEwState; waveExit:-1|0|1; confirmedPivots:number} {
  validateKrakenEw(genes,input);
  const state: KrakenEwState = input ? { ...input, window: [...input.window], pivots: [...input.pivots] }
    : { samples:0, previousClose:0, atr:0, trSum:0, window:[],pivots:[],bias:0 };
  const {high,low,close}=candle;
  if (![high,low,close].every(v=>Number.isFinite(v)&&v>0) || low>close || high<close || high<low) throw new RangeError("Invalid completed EW candle.");
  const index=state.samples, previous=state.previousClose || close;
  const tr=Math.max(high-low,Math.abs(high-previous),Math.abs(low-previous));
  state.trSum+=tr;
  if(index===genes.atrLength-1)state.atr=state.trSum/genes.atrLength;
  else if(index>=genes.atrLength)state.atr=(state.atr*(genes.atrLength-1)+tr)/genes.atrLength;
  state.samples+=1;state.previousClose=close;
  state.window.push({index,high,low,close});if(state.window.length>2*genes.pivotLength+1)state.window.shift();
  let confirmedPivots=0;
  if(state.window.length===2*genes.pivotLength+1 && index>=genes.atrLength-1){
    const center=state.window[genes.pivotLength]!;
    for(const isHigh of [true,false]){
      const price=isHigh?center.high:center.low;
      const valid=state.window.every(other=>other.index===center.index || (isHigh
        ? other.index<center.index?other.high<=price:other.high<price
        : other.index<center.index?other.low>=price:other.low>price));
      if(!valid)continue;
      const last=state.pivots[state.pivots.length-1];
      if(!last || (last.high!==isHigh && Math.abs(price-last.price)>=state.atr*genes.minMoveAtr)){
        state.pivots.push({index:center.index,price,high:isHigh});confirmedPivots+=1;
        if(state.pivots.length>100)state.pivots.shift();
      }else if(last.high===isHigh && (isHigh?price>last.price:price<last.price))state.pivots[state.pivots.length-1]={index:center.index,price,high:isHigh};
    }
  }
  let waveExit:-1|0|1=0;
  if(confirmedPivots && state.pivots.length>=3){
    const [a,b,c]=state.pivots.slice(-3) as [Pivot,Pivot,Pivot];
    const bull=a.price<b.price,retracement=Math.abs(c.price-b.price)/Math.abs(b.price-a.price);
    if(retracement>=genes.retraceMin && retracement<=genes.retraceMax && (bull?c.price>a.price:c.price<a.price))state.bias=bull?1:-1;
    if(state.pivots.length>=6){
      const [p0,p1,p2,p3,p4,p5]=state.pivots.slice(-6) as [Pivot,Pivot,Pivot,Pivot,Pivot,Pivot];
      const bull=p0.price<p1.price,w1=Math.abs(p1.price-p0.price),w3=Math.abs(p3.price-p2.price),w5=Math.abs(p5.price-p4.price);
      const holds=bull?p2.price>p0.price&&p4.price>p1.price:p2.price<p0.price&&p4.price<p1.price;
      if(holds && !(w3<w1 && w3<w5))waveExit=bull?1:-1;
    }
  }
  return {state,waveExit,confirmedPivots};
}
