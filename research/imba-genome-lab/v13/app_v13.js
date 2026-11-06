/* Quant Genome v13 — every candidate belongs to an individually simulated daily curve.
 * Analytical 1x charts, research 10x treasury, no TradingView execution claims.
 */
/* Quant Genome v13 — every candidate belongs to an individually simulated daily curve.
 * Analytical 1x charts, research 10x treasury, no TradingView execution claims.
 */
const $=id=>document.getElementById(id);
const N=META.length;
let DAYS=0, W=0, DATES=[], F=null, PREFIX=null;
const FAMILY_COLORS={none:'#75bafa',tp:'#f2af66',ttp:'#65d7ad',both:'#cf9bff'};
const GENE_ORDER=[['imba','⌁','IMBA','#48cbd5'],['renko','▦','RENKO','#70a8ed'],['candle','▤','CANDLES','#78d3ef'],['tp','↗','TP','#ecb56e'],['ttp','↝','TTP','#b995ef']];
const EXIT_NAMES=['Ohne TP/TTP','TP 6 %','TP 4 %','TP 3 %','TP6 + TTP3/1,5','TP 2 %','TP4 + TTP2/1','TTP 3/1,5','TTP 2/1'];
const AXES={s:'Sensitivity',b:'Renko-Box (USD)',tp:'TP-Ziel (%)',act:'TTP-Aktivierung (%)',trail:'TTP-Abstand (%)',exit:'Exit-Programm'};
const state={priceMode:'renko_usd',day:0,horizon:30,x:'s',y:'b',mode:'all4158',family:'all',familyEnabled:{none:true,tp:true,ttp:true,both:true},box:'all',dataset:'2026',
  aggregate:'max',chart:'3d',points:'top6',guard:30,rank:14,topN:6,stopScope:'strategy',corrLimit:null,corrFallback:false,manual:new Set(),selected:-1,
  playing:null,dir:1,port:null,portKey:'',filtered:[],gridN:0,cacheKey:'',grid:null,rendering:false,pending:false,
  smooth:1,playMs:650,sidebarOpen:true,surfaceMode:'overlay',layers:{none:true,tp:true,ttp:true,both:true},layerOpacity:0.62};
const IDMAP=new Map(META.map((m,i)=>[m.id,i]));
const YEAR_RAW_POOLS={"2022":[3169,1557,2971,2773,2664,1737,2286,2853,1917,1728,1746,1377,2575,2774,2972,1594,1397,2576,2106,622,2277,1539,1926,1566,2973,2475,1179,3618,1396,2578,2295,2775,2655,2466,1398,1755,3807,1521,820,2835,3222,2844,1368,2776,1530,1792,3249,2088,1796,2871,2974,1011,1701,1953,244,1399,1170,3609,1797,2577,1019,1710,64,2682,1548,3231,2259,271,4005,2448,1795,2313,1719,1908,1009,3429,3368,109,1764,1794,3170,823,2097,670,1018,1935,2646,3240,2673,1575,2691,3171,2862,625,2304,1596,1597,3600,4023,3042],"2023":[3645,3267,2700,261,36,3834,3456,2331,2889,1017,3078,4032,2520,2709,2340,3087,639,1971,2160,2349,2529,2898,3465,2538,3843,828,3276,3303,3870,3681,3492,2934,2745,2565,3123,2943,3132,2718,3312,3501,3114,3654,4059,2754,90,3672,3879,252,2367,4041,2169,2358,2547,3096,3285,3474,2727,3852,243,2907,2115,3051,1188,2484,3816,2925,2736,1404,2295,3627,1395,1755,2916,279,1935,4050,801,3240,2673,2556,63,3366,2862,3321,621,45,792,3483,4068,99,1953,4014,3510,4023,3861,3294,3249,3141,2304,999],"2024":[966,1864,775,3025,786,1878,955,2458,2262,1522,2047,1137,2836,970,787,3030,790,3027,1486,967,1509,1651,82,1650,1126,766,1884,965,910,388,1338,784,3001,946,1138,1342,1306,748,1849,1297,1702,3412,1141,2431,820,442,1675,1135,1882,2038,1875,1111,3784,1846,2812,1876,1506,1315,2260,3026,1657,562,1877,1108,1891,568,3000,964,739,1654,262,580,3215,1099,1272,1880,751,2044,3223,2050,1495,2415,1840,1084,2449,2265,2046,1879,2071,1090,1504,3199,1513,969,1508,3028,446,372,1291,3196],"2025":[2838,2837,2839,2835,2842,3996,1336,1028,751,3406,3226,1029,922,2830,3225,1339,931,1156,918,651,3613,3609,1327,1335,3809,3612,1325,729,840,2443,650,4000,3604,580,3229,1218,2836,3224,1892,2833,747,1333,3802,3222,3409,576,1033,462,934,3415,1309,567,748,925,3999,738,4003,754,655,3963,2448,1165,2829,461,3616,3422,391,2388,2056,3217,2840,919,1159,1705,2841,1330,2071,3405,1334,1172,1489,3998,583,742,3805,733,1894,1278,1155,839,281,273,2846,3610,577,1291,1102,3404,2828,2648]};
const SCREENSHOT_PRESETS={"original_2026_14d_20":{"label":"Original 462 · 14 Tage · Einzelstopp 20 %","pool":"original462","rank":14,"guard":20,"screenshot":"Screenshot 2 (09:41:04)","wealth_at_1000":9367.04925136168,"verified":true},"balanced_2026_30d_30":{"label":"Top100 Balanced · 30 Tage · Einzelstopp 30 %","pool":"balanced100","rank":30,"guard":30,"screenshot":"Screenshot 1 (09:39:49)","wealth_at_1000":7242.873165803406,"verified":true},"balanced_2026_30d_25":{"label":"Top100 Balanced · 30 Tage · Einzelstopp 25 %","pool":"balanced100","rank":30,"guard":25,"screenshot":"Screenshot 3 (09:41:58)","wealth_at_1000":10654.973910559,"verified":true},"raw_2026_30d_30_comparison":{"label":"Top100 Raw · 30 Tage · Einzelstopp 30 % (Vergleich)","pool":"top100","rank":30,"guard":30,"screenshot":null,"wealth_at_1000":3493.6135685711342,"verified":true,"note":"Keiner der drei Screenshots zeigt Raw; Vergleichsannahme aus früheren Tests."}};
function decodeF32(b64){const raw=atob(b64), bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);return new Float32Array(bytes.buffer);}
async function loadYear(year){
  if(!YEAR_GZIP[year])throw Error('Keine Backtest-Daten für '+year);
  if(typeof DecompressionStream!=='function')throw Error('Dieser Browser kann gzip-Daten nicht entpacken; bitte aktuelles Chrome verwenden.');
  const b64=YEAR_GZIP[year],raw=atob(b64), bytes=new Uint8Array(raw.length);
  for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);
  const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  const arrbuf=await new Response(stream).arrayBuffer();
  DAYS=YEAR_LENGTHS[year];W=90+DAYS;
  if(arrbuf.byteLength!==N*W*4)throw Error(`Jahresdaten fehlerhaft ${year}: ${arrbuf.byteLength} Bytes`);
  F=new Float32Array(arrbuf);
  const start=new Date(Date.UTC(Number(year),0,1));
  DATES=Array.from({length:DAYS},(_,i)=>{const d=new Date(start.getTime()+i*86400000);return d.toISOString().slice(0,10);});
  PREFIX=new Float64Array(N*(W+1));
  for(let i=0;i<N;i++){const src=i*W,dst=i*(W+1);for(let j=0;j<W;j++)PREFIX[dst+j+1]=PREFIX[dst+j]+Math.log(Math.max(F[src+j],1e-15));}
  state.dataset=year;state.day=DAYS-1;state.portKey='';state.port=null;state.selected=-1;
  $('slider').max=String(DAYS-1);$('slider').value=String(state.day);$('dataset').value=year;
  $('inspectTitle').textContent='Heatmap-Zelle oder Bot anklicken';
}
function dayFactor(i,d){return Number(F[i*W+90+d]);}
function rPct(i,end,window){ // end = index of last included day relative to 2026 Jan1; -1 is 2025-12-31
  const p=90+end+1,left=window?Math.max(0,p-window):90;
  if(p<=left)return 0;
  const start=i*(W+1);
  return 100*Math.expm1(PREFIX[start+p]-PREFIX[start+left]);
}
function value(i){return rPct(i,state.day,state.horizon);}
function score(i,day){return rPct(i,day-1,state.rank);}
function number(v,dec=1){return Number(v).toLocaleString('de-DE',{minimumFractionDigits:dec,maximumFractionDigits:dec});}
function euro(v){return Number(v).toLocaleString('de-DE',{style:'currency',currency:'EUR',maximumFractionDigits:2});}
function pct(v){return (v>0?'+':'')+number(v,1)+' %';}
function geneValue(i,k){const m=META[i];return m[k];}
function realAxisValue(i,k){const m=META[i];if(k==='tp')return m.tpValue;if(k==='act')return m.act;if(k==='trail')return m.trail;return Number(m[k]);}
function getPresetIds(){
 if(state.mode==='original462')return ORIGINAL;
 if(state.mode==='top100')return RAW100;
 if(state.mode==='balanced100')return BAL100;
 if(state.mode==='manual')return [...state.manual];
 if(state.mode.startsWith('rawYear')){const vintage=state.mode.slice(7);return Number(state.dataset)>Number(vintage)?(YEAR_RAW_POOLS[vintage]||[]):[];}
 return ALL;
}
function pickAxisCoords(vals,n){
 const uniq=[...new Set(vals)].sort((a,b)=>a-b);
 if(uniq.length<=n)return uniq;
 const chosen=new Set();for(let j=0;j<n;j++)chosen.add(uniq[Math.round(j*(uniq.length-1)/(n-1))]);
 return [...chosen];
}
// Always build the exact tested universe first; gene-family gates are applied afterwards.
function eligibleBase(){
 let ids=getPresetIds().filter(i=>META[i].priceMode===state.priceMode);
 if(state.box!=='all' && state.priceMode!=='candle')ids=ids.filter(i=>Math.abs(META[i].b-Number(state.box))<.00001);
 if(state.mode==='grid20'||state.mode==='grid30'){
  const k=state.mode==='grid20'?20:30;
  const xv=new Set(pickAxisCoords(ids.map(i=>realAxisValue(i,state.x)),k));
  const yv=new Set(pickAxisCoords(ids.map(i=>realAxisValue(i,state.y)),k));
  ids=ids.filter(i=>xv.has(realAxisValue(i,state.x))&&yv.has(realAxisValue(i,state.y)));
 }
 return ids;
}
function eligible(){return eligibleBase().filter(i=>state.familyEnabled[META[i].family]);}
function familySelectionMode(){const fams=['none','tp','ttp','both'].filter(f=>state.familyEnabled[f]);return fams.length===4?'all':fams.length===1?fams[0]:'custom';}
function selectFamilyOnly(fam){for(const f of ['none','tp','ttp','both'])state.familyEnabled[f]=f===fam;state.family=fam;setControls();changePool();}
function setAllFamilies(){for(const f of ['none','tp','ttp','both'])state.familyEnabled[f]=true;state.family='all';setControls();changePool();}
function setFamilyDropdown(value){if(value==='all'){setAllFamilies();return;}if(['none','tp','ttp','both'].includes(value)){selectFamilyOnly(value);return;}state.family=familySelectionMode();setControls();changePool();}
function populateSelector(sel,items,selected){$(sel).innerHTML=items.map(([v,txt])=>`<option value="${v}">${txt}</option>`).join('');$(sel).value=selected;}
function initOptions(){
 populateSelector('axisX',Object.entries(AXES),'tp');populateSelector('axisY',Object.entries(AXES),'s');
 populateSelector('boxFilter',[['all','Alle Boxgrößen (Aggregation)'],...BOXES.map(b=>[b.toFixed(2),b.toFixed(2)+' USD (exakt)'])],'all');
}
function sameCell(a,b){return Number(a)===Number(b);}
function buildHeatmap(){
 const ids=state.filtered;
 const xs=[...new Set(ids.map(i=>realAxisValue(i,state.x)))].sort((a,b)=>a-b);
 const ys=[...new Set(ids.map(i=>realAxisValue(i,state.y)))].sort((a,b)=>a-b);
 const groups=new Map();
 for(const i of ids){const x=realAxisValue(i,state.x),y=realAxisValue(i,state.y);const key=x+'|'+y;
  if(!groups.has(key))groups.set(key,[]);groups.get(key).push(i);}
 const z=[],info=[],custom=[],counts=[],winners=[];
 let occupied=0;
 for(const y of ys){const row=[],ir=[],cr=[],nr=[],wr=[];
  for(const x of xs){const botids=groups.get(x+'|'+y)||[];
   if(!botids.length){row.push(null);ir.push('Nicht getestet');cr.push(null);nr.push(0);wr.push(-1);continue;}
   occupied++;let vals=botids.map(i=>value(i));let best=botids[0],bestValue=-Infinity,sum=0;
   for(let k=0;k<botids.length;k++){sum+=vals[k];if(vals[k]>bestValue){bestValue=vals[k];best=botids[k];}}
   const zval=state.aggregate==='max'?bestValue:sum/vals.length;
   row.push(zval);cr.push(best);nr.push(botids.length);wr.push(best);
   ir.push(`${botids.length} tatsächlich berechnete Strategien<br>Wert (${state.aggregate==='max'?'beste':'Durchschnitt'}): ${pct(zval)}<br>Bester Bot: ${META[best].id}<br>Rendite: ${pct(bestValue)}`);
  }
  z.push(row);info.push(ir);custom.push(cr);counts.push(nr);winners.push(wr);
 }
 return {xs,ys,z,info,custom,counts,winners,groups,occupied};
}
function buildFamilyHeatmap(fam){
 const xs=state.grid.xs,ys=state.grid.ys;
 const groups=new Map();
 for(const i of state.filtered){if(META[i].family!==fam)continue;
   const key=realAxisValue(i,state.x)+'|'+realAxisValue(i,state.y);
   if(!groups.has(key))groups.set(key,[]);groups.get(key).push(i);
 }
 const z=[],info=[],custom=[];let occupied=0;const seenX=new Set(),seenY=new Set();
 for(const y of ys){const rz=[],ri=[],rc=[];
   for(const x of xs){const ids=groups.get(x+'|'+y)||[];
     if(!ids.length){rz.push(null);ri.push('Keine getestete Kombination');rc.push(null);continue;}
     seenX.add(x);seenY.add(y);occupied++;
     let sum=0,best=ids[0],mx=-Infinity;
     for(const i of ids){const v=value(i);sum+=v;if(v>mx){mx=v;best=i;}}
     const v=state.aggregate==='max'?mx:sum/ids.length;
     rz.push(v);rc.push(best);
     ri.push(`${familyName(fam)} · ${ids.length} echte Strategien<br>Rendite ${pct(v)}<br>Bester Bot ${META[best].id}<br>Bestwert ${pct(mx)}`);
   }
   z.push(rz);info.push(ri);custom.push(rc);
 }
 return {xs,ys,z,info,custom,occupied,seenX,seenY};
}
function pearsonPrevious28(a,b,day){
  // 28 COMPLETE daily 1× returns before selection date; no current-day future prices.
  const start=90+day-28,end=90+day,n=28,baseA=a*W,baseB=b*W;
  let sx=0,sy=0,sxx=0,syy=0,sxy=0;
  for(let t=start;t<end;t++){
    const x=F[baseA+t]-1,y=F[baseB+t]-1;
    sx+=x;sy+=y;sxx+=x*x;syy+=y*y;sxy+=x*y;
  }
  const xx=sxx-sx*sx/n,yy=syy-sy*sy/n;
  if(xx<1e-18||yy<1e-18)return 0;
  return Math.max(-1,Math.min(1,(sxy-sx*sy/n)/Math.sqrt(xx*yy)));
}
function applyCorrelationCap(ranked,day,topN=state.topN,corrLimit=state.corrLimit,corrFallback=state.corrFallback){
 if(corrLimit===null||!Number.isFinite(corrLimit))return ranked.slice(0,topN);
 const selected=[];
 for(const candidate of ranked){
  if(selected.length>=topN)break;
  if(selected.every(s=>pearsonPrevious28(s[0],candidate[0],day)<=corrLimit))selected.push(candidate);
 }
 if(corrFallback&&selected.length<topN){
  const chosen=new Set(selected.map(s=>s[0]));
  for(const c of ranked){if(selected.length>=topN)break;if(!chosen.has(c[0])){selected.push(c);chosen.add(c[0]);}}
 }
 return selected;
}
function simulatePortfolio(config,selections=null,summaryOnly=false){
 const pool=config.pool,rank=config.rank,guard=config.guard,topN=config.topN;
 const stopScope=config.stopScope,corrLimit=config.corrLimit,corrFallback=config.corrFallback;
 let reserve=1000,cash=0,basis=0,paid=0,payoutN=0,stopN=0,milestone=0;
 const h=[0,0,0,0,0,0],ids=[-1,-1,-1,-1,-1,-1],stopped=[false,false,false,false,false,false];
 const idx=[1,1,1,1,1,1],hi=[1,1,1,1,1,1];
 let sleeveIndex=1,sleevePeak=1,sleeveStopped=false;
 let wealthPeak=1000,maxDrawdownPct=0,wealth=1000;
 const results=summaryOnly?null:[],sum=a=>a.reduce((x,y)=>x+y,0);
 for(let d=0;d<DAYS;d++){
  const date=DATES[d],newMonth=date.endsWith('-01'),newCycle=d%14===0;
  let sub=0;
  if(newMonth){sub=Math.min(reserve,Math.max(100,.2*reserve));reserve-=sub;cash+=sub;basis+=sub;}
  if(newCycle){
   cash+=sum(h);h.fill(0);ids.fill(-1);stopped.fill(false);idx.fill(1);hi.fill(1);
   sleeveIndex=1;sleevePeak=1;sleeveStopped=false;
   let chosen;
   if(selections)chosen=selections.get(rank+'|'+topN+'|'+d)||[];
   else{
    const ranked=pool.map(i=>[i,rPct(i,d-1,rank)]).filter(v=>v[1]>0).sort((a,b)=>b[1]-a[1]||a[0]-b[0]);
    chosen=applyCorrelationCap(ranked,d,topN,corrLimit,corrFallback).map(v=>v[0]);
   }
   chosen.forEach((id,k)=>ids[k]=id);
   if(chosen.length&&cash>0){for(let k=0;k<chosen.length;k++)h[k]=cash/chosen.length;cash=0;}
  }else if(sub>0&&!sleeveStopped){
   const active=ids.filter((i,k)=>i>=0&&!stopped[k]).length;
   if(active){for(let k=0;k<topN;k++)if(ids[k]>=0&&!stopped[k])h[k]+=sub/active;cash-=sub;}
  }
  const beforeReturns=cash+sum(h);
  for(let k=0;k<topN;k++)if(ids[k]>=0&&!stopped[k]&&h[k]>1e-10){
   const factor=Math.max(0,1+10*(dayFactor(ids[k],d)-1));h[k]*=factor;idx[k]*=factor;hi[k]=Math.max(hi[k],idx[k]);
  }
  const afterReturns=cash+sum(h);
  // Portfolio stop uses performance-only trading capital, not the protected reserve.
  // New deposits and profit transfers must NOT generate artificial drawdowns.
  if(stopScope==='portfolio'&&!sleeveStopped&&beforeReturns>1e-10){
   sleeveIndex*=Math.max(0,afterReturns/beforeReturns);
   sleevePeak=Math.max(sleevePeak,sleeveIndex);
  }
  const nav=cash+sum(h),profit=Math.max(0,nav+paid-basis),ratio=basis>0?profit/basis:0;
  if(basis>0&&ratio>=Math.max(1,milestone+1)-1e-10){
   const take=Math.min(nav*.5,.5*profit);
   if(take>1e-10){
    const c=Math.min(cash,take);cash-=c;const rest=take-c;
    if(rest>1e-10&&sum(h)>0){
     const scale=Math.max(0,1-rest/sum(h));for(let k=0;k<topN;k++)h[k]*=scale;
    }
    reserve+=take;paid+=take;payoutN++;milestone++;
   }
  }
  if(guard>0){
   if(stopScope==='portfolio'){
    if(!sleeveStopped&&sleeveIndex/Math.max(sleevePeak,1e-12)-1<=-guard/100){
     sleeveStopped=true;cash+=sum(h);h.fill(0);
     for(let k=0;k<topN;k++)if(ids[k]>=0)stopped[k]=true;
     stopN++;
    }
   }else{
    for(let k=0;k<topN;k++)if(ids[k]>=0&&!stopped[k]&&idx[k]/Math.max(hi[k],1e-12)-1<=-guard/100){
     stopped[k]=true;cash+=h[k];h[k]=0;stopN++;
    }
   }
  }
  const trading=cash+sum(h),live=ids.filter((i,k)=>i>=0&&!stopped[k]&&h[k]>1e-10);
  wealth=reserve+trading;wealthPeak=Math.max(wealthPeak,wealth);
  maxDrawdownPct=Math.min(maxDrawdownPct,100*(wealth/Math.max(wealthPeak,1e-12)-1));
  if(!summaryOnly)results.push({date,wealth,reserve,trading,cash,ids:ids.slice(),h:h.slice(),stopped:stopped.slice(),live,payoutN,stopN,
    score:ids.map(i=>i>=0?rPct(i,Math.floor(d/14)*14-1,rank):null)});
 }
 return summaryOnly?{endWealth:wealth,maxDrawdownPct,payoutN,stopN}:results;
}
function choosePortfolio(){
 const pool=state.filtered,hash=pool.join(',')+'|'+state.guard+'|'+state.rank+'|'+state.topN+'|'+state.stopScope+'|'+state.corrLimit+'|'+state.corrFallback;
 if(hash===state.portKey&&state.port)return state.port;
 state.port=simulatePortfolio({pool,rank:state.rank,guard:state.guard,topN:state.topN,stopScope:state.stopScope,
   corrLimit:state.corrLimit,corrFallback:state.corrFallback});
 state.portKey=hash;
 return state.port;
}
function refreshPool(){state.filtered=eligible();state.grid=buildHeatmap();state.portKey='';choosePortfolio();setStatus();renderChart();}
function familyName(f){return ({none:'Nur IMBA',tp:'TP ohne TTP',ttp:'TTP ohne TP',both:'TP + TTP'})[f]||'Alle';}
function setStatus(){const d=choosePortfolio()[state.day];const len=state.filtered.length;const chosen=d.ids.filter(i=>i>=0).length;
 $('candidateCount').textContent=number(len,0);$('cellCount').textContent=number(state.grid.occupied,0);$('activeCount').textContent=d.live.length+'/'+state.topN;
 $('wealth').textContent=euro(d.wealth);$('wealthChange').textContent=pct(100*(d.wealth/1000-1));
 $('reserve').textContent=euro(d.reserve);$('trading').textContent=euro(d.trading);
 $('payouts').textContent=d.payoutN;$('stops').textContent=d.stopN;
 $('dayLabel').textContent=DATES[state.day];$('slider').value=state.day;
 $('liveNote').textContent=`${chosen} gewählt · ${d.live.length} handeln · ${chosen-d.live.length} gestoppt/ohne Einsatz`;
 $('plotCountNote').textContent=`${len} echte Bot-Kurven · ${state.grid.occupied} belegte Parameterzellen · ${state.priceMode==='candle'?'5-Minuten-Candles':state.priceMode==='renko_pct'?'Prozent-Renko':'USD-Renko'} · ${state.box==='all'?'Alle getesteten Größen': 'Box '+state.box} · 10× nur im Portfolio`;
 $('corrInfo').textContent=state.corrLimit===null?'Korrelationsfilter ausgeschaltet':('Pearson-Korrelation der letzten 28 vollständigen Tage ≤ '+state.corrLimit.toFixed(2)+(state.corrFallback?' · Aufstockung kann den Grenzwert verletzen':' · strikt, auch weniger als '+state.topN+' Bots möglich'));
 $('dataWarning').textContent=`${state.dataset}: Echte Tagesrenditen für alle 7.326 Gen-Kombinationen im gewählten Marktmodus. 10× Portfolio mit eigenem Startkapital zu Jahresbeginn. Keine TradingView-/Liquidationsparität.`;
 if(state.mode.startsWith('rawYear')&&Number(state.dataset)<=Number(state.mode.slice(7)))$('dataWarning').textContent+=' Trainingsfamilie darf erst in den folgenden Jahren gehandelt werden: kein Lookahead; deshalb keine Kandidaten.';
 renderGeneCards(d);renderTop6(d);renderEquityChart();renderFamilyBreakdown(d);
 const shot=$('activeSnapshot');if(shot){const match=Object.values(SCREENSHOT_PRESETS).find(x=>state.topN===6&&state.stopScope==='strategy'&&x.pool===state.mode&&x.guard===state.guard&&x.rank===state.rank&&state.dataset==='2026'&&state.priceMode==='renko_usd'&&state.box==='all'&&state.corrLimit===null&&Object.values(state.familyEnabled).every(Boolean));shot.textContent=match?'Aktiv: '+match.label:'Aktive Einstellungen: individuell verändert / anderes Jahr';}
}
function renderFamilyBreakdown(d){
 const base=eligibleBase(), selected=d.ids.filter(i=>i>=0), live=d.live;
 const allBase=Object.fromEntries(['none','tp','ttp','both'].map(f=>[f,base.filter(i=>META[i].family===f).length]));
 const familyColors=FAMILY_COLORS;
 const labels={none:'IMBA pur',tp:'IMBA + TP',ttp:'IMBA + TTP',both:'IMBA + TP + TTP'};
 const totalEligible=state.filtered.length, listed=selected.length, trading=live.length;
 const uniqueCells=state.grid.occupied;
 const modeNames={original462:'Original 462',top100:'Top 100 aus 2022 · höchste Rendite',balanced100:'Top 100 aus 2022 · je 25 pro Exit-Familie',all4158:'Alle wirklich berechneten Strategien',grid20:'20×20 Parameter-Filter',grid30:'30×30 Parameter-Filter',manual:'Manuell gesetzte Bots',rawYear2022:'Raw Top100 · Training 2022',rawYear2023:'Raw Top100 · Training 2023',rawYear2024:'Raw Top100 · Training 2024',rawYear2025:'Raw Top100 · Training 2025'};
 const totalVisible=state.points==='all'?totalEligible:state.points==='manual'?state.manual.size:listed;
 const title=$('poolAuditTitle');if(title)title.textContent=modeNames[state.mode]||state.mode;
 const table=$('familyAudit');if(!table)return;
 table.innerHTML=['none','tp','ttp','both'].map(f=>{
  const count=allBase[f],enabled=!!state.familyEnabled[f],sel=selected.filter(i=>META[i].family===f).length,going=live.filter(i=>META[i].family===f).length;
  return `<div class="famrow ${enabled?'famOn':'famOff'}" style="--fc:${familyColors[f]}">
   <label class="famcheck"><input type="checkbox" data-trade-family="${f}" ${enabled?'checked':''}><span class="famdot"></span><strong>${labels[f]}</strong></label>
   <button type="button" class="solo" data-family-only="${f}" title="Nur diese Familie handeln">SOLO</button>
   <div class="famstats"><span><b>${count}</b> vorhanden</span><span><b>${enabled?count:0}</b> im Pool</span><span><b>${sel}</b> gewählt</span><span class="gold"><b>${going}</b> handeln</span></div>
  </div>`;
 }).join('');
 const breakdown=$('poolAuditSummary');
 if(breakdown)breakdown.innerHTML=`<strong>${number(totalEligible,0)}</strong> wirklich simulierte Kandidaten im Top-6-Ranking · <strong>${listed}/${state.topN}</strong> gewählt · <strong>${trading}/${state.topN}</strong> handeln.<br>
  <strong>${number(uniqueCells,0)}</strong> belegte Koordinaten mit <strong>${number(totalVisible,0)}</strong> aktuell eingeblendeten Bot-Markern. <span style="color:#9cb5cd">Mehrere echte Bots können dieselbe XY-Stelle besetzen.</span>`;
 const modeInfo=$('rasterExplain');
 if(modeInfo){
  const xs=new Set(state.filtered.map(i=>realAxisValue(i,state.x))),ys=new Set(state.filtered.map(i=>realAxisValue(i,state.y)));
  const pre=getPresetIds().filter(i=>META[i].priceMode===state.priceMode && (state.box==='all'||state.priceMode==='candle'||Math.abs(META[i].b-Number(state.box))<.00001));
  const limiting=state.mode==='grid20'||state.mode==='grid30';
  modeInfo.innerHTML=limiting?`<strong>Raster ist ein Filter, kein Backtest-Generator:</strong> ${state.mode==='grid20'?'20×20':'30×30'} angeforderte Achsenwerte, tatsächlich ${xs.size} × ${ys.size} unterschiedliche Achsenwerte und ${uniqueCells} belegte Zellen. Von ${pre.length} möglichen getesteten Varianten in diesem Marktmodus sind ${totalEligible} nach Familien- und Rasterfiltern zugelassen. Fehlende Kombinationen sind nicht erfunden.`:
   `Diese ${number(totalEligible,0)} Kandidaten sind eigenständige, vorab berechnete Strategien. Auf den Achsen nicht dargestellte Gene werden pro Koordinate aggregiert; deshalb sind ${uniqueCells} Heatmap-Zellen nicht gleich ${totalEligible} Strategien.`;
 }
 const heatmapMath=$('heatmapMath');
 if(heatmapMath){const w=state.horizon===0?'seit Jahresbeginn':state.horizon+' Tage';
   heatmapMath.innerHTML=`<strong>So entsteht die Fläche:</strong> X=${AXES[state.x]}, Y=${AXES[state.y]}, Höhe/Farbe=${state.aggregate==='max'?'beste':'mittlere'} tatsächlich zurückgerechnete <strong>1×-Rendite</strong> über ${w}. Pro Zelle liegen ${state.grid.occupied===state.filtered.length?'genau ein':'ein oder mehrere'} echte Bots. Die Top-${state.topN}-Neuwahl erfolgt alle 14 Tage nach dem <strong>${state.rank}-Tage-Ranking vor dem Auswahlstichtag</strong>, nicht nach der hier dargestellten aktuellen Rendite. Optische Glättung ändert keine Bot-Rendite.`;
 }
 const pointHint=$('pointHint');
 if(pointHint)pointHint.textContent=`Punktmodus: ${state.points==='all'?'alle tatsächlichen Kandidaten':state.points==='manual'?'manuelle Punkte':'nur gewählte Top '+state.topN} · ${number(totalVisible,0)} sichtbare Marker · Top-${state.topN}-Auswahl nutzt immer alle ${number(totalEligible,0)} zulässigen Strategien (nicht nur die eingeblendeten).`;
 table.querySelectorAll('[data-trade-family]').forEach(el=>el.onchange=e=>{
  state.familyEnabled[e.target.dataset.tradeFamily]=e.target.checked;
  state.family=familySelectionMode();setControls();changePool();
 });
 table.querySelectorAll('[data-family-only]').forEach(el=>el.onclick=()=>selectFamilyOnly(el.dataset.familyOnly));
}
function renderGeneCards(d){const scope=$('geneScope').value;
 const list=scope==='active'?d.live:scope==='selected'?d.ids.filter(i=>i>=0):state.filtered;
 $('genes').innerHTML=GENE_ORDER.map(([key,icon,label,color])=>{
   const count=list.filter(i=>key==='imba'||(key==='renko'&&META[i].priceMode!=='candle')||(key==='candle'&&META[i].priceMode==='candle')||META[i][key]===true).length;
   return `<div class="gene" style="--gene:${color}"><span>${icon} ${label}</span><small>${count}</small></div>`;
 }).join('');
}
function renderTop6(d){const el=$('top6List');
 const items=d.ids.map((i,k)=>i<0?null:{i,k}).filter(Boolean);
 el.innerHTML=items.length?items.map(({i,k})=>{const m=META[i],on=d.live.includes(i),p=d.h[k];
  return `<button class="botrow" data-id="${i}"><span class="statusDot" style="background:${on?'#e7be67':'#67778c'}"></span><span>${m.id}</span><strong>${on?euro(p):'Pause'}</strong></button>`;
 }).join(''):'<div class="muted">Kein Bot hat eine positive Vorperiodenrendite.</div>';
 el.querySelectorAll('button').forEach(x=>x.onclick=()=>inspect(+x.dataset.id));
}
function renderEquityChart(){const p=choosePortfolio(),d=state.day,lo=Math.max(0,d-90),arr=p.slice(lo,d+1);
 const values=arr.map(a=>a.wealth),mi=Math.min(...values),ma=Math.max(...values),scale=Math.max(ma-mi,1);
 const path=values.map((v,j)=>`${arr.length===1?0:j/(arr.length-1)*290},${42-(v-mi)/scale*36}`).join(' ');
 $('spark').innerHTML=`<svg viewBox="0 0 290 45" preserveAspectRatio="none" width="100%" height="45"><polyline points="${path}" fill="none" stroke="#5dc9cc" stroke-width="2"/></svg>`;
}
function inspect(id){if(!Number.isInteger(id)||id<0||id>=N)return;state.selected=id;
 const m=META[id],d=choosePortfolio()[state.day];const isLive=d.live.includes(id),chosen=d.ids.includes(id);
 $('inspectTitle').textContent=m.id;
 $('inspectGenes').innerHTML=`<span>⌁ IMBA S${m.s}</span><span>${m.priceMode==='candle'?'▤ Candles 5m':m.priceMode==='renko_pct'?'▦ Renko '+number(m.b,1)+' %':'▦ Renko '+number(m.b,2)+' USD'}</span>`+
 (m.tp?`<span>↗ TP ${m.tpValue}%</span>`:'')+(m.ttp?`<span>↝ TTP ${m.act}/${m.trail}%</span>`:'');
 $('inspectReturn').textContent=`${state.dataset} YTD: ${pct(rPct(id,state.day,0))} · 14T: ${pct(rPct(id,state.day,14))} · 30T: ${pct(rPct(id,state.day,30))} · 90T: ${pct(rPct(id,state.day,90))}`;
 $('inspectStatus').textContent=isLive?'● handelt':chosen?'■ gewählt, pausiert':'○ Kandidat / nicht ausgewählt';
 $('manualAdd').disabled=state.manual.has(id);$('manualRemove').disabled=!state.manual.has(id);
}
function layoutAxes(){const axis=k=>({title:{text:k==='b'?(state.priceMode==='renko_pct'?'Renko-Box (%)':state.priceMode==='candle'?'Candles · ohne Box':'Renko-Box (USD)'):AXES[k],font:{size:12}},gridcolor:'#304b61',zerolinecolor:'#304b61',showbackground:true,backgroundcolor:'#102437'});
 const x=axis(state.x),y=axis(state.y);
 if(state.x==='exit'){x.tickvals=[0,1,2,3,4,5,6,7,8];x.ticktext=['none','TP6','TP4','TP3','TP6+TTP','TP2','TP4+TTP','TTP3','TTP2'];}
 if(state.y==='exit'){y.tickvals=[0,1,2,3,4,5,6,7,8];y.ticktext=['none','TP6','TP4','TP3','TP6+TTP','TP2','TP4+TTP','TTP3','TTP2'];}
 return {x,y};}
// Display-only smoothing: 3x3 weighted convolution on existing cells;
// original backtest data, hover text, portfolio ledger and selection remain unchanged.
function softenedMesh(matrix,passes){
 if(!passes)return matrix;
 let z=matrix.map(row=>row.slice());
 for(let pass=0;pass<passes;pass++){
  const src=z;z=src.map((row,y)=>row.map((v,x)=>{
   if(v===null||!Number.isFinite(v))return null;
   let sum=4*v,weight=4;
   for(const [dy,dx] of [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]]){
    const w=(dx===0||dy===0)?2:1;
    const neighbor=src[y+dy]?.[x+dx];
    if(neighbor!==null && Number.isFinite(neighbor)){sum+=w*neighbor;weight+=w;}
   }
   return sum/weight;
  }));
 }
 return z;
}
function surfaceStyle(){return {lighting:{ambient:.85,diffuse:.72,roughness:.91,specular:.09,fresnel:.07},lightposition:{x:500,y:-250,z:900},hidesurface:false};}
function makePlotData(){const g=state.grid,{x,y}=layoutAxes(),is3=state.chart==='3d';
 const traces=[];
 const scale=[[0,'#ca596c'],[.38,'#75465f'],[.5,'#23455f'],[.65,'#26796f'],[1,'#65cea7']];
 let overlayShapes=0,degenerate=[];
 const overlay=is3&&state.surfaceMode==='overlay';
 if(overlay){
   for(const fam of ['none','tp','ttp','both']){
     if(!state.layers[fam]||!state.familyEnabled[fam])continue;
     const layer=buildFamilyHeatmap(fam);
     if(!layer.occupied)continue;
     const color=FAMILY_COLORS[fam];
     if(layer.seenX.size>=2&&layer.seenY.size>=2){
       traces.push({type:'surface',x:layer.xs,y:layer.ys,z:softenedMesh(layer.z,state.smooth),...surfaceStyle(),
         colorscale:[[0,color],[1,color]],opacity:state.layerOpacity,connectgaps:false,
         customdata:layer.custom,hovertext:layer.info,hoverinfo:'text',showscale:false,
         showlegend:false,name:familyName(fam)});
       overlayShapes++;
     }else{
       // A 1D or singleton family cannot form an actual 2D surface; show the real cells as line/dots.
       degenerate.push(familyName(fam));
       const dots=[];
       layer.z.forEach((row,iy)=>row.forEach((v,ix)=>{if(v!==null&&Number.isFinite(v))dots.push([layer.xs[ix],layer.ys[iy],v,layer.info[iy][ix],layer.custom[iy][ix]]);}));
       dots.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
       traces.push({type:'scatter3d',mode:'markers+lines',x:dots.map(d=>d[0]),y:dots.map(d=>d[1]),z:dots.map(d=>d[2]),
         customdata:dots.map(d=>d[4]),hovertext:dots.map(d=>d[3]),hoverinfo:'text',
         marker:{color,size:5,opacity:.94},line:{color,width:4},name:familyName(fam),showlegend:false});
     }
   }
 }else if(g.xs.length&&g.ys.length){
   if(is3&&g.xs.length>=2&&g.ys.length>=2){
     traces.push({type:'surface',x:g.xs,y:g.ys,z:softenedMesh(g.z,state.smooth),...surfaceStyle(),colorscale:scale,opacity:.90,connectgaps:false,
       customdata:g.custom,hovertext:g.info,hoverinfo:'text',name:'Getestete Varianten',showscale:true,colorbar:{title:'1× %',len:.48,thickness:11}});
   }else if(!is3){
     traces.push({type:'heatmap',x:g.xs,y:g.ys,z:g.z,colorscale:scale,connectgaps:false,
       customdata:g.custom,hovertext:g.info,hoverinfo:'text',name:'Getestete Varianten',showscale:true,colorbar:{title:'1× %',thickness:11}});
   }
 }
 const note=$('layerNote');
 if(note)note.textContent=overlay
   ? `${overlayShapes} separate Flächen${degenerate.length?', als Linien/Punkte: '+degenerate.join(', '):''} · Sichten unabhängig · Nur echte Gen-Kombinationen`
   : (is3?'Einzeloberfläche · aktives Exit-Gen grenzt den Kandidatenpool ein':'2D: Aggregierte Heatmap der gewählten Familien');
 const portfolio=choosePortfolio()[state.day],live=new Set(portfolio.live),chosen=new Set(portfolio.ids.filter(i=>i>=0));
 let dotIds=state.points==='all'?state.filtered:state.points==='manual'?[...state.manual]:portfolio.ids.filter(i=>i>=0);
 const dots=dotIds.map(i=>({i,m:META[i],v:value(i),trade:live.has(i),chosen:chosen.has(i)}));
 if(dots.length){const args={x:dots.map(a=>realAxisValue(a.i,state.x)),y:dots.map(a=>realAxisValue(a.i,state.y)),
   customdata:dots.map(a=>a.i),hovertext:dots.map(a=>`${a.m.id}<br>Familie: ${familyName(a.m.family)}<br>${pct(a.v)} (${state.dataset}, 1×)<br>${a.trade?'HANDELT':a.chosen?'GEWÄHLT / GESTOPPT':'IM TESTPOOL'}`),hoverinfo:'text',
   marker:{size:dots.map(a=>a.trade?9:a.chosen?7:Math.min(6,2.5+Math.abs(a.v)/28)),
   color:dots.map(a=>a.trade?'#ffcf67':a.chosen?'#96abc4':a.v>=0?'#68d2ab':'#d56d85'),
   line:{color:'#0e2336',width:.7},opacity:state.points==='all'?.72:1},name:'Echte Strategien',showlegend:false};
   if(is3){args.z=dots.map(a=>a.v+.6);args.type='scatter3d';args.mode='markers';}
   else{args.type='scatter';args.mode='markers';}
   traces.push(args);
 }
 const vals=g.z.flat().filter(Number.isFinite), bg='#102034',zrange=[Math.min(-50,...vals),Math.max(80,...vals)];
 const l=is3?{paper_bgcolor:bg,plot_bgcolor:bg,font:{color:'#c9d9ed',size:12},margin:{l:0,r:0,t:24,b:0},uirevision:'camera',transition:{duration:Math.min(state.playMs,240),easing:'cubic-in-out'},showlegend:false,
  scene:{xaxis:x,yaxis:y,zaxis:{title:'Rendite 1× (%)',gridcolor:'#304b61',backgroundcolor:'#102437',range:zrange},camera:{eye:{x:1.68,y:-1.55,z:.86}}}}:
  {paper_bgcolor:bg,plot_bgcolor:bg,font:{color:'#c9d9ed',size:12},margin:{l:78,r:50,t:16,b:72},uirevision:'axes',showlegend:false,
   xaxis:{...x,gridcolor:'#243b52',zeroline:false,automargin:true},yaxis:{...y,gridcolor:'#243b52',zeroline:false,automargin:true}};
 return {traces,l};
}
function renderChart(){
 if(state.rendering){state.pending=true;return;}
 state.rendering=true;
 state.grid=buildHeatmap();
 const {traces,l}=makePlotData();
 Plotly.react($('chart'),traces,l,{responsive:true,scrollZoom:true,displaylogo:false}).then(()=>{
   if(!$('chart')._quanthandlers){$('chart')._quanthandlers=true;
    $('chart').on('plotly_click', ev=>{
      const pt=ev?.points?.[0];if(!pt)return;
      const raw=pt.customdata;
      if(Number.isInteger(raw)&&raw>=0&&META[raw]){inspect(raw);return;}
      const group=state.grid.groups.get(Number(pt.x)+'|'+Number(pt.y));
      if(group?.length){let best=group[0];for(const i of group)if(value(i)>value(best))best=i;inspect(best);}
    });
   }
 }).catch(err=>{console.error(err);if(state.chart==='3d'){state.chart='2d';$('chartType').value='2d';state.pending=true;}else $('plotCountNote').textContent='Grafikfehler: '+err.message;})
 .finally(()=>{state.rendering=false;if(state.pending){state.pending=false;renderChart();}});
}
function setDay(n){state.day=Math.max(0,Math.min(DAYS-1,n));state.grid=buildHeatmap();setStatus();renderChart();if(state.selected>=0)inspect(state.selected);}
function setControls(){
 $('axisX').value=state.x;$('axisY').value=state.y;$('exitFamily').value=familySelectionMode();
 $('boxFilter').value=state.box;$('poolMode').value=state.mode;$('ranking').value=String(state.rank);
 $('guard').value=state.guard;$('stopVal').textContent=state.guard===0?'Aus':'−'+state.guard+' %';
 if($('topN')){$('topN').value=state.topN;$('topNValue').textContent=String(state.topN);}
 if($('stopScope'))$('stopScope').value=state.stopScope;
 $('correlationLimit').value=state.corrLimit===null?'off':String(state.corrLimit);$('corrFallback').checked=state.corrFallback;
}
function changePool(){cancelTradeSweep();if($('tradeSweepResults'))$('tradeSweepResults').replaceChildren();state.portKey='';state.filtered=eligible();state.grid=buildHeatmap();state.port=null;setStatus();
 if(state.priceMode==='renko_pct'&&!state.filtered.length){
  $('dataWarning').textContent+=' Keine berechneten %-Renko-Gegenstücke im aktuellen Pool/Boxfilter. Wähle „Alle wirklich berechneten Strategien“ und „Alle gerechneten Werte“, oder einen anderen Exit-Filter.';
 }
 renderChart();}
function applyPreset(n){
 if(n==='tp'){state.x='tp';state.y='s';state.family='tp';state.familyEnabled={none:false,tp:true,ttp:false,both:false};state.box='all';state.mode='all4158';}
 else if(n==='ttp'){state.x='act';state.y='s';state.family='ttp';state.familyEnabled={none:false,tp:false,ttp:true,both:false};state.box='all';state.mode='all4158';}
 else if(n==='tpttp'){state.x='tp';state.y='s';state.family='both';state.familyEnabled={none:false,tp:false,ttp:false,both:true};state.box='all';state.mode='all4158';}
 else if(n==='original'){state.x='s';state.y='b';state.family='none';state.familyEnabled={none:true,tp:false,ttp:false,both:false};state.box='all';state.priceMode='renko_usd';state.mode='original462';}
 else if(n==='all'){state.x='s';state.y='b';state.family='all';state.familyEnabled={none:true,tp:true,ttp:true,both:true};state.box='all';state.mode='all4158';state.surfaceMode='overlay';$('surfaceMode').value='overlay';}
 else if(n==='top100'){state.x='s';state.y='b';state.family='all';state.familyEnabled={none:true,tp:true,ttp:true,both:true};state.box='all';state.mode='top100';}
 syncModeControls();setControls();changePool();
}
function serialize(){return {version:13,priceMode:state.priceMode,date:DATES[state.day],year:state.dataset,smooth:state.smooth,surfaceMode:state.surfaceMode,layers:state.layers,layerOpacity:state.layerOpacity,axisX:state.x,axisY:state.y,family:state.family,box:state.box,
 mode:state.mode,familyEnabled:state.familyEnabled,horizon:state.horizon,rank:state.rank,topN:state.topN,stop:state.guard,stopScope:state.stopScope,corrLimit:state.corrLimit,corrFallback:state.corrFallback,selected:[...state.manual].map(i=>META[i].id)};}
function downloadJSON(){const b=new Blob([JSON.stringify(serialize(),null,2)],{type:'application/json'}),u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download='quant_genome_v13_preset.json';a.click();URL.revokeObjectURL(u);}
function supportsWebGL(){try{const c=document.createElement('canvas');return !!(c.getContext('webgl2')||c.getContext('webgl')||c.getContext('experimental-webgl'));}catch(e){return false;}}

function modeLabel(){return state.priceMode==='candle'?'5-Minuten-Candles (kein Renko)':state.priceMode==='renko_pct'?'Renko · Prozent vom Kurs':'Renko · feste USD-Box';}
function syncModeControls(){
 $('useCandles').checked=state.priceMode==='candle';$('usePercent').checked=state.priceMode==='renko_pct';
 $('modeTag').textContent=modeLabel();
 $('modeExplanation').textContent=state.priceMode==='candle'?'Normaler 5m-OHLC-Kerzen-IMBA: S×10 Candles, keine Boxgröße.':
  state.priceMode==='renko_pct'?'Nur tatsächlich berechnete %-Renko-Strategien. Bei einem USD-Pool werden Sensitivity und Exit-Programm exakt abgeglichen; die Prozent-Box wird unabhängig gewählt (keine USD-Umrechnung).':
  'Alle tatsächlich berechneten Renko-Boxen verfügbar. Einzeln per Regler oder alle zusammen.';
 $('useAllBoxes').disabled=state.priceMode==='candle';
 if(state.priceMode==='candle')state.box='all';
 $('useAllBoxes').checked=state.box==='all';
 $('boxStep').disabled=state.priceMode==='candle'||state.box==='all';
 $('boxStepLabel').textContent=state.priceMode==='candle'?'kein Renko':(Number($('boxStep').value)/10).toFixed(1)+(state.priceMode==='renko_pct'?' %':' USD');
 const boxes=state.priceMode==='renko_pct'?[...new Set(PERCENT_MODE_IDS.map(i=>Number(META[i].b)))].sort((a,b)=>a-b):
    state.priceMode==='candle'?[]:[...new Set(META.filter(m=>m.priceMode==='renko_usd').map(m=>m.b))].sort((a,b)=>a-b);
 const prev=state.box;
 populateSelector('boxFilter',[['all','Alle gerechneten Werte'],...boxes.map(b=>[b.toFixed(2),b.toFixed(2)+(state.priceMode==='renko_pct'?' %':' USD')])], prev);
 if($('boxFilter').value!==prev){state.box='all';$('boxFilter').value='all';$('useAllBoxes').checked=true;}
 if(state.x==='b'&&state.priceMode==='candle'){state.x='tp';state.y='s';$('axisX').value=state.x;$('axisY').value=state.y;}
}
function switchPriceMode(kind){
 if(!['candle','renko_pct','renko_usd'].includes(kind))return;
 state.priceMode=kind; state.box='all';
 if(kind==='candle' && (state.x==='b'||state.y==='b')){state.x='tp';state.y='s';}
 syncModeControls();setControls();changePool();
}


let activeTradeSweep=null;
function installTradeLabControls(){
 const guardInput=$('guard'),anchor=guardInput.closest('.item')||guardInput.parentElement;
 if(!anchor||!anchor.parentElement)throw Error('Stopp-Regler nicht gefunden.');
 const node=document.createElement('div');node.id='tradeSweepControls';node.className='tradeSweepControls';
 node.innerHTML='<div class="tradeSweepPair"><div><label for="topN">HANDELNDE TOP-N STRATEGIEN</label>'+
  '<div class="tradeSweepRange"><input type="range" id="topN" min="1" max="6" step="1" value="6" aria-label="Anzahl Top-Strategien">'+
  '<strong id="topNValue">6</strong></div></div>'+
  '<div><label for="stopScope">STOPP GILT FÜR</label>'+
  '<select id="stopScope"><option value="strategy">Jede Strategie einzeln</option>'+
  '<option value="portfolio">Gesamtes Handelskapital</option></select></div></div>'+
  '<p class="muted">Der bestehende Stopp-Prozentsatz gilt wahlweise für einzelne Strategien oder das gesamte Handelskapital ohne Reserve. Ein Gesamtstopp pausiert bis zur nächsten 14-Tage-Rotation.</p>'+
  '<div class="tradeSweepButtons"><button type="button" id="runTradeSweep">Automatisch optimieren · Sweep</button>'+
  '<button type="button" id="cancelTradeSweep" disabled>Abbrechen</button></div>'+
  '<div id="tradeSweepProgress" class="muted" aria-live="polite">Testet den ausgewählten Strategienpool im gewählten Jahr. Ranking, Top-N, Stoppgrenze und Stoppmodus werden kombiniert.</div>'+
  '<div id="tradeSweepResults"></div>';
 anchor.insertAdjacentElement('afterend',node);
 const style=document.createElement('style');style.textContent=
  '.tradeSweepControls{grid-column:1/-1;border:1px solid #3c6b76;border-radius:11px;padding:12px;background:#102e3d;margin:9px 0}' +
  '.tradeSweepPair{display:grid;grid-template-columns:1fr 1fr;gap:12px}' +
  '.tradeSweepPair label{display:block;font-size:10px;color:#9dc8d4;font-weight:700;margin-bottom:7px}' +
  '.tradeSweepRange{display:flex;align-items:center;gap:12px}.tradeSweepRange input{flex:1;accent-color:#65d7ad}.tradeSweepRange strong{min-width:24px;text-align:center;color:#92e8d8}' +
  '.tradeSweepPair select{max-width:100%;width:100%}.tradeSweepButtons{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}' +
  '#runTradeSweep{background:#17565a;border-color:#6ed5bd;color:#e6fff7;font-weight:700}' +
  '#tradeSweepResults{font-size:12px}.tradeSweepResult{padding:8px 0;border-bottom:1px solid #315166;display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap}' +
  '.tradeSweepResult button{font-size:11px;padding:6px 9px}.tradeSweepResult strong{color:#d9f5df}' +
  '@media(max-width:600px){.tradeSweepPair{grid-template-columns:1fr}}';
 document.head.appendChild(style);
}
function cancelTradeSweep(message){
 if(!activeTradeSweep)return;
 activeTradeSweep.cancelled=true;activeTradeSweep=null;
 if($('runTradeSweep'))$('runTradeSweep').disabled=false;
 if($('cancelTradeSweep'))$('cancelTradeSweep').disabled=true;
 if(message&&$('tradeSweepProgress'))$('tradeSweepProgress').textContent=message;
}
function sweepGuardOptions(){
 const control=$('guard'),lo=Number(control.min)||0,hi=Number(control.max)||60;
 const step=Math.max(5,Number(control.step)||5),out=[];
 for(let x=lo;x<=hi+1e-9;x+=step)out.push(Number(x.toFixed(3)));
 for(const v of [0,state.guard])if(v>=lo&&v<=hi&&!out.includes(v))out.push(v);
 return out.sort((a,b)=>a-b);
}
async function runTradeSweep(){
 cancelTradeSweep();
 const token={cancelled:false};activeTradeSweep=token;
 const pool=state.filtered.slice(),year=state.dataset,mode=state.mode,priceMode=state.priceMode;
 const ranks=[...new Set([...$('ranking').options].map(o=>Number(o.value)).filter(Number.isFinite))];
 const guards=sweepGuardOptions(),topNs=[1,2,3,4,5,6],scopes=['strategy','portfolio'];
 const corrLimit=state.corrLimit,corrFallback=state.corrFallback;
 const total=ranks.length*guards.length*topNs.length*scopes.length;
 const progress=$('tradeSweepProgress'),results=$('tradeSweepResults');
 $('runTradeSweep').disabled=true;$('cancelTradeSweep').disabled=false;
 results.textContent='';
 const yieldFrame=()=>new Promise(resolve=>setTimeout(resolve,0));
 try{
  if(!pool.length||!ranks.length)throw Error('Kein gültiger Kandidatenpool oder Rankingzeitraum.');
  const selections=new Map();
  let ready=0,cycles=ranks.length*Math.ceil(DAYS/14);
  progress.textContent='Ranking-Auswahl vorbereiten: 0 / '+cycles+' Rotationen';
  for(const rank of ranks){
   for(let d=0;d<DAYS;d+=14){
    if(token.cancelled)return;
    // Every rebalance ranks using completed PAST returns; parameter search itself is retrospective.
    const ranked=pool.map(i=>[i,rPct(i,d-1,rank)]).filter(v=>v[1]>0)
      .sort((a,b)=>b[1]-a[1]||a[0]-b[0]);
    for(const topN of topNs){
     selections.set(rank+'|'+topN+'|'+d,
       applyCorrelationCap(ranked,d,topN,corrLimit,corrFallback).map(p=>p[0]));
    }
    ready++;
    if(ready%3===0){
     progress.textContent='Ranking-Auswahl vorbereiten: '+ready+' / '+cycles+' Rotationen';
     await yieldFrame();
    }
   }
  }
  const sorted=[];
  let tested=0;
  for(const rank of ranks)for(const topN of topNs)for(const stopScope of scopes)for(const guard of guards){
   if(token.cancelled)return;
   const config={pool,rank,topN,stopScope,guard,corrLimit,corrFallback};
   const stats=simulatePortfolio(config,selections,true);
   sorted.push({...stats,rank,topN,stopScope,guard});
   tested++;
   if(tested%16===0){
    progress.textContent='Sweep: '+tested+' / '+total+' Kombinationen ('+year+' · '+mode+' · '+priceMode+')';
    await yieldFrame();
   }
  }
  if(token.cancelled)return;
  sorted.sort((a,b)=>b.endWealth-a.endWealth||b.maxDrawdownPct-a.maxDrawdownPct||a.guard-b.guard);
  progress.textContent=total+' Kombinationen abgeschlossen · '+pool.length+' Kandidaten · '+year+'. Ranking und Stopp wurden rückblickend optimiert (In-Sample, keine Prognose).';
  const wrapper=document.createElement('div');
  const note=document.createElement('p');note.className='muted';
  note.textContent='Beste 5 nach Endvermögen. MaxDD bezieht sich auf das gesamte Depot inklusive Reserve. Optimierung gilt nur für das angezeigte Testjahr.';
  wrapper.appendChild(note);
  for(let j=0;j<Math.min(5,sorted.length);j++){
   const item=sorted[j],row=document.createElement('div');row.className='tradeSweepResult';
   const label=document.createElement('span');
   label.textContent=(j+1)+'. '+item.rank+' Tage · Top '+item.topN+' · Stopp '+item.guard+' % · '+
    (item.stopScope==='strategy'?'Einzelstopp':'Gesamtstopp');
   const scoreEl=document.createElement('strong');
   scoreEl.textContent=euro(item.endWealth)+' · MaxDD '+number(item.maxDrawdownPct,1)+' %';
   const btn=document.createElement('button');btn.type='button';btn.textContent='Übernehmen';
   btn.onclick=()=>{
    state.rank=item.rank;state.guard=item.guard;state.topN=item.topN;state.stopScope=item.stopScope;
    setControls();changePool();
    progress.textContent='Übernommen: '+label.textContent+' · Historischer Endwert '+euro(item.endWealth)+
      '. Einstellungen bitte außerhalb des Optimierungsjahrs prüfen.';
   };
   row.append(label,scoreEl,btn);wrapper.appendChild(row);
  }
  results.replaceChildren(wrapper);
 }catch(err){
  if(!token.cancelled){console.error(err);progress.textContent='Sweep fehlgeschlagen: '+err.message;}
 }finally{
  if(activeTradeSweep===token){
   activeTradeSweep=null;$('runTradeSweep').disabled=false;$('cancelTradeSweep').disabled=true;
  }
 }
}

function bindUI(){
 initOptions();
 installTradeLabControls();
 syncModeControls();
 $('useCandles').onchange=e=>switchPriceMode(e.target.checked?'candle':'renko_usd');
 $('usePercent').onchange=e=>switchPriceMode(e.target.checked?'renko_pct':'renko_usd');
 $('useAllBoxes').onchange=e=>{state.box=e.target.checked?'all':(Number($('boxStep').value)/10).toFixed(2);syncModeControls();changePool();};
 $('boxStep').oninput=e=>{state.box=(Number(e.target.value)/10).toFixed(2);$('boxStepLabel').textContent=(Number(e.target.value)/10).toFixed(1)+(state.priceMode==='renko_pct'?' %':' USD');};
 $('boxStep').onchange=()=>{state.box=(Number($('boxStep').value)/10).toFixed(2);$('useAllBoxes').checked=false;syncModeControls();changePool();};
 if(!supportsWebGL()){state.chart='2d';$('chartType').value='2d';}
 $('datePrev').onclick=()=>setDay(state.day-1);$('dateNext').onclick=()=>setDay(state.day+1);
 $('slider').oninput=e=>setDay(+e.target.value);
 function stop(){if(state.playing!==null){cancelAnimationFrame(state.playing);state.playing=null;}}
 $('pause').onclick=stop;
 function play(dir){
  stop();state.dir=dir;
  // requestAnimationFrame coalesces updates; never queue up expensive WebGL plots.
  let elapsed=0,last=0;
  function tick(ts){
   if(state.playing===null)return;
   if(!last)last=ts;
   elapsed+=Math.max(0,ts-last);last=ts;
   if(elapsed>=state.playMs){
    elapsed%=state.playMs;
    const nx=state.day+dir;
    if(nx<0||nx>=DAYS){stop();return;}
    setDay(nx);
   }
   state.playing=requestAnimationFrame(tick);
  }
  state.playing=requestAnimationFrame(tick);
 }
 $('playFwd').onclick=()=>play(1);$('playRev').onclick=()=>play(-1);
 $('presetTP').onclick=()=>applyPreset('tp');$('presetTTP').onclick=()=>applyPreset('ttp');
 $('presetBoth').onclick=()=>applyPreset('tpttp');$('preset462').onclick=()=>applyPreset('original');
 $('presetAll').onclick=()=>applyPreset('all');$('preset100').onclick=()=>applyPreset('top100');
 for(const [el,field] of [['axisX','x'],['axisY','y'],['boxFilter','box'],['poolMode','mode']])
   $(el).onchange=async e=>{state[field]=e.target.value;
    if(field==='mode'&&state.mode.startsWith('rawYear') && Number(state.dataset)<=Number(state.mode.slice(7))){
       const targetYear=String(Number(state.mode.slice(7))+1);await loadYear(targetYear);}
    if(field==='mode'&&(['original462','top100','balanced100'].includes(state.mode)||state.mode.startsWith('rawYear'))&&state.priceMode==='candle'){
     state.priceMode='renko_usd';state.box='all';syncModeControls();
    }
    if(field==='box'){if(state.box!=='all')$('boxStep').value=Math.round(Number(state.box)*10);syncModeControls();}
    if((field==='x'||field==='y')&&state.x===state.y){state[field==='x'?'y':'x']=field==='x'?'s':'b';}
    setControls();changePool();};
 $('exitFamily').onchange=e=>setFamilyDropdown(e.target.value);
 $('familyAllBtn').onclick=setAllFamilies;
 $('dataset').onchange=async e=>{const year=e.target.value;try{$('dataset').disabled=true;$('dataWarning').textContent='Lade vollständige Tageskurven für '+year+' …';if(state.playing!==null){cancelAnimationFrame(state.playing);state.playing=null;}await loadYear(year);changePool();}catch(err){console.error(err);alert('Jahr konnte nicht geladen werden: '+err.message);}finally{$('dataset').disabled=false;}};
 $('horizon').onchange=e=>{state.horizon=Number(e.target.value);state.grid=buildHeatmap();setStatus();renderChart();};
 $('aggregate').onchange=e=>{state.aggregate=e.target.value;state.grid=buildHeatmap();setStatus();renderChart();};
 $('chartType').onchange=e=>{state.chart=e.target.value;if(state.chart==='3d'&&!supportsWebGL()){state.chart='2d';$('chartType').value='2d';$('plotCountNote').textContent='WebGL nicht verfügbar – 2D-Heatmap aktiviert.';}renderChart();};
 $('pointsMode').onchange=e=>{state.points=e.target.value;if(state.points==='manual'){state.mode='manual';setControls();changePool();}else{setStatus();renderChart();}};
 $('smoothLevel').onchange=e=>{state.smooth=Number(e.target.value);renderChart();};
 $('sidebarToggle').onclick=()=>{
  state.sidebarOpen=!state.sidebarOpen;
  document.querySelector('.cols').classList.toggle('sidebarHidden',!state.sidebarOpen);
  $('sidebarToggle').textContent=state.sidebarOpen?'Seitenmenüs ausblenden':'Seitenmenüs einblenden';
  setTimeout(()=>{Plotly.Plots.resize($('chart'));},100);
 };
 document.querySelectorAll('aside.right > section.panel').forEach((panel,i)=>{
  const title=panel.querySelector('h2');if(!title)return;
  title.setAttribute('role','button');title.setAttribute('tabindex','0');
  title.setAttribute('aria-expanded',i<3?'true':'false');
  panel.classList.toggle('folded',i>=3);
  const toggle=()=>{panel.classList.toggle('folded');title.setAttribute('aria-expanded',String(!panel.classList.contains('folded')));};
  title.addEventListener('click',toggle);
  title.addEventListener('keydown',ev=>{if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();toggle();}});
 });
 $('speed').onchange=e=>{state.playMs=Number(e.target.value);};
 $('surfaceMode').onchange=e=>{state.surfaceMode=e.target.value;renderChart();};
 $('surfaceOpacity').oninput=e=>{state.layerOpacity=Number(e.target.value)/100;renderChart();};
 document.querySelectorAll('[data-layer]').forEach(el=>el.onchange=e=>{state.layers[e.target.dataset.layer]=e.target.checked;renderChart();});
 $('ranking').onchange=e=>{state.rank=Number(e.target.value);changePool();};
 $('guard').oninput=e=>{$('stopVal').textContent=e.target.value==='0'?'Aus':'−'+e.target.value+' %';};
 $('guard').onchange=e=>{state.guard=Number(e.target.value);changePool();};
 $('topN').oninput=e=>{$('topNValue').textContent=e.target.value;};
 $('topN').onchange=e=>{state.topN=Math.max(1,Math.min(6,Number(e.target.value)));changePool();};
 $('stopScope').onchange=e=>{state.stopScope=e.target.value==='portfolio'?'portfolio':'strategy';changePool();};
 $('runTradeSweep').onclick=runTradeSweep;
 $('cancelTradeSweep').onclick=()=>cancelTradeSweep('Sweep abgebrochen.');
 $('correlationLimit').onchange=e=>{state.corrLimit=e.target.value==='off'?null:Number(e.target.value);changePool();};
 $('corrFallback').onchange=e=>{state.corrFallback=e.target.checked;changePool();};
 $('geneScope').onchange=()=>renderGeneCards(choosePortfolio()[state.day]);
 $('manualAdd').onclick=()=>{if(state.selected>=0){state.manual.add(state.selected);inspect(state.selected);updateManual();if(state.mode==='manual')changePool();}};
 $('manualRemove').onclick=()=>{if(state.selected>=0){state.manual.delete(state.selected);inspect(state.selected);updateManual();if(state.mode==='manual')changePool();}};
 $('manualClear').onclick=()=>{state.manual.clear();updateManual();if(state.mode==='manual')changePool();};
 $('savePreset').onclick=downloadJSON;
 $('loadPreset').onchange=async e=>{const file=e.target.files[0];if(!file)return;
  try{const x=JSON.parse(await file.text());if(![7,8,9,10,11,12,13].includes(x.version))throw Error('Bitte eine v7/v8/v9 Preset-Datei wählen.');
   state.manual=new Set((x.selected||[]).map(v=>IDMAP.get(v)).filter(Number.isInteger));
   if(x.year&&String(x.year)!==state.dataset){await loadYear(String(x.year));}else if(x.date&&!DATES.includes(x.date)&&YEAR_LENGTHS[String(x.date).slice(0,4)])await loadYear(String(x.date).slice(0,4));
   if(DATES.includes(x.date))state.day=DATES.indexOf(x.date);
   if(x.smooth!==undefined){state.smooth=Number(x.smooth);$('smoothLevel').value=String(state.smooth);}if(x.surfaceMode)state.surfaceMode=x.surfaceMode;if(x.layers)Object.assign(state.layers,x.layers);if(x.layerOpacity)state.layerOpacity=x.layerOpacity;
   $('surfaceMode').value=state.surfaceMode;$('surfaceOpacity').value=String(Math.round(state.layerOpacity*100));
   document.querySelectorAll('[data-layer]').forEach(el=>el.checked=!!state.layers[el.dataset.layer]);
   state.x=AXES[x.axisX]?x.axisX:'tp';state.y=AXES[x.axisY]?x.axisY:'s';
   state.family=['all','none','tp','ttp','both','custom'].includes(x.family)?x.family:'all';
    state.familyEnabled=x.familyEnabled?Object.assign({none:false,tp:false,ttp:false,both:false},x.familyEnabled):Object.fromEntries(['none','tp','ttp','both'].map(f=>[f,state.family==='all'||state.family===f]));
    state.family=familySelectionMode();
   state.priceMode=['candle','renko_pct','renko_usd'].includes(x.priceMode)?x.priceMode:'renko_usd';
   state.box=x.box||'all';state.mode=x.mode||'all4158';state.horizon=x.horizon??30;
   state.rank=x.rank??14;state.guard=x.stop??30;state.topN=Math.max(1,Math.min(6,Number(x.topN)||6));state.stopScope=x.stopScope==='portfolio'?'portfolio':'strategy';state.corrLimit=(x.corrLimit===null||x.corrLimit===undefined)?null:Number(x.corrLimit);state.corrFallback=!!x.corrFallback;
   syncModeControls();setControls();$('horizon').value=state.horizon;$('slider').value=state.day;updateManual();changePool();
  }catch(err){alert('Preset konnte nicht geladen werden: '+err.message);}
 };
 $('downloadPool').onclick=()=>{const lines=['strategy_id,price_mode,sensitivity,renko_box_value,exit_program,tp_pct,ttp_activation_pct,ttp_trail_pct,return_2026_ytd_pct,return_2022_pct'];
  for(const i of state.filtered){const m=META[i];lines.push([m.id,m.priceMode,m.s,m.b,m.exitKey,m.tpValue,m.act,m.trail,rPct(i,DAYS-1,0)].join(','));}
  const b=new Blob([lines.join('\n')],{type:'text/csv'}),u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download='quant_genome_v9_echte_kandidaten.csv';a.click();URL.revokeObjectURL(u);
 };
 state.points='top6';state.filtered=eligible();state.grid=buildHeatmap();
 setControls();setStatus();renderChart();updateManual();
}
function updateManual(){$('manualCount').textContent=state.manual.size+' individuell markierte Strategien';}


// 2026 favorite settings from screenshots and evidence charts, NOT live trading changes.
let researchMode='mix';
let reportIsOpen=false;
function saved2026Info(key){
 const p=SCREENSHOT_PRESETS[key];const info=$('snapshotNote');
 if(!p){info.textContent='Bitte ein Preset wählen.';return;}
 const original=p.screenshot?`Gesichert aus ${p.screenshot}`:'Vergleichsvariante – nicht in den drei Screenshots enthalten';
 info.innerHTML=`<strong>${p.label}</strong><br>${original}. Originalwert bei 1.000 € bis 04.11.2026: <b>${euro(p.wealth_at_1000)}</b>.<br>Nur 2026 · fester USD-Renko · Auswahl alle 14 Tage · positiver Ranking-Score erforderlich.`;
}
async function loadSaved2026(key){
 const p=SCREENSHOT_PRESETS[key];if(!p)return;
 if(state.playing!==null){cancelAnimationFrame(state.playing);state.playing=null;}
 if(state.dataset!=='2026')await loadYear('2026');
 state.mode=p.pool;state.rank=p.rank;state.guard=p.guard;state.topN=6;state.stopScope='strategy';state.corrLimit=null;state.corrFallback=false;
 state.family='all';state.familyEnabled={none:true,tp:true,ttp:true,both:true};
 state.priceMode='renko_usd';state.box='all';state.x='s';state.y='b';state.day=DAYS-1;
 state.chart=supportsWebGL()?'3d':'2d';state.points='top6';state.horizon=30;
 $('chartType').value=state.chart;$('horizon').value=30;$('pointsMode').value='top6';
 $('slider').value=state.day;
 syncModeControls();setControls();changePool();saved2026Info(key);
 $('activeSnapshot').textContent=`Aktiv: ${p.label}`;
}
function chartTheme(title,extra={}){return {...{
 title:{text:title,font:{size:16,color:'#dcecf5'},x:0.03},paper_bgcolor:'#0c2336',plot_bgcolor:'#102b40',
 font:{family:'system-ui, Arial',color:'#c9deed',size:12},margin:{t:55,l:65,r:35,b:55},
 xaxis:{gridcolor:'#2b4758',zerolinecolor:'#335668',title:''},
 yaxis:{gridcolor:'#2b4758',zerolinecolor:'#335668',title:'Depotwert (€)',tickformat:',.0f'},
 legend:{orientation:'h',x:0,y:1.12,font:{size:10}},hovermode:'x unified',uirevision:'quant-v13-'+researchMode
 },...extra};}
const chartOpts={responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']};
function renderResearchCharts(){
 if(!reportIsOpen)return;
 const b=EVIDENCE_CHARTS;
 const names={original:'Original 462 · 333,33 €',raw:'Top100 Raw · 333,33 €',balanced:'Balanced B · 333,34 €',mixed:'Drei-Büschel-Mix',previous:'Bisheriger Mix (Balanced −30 %)'};
 const color={original:'#61bdd8',raw:'#f3ad71',balanced:'#c3a1f4',mixed:'#e7d27a',previous:'#8493a7'};
 const el=$('researchChart');
 const note=$('researchChartNote');
 if(researchMode==='mix'){
  const traces=['original','raw','balanced','mixed','previous'].map(k=>({type:'scatter',mode:'lines',x:b.mix.dates,y:b.mix[k],name:names[k],
   line:{color:color[k],width:k==='mixed'?3:k==='previous'?1.5:1.9,dash:k==='previous'?'dot':'solid'}}));
  Plotly.react(el,traces,chartTheme('2026 · Ein Depot, drei unabhängige Büschel'),chartOpts);
  note.textContent='Start: 333,33 € + 333,33 € + 333,34 € = 1.000 €. Der Mix ist die Summe der drei getrennt verwalteten Kapitalkurven. Die ausgepunktete Variante verwendet Balanced −30 % statt −25 %.';
 }else if(researchMode==='sharpe'){
  const stats=b.stats;const name=stats.map(s=>s.portfolio),cols=['#61bdd8','#f3ad71','#c3a1f4','#e7d27a'];
  const traces=[{type:'bar',name:'Sharpe (365d)',x:name,y:stats.map(v=>v.sharpe_365d),marker:{color:cols},text:stats.map(v=>v.sharpe_365d.toFixed(2)),textposition:'outside',hovertemplate:'%{x}: Sharpe %{y:.3f}<extra></extra>'}];
  Plotly.react(el,traces,chartTheme('2026 · Sharpe Ratio, einheitlich 1.000 € Startkapital',{
    margin:{t:55,l:60,r:25,b:115},xaxis:{tickangle:-18,gridcolor:'#2b4758'},yaxis:{title:'Sharpe · 365 Tage',gridcolor:'#2b4758'},showlegend:false}),chartOpts);
  note.innerHTML='<b>Endwerte / maximaler Drawdown:</b> '+stats.map(s=>`${s.portfolio}: ${euro(s.end_eur)} / ${s.max_drawdown_pct.toFixed(1)} %`).join(' · ')+'<br>Vergleich basiert auf täglichen Depotwertänderungen (risikofrei = 0, 365 Tage annualisiert).';
 }else{
  const family=b.annual;
  const tr=[['carry_2022','2022er Familie behalten','#f3ad71'],['annual_rotate','Jährlich neue Top 100','#70d6ab'],['accumulate_past','Frühere Jahrgänge sammeln','#c3a1f4']].map(([key,name,col])=>({
    type:'scatter',mode:'lines',x:family[key].dates,y:family[key].wealth,name,line:{color:col,width:2}}));
  Plotly.react(el,tr,chartTheme('2023–2026 · Top-100-Jahresfamilien: durchgehendes Kapital',{
    yaxis:{title:'Depotwert (€), logarithmisch',type:'log',gridcolor:'#2b4758',range:[-2,4.5]},
    legend:{orientation:'h',x:0,y:1.14,font:{size:10}}}),chartOpts);
  note.textContent='Separate Forschungsreihe: 1.000 € Anfang 2023, danach ohne Neustart, 14-Tage-Ranking, 30-%-Einzelstopps. Logarithmische Euro-Skala. Jahrgänge werden ausschließlich aus abgeschlossenen Vorjahren ausgewählt.';
 }
}
function bindSavedEvidence(){
 const selector=$('snapshotPreset');
 selector.onchange=e=>saved2026Info(e.target.value);
 $('applySnapshot').onclick=()=>loadSaved2026(selector.value);
 saved2026Info(selector.value);
 $('researchDetails').addEventListener('toggle',()=>{
  reportIsOpen=$('researchDetails').open;
  if(reportIsOpen)requestAnimationFrame(renderResearchCharts);
 });
 $('chartsJump').onclick=()=>{
  const panel=$('researchDetails');panel.open=true;panel.scrollIntoView({behavior:'smooth',block:'start'});
  reportIsOpen=true;requestAnimationFrame(renderResearchCharts);
 };
 for(const btn of document.querySelectorAll('[data-research-mode]')){
  btn.onclick=()=>{
   researchMode=btn.dataset.researchMode;
   document.querySelectorAll('[data-research-mode]').forEach(b=>b.classList.toggle('active',b===btn));
   renderResearchCharts();
  };
 }
}

/* v13 research extension: matched REAL legacy TTP matrices and proposed Kraken TTP settings.
 * Never infer uncomputed 0%-activation results from legacy 2%/3%-activation daily factors.
 */
const TTP_RESEARCH_DEFAULTS={activationPct:0,trailPct:1.5,closeFractionPct:100,action:'flip'};
state.krakenTtp={...TTP_RESEARCH_DEFAULTS};
function matchedExitRaster(act,trail){
 const map=new Map();
 META.forEach((m,i)=>{
  if(m.priceMode!=='renko_usd'||m.family!=='ttp')return;
  if(Math.abs(Number(m.act)-act)>1e-5||Math.abs(Number(m.trail)-trail)>1e-5)return;
  map.set(String(m.s)+'|'+Number(m.b).toFixed(5),i);
 });
 const pure=[],ttp=[];
 for(const i of ORIGINAL){
  const m=META[i];
  if(m.family!=='none'||m.priceMode!=='renko_usd')continue;
  const j=map.get(String(m.s)+'|'+Number(m.b).toFixed(5));
  if(j!==undefined){pure.push(i);ttp.push(j);}
 }
 return {pure,ttp,mixed:[...pure,...ttp]};
}
const HIST_EXIT_315=matchedExitRaster(3,1.5);
const HIST_EXIT_21=matchedExitRaster(2,1);
const RESEARCH_POOL_VARIANTS={
 pure315:HIST_EXIT_315.pure,ttp315:HIST_EXIT_315.ttp,mixed315:HIST_EXIT_315.mixed,
 pure21:HIST_EXIT_21.pure,ttp21:HIST_EXIT_21.ttp,mixed21:HIST_EXIT_21.mixed
};
const PREVIOUS_GET_IDS=getPresetIds;
getPresetIds=function(){return RESEARCH_POOL_VARIANTS[state.mode]||PREVIOUS_GET_IDS();};

// Link USD-pool selections to real, precomputed percentage-Renko counterparts.
// Sensitivity and exit-program identity must match exactly; % brick sizes stay
// independent parameters, not an approximate USD-to-percent backtest.
const PERCENT_MODE_IDS=ALL.filter(i=>META[i].priceMode==='renko_pct');
const PERCENT_GENES=new Map();
function correspondingGeneKey(m){return String(m.s)+'|'+String(m.family)+'|'+String(m.exitKey);}
for(const i of PERCENT_MODE_IDS){
 const k=correspondingGeneKey(META[i]);
 if(!PERCENT_GENES.has(k))PERCENT_GENES.set(k,[]);
 PERCENT_GENES.get(k).push(i);
}
const PRE_PERCENT_GET_IDS=getPresetIds;
getPresetIds=function(){
 const selected=PRE_PERCENT_GET_IDS();
 if(state.priceMode!=='renko_pct'||!selected.length)return selected;
 // These universal pools already contain their own percentage candidates.
 if(['all4158','grid20','grid30'].includes(state.mode))return selected;
 const out=new Set();
 for(const id of selected){
  const m=META[id];if(!m)continue;
  if(m.priceMode==='renko_pct'){out.add(id);continue;}
  for(const j of PERCENT_GENES.get(correspondingGeneKey(m))||[])out.add(j);
 }
 return [...out];
};
const PREVIOUS_SERIALIZE=serialize;
serialize=function(){return {...PREVIOUS_SERIALIZE(),krakenTtp:{...state.krakenTtp}};};
function normalizeResearchTtp(x){
 const v=x||{};
 const trail=Number(v.trailPct);
 return {
  activationPct:[0,.5,1,2,3].includes(Number(v.activationPct))?Number(v.activationPct):0,
  trailPct:Number.isFinite(trail)?Math.max(.5,Math.min(4,Math.round(trail*2)/2)):1.5,
  closeFractionPct:Number(v.closeFractionPct)===50?50:100,
  action:v.action==='close'?'close':'flip'
 };
}
function renderTtpControls(){
 const v=state.krakenTtp;
 if(!$('krakenTtpTrail'))return;
 $('krakenTtpTrail').value=String(v.trailPct);$('krakenTtpTrailValue').textContent=number(v.trailPct,1)+' %';
 $('krakenTtpActivation').value=String(v.activationPct);
 $('krakenTtpAction').value=v.action;
 $('krakenTtpFraction').value=String(v.closeFractionPct);
 $('krakenTtpFraction').disabled=v.action==='flip';
 if($('historicalPairsInfo'))$('historicalPairsInfo').textContent=
  'Tatsächlich berechnet: '+HIST_EXIT_315.pure.length+' paarige IMBA + '+HIST_EXIT_315.ttp.length+
  ' TTP 3/1,5 sowie '+HIST_EXIT_21.pure.length+' IMBA + '+HIST_EXIT_21.ttp.length+
  ' TTP 2/1. Nicht das neue Kraken-TTP 0/1,5.';
}
const PREVIOUS_SET_CONTROLS=setControls;
setControls=function(){PREVIOUS_SET_CONTROLS();if($('researchPairPool')&&RESEARCH_POOL_VARIANTS[state.mode])$('researchPairPool').value=state.mode;renderTtpControls();};
function historicalCompare(){
 const out=$('historicalCompareOutput'),chart=$('historicalCompareChart');
 if(state.priceMode!=='renko_usd'){out.textContent='Nur feste USD-Renko-Boxen können mit dem 462er-Raster verglichen werden.';return;}
 const test=[
  ['IMBA pur · paarig',HIST_EXIT_315.pure,null,false],
  ['TTP Alt 3%/1,5%',HIST_EXIT_315.ttp,null,false],
  ['IMBA + TTP Alt',HIST_EXIT_315.mixed,null,false],
  ['Aktueller Pool · ρ≤0,85',state.filtered,.85,false],
  ['Aktueller Pool · ρ≤0,90',state.filtered,.9,false]
 ];
 const traces=[],labels=[];
 for(const [name,base,limit,fallback] of test){
  const pool=base.filter(i=>state.box==='all'||Math.abs(Number(META[i].b)-Number(state.box))<1e-5);
  if(!pool.length){labels.push(name+': keine simulierten Daten');continue;}
  const rows=simulatePortfolio({pool,rank:state.rank,topN:state.topN,guard:state.guard,stopScope:state.stopScope,
    corrLimit:limit===null?state.corrLimit:limit,corrFallback:fallback});
  let peak=1000,dd=0;
  for(const r of rows){peak=Math.max(peak,r.wealth);dd=Math.min(dd,100*(r.wealth/peak-1));}
  labels.push(name+': '+euro(rows[rows.length-1].wealth)+' · DD '+number(dd,1)+' % · n='+pool.length);
  traces.push({type:'scatter',mode:'lines',name,x:rows.map(r=>r.date),y:rows.map(r=>r.wealth)});
 }
 out.textContent=labels.join(' | ')+' · '+state.dataset+' · Top '+state.topN+
  ' · '+state.rank+' Tage Ranking · Stopp '+state.guard+' % · '+state.stopScope+
  '. Vergleich mit gleichen Portfolioregeln, historische Alt-TTP-Signale. Keine Kraken-Flip-Resultate.';
 if(traces.length&&typeof Plotly!=='undefined'){
  chart.style.height='390px';
  Plotly.react(chart,traces,{paper_bgcolor:'#0c2336',plot_bgcolor:'#102b40',
   font:{color:'#c9deed',size:11},margin:{l:60,r:20,t:50,b:55},
   xaxis:{gridcolor:'#284e62'},yaxis:{title:'Depotwert (€)',gridcolor:'#284e62'},
   legend:{orientation:'h',x:0,y:1.18},hovermode:'x unified'},
   {responsive:true,displaylogo:false});
 }
}
async function compareTtpYears(){
 const btn=$('historicalAllYearsBtn'),out=$('historicalAllYearsOutput'),savedYear=state.dataset,savedDay=state.day;
 const savedSelected=state.selected;
 btn.disabled=true;out.textContent='Jahresvergleich laden …';
 const series=[
  ['IMBA',HIST_EXIT_315.pure,null,false],
  ['TTP Alt',HIST_EXIT_315.ttp,null,false],
  ['Mix',HIST_EXIT_315.mixed,null,false],
  ['Korr. 0,85 strikt',null,.85,false],
  ['Korr. 0,90 auffüllen',null,.9,true]
 ];
 const rows=[],failures=[];
 try{
  for(const year of ['2022','2023','2024','2025','2026']){
   out.textContent='Prüfe '+year+' … ('+(rows.length+1)+'/5)';
   await loadYear(year);
   const currentPool=eligible();
   const vals=series.map(([name,base,cap,fallback])=>{
    const pool=(base||currentPool).filter(i=>base===null||
      state.box==='all'||Math.abs(Number(META[i].b)-Number(state.box))<1e-5);
    if(!pool.length)return null;
    const x=simulatePortfolio({pool,rank:state.rank,topN:state.topN,guard:state.guard,
     stopScope:state.stopScope,corrLimit:cap===null?state.corrLimit:cap,
     corrFallback:cap===null?state.corrFallback:fallback},null,true);
    return {value:x.endWealth,dd:x.maxDrawdownPct,count:pool.length};
   });
   rows.push([year,...vals]);
   await new Promise(resolve=>setTimeout(resolve,0));
  }
 }catch(e){failures.push(e.message);}
 finally{
  try{await loadYear(savedYear);state.day=Math.min(savedDay,DAYS-1);state.selected=savedSelected;changePool();}
  catch(e){failures.push('Wiederherstellen: '+e.message);}
  btn.disabled=false;
 }
 const table=document.createElement('table');table.style.cssText='width:100%;font-size:11px;border-collapse:collapse';
 const title=document.createElement('tr');
 for(const h of ['Jahr',...series.map(s=>s[0])]){
  const cell=document.createElement('th');cell.textContent=h;cell.style.padding='5px';title.appendChild(cell);
 }
 table.appendChild(title);
 for(const row of rows){
  const tr=document.createElement('tr');
  for(const value of row){
   const td=document.createElement('td');td.style.cssText='padding:6px;border-top:1px solid #385769';
   td.textContent=typeof value==='string'?value:value?euro(value.value)+' / '+number(value.dd,1)+' % DD':'–';
   tr.appendChild(td);
  }
  table.appendChild(tr);
 }
 out.replaceChildren(document.createTextNode(
  'Einzeln gestartete Jahresdepots zu je 1.000 €, kein Jahresübertrag; aktuelle Top-N/Ranking/Stopp-Regeln. '+
  'TTP Alt 3 % / 1,5 % ist nicht der neue Kraken-Flip. '+(failures.length?'Fehler: '+failures.join('; '):'')
 ),table);
}

function installTTPResearch(){
 const base=$('tradeSweepControls');
 if(!base)throw Error('Top-N-Trade-Sweep UI fehlt');
 const panel=document.createElement('section');panel.className='tradeSweepControls';panel.id='ttpResearchControls';
 panel.innerHTML='<h3>462er-Raster · IMBA gegen TTP / Korrelation</h3>'+
  '<p class="muted">Gleiche Sensitivity-/Box-Koordinaten. Nur bereits vorhandene Alt-TTP-Ergebnisse (Aktivierung 3 % oder 2 %) sind in den Tageskurven enthalten.</p>'+
  '<label for="researchPairPool">Historisch vorhandener Pool</label>'+
  '<select id="researchPairPool"><option value="pure315">462 IMBA pur · paarig</option>'+
  '<option value="ttp315">462 TTP Alt · Aktivierung 3 % / Trail 1,5 %</option>'+
  '<option value="mixed315">Bis 924 IMBA + TTP Alt 3/1,5</option>'+
  '<option value="pure21">462 IMBA pur · paarig mit TTP 2/1</option>'+
  '<option value="ttp21">462 TTP Alt · Aktivierung 2 % / Trail 1 %</option>'+
  '<option value="mixed21">Bis 924 IMBA + TTP Alt 2/1</option></select>'+
  '<p id="historicalPairsInfo" class="muted"></p>'+
  '<div class="tradeSweepButtons"><button id="researchCorr85" type="button">ρ≤0,85 strikt</button>'+
  '<button id="researchCorr90" type="button">ρ≤0,90 auffüllen</button>'+
  '<button id="researchCorrOff" type="button">Korrelation aus</button></div>'+
  '<button type="button" id="historicalCompareBtn">Historische Pools vergleichen</button>'+
  '<button type="button" id="historicalAllYearsBtn">Vergleich 2022–2026</button>'+
  '<div id="historicalAllYearsOutput" class="muted"></div>'+
  '<p id="historicalCompareOutput" class="muted" aria-live="polite"></p><div id="historicalCompareChart"></div>'+
  '<details><summary>Neue Kraken-TTP-Einstellungen (Research-Preset, Backtest ausstehend)</summary>'+
  '<p class="muted">Keine fiktiven Erträge: Änderungen hier werden gespeichert, aber nicht auf die vorhandenen Alt-TTP-Tagesrenditen angewendet.</p>'+
  '<label for="krakenTtpActivation">Aktivierung</label><select id="krakenTtpActivation">'+
  '<option value="0">0 % sofort</option><option value="0.5">0,5 %</option>'+
  '<option value="1">1 %</option><option value="2">2 %</option><option value="3">3 %</option></select>'+
  '<label for="krakenTtpTrail">Trailing-Abstand: <strong id="krakenTtpTrailValue">1,5 %</strong></label>'+
  '<input id="krakenTtpTrail" type="range" min="0.5" max="4" step="0.5" value="1.5">'+
  '<label for="krakenTtpAction">Aktion beim TTP-Stopp</label>'+
  '<select id="krakenTtpAction"><option value="flip">Kraken-Flip → WAIT</option>'+
  '<option value="close">Schließen / Teilgewinn</option></select>'+
  '<label for="krakenTtpFraction">Geschlossener Anteil (nur Schließen)</label>'+
  '<select id="krakenTtpFraction"><option value="100">100 %</option><option value="50">50 %</option></select>'+
  '<p class="muted">Kraken-Flip schließt die alte Position vollständig und eröffnet die Gegenrichtung. Der neue Trade ist zunächst WAIT (SL), bis IMBA erneut TTP aktiviert. Für einen veränderbaren TTP braucht es neue Intraday-Trades und neue Tageskurven.</p>'+
  '</details>';
 base.insertAdjacentElement('afterend',panel);
 const pool=$('researchPairPool');
 const poolDropdown=$('poolMode');
 const poolLabels={
  pure315:'IMBA pur · TTP 3/1,5 gepaart',ttp315:'TTP Alt 3% / 1,5% · gepaart',
  mixed315:'IMBA + TTP Alt 3/1,5',pure21:'IMBA pur · TTP 2/1 gepaart',
  ttp21:'TTP Alt 2% / 1%',mixed21:'IMBA + TTP Alt 2/1'
 };
 for(const [value,label] of Object.entries(poolLabels)){
  if(![...poolDropdown.options].some(o=>o.value===value)){
   const o=document.createElement('option');o.value=value;o.textContent=label;poolDropdown.appendChild(o);
  }
 }
 const nativePoolChange=poolDropdown.onchange;
 poolDropdown.onchange=async e=>{
  if(RESEARCH_POOL_VARIANTS[e.target.value]){
   const mode=e.target.value;const fam=mode.startsWith('pure')?'none':mode.startsWith('ttp')?'ttp':'mix';
   state.priceMode='renko_usd';state.box='all';
   state.familyEnabled={none:fam!=='ttp',tp:false,ttp:fam!=='none',both:false};
   state.family=familySelectionMode();
   syncModeControls();
  }
  await nativePoolChange(e);
 };
 pool.onchange=e=>{
  state.mode=e.target.value;state.priceMode='renko_usd';state.box='all';
  const fam=e.target.value.startsWith('pure')?'none':e.target.value.startsWith('ttp')?'ttp':'mix';
  state.familyEnabled={none:fam!=='ttp',tp:false,ttp:fam!=='none',both:false};
  state.family=familySelectionMode();
  if($('poolMode')&&![...$('poolMode').options].some(o=>o.value===state.mode)){
   const opt=document.createElement('option');opt.value=state.mode;opt.textContent=e.target.selectedOptions[0].textContent;$('poolMode').appendChild(opt);
  }
  syncModeControls();setControls();changePool();
 };
 $('researchCorr85').onclick=()=>{state.corrLimit=.85;state.corrFallback=false;setControls();changePool();};
 $('researchCorr90').onclick=()=>{state.corrLimit=.9;state.corrFallback=true;setControls();changePool();};
 $('researchCorrOff').onclick=()=>{state.corrLimit=null;state.corrFallback=false;setControls();changePool();};
 $('historicalCompareBtn').onclick=historicalCompare;
 $('historicalAllYearsBtn').onclick=compareTtpYears;
 $('krakenTtpActivation').onchange=e=>{state.krakenTtp.activationPct=Number(e.target.value);};
 $('krakenTtpTrail').oninput=e=>{state.krakenTtp.trailPct=Number(e.target.value);renderTtpControls();};
 $('krakenTtpAction').onchange=e=>{state.krakenTtp.action=e.target.value==='close'?'close':'flip';renderTtpControls();};
 $('krakenTtpFraction').onchange=e=>{state.krakenTtp.closeFractionPct=Number(e.target.value)===50?50:100;};
 const oldLoad=$('loadPreset').onchange;
 $('loadPreset').onchange=async e=>{
  const file=e.target.files[0];
  if(file){try{const x=JSON.parse(await file.text());state.krakenTtp=normalizeResearchTtp(x.krakenTtp);}catch(err){console.error(err);}}
  await oldLoad(e);renderTtpControls();
 };
 const select=$('correlationLimit');
 for(const val of [.5,.65,.7,.75,.8,.95]){
  if(![...select.options].some(o=>o.value===String(val))){
   const opt=document.createElement('option');opt.value=String(val);opt.textContent='Maximal +'+val.toFixed(2);select.appendChild(opt);
  }
 }
 if(state.corrLimit!==null)select.value=String(state.corrLimit);
 renderTtpControls();
}

async function startApp(){try{await loadYear('2026');bindUI();bindSavedEvidence();installTTPResearch();window.__QUANT_V13_READY=true;}catch(e){console.error('Quant v13 failed to load',e);alert('Daten konnten nicht geladen werden: '+e.message);}}
startApp();