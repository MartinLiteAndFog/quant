import { useId, useState } from 'react';
import { fundingSettings, type HistoricalSession, type SolArchive } from './historicalGarden';
import { IsometricGarden } from './IsometricGarden';
import { GeneLibrary } from './GeneLibraryPanel';
import { getKrakenProfile, KRAKEN_STARTER_PROFILES } from './krakenProfiles';
import type { BeeMode } from './simulation';
import './game.css';

const money = (n: number) => `${n.toLocaleString('de-DE', {minimumFractionDigits: 2, maximumFractionDigits: 4})} USDT`;
export type GameSurfaceProps = {
  session: HistoricalSession; archive: SolArchive; selectedId: string; onSelect: (id:string)=>void;
  playing: boolean; busy: boolean; error: string|null; saveStatus:string; startDate:string;
  onDate:(date:string)=>void; onRestart:()=>void; onAdvance:(days:number)=>void; onPlay:()=>void;
  onPause:()=>void; onPolicy:(policy:'mixed'|BeeMode)=>void; speed:number; onSpeed:(speed:number)=>void;
  profileIds:readonly string[]; onProfiles:(ids:readonly string[])=>void;
  onOpenMenu:()=>void;
};

/** Game composition uses ledger values directly; no decorative price or performance data. */
export function GameSurface(p:GameSurfaceProps) {
  const scope=useId();
  const asset=p.archive.symbol.split("-")[0];
  const [motion, setMotion] = useState(true);
  const [library, setLibrary] = useState(false);
  const [tab,setTab] = useState('Handel');
  const tabs=['Handel','Gene','Analyse','Abstammung'];
  const [plane,setPlane] = useState<'root-leaf'|'water-sun'|'trading'>('root-leaf');
  const {session:s,archive:a} = p, w=s.world, c=w.capital;
  const plant=w.plants.find(x=>x.id===p.selectedId) ?? w.plants.find(x=>x.status==='active') ?? w.plants[0];
  const genome=plant ? s.genomes[plant.id] : undefined, record=plant ? s.records[plant.id] : undefined;
  const state=plant ? s.krakenStates?.[plant.id] : undefined;
  const profile=state ? (s.krakenProfiles?.[plant!.id] ?? getKrakenProfile(state.profileId)) : undefined;
  const funding=s.funding;
  const fundingRule=fundingSettings(funding);
  const fundingLabel=!funding ? "Keine automatischen Einzahlungen" : fundingRule.mode==="topup"
    ? `Auffüllen auf ${fundingRule.topUpTargetEur} € bei Unterschreitung (Tagesende)`
    : fundingRule.mode==="monthly" ? `${fundingRule.monthlyEur} € monatlicher Zuschuss` : "Keine automatischen Einzahlungen";
  const deposits=funding?.totalUsdt ?? 0, pnl=c.navUsdt-100-deposits;
  const alive=w.plants.filter(x=>x.status==='active').length;
  const parameters=profile ? {...profile} : {...genome};
  const candidate=plant ? {name:`${plant.name} · ${profile?.name ?? genome?.name ?? 'Genom'}`,source:profile?.sourceFiles.join(', ') ?? genome?.sourceId ?? 'Cloudgarden',parameters} : undefined;
  const recent=s.trades.filter(t=>t.plantId===plant?.id).slice(-30).reverse();
  const progress=100*s.loop.archiveIndex/a.days.length;
  return <main className="cg-game">
    <header className="cg-hud"><div className="cg-calendar"><button type="button" className="cg-menu-trigger" onClick={p.onOpenMenu}>☰ Start / Einstellungen</button><h1>Ein Garten lernt handeln.</h1><p>Archivdatum {s.nextDate} · Umlauf {s.loop.cycle} · {s.loop.totalDays.toLocaleString('de-DE')} simulierte Tage</p></div>
      <dl className="cg-meters"><div><dt>Kapital</dt><dd>{money(c.navUsdt)}</dd></div><div><dt>Handelsergebnis nach Kosten</dt><dd className={pnl<0?'cg-loss':''}>{money(pnl)}</dd></div><div><dt>Einzahlungen</dt><dd>{money(deposits)}</dd></div><div><dt>Ausführungen</dt><dd>{s.loop.totalFills.toLocaleString('de-DE')}</dd></div></dl>
      <div className="cg-play"><button className="cg-primary" onClick={p.onPlay} disabled={p.busy&&!p.playing}>{p.playing?'Pause':'Starten'}</button><button onClick={p.onPause} disabled={!p.playing&&!p.busy}>Nach diesem Tag pausieren</button><button onClick={()=>p.onAdvance(1)} disabled={p.busy}>Ein Tag</button><button type="button" onClick={p.onRestart} disabled={p.busy||p.playing} title="Neue Simulation am gewählten Startdatum; der bisherige Spielstand wird ersetzt">↺ Tag 0</button><label>Tempo<select value={p.speed} onChange={e=>p.onSpeed(Number(e.target.value))}><option value={1}>Ruhig</option><option value={8}>8 Tage</option><option value={32}>32 Tage</option></select></label><button onClick={()=>setLibrary(true)}>Gen-Datenbank</button><label><input type="checkbox" checked={motion} onChange={e=>setMotion(e.target.checked)}/> Bewegung</label></div>
      <progress aria-label="Fortschritt durch das Datenarchiv" max={100} value={progress}/><div className="cg-status" role="status">{p.busy?'Kerzen auswerten und Kapital abstimmen …':p.saveStatus}{s.loop.capitalHalted?' · Neue Positionen derzeit gesperrt':''}</div>
      {p.error&&<p role="alert" className="cg-error">{p.error}</p>}
    </header>
    <div className="cg-playfield"><section className="cg-landscape" aria-label="Lebender Strategiegarten"><div className="cg-map-heading"><h2>{alive} lebende Pflanzen</h2><span>Kapitalzuweisung {money(c.deployedCapitalUsdt)} · Gebühren {money(c.paperFeesUsdt)}</span></div>
      <IsometricGarden world={w} selectedId={plant?.id??''} onSelect={p.onSelect} running={p.playing} motionEnabled={motion} plane={plane} onPlaneChange={setPlane} capitalBaselineUsdt={100+deposits}/>
      <div className="cg-garden-settings"><label>Bestäubung<select value={w.conceptionPolicy} disabled={p.busy} onChange={e=>p.onPolicy(e.target.value as 'mixed'|BeeMode)}>{['mixed','random','local','hive','hybrid'].map(x=><option key={x} value={x}>{({mixed:'Gemischt',random:'Zufällig',local:'Nachbarschaft',hive:'Bienenstock',hybrid:'Hybrid'})[x]}</option>)}</select></label><p>{w.dynamicEcology ? "Samen entstehen durch Bestäubung und brauchen freies Kapital. Fressfeinde sind im Startprofil aus, über Start / Einstellungen zuschaltbar. Höchstens 50 Pflanzen leben gleichzeitig." : "Dieser ältere Spielstand nutzt noch 20 einmalige Saatplätze. Eine neue Welt aktiviert dynamische Vermehrung und monatliche Einzahlungen."}</p></div>
    </section>
    <aside className="cg-inspector" aria-label="Informationen zur ausgewählten Pflanze">
      {plant?<><header><span>{plant.phenotype} · {plant.status==='active'?'lebend':'archiviert'} · {plant.physiology}</span><h2>{plant.name}</h2><p>{profile?.name??genome?.name??'Strategiepflanze'}</p><button onClick={()=>setLibrary(true)}>Strategiebausteine sammeln</button></header>
      <div className="cg-tabs" role="tablist" aria-label="Pflanzeninformationen">{tabs.map((t,index)=><button key={t} role="tab" tabIndex={tab===t?0:-1} onKeyDown={e=>{const move=e.key==='ArrowRight'?1:e.key==='ArrowLeft'?-1:0;const next=e.key==='Home'?0:e.key==='End'?tabs.length-1:move?(index+move+tabs.length)%tabs.length:-1;if(next>=0){e.preventDefault();setTab(tabs[next]);document.getElementById(`${scope}-tab-${tabs[next]}`)?.focus();}}} aria-selected={tab===t} id={`${scope}-tab-${t}`} aria-controls={`${scope}-inspector-panel`} onClick={()=>setTab(t)}>{t}</button>)}</div>
      <section id={`${scope}-inspector-panel`} role="tabpanel" aria-labelledby={`${scope}-tab-${tab}`}>
      {tab==='Handel'&&<><h3>So handelt diese Pflanze</h3><p>{profile?`${profile.mode==='CT'?'Countertrend mit TTP und WAIT-Zustand':'Trendfolge mit Teilgewinn und TP2'}. Abgeschlossene 5-Minuten-Kerzen bilden einen Renko-Proxy. Ausführung erst am folgenden Minuten-Open.`:'IMBA-Kanal mit optional bestätigten Elliott-Wave-Modulen; Signale aus abgeschlossenen Kerzen, spätere Ausführung.'}</p>
      <dl className="cg-values"><div><dt>Zugewiesenes Kapital</dt><dd>{money(plant.allocationUsdt)}</dd></div><div><dt>Position</dt><dd>{state?`${state.quantity.toFixed(8)} ${asset} · ${state.mode}`:'Täglich glattgestellt'}</dd></div>{state&&<><div><dt>Einstieg</dt><dd>{money(state.entryPrice)}</dd></div><div><dt>Offenes Ergebnis</dt><dd>{money(state.quantity*(state.lastMarkPrice-state.entryPrice))}</dd></div></>}</dl>
      <h3>Vollständige Einstellungen</h3><p>Quellparameter und wirksame Sicherheitsbegrenzungen sind separat aufgeführt. Kraken bezeichnet die Vorlage; das {a.symbol}-Archiv hat keinen verifizierten Handelsplatz.</p><ParameterTree value={parameters}/><h3>Herkunft</h3><p className="cg-source">{profile?.sourceFiles.join(' · ')??genome?.sourceId}</p><p>{profile ? "Die Vorlage sieht bis zu 10× Hebel vor; Cloudgarden begrenzt auf 1× und Schutzstopp." : "Diese Kerzenvariante verwendet höchstens 1× und einen Schutzstopp."} Keine nachgewiesene TradingView-Parität.</p></>}
      {tab==='Gene'&&<><h3>Lebensraum und Wuchs</h3><ParameterTree value={plant.genes}/><h3>{profile?'Wirksame Handelsgene':'Handelsgene'}</h3><ParameterTree value={profile?{channelLookback:profile.channelLookback,ttpTrailPct:profile.ttpTrailPct,partialTpPct:profile.partialTpPct,waitMinSlPct:profile.waitMinSlPct,waitMaxSlPct:profile.waitMaxSlPct,hardSlPct:profile.hardSlPct}:genome??{}}/>{profile&&<details><summary>EW-Genmaterial und Herkunft</summary><ParameterTree value={genome??{}}/><p>Bestätigte EW-Pivots können Einstiege filtern und Impuls-Exits auslösen. Die wirksamen Schalter stehen unter Handel → Vollständige Einstellungen → ew. Die Erkennung wartet auf die rechte Bestätigungslänge.</p></details>}<h3>Ökologische Signalmerkmale</h3><ParameterTree value={plant.tradingTraits}/><p>Die Ebene zeigt die jeweils beschrifteten Gene. Der Wurzelpunkt bleibt an der exakten Position; ausweichender Wuchs verändert keine Zahlen.</p><button onClick={()=>setLibrary(true)}>In der Gen-Datenbank öffnen</button></>}
      {tab==='Analyse'&&<><h3>Gemessene Handelsaktivität</h3>{record?.equityHistory&&<EquityCurve points={record.equityHistory}/>}<dl className="cg-values"><div><dt>Nettoergebnis</dt><dd>{money(record?.netPnl??0)}</dd></div><div><dt>Gebühren</dt><dd>{money(record?.fees??0)}</dd></div><div><dt>Ausführungen</dt><dd>{record?.fills??0}</dd></div><div><dt>Abgelehnt</dt><dd>{record?.rejected??0}</dd></div>{record?.observedDays!==undefined&&<><div><dt>Gemessene Handelstage</dt><dd>{record.observedDays}</dd></div><div><dt>Größter Tagesend-Rückgang</dt><dd>{record.maxDrawdownPct?.toFixed(2)} %</dd></div><div><dt>Realisiertes Preisergebnis</dt><dd>{money(record.realizedPnl??0)}</dd></div><div><dt>Offenes Preisergebnis</dt><dd>{money(record.unrealizedPnl??0)}</dd></div></>}<div><dt>Beobachtungstage der Ökologie</dt><dd>{plant.evidenceDays}</dd></div><div><dt>Vitalität / Fitness</dt><dd>{plant.vigor.toFixed(1)} / {plant.fitness.toFixed(1)}</dd></div></dl><p>Fitness und Vitalität sind ökologische Modellwerte, kein Rentabilitätsnachweis. Eine Trefferquote oder Sharpe-Ratio wird aus dem gekürzten Ausführungsspeicher nicht abgeleitet.</p>
      <h3>Letzte Ausführungen dieser Pflanze</h3>{recent.length?<ol className="cg-fills">{recent.map((t,i)=><li key={`${t.executedAt}-${i}`}><time>{t.executedAt.replace('T',' ').slice(0,16)}</time><b>{t.side} · {t.filledQuantity} {asset}</b><span>{t.price} USDT · Gebühr {t.fee}</span></li>)}</ol>:<p>Noch keine Ausführungen gespeichert. Die Strategie kann noch auf ein bestätigtes Signal warten.</p>}</>}
      {tab==='Abstammung'&&<><h3>Eltern und Nachkommen</h3><p>{plant.parents?.join(' × ')??(plant.conceivedBy?'Aus dem Quellgen-Pool bestäubt':'Anfangsvariante aus einer Quellstrategie')}</p><p>Bestäubung: {plant.conceivedBy??'Startpopulation'}</p><ul>{w.plants.filter(x=>x.parents?.includes(plant.id)).map(x=><li key={x.id}><button onClick={()=>p.onSelect(x.id)}>{x.name} · {x.status}</button></li>)}</ul><h3>Saat und Kapital</h3><p>Bestäubung kombiniert vorhandene Gene mit begrenzter Zufallsmutation. Neue Pflanzen erhalten nur verfügbares Kapital. {w.dynamicEcology ? "Es gibt keine festen Pflanzplätze. Die Berechnung begrenzt gleichzeitig lebende Pflanzen auf 50." : "Dieser ältere Spielstand besitzt 20 einmalig nutzbare Saatplätze; starte eine neue Welt für die dynamische Population."}</p><dl className="cg-values"><div><dt>Freies Kapital</dt><dd>{money(c.freeCapitalUsdt)}</dd></div><div><dt>Geschützte Reserve</dt><dd>{money(c.protectedReserveUsdt)}</dd></div></dl><h3>Letzte Ereignisse</h3><ol className="cg-fills">{w.events.filter(e=>e.actor===plant.id||e.subject===plant.id).slice(-12).reverse().map(e=><li key={e.id}>Tag {e.day} · {e.type}<small>{JSON.stringify(e.detail)}</small></li>)}</ol></>}
      </section></>:<p>Wähle eine Pflanze im Garten.</p>}
    </aside></div>
    <details className="cg-method"><summary>Daten, Forschungsgrenzen und neue Welt</summary><p>{a.totalBars.toLocaleString('de-DE')} echte Minuten · {a.firstDate} bis {a.lastDate} · {a.missingMinutes} fehlende Minuten bleiben Lücken. Quelle SHA {a.sourceSha256.slice(0,16)}…</p><p>Wiederholte Archivdurchläufe sind Training auf denselben Daten. Kein unabhängiger Nachweis einer profitablen Strategie: not_comparable. Preise sind historisch, Ausführungen und Kosten modelliert; Funding fehlt. Finanzierung: {fundingLabel}. Einzahlungen sind externe Kapitalzuflüsse, kein Gewinn. Verwendeter fester Simulationskurs: {funding?.eurUsdt??1} USDT je EUR; kein historischer Devisenkurs.</p><div className="cg-new-world"><label>Startdatum<input type="date" min={a.firstDate} max={a.lastDate} value={p.startDate} onChange={e=>p.onDate(e.target.value)} disabled={p.busy}/></label>{p.profileIds.map((id,i)=><label key={i}>Anfangsvariante {i+1}<select value={id} disabled={p.busy} onChange={e=>p.onProfiles(p.profileIds.map((x,j)=>j===i?e.target.value:x))}>{KRAKEN_STARTER_PROFILES.map(x=><option key={x.id} value={x.id}>{x.name}</option>)}</select></label>)}<button onClick={p.onRestart} disabled={p.busy}>Neue Welt mit diesen Varianten</button></div></details>
    <GeneLibrary open={library} onClose={()=>setLibrary(false)} candidate={candidate}/>
  </main>;
}
function ParameterTree({value}:{value:Record<string,unknown>}) {
  return <dl className="cg-parameters">{Object.entries(value).map(([key,item])=><div key={key}><dt>{key}</dt><dd>{item&&typeof item==='object'?<ParameterTree value={item as Record<string,unknown>}/>:typeof item==='boolean'?(item?'an':'aus'):String(item??'—')}</dd></div>)}</dl>;
}

function EquityCurve({points}:{points:readonly {day:number;equity:number;netPnl:number}[]}) {
  if(!points.length)return null;
  const values=points.map(x=>x.netPnl),low=Math.min(0,...values),high=Math.max(0,...values),range=Math.max(.01,high-low);
  const y=(value:number)=>18+(high-value)/range*100;
  const path=points.map((point,index)=>`${20+index/Math.max(1,points.length-1)*280},${y(point.netPnl)}`).join(' ');
  return <figure className="cg-curve"><figcaption>Kumuliertes Nettoergebnis · letzte {points.length} Handelstage</figcaption><svg viewBox="0 0 320 150" role="img" aria-label={`Nettoergebnis von ${money(values[0]!)} bis ${money(values.at(-1)!)}, Simulationstag ${points[0]!.day} bis ${points.at(-1)!.day}`}><line x1="20" x2="300" y1={y(0)} y2={y(0)} stroke="#a9b8a0" strokeDasharray="3 4"/><polyline points={path} fill="none" stroke="#466743" strokeWidth="2"/><text x="20" y="142">Tag {points[0]!.day}</text><text x="300" y="142" textAnchor="end">Tag {points.at(-1)!.day}</text></svg><p>{money(values[0]!)} → {money(values.at(-1)!)}</p></figure>;
}
