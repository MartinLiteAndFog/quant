import { useState } from "react";
import { type EcologyControls, GENTLE_ECOLOGY } from "./simulation";
import type { SolArchive } from "./historicalGarden";
import type { FundingSettings } from "./fundingPolicy";
import { observedStartDate, type StartPreset } from "./gameStart";
import "./gameMenu.css";

type Symbol = "SOL-USDT" | "BTC-USDT";

export type GameMenuProps = {
  archive: SolArchive | null;
  symbol: Symbol;
  onSymbol: (symbol: Symbol) => void;
  startDate: string;
  onStartDate: (date: string) => void;
  controls?: EcologyControls;
  onControls: (controls: EcologyControls) => void;
  evolutionEnabled: boolean;
  onEvolution: (enabled: boolean) => void;
  funding: FundingSettings;
  onFunding: (settings: FundingSettings) => void;
  hasProgress: boolean;
  canClose: boolean;
  busy: boolean;
  error: string | null;
  onNew: () => void;
  onContinue: () => void;
};

const presets: readonly [StartPreset, string][] = [
  ["last30", "Letzte 30 Beobachtungstage"],
  ["last90", "Letzte 90 Beobachtungstage"],
  ["last365", "Letzte 365 Beobachtungstage"],
  ["first", "Ab Archivbeginn"],
];

export function GameMenu(p: GameMenuProps) {
  const rules = p.controls ?? GENTLE_ECOLOGY;
  const [rulesOpen, setRulesOpen] = useState(true);
  const [fundingOpen, setFundingOpen] = useState(true);
  return <div className="cg-menu-backdrop">
    <section className="cg-menu" role="dialog" aria-modal="true" aria-labelledby="cg-menu-heading">
      <div className="cg-menu-head"><div className="cg-menu-logo">◌ <strong>CLOUDGARDEN</strong><small>STRATEGY ECOSYSTEM</small></div>
        <span className="cg-menu-status">HISTORICAL PAPER · KEIN LIVE-HANDEL</span></div>
      <div className="cg-menu-content">
        <div className="cg-menu-intro"><span>NEUE EXPEDITION / FORTSETZEN</span>
          <h1 id="cg-menu-heading">Ein Garten. Viele Strategien.</h1>
          <p>Wähle das echte Datenarchiv, einen Startpunkt und die Regeln des Ökosystems. Die Handels-Engine bleibt unverändert.</p>
        </div>
        <div className="cg-menu-columns">
          <div className="cg-menu-group">
            <h2>01 · Datenwelt</h2>
            <div className="cg-menu-options" role="group" aria-label="Marktarchiv">
              {(["SOL-USDT","BTC-USDT"] as const).map(symbol=><button key={symbol} type="button" aria-pressed={p.symbol===symbol}
                onClick={()=>p.onSymbol(symbol)}>{symbol.startsWith("SOL")?"SOLANA":"BITCOIN"} <small>{symbol}</small></button>)}
            </div>
            {p.archive ? <dl className="cg-menu-source">
              <div><dt>Quelle</dt><dd>{p.archive.source}</dd></div>
              <div><dt>Zeitraum (UTC)</dt><dd>{p.archive.firstDate} — {p.archive.lastDate}</dd></div>
              <div><dt>Beobachtete Minuten</dt><dd>{p.archive.totalBars.toLocaleString("de-DE")}</dd></div>
              <div><dt>Identität</dt><dd title={p.archive.sourceSha256}>SHA-256: {p.archive.sourceSha256.slice(0,16)}…</dd></div>
              <div><dt>Datenlücken</dt><dd>{p.archive.missingMinutes.toLocaleString("de-DE")} Minuten · nicht aufgefüllt</dd></div>
            </dl> : <p className="cg-menu-no-data" role="status">{p.error ?? "Lade und prüfe das lokale Archiv …"}</p>}
          </div>
          <div className="cg-menu-group">
            <h2>02 · Einstiegspunkt</h2>
            <div className="cg-menu-presets">
              {presets.map(([preset,label])=><button type="button" key={preset} disabled={!p.archive}
                onClick={()=>p.archive&&p.onStartDate(observedStartDate(p.archive,preset))}
                aria-pressed={!!p.archive&&p.startDate===observedStartDate(p.archive,preset)}>{label}</button>)}
            </div>
            <label className="cg-menu-date">Freies Startdatum (UTC)
              <input type="date" min={p.archive?.firstDate} max={p.archive?.lastDate}
                value={p.startDate} onChange={e=>p.onStartDate(e.target.value)} disabled={!p.archive}/>
            </label>
            <p>Tag 0 ist der gewählte Starttag. Das Spiel berechnet alle folgenden Trades; es überspringt keine PnL-Buchungen.</p>
          </div>
          <div className="cg-menu-group">
            <h2>03 · Spielregeln</h2>
            <details className="cg-menu-rules" open={rulesOpen} onToggle={e=>setRulesOpen(e.currentTarget.open)}>
              <summary>Ökologie / Debug-Profil <span>ANPASSBAR</span></summary>
              <label className="cg-menu-rule"><span>Fressfeinde <small>Aus: keine pflanzenbedingte Archivierung durch Räuber</small></span>
                <input type="checkbox" checked={rules.predatorsEnabled}
                  onChange={e=>p.onControls({...rules,predatorsEnabled:e.target.checked})}/></label>
              <label className="cg-menu-rule"><span>Vitalitätsmalus <small>Nur negative Vitalitätsänderung; 0 = keine Strafe</small></span>
                <input type="range" min="0" max="1" step=".1" value={rules.vitalityMalus}
                  onChange={e=>p.onControls({...rules,vitalityMalus:Number(e.target.value)})}/>
                <output>{Math.round(rules.vitalityMalus*100)} %</output></label>
              <label className="cg-menu-rule"><span>Evolution / Samen <small>Geburten bleiben kapitalbegrenzt</small></span>
                <input type="checkbox" checked={p.evolutionEnabled} onChange={e=>p.onEvolution(e.target.checked)}/></label>
              <p>Standard: Räuber aus, Vitalitätsmalus null. Diese Einstellung verhindert keine handelsbedingten Verluste, Kapitalgrenzen oder die Populationsobergrenze.</p>
            </details>
            <details className="cg-menu-rules cg-funding-rules" open={fundingOpen} onToggle={e=>setFundingOpen(e.currentTarget.open)}>
              <summary>Kapital / Einzahlungen <span>{p.funding.mode==="topup"?"AUFSTOCKUNG AN":p.funding.mode==="monthly"?"MONATLICH AN":"AUS"}</span></summary>
              <label className="cg-menu-rule"><span>Monatlicher Kapitalzuschuss <small>Standardmäßig aus. Nur bei Aktivierung jeden neuen Archivmonat einzahlen.</small></span>
                <input type="checkbox" checked={p.funding.mode==="monthly"} onChange={e=>p.onFunding({...p.funding,mode:e.target.checked?"monthly":"off"})}/></label>
              <label className="cg-menu-rule"><span>Betrag pro Monat (€)</span>
                <input className="cg-funding-number" type="number" min="1" max="10000" step="10"
                  disabled={p.funding.mode!=="monthly"} value={p.funding.monthlyEur}
                  onChange={e=>{const n=e.currentTarget.valueAsNumber;if(Number.isFinite(n)&&n>0&&n<=10000)p.onFunding({...p.funding,monthlyEur:n});}}/></label>
              <label className="cg-menu-rule"><span>Depot auf Zielwert auffüllen <small>Standardmäßig an. Nach jedem Handelstag nur die Differenz einzahlen, falls das Depot darunter liegt.</small></span>
                <input type="checkbox" checked={p.funding.mode==="topup"} onChange={e=>p.onFunding({...p.funding,mode:e.target.checked?"topup":"off"})}/></label>
              <label className="cg-menu-rule"><span>Zielkapital (€)</span>
                <input className="cg-funding-number" type="number" min="1" max="10000" step="10"
                  disabled={p.funding.mode!=="topup"} value={p.funding.topUpTargetEur}
                  onChange={e=>{const n=e.currentTarget.valueAsNumber;if(Number.isFinite(n)&&n>0&&n<=10000)p.onFunding({...p.funding,topUpTargetEur:n});}}/></label>
              <p>Beide Automatiken schließen sich aus. Beispiel: 93 € Depot → 7 € Aufstockung auf 100 €.
                Gemessen wird das gesamte Depot einschließlich offener Positionen nach der Tagesabrechnung.
                Fester Forschungskurs: {p.funding.eurUsdt} USDT je €. Externe Einzahlungen zählen niemals als Handelsgewinn.</p>
            </details>
          </div>
        </div>
        {p.error&&p.archive&&<p className="cg-menu-error" role="alert">{p.error}</p>}
        <div className="cg-menu-actions">
          <button className="cg-menu-primary" type="button" onClick={p.onNew} disabled={!p.archive||p.busy}>
            {p.hasProgress?"NEU BEGINNEN · TAG 0":"SPIEL STARTEN"} <span>→</span></button>
          {p.canClose&&<button type="button" className="cg-menu-continue" onClick={p.onContinue} disabled={p.busy}>
            Bisherigen Spielstand fortsetzen</button>}
        </div>
        {p.hasProgress&&<p className="cg-menu-caution">Ein neues Spiel ersetzt nur den lokalen Spielstand dieses Markts. Für einen unveränderten historischen Verlauf müssen die Tage erneut durchlaufen werden.</p>}
      </div>
    </section>
  </div>;
}
