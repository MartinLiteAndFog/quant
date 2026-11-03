import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { setConceptionPolicy, setEcologyControls, setEcologyEvolution, GENTLE_ECOLOGY, type EcologyControls, type BeeMode, type GardenPlant, type SimulationWorld } from "./simulation";
import { advanceHistoricalSession, createHistoricalSession, HISTORICAL_SAVE_KEY, loadSolArchive, validateHistoricalSession,
  configureHistoricalFunding, fundingSettings, type HistoricalSession, type SolArchive } from "./historicalGarden";
import { DEFAULT_FUNDING_SETTINGS, type FundingSettings } from "./fundingPolicy";
import "./historical.css";
import { GameSurface } from "./GameSurface";
import { loadHistoricalSnapshot, saveHistoricalSnapshot } from "./historicalStorage";
import { GameMenu } from "./GameMenu";
import { observedStartDate, requireObservedDate } from "./gameStart";

export type HistoricalGardenControls = {
  world: SimulationWorld; selected: GardenPlant; selectedId: string; playing: boolean; busy: boolean;
  onSelect: (id: string) => void; onAdvance: (days: number) => void; onTogglePlay: () => void;
  onReset: () => void; onPolicy: (policy: "mixed" | BeeMode) => void;
};


/** Historical trading is the default playable path; legacy ecology fixtures remain separate. */
export function HistoricalTrading(_props: { renderGarden?: (controls: HistoricalGardenControls) => ReactNode }) {
  const [playing,setPlaying]=useState(false);
  const [symbol, setSymbol] = useState<"SOL-USDT" | "BTC-USDT">("SOL-USDT");
  return <div className="cg-single-garden"><HistoricalGarden key={symbol} symbol={symbol}
    onSymbol={(value)=>{setPlaying(false);setSymbol(value);}}
    sharedPlaying={playing} setSharedPlaying={setPlaying}/></div>;
}
function HistoricalGarden({symbol,onSymbol,sharedPlaying,setSharedPlaying}:{symbol:"SOL-USDT"|"BTC-USDT";onSymbol:(symbol:"SOL-USDT"|"BTC-USDT")=>void;sharedPlaying:boolean;setSharedPlaying:(value:boolean)=>void}) {
  const snapshotKey=symbol==="SOL-USDT"?HISTORICAL_SAVE_KEY:`${HISTORICAL_SAVE_KEY}.btc`;
  const [archive, setArchive] = useState<SolArchive | null>(null);
  const [session, setSession] = useState<HistoricalSession | null>(null);
  const [startDate, setStartDate] = useState("");
  const [menuOpen, setMenuOpen] = useState(true);
  const [started, setStarted] = useState(false);
  const [hasSnapshot, setHasSnapshot] = useState(false);
  const [ecologyControls, setControls] = useState<EcologyControls>(GENTLE_ECOLOGY);
  const [funding, setFunding] = useState<FundingSettings>(DEFAULT_FUNDING_SETTINGS);
  const [evolutionEnabled, setEvolutionEnabled] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [profileIds, setProfileIds] = useState<readonly string[]>(["kraken_legacy", "kraken_17_2_close", "kraken_17_2_reverse"]);
  const [selectedId, setSelectedId] = useState("parent-01");
  const playing=sharedPlaying, setPlaying=setSharedPlaying;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState("Lokales SOL-Archiv laden …");
  const current = useRef<HistoricalSession | null>(null);
  const locked = useRef(false);
  const stop = useRef(false);
  const saveEnabled = useRef(true);
  const generation = useRef(0);

  useEffect(() => {
    generation.current += 1;
    let cancelled = false;
    void loadSolArchive(symbol).then(async (a) => {
      if (cancelled) return;
      const initialDate = observedStartDate(a, "last30");
      let restored = createHistoricalSession(a, initialDate);
      let recovered = false;
      try {
        const raw = await loadHistoricalSnapshot(snapshotKey);
        if (cancelled) return;
        if (raw) { restored = validateHistoricalSession(raw, a); recovered = true; }
        // Existing saves keep their financial history and can disable historical predation on resume.
        if (!restored.world.ecologyControls) restored = {...restored, world:setEcologyControls(restored.world, GENTLE_ECOLOGY)};
        setSaveStatus(raw ? "Gespeicherte Welt wiederhergestellt" : "Welt bereit");
      } catch {
        saveEnabled.current = false;
        setSaveStatus("Älterer Spielstand bleibt geschützt. Eine neue Welt ersetzt ihn.");
      }
      current.current = restored; setSession(restored); setArchive(a); setStartDate(restored.startDate);
      setHasSnapshot(recovered); setControls(restored.world.ecologyControls ?? GENTLE_ECOLOGY);
      setFunding(fundingSettings(restored.funding));
      setEvolutionEnabled(!restored.world.evolutionPaused);
      if (restored.initialProfileIds) setProfileIds(restored.initialProfileIds);
    }).catch((e: unknown) => { if (!cancelled) setError(e instanceof Error ? e.message : "SOL history could not be loaded."); });
    return () => { cancelled = true; stop.current = true; generation.current += 1; };
  }, []);

  useEffect(() => {
    if (!session || !started || !saveEnabled.current) return;
    let cancelled=false;
    void saveHistoricalSnapshot(session,snapshotKey).then(()=>{if(!cancelled)setSaveStatus("Welt auf diesem Gerät gespeichert");}).catch(()=>{if(!cancelled)setSaveStatus("Speicher nicht verfügbar; halte dieses Fenster für den Fortschritt offen");});
    return ()=>{cancelled=true;};
  }, [session, started]);

  const advance = useCallback(async (days: number, autoplay = false) => {
    if (locked.current || !archive || !current.current) return;
    if (!Number.isInteger(days) || days < 1 || days > 140) return;
    locked.current = true; stop.current = false; setBusy(true); setError(null);
    const lifetime = generation.current;
    if (!autoplay) setPlaying(false);
    try {
      for (let i = 0; i < days && !stop.current; i += 1) {
        const next = await advanceHistoricalSession(current.current!, archive);
        if (lifetime !== generation.current) return;
        current.current = next; setSession(next);
        // Let the browser paint each finished historical day and its new NAV.
        await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
      }
    } catch (e) { if (lifetime === generation.current) { setPlaying(false); setError(e instanceof Error ? e.message : "Historical playback stopped."); } }
    finally { locked.current = false; if (lifetime === generation.current) setBusy(false); }
  }, [archive]);

  useEffect(() => {
    if (!playing || busy || !session) return;
    const timer = window.setTimeout(() => { void advance(speed, true); }, speed === 1 ? 700 : 30);
    return () => window.clearTimeout(timer);
  }, [playing, busy, session, advance, speed]);

  const restart = () => {
    if (!archive || locked.current) return;
    if (started && current.current && current.current.world.day > 0
      && !window.confirm("Den bisherigen Spielstand ersetzen und bei Tag 0 neu beginnen?")) return;
    try {
      const observed = requireObservedDate(archive,startDate);
      const fresh = createHistoricalSession(archive, observed, { profileIds, fundingMode: funding.mode,
        monthlyEur: funding.monthlyEur, topUpTargetEur: funding.topUpTargetEur, eurUsdt: funding.eurUsdt });
      const next = {...fresh, world:setEcologyEvolution(setEcologyControls(fresh.world,ecologyControls),evolutionEnabled)};
      saveEnabled.current = true; stop.current = true; setPlaying(false); setSelectedId("parent-01");
      current.current = next; setSession(next); setError(null); setHasSnapshot(true);
      setStarted(true); setMenuOpen(false);
    } catch (e) { setError(e instanceof Error ? e.message : "Choose an available date."); }
  };
  const continueGame = () => {
    if (!current.current || locked.current) return;
    const next = configureHistoricalFunding({...current.current,world:setEcologyEvolution(setEcologyControls(current.current.world,ecologyControls),evolutionEnabled)},funding);
    current.current = next;setSession(next);setError(null);
    setStarted(true);setMenuOpen(false);
  };
  const changeFunding = (value: FundingSettings) => {
    setFunding(value);
    if (started && current.current && !locked.current) {
      const next = configureHistoricalFunding(current.current,value);
      current.current=next;setSession(next);
    }
  };
  const changeRules = (value: EcologyControls) => {
    setControls(value);
    if (started && current.current && !locked.current) {
      const next = {...current.current,world:setEcologyControls(current.current.world,value)};
      current.current=next;setSession(next);
    }
  };
  const changeEvolution = (enabled:boolean) => {
    setEvolutionEnabled(enabled);
    if (started && current.current && !locked.current) {
      const next = {...current.current,world:setEcologyEvolution(current.current.world,enabled)};
      current.current=next;setSession(next);
    }
  };
  const onPolicy = (policy: "mixed" | BeeMode) => {
    if (!current.current || locked.current) return;
    const next = { ...current.current, world: setConceptionPolicy(current.current.world, policy) };
    current.current = next; setSession(next);
  };

  if (!session || !archive) return <><GameMenu archive={null} symbol={symbol}
    onSymbol={onSymbol} startDate="" onStartDate={setStartDate}
    controls={ecologyControls} onControls={changeRules} evolutionEnabled={evolutionEnabled}
    onEvolution={changeEvolution} funding={funding} onFunding={changeFunding} hasProgress={false} canClose={false} busy={true}
    error={error} onNew={()=>{}} onContinue={()=>{}}/></>;
  return <section aria-label={`${symbol} Garten`}><h2 className="cg-garden-title">{symbol} · eigener Garten</h2><p className="cg-garden-disclosure">Eigene Population und eigenes Kapital · {funding.mode==="topup"?`Nach dem Handelstag auf ${funding.topUpTargetEur} € auffüllen` : funding.mode==="monthly"?`${funding.monthlyEur} € monatlicher Zuschuss`:"Keine automatischen Einzahlungen"}. {symbol==="BTC-USDT"?"SOL-Strategievorlagen auf Bitcoin: unveränderte Parameter als expliziter Transferversuch, keine bestätigte Eignung.":"SOL-Strategievorlagen auf dem SOL-Archiv."}</p><GameSurface session={session} archive={archive} selectedId={selectedId} onSelect={setSelectedId}
    playing={playing} busy={busy} error={error} saveStatus={saveStatus} startDate={startDate}
    onDate={setStartDate} onRestart={restart} onAdvance={(days)=>{void advance(days);}}
    onPlay={()=>{stop.current=playing;setPlaying(!playing);}}
    onPause={()=>{stop.current=true;setPlaying(false);}} onPolicy={onPolicy}
    speed={speed} onSpeed={setSpeed} profileIds={profileIds} onProfiles={setProfileIds}
    onOpenMenu={()=>{stop.current=true;setPlaying(false);setMenuOpen(true);}}/>
    {menuOpen&&<GameMenu archive={archive} symbol={symbol} onSymbol={onSymbol}
      startDate={startDate} onStartDate={(date)=>{setError(null);setStartDate(date);}}
      controls={ecologyControls} onControls={changeRules}
      evolutionEnabled={evolutionEnabled} onEvolution={changeEvolution} funding={funding} onFunding={changeFunding}
      hasProgress={hasSnapshot || started} canClose={hasSnapshot || started}
      busy={busy} error={error} onNew={restart} onContinue={continueGame}/>}
    </section>;
}
