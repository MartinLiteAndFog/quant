import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { setConceptionPolicy, type BeeMode, type GardenPlant, type SimulationWorld } from "./simulation";
import { advanceHistoricalSession, createHistoricalSession, HISTORICAL_SAVE_KEY, loadSolArchive, validateHistoricalSession,
  type HistoricalSession, type SolArchive } from "./historicalGarden";
import "./historical.css";
import { GameSurface } from "./GameSurface";
import { loadHistoricalSnapshot, saveHistoricalSnapshot } from "./historicalStorage";

export type HistoricalGardenControls = {
  world: SimulationWorld; selected: GardenPlant; selectedId: string; playing: boolean; busy: boolean;
  onSelect: (id: string) => void; onAdvance: (days: number) => void; onTogglePlay: () => void;
  onReset: () => void; onPolicy: (policy: "mixed" | BeeMode) => void;
};


/** Historical trading is the default playable path; legacy ecology fixtures remain separate. */
export function HistoricalTrading(_props: { renderGarden?: (controls: HistoricalGardenControls) => ReactNode }) {
  const [playing,setPlaying]=useState(false);
  return <div className="cg-two-gardens"><HistoricalGarden symbol="SOL-USDT" sharedPlaying={playing} setSharedPlaying={setPlaying}/></div>;
}
function HistoricalGarden({symbol,sharedPlaying,setSharedPlaying}:{symbol:"SOL-USDT"|"BTC-USDT";sharedPlaying:boolean;setSharedPlaying:(value:boolean)=>void}) {
  const snapshotKey=symbol==="SOL-USDT"?HISTORICAL_SAVE_KEY:`${HISTORICAL_SAVE_KEY}.btc`;
  const [archive, setArchive] = useState<SolArchive | null>(null);
  const [session, setSession] = useState<HistoricalSession | null>(null);
  const [startDate, setStartDate] = useState("");
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
      const initialDate = a.firstDate;
      let restored = createHistoricalSession(a, initialDate);
      try {
        const raw = await loadHistoricalSnapshot(snapshotKey);
        if (cancelled) return;
        if (raw) restored = validateHistoricalSession(raw, a);
        setSaveStatus(raw ? "Gespeicherte Welt wiederhergestellt" : "Welt bereit");
      } catch {
        saveEnabled.current = false;
        setSaveStatus("Älterer Spielstand bleibt geschützt. Eine neue Welt ersetzt ihn.");
      }
      current.current = restored; setSession(restored); setArchive(a); setStartDate(restored.startDate);
      if (restored.initialProfileIds) setProfileIds(restored.initialProfileIds);
    }).catch((e: unknown) => { if (!cancelled) setError(e instanceof Error ? e.message : "SOL history could not be loaded."); });
    return () => { cancelled = true; stop.current = true; generation.current += 1; };
  }, []);

  useEffect(() => {
    if (!session || !saveEnabled.current) return;
    let cancelled=false;
    void saveHistoricalSnapshot(session,snapshotKey).then(()=>{if(!cancelled)setSaveStatus("Welt auf diesem Gerät gespeichert");}).catch(()=>{if(!cancelled)setSaveStatus("Speicher nicht verfügbar; halte dieses Fenster für den Fortschritt offen");});
    return ()=>{cancelled=true;};
  }, [session]);

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
    try {
      const next = createHistoricalSession(archive, startDate, { profileIds });
      saveEnabled.current = true; stop.current = true; setPlaying(false); setSelectedId("parent-01");
      current.current = next; setSession(next); setError(null);
    } catch (e) { setError(e instanceof Error ? e.message : "Choose an available date."); }
  };
  const onPolicy = (policy: "mixed" | BeeMode) => {
    if (!current.current || locked.current) return;
    const next = { ...current.current, world: setConceptionPolicy(current.current.world, policy) };
    current.current = next; setSession(next);
  };

  if (!session || !archive) return <div className="page-content"><section className="historical-panel"><h1>{symbol} · Archiv laden</h1>
    <p>{error ?? saveStatus}</p></section></div>;
  return <section aria-label={`${symbol} Garten`}><h2 className="cg-garden-title">{symbol} · eigener Garten</h2><p className="cg-garden-disclosure">Eigene Population und eigenes Kapital · 100 € Zufluss pro Monat und Garten. {symbol==="BTC-USDT"?"SOL-Strategievorlagen auf Bitcoin: unveränderte Parameter als expliziter Transferversuch, keine bestätigte Eignung.":"SOL-Strategievorlagen auf dem SOL-Archiv."}</p><GameSurface session={session} archive={archive} selectedId={selectedId} onSelect={setSelectedId}
    playing={playing} busy={busy} error={error} saveStatus={saveStatus} startDate={startDate}
    onDate={setStartDate} onRestart={restart} onAdvance={(days)=>{void advance(days);}}
    onPlay={()=>{stop.current=playing;setPlaying(!playing);}}
    onPause={()=>{stop.current=true;setPlaying(false);}} onPolicy={onPolicy}
    speed={speed} onSpeed={setSpeed} profileIds={profileIds} onProfiles={setProfileIds}/></section>;
}
