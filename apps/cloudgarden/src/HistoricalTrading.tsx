import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { setConceptionPolicy, type BeeMode, type GardenPlant, type SimulationWorld } from "./simulation";
import { advanceHistoricalSession, createHistoricalSession, HISTORICAL_SAVE_KEY, loadSolArchive, validateHistoricalSession,
  type HistoricalSession, type SolArchive } from "./historicalGarden";
import "./historical.css";

export type HistoricalGardenControls = {
  world: SimulationWorld; selected: GardenPlant; selectedId: string; playing: boolean; busy: boolean;
  onSelect: (id: string) => void; onAdvance: (days: number) => void; onTogglePlay: () => void;
  onReset: () => void; onPolicy: (policy: "mixed" | BeeMode) => void;
};
const money = (n: number) => `${n.toFixed(4)} USDT`;

/** Historical trading is the default playable path; legacy ecology fixtures remain separate. */
export function HistoricalTrading({ renderGarden }: { renderGarden: (controls: HistoricalGardenControls) => ReactNode }) {
  const [archive, setArchive] = useState<SolArchive | null>(null);
  const [session, setSession] = useState<HistoricalSession | null>(null);
  const [startDate, setStartDate] = useState("2026-01-01");
  const [selectedId, setSelectedId] = useState("parent-01");
  const [playing, setPlaying] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState("Loading local SOL history…");
  const current = useRef<HistoricalSession | null>(null);
  const locked = useRef(false);
  const stop = useRef(false);
  const saveEnabled = useRef(true);
  const generation = useRef(0);

  useEffect(() => {
    generation.current += 1;
    let cancelled = false;
    void loadSolArchive().then((a) => {
      if (cancelled) return;
      const initialDate = a.days.some((d) => d.date === "2026-01-01" && d.barCount === 1440 && !d.missingMinutes)
        ? "2026-01-01" : a.days.find((d) => d.barCount === 1440 && !d.missingMinutes)!.date;
      let restored = createHistoricalSession(a, initialDate);
      try {
        const raw = localStorage.getItem(HISTORICAL_SAVE_KEY);
        const oldWorld = localStorage.getItem("cloudgarden.historical.v2") ?? localStorage.getItem("cloudgarden.historical.v1");
        if (raw) restored = validateHistoricalSession(JSON.parse(raw), a);
        setSaveStatus(raw ? "G2 historical world restored on this device"
          : oldWorld ? "G2 world ready; previous historical save is preserved separately"
            : "G2 historical world ready");
      } catch {
        saveEnabled.current = false;
        setSaveStatus("Previous historical save is protected. Start a new world to replace it.");
      }
      current.current = restored; setSession(restored); setArchive(a); setStartDate(restored.startDate);
    }).catch((e: unknown) => { if (!cancelled) setError(e instanceof Error ? e.message : "SOL history could not be loaded."); });
    return () => { cancelled = true; stop.current = true; generation.current += 1; };
  }, []);

  useEffect(() => {
    if (!session || !saveEnabled.current) return;
    try { localStorage.setItem(HISTORICAL_SAVE_KEY, JSON.stringify(session)); setSaveStatus("Historical world saved on this device"); }
    catch { setSaveStatus("Storage unavailable; keep this tab open to retain progress"); }
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
    const timer = window.setTimeout(() => { void advance(1, true); }, 700);
    return () => window.clearTimeout(timer);
  }, [playing, busy, session, advance]);

  const restart = () => {
    if (!archive || locked.current) return;
    try {
      const next = createHistoricalSession(archive, startDate);
      saveEnabled.current = true; stop.current = true; setPlaying(false); setSelectedId("parent-01");
      current.current = next; setSession(next); setError(null);
    } catch (e) { setError(e instanceof Error ? e.message : "Choose an available date."); }
  };
  const onPolicy = (policy: "mixed" | BeeMode) => {
    if (!current.current || locked.current) return;
    const next = { ...current.current, world: setConceptionPolicy(current.current.world, policy) };
    current.current = next; setSession(next);
  };

  if (!session || !archive) return <div className="page-content"><section className="historical-panel"><h1>Loading your SOL archive</h1>
    <p>{error ?? saveStatus}</p></section></div>;
  const selected = session.world.plants.find((p) => p.id === selectedId) ?? session.world.plants[0]!;
  const totalFills = Object.values(session.records).reduce((sum, r) => sum + r.fills, 0);
  const net = session.world.capital.navUsdt - 100;
  const values = [100, ...session.history.map((h) => h.nav)];
  const min = Math.min(...values), max = Math.max(...values), range = Math.max(.01, max - min);
  const path = values.map((n, i) => `${i * 800 / Math.max(1, values.length - 1)},${90 - (n - min) * 75 / range}`).join(" ");
  return <>
    <div className="page-content historical-top">
      <section className="historical-panel">
        <div className="historical-heading"><div><span className="eyebrow">HISTORICAL SOL · LOCAL PAPER TRADING</span>
          <h1>Your strategies are trading.</h1><p>Play advances every funded plant over actual SOL minute candles. IMBA parents pass confirmed EW modules to their offspring.</p></div>
          <div className="historical-next"><span>Next UTC trading day</span><b>{session.nextDate}</b><small>{busy ? "Executing and reconciling…" : saveStatus}</small></div></div>
        <div className="historical-metrics"><div><span>Paper NAV</span><b>{money(session.world.capital.navUsdt)}</b></div>
          <div><span>Net PnL after costs</span><b className={net < 0 ? "negative" : "positive"}>{net >= 0 ? "+" : ""}{money(net)}</b></div>
          <div><span>Executed fills</span><b>{totalFills.toLocaleString()}</b></div><div><span>Fees paid</span><b>{money(session.world.capital.paperFeesUsdt)}</b></div></div>
        <svg className="historical-equity" viewBox="0 0 800 100" role="img" aria-label={`Paper equity from 100 to ${session.world.capital.navUsdt.toFixed(4)} USDT`} preserveAspectRatio="none">
          <line x1="0" y1="95" x2="800" y2="95" stroke="#d9e2d4" /><polyline points={path} fill="none" stroke="#668463" strokeWidth="2" vectorEffect="non-scaling-stroke" /></svg>
        <div className="historical-controls"><label>Start a new historical world<input aria-label="SOL historical start date" type="date" value={startDate}
          min={archive.firstDate} max={archive.lastDate} onChange={(e) => setStartDate(e.target.value)} disabled={busy} /></label>
          <button onClick={restart} disabled={busy}>Start from this date</button><button onClick={() => { stop.current = true; setPlaying(false); }} disabled={!busy && !playing}>Pause after this day</button>
          <span>{archive.totalBars.toLocaleString()} actual minutes · {archive.firstDate} → {archive.lastDate}</span></div>
        {error && <p className="historical-error" role="alert">{error}</p>}
        <p className="historical-method">Source: local SOL-USDT archive · SHA {archive.sourceSha256.slice(0, 12)}… · {archive.missingMinutes.toLocaleString()} missing archive minutes, never filled.
          Historical prices; modeled executions and assumed fees, spread and slippage. Venue/instrument unverified; funding unavailable and excluded. Positions flatten daily, exposure targets use 50% of each plant’s allocation, capped at 1×. Stop signals use completed closes and may fill beyond their trigger after a gap. Source-derived candle variants; full TradingView/Renko/HTF parity is unproven. Research statistics are separate from the game. No statistical verdict controls Play, plant growth, breeding or predation.</p>
      </section>
    </div>
    {renderGarden({ world: session.world, selected, selectedId: selected.id, playing, busy,
      onSelect: setSelectedId, onAdvance: (days) => { void advance(days); }, onTogglePlay: () => { stop.current = playing; setPlaying(!playing); }, onReset: restart, onPolicy })}
    <div className="page-content historical-bottom"><section className="historical-panel">
      <h2>Inherited strategies & actual paper fills</h2><p className="historical-method">Ecology remains a simulated model. The paper ledger below is calculated from historical SOL prices. No performance promotion is inferred.</p>
      <div className="historical-scroll"><table className="historical-table"><thead><tr><th>Plant / base</th><th>IMBA lookback</th><th>EW genes</th><th>Stop trigger</th><th>Fills</th><th>Net PnL</th><th>Fees</th><th>Rejected</th></tr></thead>
        <tbody>{session.world.plants.map((plant) => { const genome = session.genomes[plant.id]!, r = session.records[plant.id]; return <tr key={plant.id}>
          <th><button className="historical-select" onClick={() => setSelectedId(plant.id)}>{plant.name}</button><small>{genome.family} · {plant.status}</small><small>{genome.sourceId}</small></th>
          <td>{genome.channelLookback} min</td><td>{genome.ewFilter ? "W3 filter" : "no filter"} · {genome.ewExit ? "W5 exit" : "no EW exit"}<small>confirmed pivot {genome.pivotLength} · ATR ×{genome.ewMinMoveAtr.toFixed(2)}</small></td>
          <td>{genome.stopLossBps} bps</td><td>{r?.fills ?? 0}</td><td>{money(r?.netPnl ?? 0)}</td><td>{money(r?.fees ?? 0)}</td><td>{r?.rejected ?? 0}</td></tr>; })}</tbody></table></div>
      <h2>Latest executions</h2>{session.trades.length ? <div className="historical-scroll"><table className="historical-table"><thead><tr><th>UTC execution</th><th>Plant</th><th>Side</th><th>Quantity</th><th>Actual next open</th><th>Paper fill price</th><th>Fee</th></tr></thead>
        <tbody>{[...session.trades].reverse().slice(0, 40).map((fill) => <tr key={`${fill.plantId}:${fill.orderId}:${fill.executedAt}`}>
          <td>{fill.executedAt.replace("T", " ").replace(".000Z", "")}</td><td>{fill.plantName}</td><td>{fill.side}</td><td>{fill.filledQuantity}</td><td>{fill.referencePrice}</td><td>{fill.price}</td><td>{fill.fee}</td></tr>)}</tbody></table></div>
        : <p>Press Play or + 1 day in the meadow to execute the first historical SOL day.</p>}
    </section></div>
  </>;
}
