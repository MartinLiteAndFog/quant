import { useEffect, useMemo, useRef, useState } from "react";
import {
  advanceSimulationDay,
  advanceSimulationDays,
  bookClosedPaperReplay,
  createSimulationWorld,
  setConceptionPolicy,
  visiblePopulation,
  type BeeMode,
  type GardenPlant,
  type SimulationWorld,
} from "./simulation";
import { runControlledExperiment, type ExperimentReport } from "./experiments";
import { type PaperReplayReport } from "./paperReplay";
import { type StrategySignalReport, type TradingSignalGenome } from "./strategySignals";
import { type SyntheticReplayEvidence } from "./replayEvidence";
import { type SyntheticReplayCycleResult } from "./replayCycle";
import { readSyntheticReplayWorkerResult } from "./replayWorkerProtocol";
import { CLOUDGARDEN_SAVE_SCHEMA_VERSION, loadCloudgardenSave, saveCloudgardenSave, type SavedExperimentInput, type SavedPaperReplayConfig } from "./saveGame";
import { HistoricalTrading } from "./HistoricalTrading";

type View = "meadow" | "understory";
type GeneAxis = "root" | "leaf" | "water" | "sun";
const axisLabels: Record<GeneAxis, string> = { root: "Root depth", leaf: "Leaf area", water: "Water tolerance", sun: "Sun tolerance" };
const money = (amount: number) => `${amount.toFixed(2)} USDT`;
const percent = (value: number) => `${Math.round(value * 100)}%`;
type StrategyControls = Pick<TradingSignalGenome, "lookbackBars" | "entryThresholdBps" | "exitThresholdBps">;
type PaperReplayRun = Readonly<{
  execution: PaperReplayReport;
  strategy: StrategySignalReport;
  genome: TradingSignalGenome;
  evidence: SyntheticReplayEvidence;
}>;

function App() {
  const [showHistorical, setShowHistorical] = useState(true);
  const [loadedSave] = useState(() => readBrowserSave());
  const [saveStatus, setSaveStatus] = useState(() => loadedSave.status === "restored"
    ? "Garden restored from this device"
    : loadedSave.status === "invalid" ? "Saved data is protected; reset the world to replace it"
      : loadedSave.status === "unavailable" ? "Local saving is unavailable in this browser" : "Saving on this device");
  const [saveEnabled, setSaveEnabled] = useState(loadedSave.status !== "invalid");
  const [world, setWorld] = useState<SimulationWorld>(() => loadedSave.save?.world ?? createSimulationWorld());
  const [selectedId, setSelectedId] = useState(() => loadedSave.save?.selectedId ?? "parent-01");
  const [view, setView] = useState<View>(() => loadedSave.save?.view ?? "meadow");
  const [playing, setPlaying] = useState(false);
  const [axisX, setAxisX] = useState<GeneAxis>(() => loadedSave.save?.axisX ?? "root");
  const [axisY, setAxisY] = useState<GeneAxis>(() => loadedSave.save?.axisY ?? "leaf");
  const [experimentInput, setExperimentInput] = useState<SavedExperimentInput | null>(() => loadedSave.save?.experimentInput ?? null);
  const [experiment, setExperiment] = useState<ExperimentReport | null>(() => experimentInputFrom(loadedSave.save?.experimentInput ?? null));
  const [experimentBusy, setExperimentBusy] = useState(false);
  const [paperReplay, setPaperReplay] = useState<PaperReplayRun | null>(() => loadedSave.save?.paperReplay ?? null);
  const [paperReplayConfig, setPaperReplayConfig] = useState<SavedPaperReplayConfig | null>(() => loadedSave.save?.paperReplayConfig ?? null);
  const [paperReplayBusy, setPaperReplayBusy] = useState(false);
  const [paperReplayError, setPaperReplayError] = useState<string | null>(null);
  const paperReplayLock = useRef(false);

  useEffect(() => {
    try {
      if (typeof window === "undefined" || !saveEnabled) return;
      saveCloudgardenSave(window.localStorage, {
        schemaVersion: CLOUDGARDEN_SAVE_SCHEMA_VERSION,
        world,
        selectedId,
        view,
        axisX,
        axisY,
        experimentInput,
        paperReplay,
        paperReplayConfig,
      });
      setSaveStatus("Saved on this device");
    } catch {
      setSaveStatus("Could not save locally; progress remains in this tab");
    }
  }, [world, selectedId, view, axisX, axisY, experimentInput, paperReplay, paperReplayConfig, saveEnabled]);

  const population = visiblePopulation(world);
  const selected = world.plants.find((plant) => plant.id === selectedId) ?? population[0] ?? world.plants[0];

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      if (!paperReplayLock.current) setWorld((current) => advanceSimulationDay(current));
    }, 700);
    return () => window.clearInterval(timer);
  }, [playing]);

  const advance = (days: number) => {
    if (!paperReplayLock.current) setWorld((current) => advanceSimulationDays(current, days));
  };
  const reset = () => {
    if (paperReplayLock.current) return;
    setSaveEnabled(true);
    setPlaying(false);
    setWorld(createSimulationWorld());
    setSelectedId("parent-01");
    setExperiment(null);
    setExperimentInput(null);
    setPaperReplay(null);
    setPaperReplayConfig(null);
    setPaperReplayError(null);
  };
  const runExperiment = async () => {
    setExperimentBusy(true);
    try {
      const input = { seeds: ["CLOUD-071", "CLOUD-113", "CLOUD-208", "CLOUD-319"], days: 140 } as const;
      const report = await runControlledExperiment(input);
      setExperiment(report);
      setExperimentInput(input);
    } finally {
      setExperimentBusy(false);
    }
  };
  const runPaperReplay = async (plantId: string, days: 7 | 30 | 90) => {
    if (paperReplayLock.current) return;
    const sourceWorld = world;
    const plant = sourceWorld.plants.find((candidate) => candidate.id === plantId && candidate.status === "active");
    if (!plant) return;
    const controls: StrategyControls = plant.tradingTraits;
    paperReplayLock.current = true;
    setPlaying(false);
    setPaperReplayBusy(true);
    setPaperReplayError(null);
    try {
      const fixtureStart = new Date(Date.parse("2024-01-01T00:00:00Z") + sourceWorld.day * 86_400_000).toISOString();
      const worker = new Worker(new URL("./replayWorker.ts", import.meta.url), { type: "module" });
      const runId = `${sourceWorld.seed}:${sourceWorld.day}:${plant.id}`;
      const { execution: report, strategy, genome, evidence } = await new Promise<SyntheticReplayCycleResult>((resolve, reject) => {
        let settled = false;
        let timeout = 0;
        const finish = () => {
          window.clearTimeout(timeout);
          worker.terminate();
        };
        const fail = (error: Error) => {
          if (settled) return;
          settled = true;
          finish();
          reject(error);
        };
        timeout = window.setTimeout(() => fail(new Error("The synthetic replay worker timed out.")), 30_000);
        worker.onmessage = (event: MessageEvent<unknown>) => {
          try {
            const result = readSyntheticReplayWorkerResult(event.data, runId, days, fixtureStart);
            if (settled) return;
            settled = true;
            finish();
            resolve(result);
          } catch (error) {
            fail(error instanceof Error ? error : new Error("The replay worker returned an invalid response."));
          }
        };
        worker.onerror = (event) => {
          fail(new Error(event.message || "The replay worker could not complete the synthetic cycle."));
        };
        worker.onmessageerror = () => fail(new Error("The replay worker response could not be decoded."));
        try {
          worker.postMessage({
            id: runId,
            input: {
              days,
              start: fixtureStart,
              gardenSeed: sourceWorld.seed,
              weekIndex: Math.floor(sourceWorld.day / 7),
              plantId: plant.id,
              allocationUsdt: plant.allocationUsdt,
              tradingTraits: controls,
            },
          });
        } catch (error) {
          fail(error instanceof Error ? error : new Error("The replay request could not be sent to its worker."));
        }
      });
      const advanced = advanceSimulationDays(sourceWorld, report.days);
      const booked = bookClosedPaperReplay(advanced, plant.id, {
        startDay: sourceWorld.day,
        days: report.days,
        datasetFingerprint: report.dataset.fingerprint,
        synthetic: report.dataset.synthetic,
        endedFlat: report.endedFlat,
        riskBreach: report.riskBreach,
        fillCount: report.fills.length,
        signalStatus: strategy.status,
        strategyId: genome.strategyId,
        signalGenome: {
          schemaVersion: genome.schemaVersion,
          lookbackBars: genome.lookbackBars,
          entryThresholdBps: genome.entryThresholdBps,
          exitThresholdBps: genome.exitThresholdBps,
          targetQuantity: genome.targetQuantity,
        },
        observations: strategy.observations,
        opportunities: strategy.opportunities,
        generatedSignals: strategy.generatedSignals,
        allocationLimitUsdt: Number(report.allocationLimit),
        startingCashUsdt: Number(report.startingCash),
        endingCashUsdt: Number(report.endingCash),
        maxMarginUsedUsdt: Number(report.maxMarginUsed),
        realizedPricePnlUsdt: Number(report.realizedPricePnl),
        feesUsdt: Number(report.fees),
        fundingUsdt: Number(report.funding),
        netPnlUsdt: Number(report.netPnl),
      });
      setWorld(booked);
      setPaperReplay({ execution: report, strategy, genome, evidence });
      setPaperReplayConfig({ plantId: plant.id, days, tradingTraits: controls });
      setSelectedId(plant.id);
    } catch (error) {
      setPaperReplayError(error instanceof Error ? error.message : "The synthetic replay could not be reconciled.");
    } finally {
      paperReplayLock.current = false;
      setPaperReplayBusy(false);
    }
  };

  return (
    <main className="app-shell cloudgarden-app">
      <aside className="rail" aria-label="Main navigation">
        <a className="brand-mark" href="#home" aria-label="Cloudgarden home"><span>C</span><i /></a>
        <div className="rail-divider" />
        <button className={`rail-button ${showHistorical ? "is-active" : ""}`} onClick={() => { setPlaying(false); setShowHistorical(true); }} aria-label="SOL historical trading" aria-pressed={showHistorical}><span className="rail-glyph">↗</span></button>
        <button className={`rail-button ${!showHistorical && view === "meadow" ? "is-active" : ""}`} onClick={() => { setShowHistorical(false); setView("meadow"); }} aria-label="Demo meadow" aria-pressed={!showHistorical && view === "meadow"}><span className="rail-glyph">✳</span></button>
        <button className={`rail-button ${!showHistorical && view === "understory" ? "is-active" : ""}`} onClick={() => { setShowHistorical(false); setView("understory"); }} aria-label="Demo understory" aria-pressed={!showHistorical && view === "understory"}><span className="rail-glyph glyph-roots">⌁</span></button>
        <div className="rail-spacer" />
        <div className="seed-mark" title={`World seed ${world.seed}`}>CG</div>
      </aside>

      <section className="app-main" id="home">
        <header className="topbar">
          <div className="wordmark"><span className="wordmark-cloud">Cloud</span><span>garden</span></div>
          <nav className="view-tabs" aria-label="World view">
            <button className={showHistorical ? "tab-active" : ""} onClick={() => { setPlaying(false); setShowHistorical(true); }}>SOL Trading</button>
            <button className={!showHistorical && view === "meadow" ? "tab-active" : ""} onClick={() => { setShowHistorical(false); setView("meadow"); }}>Meadow demo</button>
            <button className={!showHistorical && view === "understory" ? "tab-active" : ""} onClick={() => { setShowHistorical(false); setView("understory"); }}>Understory demo</button>
          </nav>
          <div className="world-status"><span className="status-leaf" /> {showHistorical ? "Historical SOL · paper execution" : "Synthetic demo"} <span className="status-separator">·</span> no exchange connection {!showHistorical && <><span className="status-separator">·</span><span aria-live="polite">{saveStatus}</span></>}</div>
        </header>

        {showHistorical ? <HistoricalTrading renderGarden={(game) => <Meadow
          historical world={game.world} population={visiblePopulation(game.world)} selected={game.selected} selectedId={game.selectedId}
          playing={game.playing} paperReplayBusy={game.busy} axisX={axisX} axisY={axisY} setAxisX={setAxisX} setAxisY={setAxisY}
          onSelect={game.onSelect} onAdvance={game.onAdvance} onTogglePlay={game.onTogglePlay} onReset={game.onReset} onPolicy={game.onPolicy}
        />} /> : view === "meadow" ? (
          <Meadow
            world={world} population={population} selected={selected} selectedId={selectedId} playing={playing}
            paperReplayBusy={paperReplayBusy}
            axisX={axisX} axisY={axisY} setAxisX={setAxisX} setAxisY={setAxisY}
            onSelect={setSelectedId} onAdvance={advance} onTogglePlay={() => { if (!paperReplayLock.current) setPlaying((current) => !current); }} onReset={reset}
            onPolicy={(policy) => { if (!paperReplayLock.current) setWorld((current) => setConceptionPolicy(current, policy)); }}
          />
        ) : (
          <Understory
            world={world} experiment={experiment} experimentBusy={experimentBusy} onRunExperiment={runExperiment}
            paperReplay={paperReplay} paperReplayConfig={paperReplayConfig} paperReplayBusy={paperReplayBusy} paperReplayError={paperReplayError} onRunPaperReplay={runPaperReplay}
            onReturn={() => setView("meadow")} onSelectPlant={(id) => { setSelectedId(id); setView("meadow"); }}
          />
        )}
      </section>
    </main>
  );
}

function Meadow({
  world, population, selected, selectedId, playing, paperReplayBusy, axisX, axisY, setAxisX, setAxisY,
  onSelect, onAdvance, onTogglePlay, onReset, onPolicy, historical = false,
}: {
  world: SimulationWorld; population: readonly GardenPlant[]; selected: GardenPlant; selectedId: string; playing: boolean; paperReplayBusy: boolean;
  axisX: GeneAxis; axisY: GeneAxis; setAxisX: (axis: GeneAxis) => void; setAxisY: (axis: GeneAxis) => void;
  onSelect: (id: string) => void; onAdvance: (days: number) => void; onTogglePlay: () => void; onReset: () => void;
  onPolicy: (policy: "mixed" | "random" | BeeMode) => void;
  historical?: boolean;
}) {
  const archivedCount = world.plants.length - population.length;
  const usedSeeds = world.capital.seedlingEntitlements.filter((seed) => seed.status === "used").length;
  const visiblePlants = useMemo(() => [...population].sort((a, b) => a.x - b.x), [population]);
  const layout = useMemo(() => visiblePlants.map((plant, index) => ({
    plant,
    // Evenly spaced visual lanes keep a late-game population legible. The gene field
    // below retains the true genome coordinates for analysis.
    x: visiblePlants.length < 2 ? 50 : 5 + index * 90 / (visiblePlants.length - 1),
  })), [visiblePlants]);

  return (
    <div className="page-content game-page">
      <div className="game-heading">
        <div>
          <div className="eyebrow"><span className="eyebrow-line" /> A living strategy ecosystem <span className="eyebrow-coordinate">{historical ? "HISTORICAL PAPER / SIMULATED ECOLOGY" : "SYNTHETIC / REPLAYABLE"}</span></div>
          <h1>A world shaped <em>by its genes.</em></h1>
          <p className="lede">Every plant, seed and unit of capital follows the same deterministic rules.</p>
        </div>
        <div className="weather-pill"><span className="sun-symbol">☼</span><span><b>Day {world.day}</b><small>{percent(world.climate.sunlight)} sunlight · {percent(world.climate.water)} water</small></span></div>
      </div>

      <section className="ledger-strip" aria-label={historical ? "Historical paper capital ledger" : "Synthetic capital ledger"}>
        <LedgerMetric label={historical ? "Paper NAV" : "Test NAV"} value={money(world.capital.navUsdt)} detail={historical ? "historical PnL · single 90/10 ledger" : "single synthetic 90/10 ledger"} />
        <LedgerMetric label="Budget ceiling" value={money(world.capital.allocationCeilingUsdt)} detail="90% maximum" />
        <LedgerMetric label="Protected reserve" value={money(world.capital.protectedReserveUsdt)} detail="never allocated" />
        <LedgerMetric label="Active / archived" value={`${population.length} / ${archivedCount}`} detail="archived plants hidden" />
      </section>

      <div className="game-layout">
        <section className="garden-column" aria-label="Cloudgarden simulation">
          <div className="garden-card">
            <div className="garden-toolbar">
              <div><span className="mini-world-icon">◌</span><span><b>Upper meadow</b><small>World seed {world.seed} · {world.plants.filter((plant) => plant.role === "parent").length} parents · {usedSeeds}/20 seedlings</small></span></div>
              <div className="toolbar-actions"><span className="fixture-tag">{historical ? "SOL ARCHIVE · PAPER" : "SYNTHETIC FIXTURE"}</span><button className="quiet-icon-button" onClick={onReset} disabled={paperReplayBusy} aria-label="Reset Cloudgarden">↺</button></div>
            </div>

            <div className={`garden-scene ${population.length > 10 ? "dense-population" : ""}`} role="group" aria-label="Select an active plant">
              <div className="scene-sky" /><div className="scene-sun" /><div className="scene-cloud cloud-left" /><div className="scene-cloud cloud-right" />
              <div className="scene-hill hill-left" /><div className="scene-hill hill-right" />
              <div className="air-note"><span /> daylight rhythm · deterministic</div>
              <div className="hive-note">🐝 {world.events.filter((event) => event.type === "conception").length} conceptions</div>
              {layout.map(({ plant, x }, index) => <PlantFigure key={plant.id} plant={plant} x={x} index={index} selected={plant.id === selectedId} onSelect={() => onSelect(plant.id)} />)}
              <div className="soil-surface"><i /></div><div className="soil-band soil-band-top" /><div className="soil-band soil-band-bottom" />
              <div className="waterline" style={{ top: `${57 - world.climate.water * 18}%` }}><span>shared synthetic water table</span></div>
              {layout.map(({ plant, x }, index) => <RootSystem key={`${plant.id}-root`} plant={plant} x={x} index={index} selected={plant.id === selectedId} />)}
              <div className="scene-caption"><span>Above ground · funded population</span><span>Below ground · roots and water access</span></div>
            </div>

            <div className="garden-controls">
              <div className="control-meta"><span>DAY <b>{String(world.day).padStart(3, "0")}</b></span><span className="control-dot" /><span>SEED <b>{world.seed}</b></span><span className="control-dot" /><span>CAPITAL <b>{money(world.capital.deployedCapitalUsdt + world.capital.reservedMarginUsdt)}</b></span></div>
              <div className="time-controls">
                <button className={`play-button ${playing ? "is-playing" : ""}`} onClick={onTogglePlay} disabled={paperReplayBusy}>{playing ? "Ⅱ Pause" : "▶ Play"}</button>
                <button className="garden-step" onClick={() => onAdvance(1)} disabled={paperReplayBusy}>+ 1 day</button>
                <button className="garden-step" onClick={() => onAdvance(7)} disabled={paperReplayBusy}>+ 7 days</button>
                <button className="garden-step garden-season" onClick={() => onAdvance(140)} disabled={paperReplayBusy}>+ 140 days</button>
              </div>
            </div>
          </div>

          <div className="ledger-details">
            <div><span>Deployed</span><b>{money(world.capital.deployedCapitalUsdt)}</b></div>
            <div><span>Reserved for seed bank</span><b>{money(world.capital.reservedMarginUsdt)}</b></div>
            <div><span>Free allocation</span><b>{money(world.capital.freeCapitalUsdt)}</b></div>
            <div><span>Available seed entitlements</span><b>{20 - usedSeeds} / 20</b></div>
          </div>

          <GeneField population={population} axisX={axisX} axisY={axisY} setAxisX={setAxisX} setAxisY={setAxisY} selectedId={selectedId} onSelect={onSelect} />
        </section>

        <aside className="plant-inspector" aria-label="Selected plant and ecosystem settings">
          <div className="inspector-topline"><span>SPECIMEN / {selected.role.toUpperCase()}</span><span>{selected.status === "archived" ? "ARCHIVED" : selected.physiology.toUpperCase()}</span></div>
          {selected.status === "archived" && <div className="archived-inspector">Archived genome record · no longer visible in the meadow or deployed.</div>}
          <h2>{selected.name}</h2>
          <p className="plant-class">{selected.phenotype} phenotype <span>·</span> {selected.conceivedBy ? `${selected.conceivedBy} bee` : "parent line"}</p>
          <div className={`plant-portrait portrait-${selected.phenotype.toLowerCase()}`}><PlantArt plant={selected} /></div>
          <div className="specimen-stats">
            <Stat label="Daily habitat fit · synthetic" value={selected.fitness} />
            <Stat label={historical ? "Trading-grown vigor" : "Vigor"} value={selected.vigor} />
            <Stat label="Synthetic evidence" value={selected.evidence} />
          </div>
          <p className="metrics-legend">{historical
            ? "Body mass grows with this plant’s net paper return. After 14 idle trading days, an inherited patience gene causes a tiny extra daily vigor loss (no fabricated PnL). Habitat fit/evidence and the legacy predators remain synthetic game traits."
            : "Habitat fit reflects today’s synthetic climate match. Vigor is the plant’s accumulated state; evidence counts only simulated observation."}</p>
          {historical && <>
            <div className="plant-allocation"><span>Trading-grown body mass</span><b>{selected.biomass.toFixed(2)} / 100</b></div>
            <div className="plant-allocation"><span>Consecutive inactive days</span><b>{selected.inactiveDays ?? 0} · decay after day 14</b></div>
            <div className="plant-allocation"><span>Inherited patience</span><b>{((selected.genes.patience ?? 0.5) * 100).toFixed(1)}%</b></div>
            <div className="plant-allocation"><span>Parents</span><b>{selected.parents?.join(" × ") ?? "Founding plant"}</b></div>
          </>}
          <div className="plant-allocation"><span>Capital entitlement</span><b>{money(selected.allocationUsdt)}</b></div>
          <div className="plant-allocation"><span>Observed in this world</span><b>{selected.evidenceDays} days</b></div>
          <div className="genome-title"><span>Genome · heritable traits</span><small>protected policies fixed</small></div>
          <GeneBar name="Root depth" value={selected.genes.root} color="blue" />
          <GeneBar name="Leaf area" value={selected.genes.leaf} color="green" />
          <GeneBar name="Water response" value={selected.genes.water} color="teal" />
          <GeneBar name="Sun response" value={selected.genes.sun} color="gold" />
          <div className="tradeoff-note">Root and leaf growth compete for daily energy. Neither gene can change the 90/10 capital boundary.</div>
          <label className="bee-policy"><span>Conception policy</span>
            <select value={world.conceptionPolicy} onChange={(event) => onPolicy(event.target.value as "mixed" | BeeMode)} disabled={paperReplayBusy}>
              <option value="mixed">Mixed modes · local / hive / hybrid</option><option value="random">Random-only baseline</option>
              <option value="local">Local pairing</option><option value="hive">Hive recombination</option><option value="hybrid">Hybrid pollination</option>
            </select>
          </label>
        </aside>
      </div>

      <footer className="game-footer"><span>{historical ? "Historical SOL prices drive local paper trades; ecology remains simulated. See the execution ledger below." : "Cloudgarden is a closed, synthetic simulation. No historical market feed, account, strategy executor or orders are connected."}</span><span>Same seed + same actions = same garden.</span></footer>
    </div>
  );
}

function LedgerMetric({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <div className="ledger-metric"><span>{label}</span><b>{value}</b><small>{detail}</small></div>;
}

function PlantFigure({ plant, x, index, selected, onSelect }: { plant: GardenPlant; x: number; index: number; selected: boolean; onSelect: () => void }) {
  return (
    <button className={`garden-plant plant-${plant.phenotype.toLowerCase()} ${plant.role === "seedling" ? "plant-seedling" : ""} ${plant.physiology === "dormant" ? "plant-dormant" : ""} ${selected ? "plant-selected" : ""}`}
      style={{ left: `${x}%`, zIndex: 8 + index }} onClick={onSelect} aria-label={`Select ${plant.name}, ${plant.phenotype}, ${Math.round(plant.vigor)} vigor`} aria-pressed={selected}>
      <span className="plant-glow" /><PlantArt plant={plant} /><span className="plant-label">{plant.role === "parent" ? plant.name : `Seed ${plant.id.slice(-2)}`}</span>
    </button>
  );
}

function PlantArt({ plant }: { plant: GardenPlant }) {
  const height = 30 + plant.biomass * 0.55;
  const spread = 20 + plant.genes.leaf * 32;
  return <span className={`plant-art art-${plant.phenotype.toLowerCase()}`} style={{ height: `${height}px`, width: `${spread}px` }} aria-hidden="true">
    <i className="plant-stem" /><i className="plant-leaf leaf-a" /><i className="plant-leaf leaf-b" /><i className="plant-leaf leaf-c" /><i className="plant-bloom" />
  </span>;
}

function RootSystem({ plant, x, index, selected }: { plant: GardenPlant; x: number; index: number; selected: boolean }) {
  return <div className={`root-network ${selected ? "root-selected" : ""}`} style={{ left: `${x}%`, height: `${18 + plant.genes.root * 75}px`, width: `${25 + plant.genes.root * 38}px`, zIndex: index + 1 }} aria-hidden="true"><i /><i /><i /><i /></div>;
}

function Stat({ label, value }: { label: string; value: number }) {
  return <div className="plant-stat"><span>{label}</span><b>{Math.round(value)}<small>/100</small></b><div><i style={{ width: `${value}%` }} /></div></div>;
}

function GeneBar({ name, value, color }: { name: string; value: number; color: string }) {
  return <div className="genome-row"><span>{name}</span><div><i className={`gene-${color}`} style={{ width: `${value * 100}%` }} /></div><b>{value.toFixed(2)}</b></div>;
}

function GeneField({ population, axisX, axisY, setAxisX, setAxisY, selectedId, onSelect }: {
  population: readonly GardenPlant[]; axisX: GeneAxis; axisY: GeneAxis; setAxisX: (axis: GeneAxis) => void; setAxisY: (axis: GeneAxis) => void;
  selectedId: string; onSelect: (id: string) => void;
}) {
  return <section className="gene-field" aria-label="Genome parameter field">
    <div className="gene-field-heading"><div><span>GENOME SPACE</span><h2>Change the axes, keep every plant.</h2></div>
      <div className="axis-selectors"><label>X <select aria-label={`Horizontal axis: ${axisLabels[axisX]}`} value={axisX} onChange={(event) => setAxisX(event.target.value as GeneAxis)}>{Object.entries(axisLabels).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label>
      <label>Y <select aria-label={`Vertical axis: ${axisLabels[axisY]}`} value={axisY} onChange={(event) => setAxisY(event.target.value as GeneAxis)}>{Object.entries(axisLabels).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label></div>
    </div>
    <div className="gene-plot"><span className="plot-y-label">{axisLabels[axisY]}</span><div className="plot-area">
      {population.map((plant) => <button key={plant.id} className={`plot-point point-${plant.phenotype.toLowerCase()} ${plant.id === selectedId ? "point-selected" : ""}`}
        style={{ left: `${plant.genes[axisX] * 100}%`, bottom: `${plant.genes[axisY] * 100}%` }} onClick={() => onSelect(plant.id)} aria-label={`Select ${plant.name}; ${axisLabels[axisX]} ${plant.genes[axisX].toFixed(2)}, ${axisLabels[axisY]} ${plant.genes[axisY].toFixed(2)}`} title={`${plant.name} · ${plant.phenotype}`} />)}
    </div><span className="plot-x-label">{axisLabels[axisX]}</span></div>
    <p className="plot-note">All {population.length} living plants remain selectable when axes change. Archived genome records stay in the research archive.</p>
  </section>;
}

function Understory({ world, experiment, experimentBusy, onRunExperiment, paperReplay, paperReplayConfig, paperReplayBusy, paperReplayError, onRunPaperReplay, onReturn, onSelectPlant }: {
  world: SimulationWorld; experiment: ExperimentReport | null; experimentBusy: boolean; onRunExperiment: () => void;
  paperReplay: PaperReplayRun | null; paperReplayConfig: SavedPaperReplayConfig | null; paperReplayBusy: boolean; paperReplayError: string | null;
  onRunPaperReplay: (plantId: string, days: 7 | 30 | 90) => void;
  onReturn: () => void; onSelectPlant: (id: string) => void;
}) {
  const active = visiblePopulation(world);
  const archived = world.plants.filter((plant) => plant.status === "archived");
  const conceptions = world.events.filter((event) => event.type === "conception");
  const archiveEvents = world.events.filter((event) => event.type === "archive");
  const candidates = [...world.plants].reverse();
  const modeCounts = (['random', 'local', 'hive', 'hybrid'] as const).map((mode) => ({ mode, count: conceptions.filter((event) => event.detail.mode === mode).length }));

  return <div className="page-content understory-page game-understory">
    <div className="understory-heading"><div><div className="eyebrow"><span className="eyebrow-line" /> The research layer · no capital promotion</div><h1>What grows <em>below.</em></h1><p className="lede">The archive and seed bank are records, not hidden individual plants in the meadow.</p></div><button className="return-button" onClick={onReturn}>← Return to meadow</button></div>
    <section className="research-status-grid">
      <LedgerMetric label="Active population" value={String(active.length)} detail="shown above ground" />
      <LedgerMetric label="Research archive" value={String(archived.length)} detail="records only · not rendered" />
      <LedgerMetric label="Conceptions" value={String(conceptions.length)} detail="all event-logged" />
      <LedgerMetric label="Seed bank" value={`${world.capital.seedlingEntitlements.filter((seed) => seed.status === "available").length} / 20`} detail="single-use funded slots" />
    </section>

    <div className="research-layout game-research-layout">
      <section className="research-main"><div className="section-title"><div><span>GENOME ARCHIVE</span><h2>Plant and lineage records</h2></div><span className="record-count">{world.plants.length} records</span></div>
        {candidates.length ? <div className="research-plant-list">{candidates.map((plant) => <button className="research-plant-row" key={plant.id} onClick={() => onSelectPlant(plant.id)} title={plant.status === "archived" ? "Open archived genome record" : "Inspect plant"}>
          <span className={`research-seed seed-${plant.phenotype.toLowerCase()}`}><i /></span><span className="research-plant-info"><b>{plant.name}</b><small>{plant.phenotype} · {plant.conceivedBy ?? "parent line"} · {plant.evidenceDays} observation days · fit {Math.round(plant.fitness)}</small></span>
          <span className="research-genome">R {plant.genes.root.toFixed(2)}<br />L {plant.genes.leaf.toFixed(2)}</span><span className={`archive-state ${plant.status === "archived" ? "archived" : plant.physiology}`}>{plant.status === "archived" ? "Archived" : plant.physiology === "dormant" ? "Dormant" : "Growing"}</span>
        </button>)}</div> : <div className="research-empty"><span>⌁</span><b>The first seeds are still forming.</b><p>Step the world forward to day 7. The first conception is funded from an existing seed entitlement.</p></div>}
        <div className="research-footnote"><span>i</span><p>Recent event window: {world.events.length} records (maximum 2,000). Archive events in this window: {archiveEvents.length}. Genome records remain available; archived plants leave the meadow.</p></div>
      </section>

      <aside className="evolution-card"><span className="evolution-kicker">CONTROLLED SYNTHETIC EXPERIMENT</span><h2>Four ways to<br /><em>spread genes.</em></h2>
        <p className="experiment-intro">Each cohort gets the same world seeds, 90 USDT allocation ceiling, and 140 daily steps. This compares game-rule outcomes only.</p>
        <div className="mode-counts">{modeCounts.map(({ mode, count }) => <div key={mode}><span>{mode}</span><b>{count}</b></div>)}</div>
        <button className="experiment-button" onClick={onRunExperiment} disabled={experimentBusy}>{experimentBusy ? "Comparing worlds…" : "Run matched experiment"}<span>↗</span></button>
        <small className="offline-note">Random · local · hive · hybrid / equal seeds</small>
      </aside>
    </div>

    <PaperReplayPanel plants={active} world={world} worldDay={world.day} navUsdt={world.capital.navUsdt} run={paperReplay} runConfig={paperReplayConfig} busy={paperReplayBusy} error={paperReplayError} onRun={onRunPaperReplay} />
    {experiment && <ExperimentResults report={experiment} />}
    <footer className="game-footer"><span>Synthetic outcomes are not trading evidence and cannot qualify a strategy.</span><span>{archived.length} plants archived · {money(world.capital.freeCapitalUsdt)} returned to free allocation</span></footer>
  </div>;
}

function PaperReplayPanel({ plants, world, worldDay, navUsdt, run, runConfig, busy, error, onRun }: {
  plants: readonly GardenPlant[]; world: SimulationWorld; worldDay: number; navUsdt: number; run: PaperReplayRun | null; runConfig: SavedPaperReplayConfig | null; busy: boolean; error: string | null;
  onRun: (plantId: string, days: 7 | 30 | 90) => void;
}) {
  const [plantId, setPlantId] = useState(runConfig?.plantId ?? plants.find((plant) => plant.role === "parent")?.id ?? plants[0]?.id ?? "");
  const [days, setDays] = useState<7 | 30 | 90>(runConfig?.days ?? 7);
  const selected = plants.find((plant) => plant.id === plantId) ?? plants[0];
  const report = run?.execution;
  return <section className="paper-replay-panel">
    <div className="section-title"><div><span>SYNTHETIC INTRADAY PAPER REPLAY</span><h2>Test an inherited signal genome</h2></div><span className="record-count">{(report?.dataset.barCount ?? days * 24 * 60).toLocaleString()} UTC minute bars{report ? " · last run" : " · selected horizon"}</span></div>
    <p className="paper-replay-intro">The selected plant’s inherited trading genes drive this causal replay over completed synthetic closes. The recipe is not alpha or market evidence.</p>
    <div className="paper-replay-controls">
      <label>Funded plant<select value={selected?.id ?? ""} onChange={(event) => setPlantId(event.target.value)} disabled={busy || plants.length === 0}>
        {plants.map((plant) => <option key={plant.id} value={plant.id}>{plant.name} · {money(plant.allocationUsdt)} available</option>)}
      </select></label>
      <label>Replay horizon<select value={days} onChange={(event) => setDays(Number(event.target.value) as 7 | 30 | 90)} disabled={busy}>
        <option value={7}>7 days · one ecology week</option><option value={30}>30 days · one ecology month</option><option value={90}>90 days · three ecology months</option>
      </select></label>
      {selected && <div className="trading-genome-readout" aria-label="Inherited trading genome">
        <span>Lookback <b>{selected.tradingTraits.lookbackBars} bars</b></span>
        <span>Entry <b>{selected.tradingTraits.entryThresholdBps} bps</b></span>
        <span>Exit <b>{selected.tradingTraits.exitThresholdBps} bps</b></span>
      </div>}
      <button className="experiment-button" onClick={() => selected && onRun(selected.id, days)} disabled={busy || !selected}>{busy ? "Generating and replaying…" : `Run ${days}-day genome · garden day ${worldDay}`}<span>↗</span></button>
    </div>
    {error && <p className="paper-replay-error" role="alert">Replay not booked: {error}</p>}
    {report && <>
      {run && <div className="paper-replay-accounting"><span>Strategy <b>Momentum v1 · {run.evidence.strategyStatus}</b></span><span>Executable opportunities / observations <b>{run.evidence.executableOpportunityObservations} / {run.evidence.observations}</b></span><span>Targets / fills / rejected <b>{run.evidence.generatedTargets} / {run.evidence.fills} / {run.evidence.rejectedSignals}</b></span><span>Target size <b>{run.genome.targetQuantity} units · 1× cap</b></span></div>}
      <div className="paper-replay-metrics">
        <LedgerMetric label="Net replay PnL" value={report.fills.length ? `${Number(report.netPnl) >= 0 ? "+" : ""}${money(Number(report.netPnl))}` : "—"} detail={report.fills.length ? "fees and funding included" : "no executed fill"} />
        <LedgerMetric label="Fills / rejected" value={`${report.fills.length} / ${report.rejectedSignals.length}`} detail="strict next-open timing" />
        <LedgerMetric label="Peak margin" value={money(Number(report.maxMarginUsed))} detail={`plant limit ${money(Number(report.allocationLimit))}`} />
        <LedgerMetric label="Garden NAV now" value={money(navUsdt)} detail="same portfolio ledger" />
      </div>
      <div className="paper-replay-accounting">{report.fills.length > 0 && <><span>Price PnL <b>{money(Number(report.realizedPricePnl))}</b></span><span>Fees <b>−{money(Number(report.fees))}</b></span><span>Funding <b>{money(Number(report.funding))}</b></span></>}<span>Last closed replay <b>{report.days} days · {report.dataset.barCount.toLocaleString()} UTC minute bars</b></span>{runConfig && <span>Run settings <b>{world.plants.find((plant) => plant.id === runConfig.plantId)?.name ?? "Archived plant"} · lookback {runConfig.tradingTraits.lookbackBars} · entry {runConfig.tradingTraits.entryThresholdBps} bps · exit {runConfig.tradingTraits.exitThresholdBps} bps</b></span>}<span>Dataset <b>{report.dataset.fingerprint.slice(0, 12)}…</b></span></div>
      {run && <div className="paper-replay-accounting"><span>Evidence <b>{run.evidence.claim}</b></span><span>Inference <b>{run.evidence.statisticalStatus} · independent N unavailable</b></span><span>Max daily drawdown <b>{run.evidence.maxDailyDrawdownUsdt === null ? "unavailable" : `${money(run.evidence.maxDailyDrawdownUsdt)} · ${((run.evidence.maxDailyDrawdownPct ?? 0) * 100).toFixed(2)}%`}</b></span><span>Cost / cash reconciliation <b>{Math.abs(run.evidence.netPnlResidualUsdt) < 0.0001 && Math.abs(run.evidence.cashResidualUsdt) < 0.0001 ? "passed" : "failed"}</b></span></div>}
      {run?.strategy.status === "no_trade" && <p className="paper-replay-error">No eligible pre-close observation crossed the strategy threshold. No trade occurred; no trading loss was invented.</p>}
      <p className="experiment-disclaimer">{report.claim} The signal recipe is versioned as synthetic Momentum v1 and is not alpha. Executable opportunities depend on this genome's synthetic entry threshold; they are not independent market opportunities. The {run?.evidence.dailyMarkCount ?? 0} daily marks provide descriptive drawdown only: no Sharpe, CVaR, confidence, OOS, fitness, or independent-sample claim is available. Benchmark: cash / no trade. Verdict: <b>not_comparable</b>. The garden clock advanced {report.days} days and the closed result was booked into the single 90/10 ledger. Habitat fit and vigor were not changed by this result.</p>
    </>}
  </section>;
}

function ExperimentResults({ report }: { report: ExperimentReport }) {
  return <section className="experiment-results"><div className="section-title"><div><span>MATCHED SEED COHORTS</span><h2>Rule-level outcomes</h2></div><span className="record-count">{report.days} days · {report.seeds.length} seeds</span></div>
    <table className="experiment-table"><thead><tr><th scope="col">Pollination</th><th scope="col">Living</th><th scope="col">Archived</th><th scope="col">Mean habitat fit · synthetic</th><th scope="col">Mean vigor</th><th scope="col">Lookback · bars</th><th scope="col">Entry · bps</th><th scope="col">Exit · bps</th><th scope="col">Capital reconciled</th></tr></thead>
      <tbody>{report.cohorts.map((cohort) => <tr key={cohort.policy}><th scope="row">{cohort.policy}</th><td>{cohort.meanActivePlants.toFixed(1)}</td><td>{cohort.meanArchivedPlants.toFixed(1)}</td><td>{cohort.meanHabitatFit.toFixed(1)}</td><td>{cohort.meanVigor.toFixed(1)}</td><td>{cohort.meanTradingLookbackBars.toFixed(1)}</td><td>{cohort.meanTradingEntryThresholdBps.toFixed(1)}</td><td>{cohort.meanTradingExitThresholdBps.toFixed(1)}</td><td>{cohort.capitalConserved ? "Yes · 90/10" : "Check"}</td></tr>)}</tbody>
    </table><p className="experiment-verdict">{report.interpretation}</p><p className="experiment-disclaimer">Gene means describe the inherited genome only. No market, return, Sharpe, or execution claim; these results compare Cloudgarden's synthetic rules and remain not comparable to market performance.</p>
  </section>;
}

export default App;

function readBrowserSave() {
  try {
    return typeof window === "undefined"
      ? { save: null, status: "unavailable" as const }
      : loadCloudgardenSave(window.localStorage);
  } catch {
    return { save: null, status: "unavailable" as const };
  }
}

function experimentInputFrom(input: SavedExperimentInput | null): ExperimentReport | null {
  if (!input) return null;
  try {
    return runControlledExperiment(input);
  } catch {
    return null;
  }
}
