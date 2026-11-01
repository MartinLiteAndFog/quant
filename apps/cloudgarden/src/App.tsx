import { useEffect, useState } from "react";
import { advanceDay, advanceDays, createWorld, phenotype, type Plant, type World } from "./world";

type View = "meadow" | "understory";

const pad = (value: number) => String(value).padStart(3, "0");
const percent = (value: number) => `${Math.round(value)}%`;

function App() {
  const [world, setWorld] = useState<World>(() => createWorld());
  const [selectedId, setSelectedId] = useState("aster-04");
  const [view, setView] = useState<View>("meadow");
  const [playing, setPlaying] = useState(false);
  const selected = world.plants.find((plant) => plant.id === selectedId) ?? world.plants[0];

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => setWorld((current) => advanceDay(current)), 1800);
    return () => window.clearInterval(timer);
  }, [playing]);

  const advanceWeek = () => setWorld((current) => advanceDays(current, 7));

  return (
    <main className="app-shell">
      <aside className="rail" aria-label="Main navigation">
        <a className="brand-mark" href="#home" aria-label="Cloudgarden home"><span>C</span><i /></a>
        <div className="rail-divider" />
        <button className={`rail-button ${view === "meadow" ? "is-active" : ""}`} onClick={() => setView("meadow")} aria-label="Meadow" aria-pressed={view === "meadow"}>
          <span className="rail-glyph glyph-meadow" aria-hidden="true">✳</span>
        </button>
        <button className={`rail-button ${view === "understory" ? "is-active" : ""}`} onClick={() => setView("understory")} aria-label="Understory" aria-pressed={view === "understory"}>
          <span className="rail-glyph glyph-roots" aria-hidden="true">⌁</span>
        </button>
        <div className="rail-spacer" />
        <div className="seed-mark" title={`World seed ${world.seed}`}>07</div>
      </aside>

      <section className="app-main" id="home">
        <header className="topbar">
          <div className="wordmark"><span className="wordmark-cloud">Cloud</span><span>garden</span></div>
          <nav className="view-tabs" aria-label="World view">
            <button className={view === "meadow" ? "tab-active" : ""} onClick={() => setView("meadow")}>Meadow</button>
            <button className={view === "understory" ? "tab-active" : ""} onClick={() => setView("understory")}>Understory</button>
          </nav>
          <div className="world-status"><span className="status-leaf" /> Synthetic world <span className="status-separator">·</span> no exchange connection</div>
        </header>

        {view === "meadow" ? (
          <Meadow
            world={world}
            selected={selected}
            selectedId={selectedId}
            playing={playing}
            onSelect={setSelectedId}
            onAdvance={() => setWorld((current) => advanceDay(current))}
            onAdvanceWeek={advanceWeek}
            onTogglePlay={() => setPlaying((current) => !current)}
            onReset={() => { setPlaying(false); setWorld(createWorld()); setSelectedId("aster-04"); }}
          />
        ) : (
          <Understory world={world} onReturn={() => setView("meadow")} />
        )}
      </section>
    </main>
  );
}

function Meadow({
  world,
  selected,
  selectedId,
  playing,
  onSelect,
  onAdvance,
  onAdvanceWeek,
  onTogglePlay,
  onReset,
}: {
  world: World;
  selected: Plant;
  selectedId: string;
  playing: boolean;
  onSelect: (id: string) => void;
  onAdvance: () => void;
  onAdvanceWeek: () => void;
  onTogglePlay: () => void;
  onReset: () => void;
}) {
  return (
    <div className="page-content meadow-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow"><span className="eyebrow-line" /> The upper meadow <span className="eyebrow-coordinate">N 04° · W 11°</span></div>
          <h1>A small world, <em>still growing.</em></h1>
          <p className="lede">Five strategy-seeds take root in one shared climate. Watch what the genes make of it.</p>
        </div>
        <div className="weather-pill"><span className="sun-symbol">☼</span><span><b>Open sky</b><small>{percent(world.sunlight * 100)} light · soft wind</small></span></div>
      </div>

      <div className="workspace-grid">
        <section className="world-column" aria-label="Cloudgarden meadow simulation">
          <div className="scene-frame">
            <div className="scene-toolbar">
              <div className="biome-name"><span className="mini-world-icon">◌</span><span><b>Cloudgarden</b><small>Meadow 01 / home world</small></span></div>
              <div className="scene-toolbar-right"><span className="fixture-tag">SIMULATION FIXTURE</span><button className="quiet-icon-button" onClick={onReset} aria-label="Reset world">↺</button></div>
            </div>

            <div className="landscape" role="group" aria-label="Select a plant in the meadow">
              <div className="sky-grain" />
              <div className="sun-disc" />
              <div className="cloud cloud-a"><i /><i /><i /></div>
              <div className="cloud cloud-b"><i /><i /><i /></div>
              <div className="distant-hill hill-a" /><div className="distant-hill hill-b" />
              <div className="flying-bee bee-a" aria-hidden="true"><span /><i /></div>
              <div className="flying-bee bee-b" aria-hidden="true"><span /><i /></div>
              <div className="garden-air-label label-air"><span className="air-dash" /> daylight rhythm</div>
              <div className="garden-air-label label-hive">hive activity <span>3</span></div>

              {world.plants.map((plant, index) => (
                <PlantFigure key={plant.id} plant={plant} index={index} selected={plant.id === selectedId} onSelect={() => onSelect(plant.id)} />
              ))}

              <div className="ground-surface"><div className="grass-edge" /></div>
              <div className="soil-layer soil-top" />
              <div className="soil-layer soil-deep" />
              <div className="soil-specks specks-one" /><div className="soil-specks specks-two" />
              <div className="water-table" style={{ top: `${(1 - world.waterline) * 74 + 25}%` }}><span>shared water table</span></div>
              {world.plants.map((plant, index) => <RootSystem key={`${plant.id}-roots`} plant={plant} index={index} selected={plant.id === selectedId} />)}
              <div className="soil-depth-label"><span>root zone</span><i /></div>
              <div className="landscape-bottom"><span>Above ground · active population</span><span>Below ground · {percent(world.waterline * 100)} waterline</span></div>
            </div>

            <div className="scene-footer">
              <div className="timeline-label"><span className="timeline-seed" /> Day <b>{pad(world.day)}</b><span className="timeline-divider" /> Seed <b>{world.seed}</b></div>
              <div className="time-controls">
                <button className="text-button" onClick={onAdvanceWeek}>+ 7 days</button>
                <button className={`play-button ${playing ? "is-playing" : ""}`} onClick={onTogglePlay} aria-label={playing ? "Pause simulation" : "Play simulation"}>{playing ? "Ⅱ" : "▶"}<span>{playing ? "Pause" : "Let time flow"}</span></button>
                <button className="step-button" onClick={onAdvance} aria-label="Advance one day">+ 1 day <span>↗</span></button>
              </div>
            </div>
          </div>

          <div className="meadow-underbar">
            <span><i className="legend-flower" /> fitness</span>
            <span><i className="legend-leaf" /> vitality</span>
            <span><i className="legend-root" /> root reach</span>
            <span className="underbar-note">A visual field guide to a deterministic toy world.</span>
          </div>
        </section>

        <aside className="inspector" aria-label="Selected plant details">
          <div className="inspector-heading"><span>SPECIMEN / {selected.family.toUpperCase()}</span><span className="specimen-index">{String(world.plants.findIndex((plant) => plant.id === selected.id) + 1).padStart(2, "0")}—05</span></div>
          <h2>{selected.name}</h2>
          <div className="specimen-subtitle">{phenotype(selected.genes)} phenotype <span>·</span> seedling line 0{world.plants.findIndex((plant) => plant.id === selected.id) + 1}</div>
          <div className={`specimen-portrait portrait-${phenotype(selected.genes).toLowerCase()}`}><PlantPortrait plant={selected} /></div>

          <div className="metric-pair">
            <div className="metric-block"><span>Vigor</span><b>{Math.round(selected.vigor)}<small>%</small></b><div className="metric-track"><i style={{ width: `${selected.vigor}%` }} /></div></div>
            <div className="metric-block"><span>Fitness</span><b>{Math.round(selected.fitness)}<small>%</small></b><div className="metric-track track-amber"><i style={{ width: `${selected.fitness}%` }} /></div></div>
          </div>
          <div className="evidence-line"><span className="evidence-mark">i</span><span>{selected.evidenceDays} synthetic observation days <small>Illustrative, not market evidence</small></span></div>

          <div className="trait-heading"><span>Genome / four visible genes</span><button aria-label="Genome help" title="Gene values are bounded simulation inputs">?</button></div>
          <GeneRow label="Growth tempo" value={selected.genes.tempo} variant="sage" />
          <GeneRow label="Canopy" value={selected.genes.canopy} variant="gold" />
          <GeneRow label="Root depth" value={selected.genes.rootDepth} variant="blue" />
          <GeneRow label="Root spread" value={selected.genes.rootSpread} variant="moss" />

          <div className="specimen-bottom"><span>Home tile</span><b>Meadow 01 <span>↗</span></b></div>
        </aside>
      </div>

      <footer className="page-footer"><span>Cloudgarden is a simulation toy. No live account, orders or exchange data.</span><span>Current cycle <b>{world.day} days</b><i /> Same seed, same weather, same outcome.</span></footer>
    </div>
  );
}

function PlantFigure({ plant, index, selected, onSelect }: { plant: Plant; index: number; selected: boolean; onSelect: () => void }) {
  const type = phenotype(plant.genes).toLowerCase();
  return (
    <button className={`plant-figure plant-${type} ${selected ? "plant-selected" : ""}`} style={{ left: `${plant.x}%`, zIndex: 6 + index }} onClick={onSelect} aria-label={`Select ${plant.name}, ${phenotype(plant.genes)} phenotype`} aria-pressed={selected}>
      <span className="plant-halo" />
      <span className="plant-crown"><i className="stem" /><i className="leaf leaf-left" /><i className="leaf leaf-right" /><i className="leaf leaf-high" /><i className="flower-core" /></span>
      <span className="plant-foot" />
      <span className="plant-caption"><b>{plant.name}</b><small>{phenotype(plant.genes)} · {Math.round(plant.vigor)} vigor</small></span>
    </button>
  );
}

function RootSystem({ plant, index, selected }: { plant: Plant; index: number; selected: boolean }) {
  const width = 24 + plant.genes.rootSpread * 56;
  const depth = 22 + plant.genes.rootDepth * 62;
  return <div className={`root-system ${selected ? "roots-selected" : ""}`} style={{ left: `${plant.x}%`, width: `${width}px`, height: `${depth}px`, zIndex: index + 2 }} aria-hidden="true"><i /><i /><i /><i /><i /></div>;
}

function PlantPortrait({ plant }: { plant: Plant }) {
  return <div className={`portrait-plant portrait-${phenotype(plant.genes).toLowerCase()}`} aria-hidden="true"><i className="portrait-stem" /><i className="portrait-leaf leaf-a" /><i className="portrait-leaf leaf-b" /><i className="portrait-leaf leaf-c" /><i className="portrait-bloom" /></div>;
}

function GeneRow({ label, value, variant }: { label: string; value: number; variant: string }) {
  return <div className="gene-row"><span>{label}</span><div className="gene-meter"><i className={`meter-${variant}`} style={{ width: `${Math.round(value * 100)}%` }} /></div><b>{value.toFixed(2)}</b></div>;
}

function Understory({ world, onReturn }: { world: World; onReturn: () => void }) {
  const candidates = [
    { code: "CG-114", form: "Deep-root / slow canopy", score: 0.78, source: "Local pollination", status: "Growing" },
    { code: "CG-208", form: "Balanced spread / high tempo", score: 0.64, source: "Seed bank", status: "Observing" },
    { code: "CG-319", form: "Wide canopy / shallow root", score: 0.52, source: "Hive suggestion", status: "Dormant" },
    { code: "CG-427", form: "Low canopy / long root", score: 0.71, source: "Local pollination", status: "Growing" },
  ];

  return (
    <div className="page-content understory-page">
      <div className="understory-heading">
        <div><div className="eyebrow"><span className="eyebrow-line" /> Below the root line</div><h1>The <em>understory.</em></h1><p className="lede">Ideas are compost here. Nothing in this room receives capital.</p></div>
        <button className="return-button" onClick={onReturn}>← Return to meadow</button>
      </div>
      <div className="understory-summary"><span className="summary-mark">⌁</span><div><b>Research bed / {world.seed}</b><p>Candidate records describe synthetic genomes and simulated conditions. They do not represent exchange-tested strategies.</p></div><span className="fixture-tag">NO CAPITAL</span></div>

      <div className="research-layout">
        <section className="research-main">
          <div className="section-title"><div><span>SEED BANK</span><h2>Four ideas in the soil</h2></div><span className="record-count">04 records</span></div>
          <div className="candidate-list">
            {candidates.map((candidate, index) => (
              <div className="candidate-row" key={candidate.code}>
                <div className={`candidate-glyph glyph-${index}`}><i /><b>{String(index + 1).padStart(2, "0")}</b></div>
                <div className="candidate-main"><b>{candidate.code}</b><span>{candidate.form}</span><small>{candidate.source}</small></div>
                <div className="candidate-score"><span>Sim fitness</span><b>{Math.round(candidate.score * 100)}</b></div>
                <span className={`candidate-status status-${candidate.status.toLowerCase()}`}>{candidate.status}</span>
              </div>
            ))}
          </div>
          <div className="research-footnote"><span>i</span><p>Scores summarize this toy world's own rules. They are not predictive and cannot be promoted to a trading system.</p></div>
        </section>

        <aside className="evolution-card">
          <span className="evolution-kicker">POLLINATION / TODAY</span>
          <h2>A little pollen<br /><em>goes a long way.</em></h2>
          <div className="bee-flight"><span className="bee-dot" /><i /><b /><small>Local bee path</small></div>
          <div className="parent-pair"><span><i className="parent-leaf leaf-one" />Aster 04</span><b>×</b><span><i className="parent-leaf leaf-two" />Cloud fern</span></div>
          <p className="offspring-note">A bounded suggestion carries two visible genes forward and mutates one small step.</p>
          <button className="pollen-button" disabled title="Breeding is a visual prototype action only">Watch pollination <span>↗</span></button>
          <small className="offline-note">No capital entitlement is created.</small>
        </aside>
      </div>
      <footer className="page-footer"><span>Synthetic research space · no live signals</span><span>World day <b>{world.day}</b><i /> deterministic seed <b>{world.seed}</b></span></footer>
    </div>
  );
}

export default App;
