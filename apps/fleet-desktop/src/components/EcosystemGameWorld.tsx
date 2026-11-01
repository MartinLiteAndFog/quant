import { useEffect, useMemo, useState } from "react";
import {
  DEMO_PLANTS, DEMO_UNDERWORLD, PARAMETER_MEADOWS, getUnassignedPlants,
  groundwaterSurface, missingAxisGenomes, normalizeAxis, selectMeadowPlants,
  type WorldPlant,
} from "../lib/ecosystemGameWorld";

const FAMILY_COLORS = ["#91c9a4", "#e1bf75", "#95afd5"];
const phenotypeName: Record<WorldPlant["phenotype"], string> = {
  grass: "Gras", flower: "Blüte", shrub: "Busch", cactus: "Kaktus", orchid: "Orchidee",
};
const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v));

/** Standalone, synthetic research world. It has no connection to order or account APIs. */
export function EcosystemGameWorld() {
  const [meadowId, setMeadowId] = useState(PARAMETER_MEADOWS[0].id);
  const [selectedId, setSelectedId] = useState(DEMO_PLANTS[0].id);
  const [paused, setPaused] = useState(true);
  const [day, setDay] = useState(1);
  const [underworld, setUnderworld] = useState(false);
  const [heightMetric, setHeightMetric] = useState<"profit" | "biomass">("profit");
  const [waterDepth, setWaterDepth] = useState(0.58);
  const meadow = PARAMETER_MEADOWS.find((item) => item.id === meadowId) ?? PARAMETER_MEADOWS[0];
  const plants = useMemo(() => selectMeadowPlants(DEMO_PLANTS, meadow), [meadow]);
  const selected = plants.find((plant) => plant.id === selectedId) ?? plants[0];
  const unassigned = getUnassignedPlants(DEMO_PLANTS);
  const axisUnassigned = plants.filter((plant) => normalizeAxis(plant.genome[meadow.xAxis.id], meadow.xAxis) === null || normalizeAxis(plant.genome[meadow.yAxis.id], meadow.yAxis) === null);
  const missingGenome = missingAxisGenomes(plants, meadow.xAxis).length + missingAxisGenomes(plants, meadow.yAxis).length;
  const waterSurface = groundwaterSurface(waterDepth);

  useEffect(() => {
    if (paused) return;
    const timer = window.setInterval(() => setDay((current) => current + 1), 1500);
    return () => window.clearInterval(timer);
  }, [paused]);

  const visible = plants.slice(0, 1200);
  return (
    <section aria-label="Ecosystem game world" className="flex h-full min-h-0 flex-col overflow-auto bg-[#101417] text-[#e7e7df]">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold">Ecosystem · Spielwelt</h2>
          <p className="mt-1 text-[11px] text-[#aab4ad]">Oberwelt · Paper-Genome-Unterwelt · <strong className="text-[#e0b870]">SYNTHETISCHE FIXTURES</strong></p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <button type="button" className="rounded border border-white/15 px-3 py-2 hover:bg-white/10" onClick={() => setPaused((p) => !p)}>{paused ? "▶ Start" : "Ⅱ Pause"}</button>
          <button type="button" className="rounded border border-white/15 px-3 py-2 hover:bg-white/10" onClick={() => setDay((d) => d + 1)}>＋ 1 Tag</button>
          <span className="rounded bg-white/5 px-3 py-2 font-mono">Tag {day}</span>
          <button type="button" className="rounded border border-white/15 px-3 py-2 hover:bg-white/10" aria-pressed={underworld} onClick={() => setUnderworld((v) => !v)}>{underworld ? "Oberwelt" : "Unterwelt"}</button>
        </div>
      </div>

      <div className="flex flex-wrap items-end gap-3 border-b border-white/10 px-4 py-3 text-xs">
        <label className="grid gap-1 text-[#aab4ad]">Gen-Wiese / Strategie
          <select className="min-w-56 rounded border border-white/15 bg-[#18201f] px-2 py-2 text-white" value={meadow.id} onChange={(event) => { setMeadowId(event.target.value); setSelectedId(""); }}>
            {PARAMETER_MEADOWS.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
          </select>
        </label>
        <div className="grid gap-1 text-[#aab4ad]"><span>Horizontale Gen-Achse</span><span className="rounded border border-white/10 px-3 py-2 text-white">{meadow.xAxis.label} · {meadow.xAxis.min}–{meadow.xAxis.max}</span></div>
        <div className="grid gap-1 text-[#aab4ad]"><span>Vertikale Gen-Achse</span><span className="rounded border border-white/10 px-3 py-2 text-white">{meadow.yAxis.label} · {meadow.yAxis.min}–{meadow.yAxis.max}</span></div>
        <label className="ml-auto grid min-w-44 gap-1 text-[#aab4ad]">Höhe
          <select className="rounded border border-white/15 bg-[#18201f] px-2 py-2 text-white" value={heightMetric} onChange={(event) => setHeightMetric(event.target.value as "profit" | "biomass")}>
            <option value="profit">Profit · sim.</option><option value="biomass">Biomasse · log.</option>
          </select>
        </label>
      </div>

      {underworld ? <div className="grid min-h-0 flex-1 gap-4 overflow-auto p-4 lg:grid-cols-[minmax(0,1fr)_280px]">
        <div className="min-h-[330px] rounded-xl border border-white/10 bg-[#111b1a] p-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2"><div><h3 className="font-medium">Unterwelt · Paper-Forschung</h3><p className="text-[11px] text-[#9da8a1]">Cluster-Heatmap; einzelne Kandidaten werden nicht gerendert.</p></div><span className="font-mono text-xs text-[#b8c8bc]">{DEMO_UNDERWORLD.reduce((sum, cell) => sum + cell.count, 0).toLocaleString("de-DE")} Kandidaten (sim.)</span></div>
          <div className="grid grid-cols-4 gap-2 sm:grid-cols-6 lg:grid-cols-8" aria-label="Research cluster heatmap">
            {DEMO_UNDERWORLD.map((cell) => <button key={cell.id} title={`${cell.id} · ${cell.count.toLocaleString()} Kandidaten · Score ${cell.medianScore.toFixed(1)} (sim.)`} className="aspect-square rounded border border-white/10 p-1 text-[10px]" style={{ backgroundColor: `rgba(${cell.medianScore >= 0 ? "70,151,101" : "164,104,83"},${0.18 + Math.min(0.48, Math.abs(cell.medianScore) / 6)})` }} onClick={() => setSelectedId("")}>{cell.count >= 1000 ? `${Math.round(cell.count / 1000)}k` : cell.count}</button>)}
          </div>
          <p className="mt-3 text-[10px] text-[#84918a]">Farbe zeigt nur den synthetischen Cluster-Score, keine gemessene oder prognostizierte Rendite.</p>
        </div>
        <aside className="rounded-xl border border-white/10 bg-[#151d1c] p-4 text-xs"><h3 className="font-medium">Research-Snapshot</h3><dl className="mt-3 space-y-2 text-[#aab4ad]"><div className="flex justify-between"><dt>Cluster</dt><dd>{DEMO_UNDERWORLD.length}</dd></div><div className="flex justify-between"><dt>Seed</dt><dd>fest · 20261101</dd></div><div className="flex justify-between"><dt>Evaluation</dt><dd>Paper-only</dd></div><div className="flex justify-between"><dt>Empirisch</dt><dd>nein</dd></div></dl></aside>
      </div> : <div className="grid min-h-0 flex-1 gap-3 overflow-auto p-3 xl:grid-cols-[minmax(0,1fr)_255px]">
        <div className="relative min-h-[390px] overflow-hidden rounded-xl border border-[#35564b] bg-[#101b19]">
          <div className="absolute inset-x-0 top-0 z-10 flex flex-wrap items-center justify-between gap-2 bg-gradient-to-b from-[#101b19]/95 to-transparent p-3 text-[10px] text-[#b4c3b9]">
            <span>{meadow.label} · {plants.length} sichtbare Pflanzen · LOD: lebende/finanzierte Population</span>
            <label className="flex items-center gap-2">Grundwasser
              <input aria-label="Grundwasser-Pegel" type="range" min="0" max="1" step="0.01" value={waterDepth} onChange={(event) => setWaterDepth(Number(event.target.value))} />
            </label>
          </div>
          <svg viewBox="0 0 900 540" preserveAspectRatio="xMidYMid meet" className="h-full min-h-[390px] w-full" role="img" aria-label={`Simulierte Gen-Wiese mit Achsen ${meadow.xAxis.label} und ${meadow.yAxis.label}`}>
            <defs><linearGradient id="eco-sky" x2="0" y2="1"><stop stopColor="#1a302c"/><stop offset="1" stopColor="#172421"/></linearGradient><linearGradient id="eco-soil" x2="0" y2="1"><stop stopColor="#493a2c"/><stop offset="1" stopColor="#201c19"/></linearGradient></defs>
            <rect width="900" height="540" fill="url(#eco-sky)" />
            <circle cx="770" cy="88" r="30" fill="#e9cd81" opacity=".87" /><circle cx="770" cy="88" r="49" fill="#e9cd81" opacity=".08" />
            <path d="M420 392 Q535 359 638 392 L710 392 Q602 372 498 410Z" fill="#08110e" opacity=".2" />
            <path d="M0 390 Q210 370 450 392 T900 382 V540 H0Z" fill="url(#eco-soil)" />
            {Array.from({ length: 7 }, (_, index) => { const y = 412 + index * 20; return <path key={y} d={`M0 ${y} Q250 ${y + 6} 450 ${y} T900 ${y + 8}`} fill="none" stroke="#8a7051" opacity=".16" />; })}
            <path d={`M0 ${540 - waterSurface * 160} Q240 ${526 - waterSurface * 160} 460 ${540 - waterSurface * 160} T900 ${534 - waterSurface * 160}`} fill="none" stroke="#64b4c2" strokeWidth="3" opacity=".85" />
            <text x="15" y={530 - waterSurface * 160} fill="#8cd2da" fontSize="10">Grundwasser · asymptotische Skala</text>
            {Array.from({ length: 11 }, (_, index) => { const t = index / 10; const x = 100 + t * 700; const y = 370 - t * 108; return <g key={index} opacity=".13"><line x1={x} y1={y} x2={x - 120} y2={y + 35} stroke="#a6c6b1"/><line x1={x} y1={y} x2={x + 110} y2={y + 35} stroke="#a6c6b1"/></g>; })}
            {visible.map((plant, index) => {
              const xRatio = normalizeAxis(plant.genome[meadow.xAxis.id], meadow.xAxis);
              const yRatio = normalizeAxis(plant.genome[meadow.yAxis.id], meadow.yAxis);
              if (xRatio === null || yRatio === null) return null;
              const x = 105 + xRatio * 690; const groundY = 366 - yRatio * 91;
              const heightRaw = heightMetric === "profit" ? Math.log1p(Math.max(0, plant.profitPct + 1.5)) : Math.log1p(plant.biomass * 9);
              const height = 18 + clamp(heightRaw, 0, 2.2) * 56;
              const color = FAMILY_COLORS[Number(plant.familyId.slice(-1)) - 1] ?? "#b5d0a3";
              const isSelected = plant.id === selected?.id;
              const spread = 5 + plant.rootSpread * 14;
              return <g key={plant.id} role="button" tabIndex={0} aria-label={`${plant.id}, ${phenotypeName[plant.phenotype]}, ${plant.profitPct >= 0 ? "+" : ""}${plant.profitPct.toFixed(1)} Prozent simuliert`} onClick={() => setSelectedId(plant.id)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") setSelectedId(plant.id); }} className="cursor-pointer">
                <ellipse cx={x} cy={groundY + 38} rx={spread} ry="7" fill="#795b43" opacity=".8" />
                <path d={`M${x} ${groundY + 4} Q${x - spread * .45} ${groundY + 18 + plant.rootDepth * 28} ${x + spread * .22} ${groundY + 34 + plant.rootDepth * 20} M${x} ${groundY + 8} Q${x + spread * .55} ${groundY + 18 + plant.rootDepth * 18} ${x - spread * .4} ${groundY + 30 + plant.rootDepth * 13}`} fill="none" stroke="#c28e60" strokeWidth={isSelected ? 1.7 : 1} opacity=".55" />
                <line x1={x} y1={groundY} x2={x} y2={groundY - height} stroke={color} strokeWidth={isSelected ? 3 : 1.8} opacity=".85" />
                {plant.phenotype === "cactus" ? <g><rect x={x - 5} y={groundY - height - 13} width="10" height="17" rx="5" fill={color}/><path d={`M${x - 3} ${groundY - height - 3} Q${x - 12} ${groundY - height - 12} ${x - 9} ${groundY - height - 20} M${x + 3} ${groundY - height - 2} Q${x + 12} ${groundY - height - 12} ${x + 9} ${groundY - height - 19}`} fill="none" stroke={color} strokeWidth="3"/></g> : <g fill={plant.profitPct >= 0 ? color : "#84907f"} opacity={isSelected ? 1 : .83}>
                  {plant.phenotype === "grass" ? <path d={`M${x} ${groundY - height + 15} Q${x - 10} ${groundY - height + 2} ${x - 11} ${groundY - height - 3} Q${x - 2} ${groundY - height + 2} ${x} ${groundY - height + 12} Q${x + 3} ${groundY - height - 2} ${x + 12} ${groundY - height - 5} Q${x + 12} ${groundY - height + 8} ${x} ${groundY - height + 15}Z`} /> : Array.from({ length: plant.phenotype === "orchid" ? 8 : plant.phenotype === "shrub" ? 5 : 6 }, (_, petal) => <ellipse key={petal} cx={x + Math.cos(petal * Math.PI / 3) * (plant.phenotype === "orchid" ? 10 : 7)} cy={groundY - height + Math.sin(petal * Math.PI / 3) * 6} rx={plant.phenotype === "shrub" ? 5 : 7} ry="4.5" />)}
                </g>}
                <circle cx={x} cy={groundY - height - 1} r={isSelected ? 3.2 : 1.8} fill={isSelected ? "#fff1c1" : color} />
                <circle cx={x} cy={groundY + 2} r="2.3" fill={color} opacity={plant.vitality < 0.5 ? ".3" : ".8"} />
                {index % 7 === 0 && <circle className="eco-bee" cx={x + 22} cy={groundY - height - 16} r="2.5" fill="#e6c568" />}
              </g>;
            })}
            <g className="eco-predator" aria-label="Schematischer Fresser-Lod">
              <path d="M170 364 l8 -7 8 7 9 2 -5 4 -18 0 -6 -3Z" fill="#dc8e65" opacity=".85" />
              <circle cx="188" cy="363" r="1" fill="#191714" />
            </g>
            <g className="eco-predator eco-predator-late" aria-label="Schematischer Fresser-Lod">
              <path d="M670 356 l8 -7 8 7 9 2 -5 4 -18 0 -6 -3Z" fill="#dc8e65" opacity=".72" />
              <circle cx="688" cy="355" r="1" fill="#191714" />
            </g>
            <text x="450" y="522" textAnchor="middle" fill="#c1c9bf" fontSize="12">{meadow.xAxis.label} · {meadow.xAxis.min}–{meadow.xAxis.max} {meadow.xAxis.step !== 1 ? `· Schritt ${meadow.xAxis.step}` : ""}</text>
            <text x="20" y="205" fill="#c1c9bf" fontSize="11" transform="rotate(-90 20 205)">{meadow.yAxis.label} · {meadow.yAxis.min}–{meadow.yAxis.max}</text>
            <text x="19" y="50" fill="#f2d381" fontSize="10">☀ Sonnenregime: synthetisch</text>
          </svg>
          <div className="pointer-events-none absolute bottom-3 right-3 rounded bg-[#0b100f]/85 px-2 py-1 text-[9px] text-[#b4c3b9]">Profit-Höhe / Wurzeln / Blattfläche separat im Detail · ≤ 1.200 Nodes</div>
        </div>
        <aside className="flex flex-col gap-3 text-xs">
          <div className="rounded-xl border border-white/10 bg-[#151d1c] p-4">
            <div className="flex items-center justify-between"><h3 className="font-medium">Pflanzen-Inspektor</h3><span className="text-[10px] text-[#e0b870]">SIM.</span></div>
            {selected ? <><p className="mt-3 text-sm">{phenotypeName[selected.phenotype]} · {selected.id}</p><p className="mt-1 text-[10px] text-[#97a39c]">{selected.familyId} · {selected.strategy}</p>
              <dl className="mt-3 space-y-2 text-[#aab4ad]"><Metric label="Netto-PnL" value={`${selected.profitPct >= 0 ? "+" : ""}${selected.profitPct.toFixed(1)} %`} /><Metric label="Vitalität" value={`${Math.round(selected.vitality * 100)} %`} /><Metric label="Beobachtung" value={`${selected.evidenceDays} Tage`} /><Metric label="Alter" value={`${selected.ageDays} Tage`} /><Metric label="Wurzel-Tiefe / -Breite" value={`${Math.round(selected.rootDepth * 100)} / ${Math.round(selected.rootSpread * 100)}`} /><Metric label="Blattfläche" value={`${Math.round(selected.leafArea * 100)} %`} /></dl>
              <div className="mt-3 border-t border-white/10 pt-3"><span className="text-[10px] uppercase tracking-wider text-[#839087]">Genom · echte Wiesenachsen</span><dl className="mt-2 space-y-1 text-[#aab4ad]">{Object.entries(selected.genome).map(([id, value]) => <Metric key={id} label={id} value={String(value)} />)}</dl></div>
            </> : <p className="mt-3 text-[#9da8a1]">Wähle eine Pflanze in der Wiese.</p>}
          </div>
          <div className="rounded-xl border border-white/10 bg-[#151d1c] p-4"><h3 className="font-medium">Population & Schutz</h3><dl className="mt-2 space-y-2 text-[#aab4ad]"><Metric label="Wiese" value={`${plants.length} Pflanzen`} /><Metric label="Genom-Achsen fehlen" value={String(missingGenome)} /><Metric label="Bienen (LOD)" value="6 · schematisch" /><Metric label="Fresser (LOD)" value="2 · schematisch" /></dl><p className="mt-3 text-[10px] leading-4 text-[#84918a]">Pflanzen ohne Gen-Achsen erscheinen hier ausdrücklich als „nicht zugeordnet“.</p>{axisUnassigned.length > 0 && <ul className="mt-2 space-y-1">{axisUnassigned.slice(0, 12).map((plant) => <li key={plant.id}><button type="button" className="text-[#d9bd78] underline" onClick={() => setSelectedId(plant.id)}>{plant.id} · Gen-Achse fehlt</button></li>)}</ul>}</div>
          <div className="rounded-xl border border-[#886f3f]/40 bg-[#2c271c] p-3 text-[10px] leading-4 text-[#dbcfab]">Alle Pflanzen, Werte, Pegel und Scores in dieser Ansicht sind reproduzierbare Demo-Fixtures. Kein empirischer Feed, keine Orders oder Konten.</div>
        </aside>
      </div>}
      <footer className="flex flex-wrap items-center justify-between gap-2 border-t border-white/10 px-4 py-2 text-[10px] text-[#8f9b93]"><span>Unzugeordnet: {unassigned.length} Strategie-Familien · {axisUnassigned.length} Pflanzen ohne Achsengen · Portfolio unverändert</span><span>Demo Seed 20261101 · Forschungssimulation</span></footer>
      <style>{"@keyframes ecoBee{0%,100%{transform:translate(0,0)}50%{transform:translate(4px,-3px)}}@keyframes ecoPredator{0%,100%{transform:translateX(0)}50%{transform:translateX(32px)}}.eco-bee{animation:ecoBee 1.8s ease-in-out infinite}.eco-predator{animation:ecoPredator 8s ease-in-out infinite}.eco-predator-late{animation-delay:-4s}"}</style>
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="flex justify-between gap-2"><dt>{label}</dt><dd className="text-right font-mono text-[#d7dfd8]">{value}</dd></div>;
}
