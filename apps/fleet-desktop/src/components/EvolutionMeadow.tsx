import { useMemo, useState, type KeyboardEvent } from "react";
import {
  BOUNDS, ROOT_COLORS, createInitialMeadow, mutateMeadow, seedCount,
  type MeadowFlower,
} from "../lib/evolutionMeadow";

/** Orthographic interactive 3D projection. DEMO ONLY: simulated profits. */
const GROUPS: Array<[string, string]> = [
  ["all", "Alle Familien"], ["blaupunkt", "Familie 1"],
  ["blaupunkt-p02", "Familie 2"], ["blaupunkt-p03", "Familie 3"],
];
const clamp = (x: number, low: number, high: number) => Math.max(low, Math.min(high, x));
const fmt = (n: number) => (n >= 0 ? "+" : "") + (n * 100).toFixed(1) + " %";

function world(f: MeadowFlower) {
  return {
    x: 2 * (f.sensitivity - BOUNDS.sensitivity[0]) / (BOUNDS.sensitivity[1] - BOUNDS.sensitivity[0]) - 1,
    y: 2 * (f.ttp - BOUNDS.ttp[0]) / (BOUNDS.ttp[1] - BOUNDS.ttp[0]) - 1,
    z: Math.max(0, f.roi),
  };
}
function project(x: number, y: number, z: number, azimuth: number) {
  const a = azimuth * Math.PI / 180;
  const rx = x * Math.cos(a) - y * Math.sin(a);
  const depth = x * Math.sin(a) + y * Math.cos(a);
  return { x: 450 + rx * 194, y: 382 + depth * 90 - Math.sqrt(Math.max(0, z)) * 255, depth };
}

export function EvolutionMeadow() {
  const [flowers, setFlowers] = useState<MeadowFlower[]>(createInitialMeadow);
  const [generation, setGeneration] = useState(1);
  const [family, setFamily] = useState("all");
  const [angle, setAngle] = useState(25);
  const [showSeeds, setShowSeeds] = useState(true);
  const [selectedId, setSelectedId] = useState("blaupunkt");
  const byId = useMemo(() => new Map(flowers.map(f => [f.id, f])), [flowers]);
  const shown = useMemo(() => flowers.filter(f => family === "all" || f.rootId === family), [flowers, family]);
  const selected = byId.get(selectedId);
  const active = shown.filter(f => seedCount(f) > 0).length;
  const sorted = useMemo(() => [...shown].sort((a, b) => {
    const p = world(a), q = world(b);
    return project(p.x, p.y, p.z, angle).depth - project(q.x, q.y, q.z, angle).depth;
  }), [shown, angle]);

  function grow() {
    const next = mutateMeadow(flowers, generation + 1);
    if (next !== flowers) {
      setFlowers(next);
      setGeneration(generation + 1);
    }
  }
  function reset() {
    setFlowers(createInitialMeadow());
    setGeneration(1);
    setSelectedId("blaupunkt");
  }
  function press(e: KeyboardEvent<SVGGElement>, id: string) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      setSelectedId(id);
    }
  }

  const corners = [
    project(-1, -1, 0, angle), project(1, -1, 0, angle),
    project(1, 1, 0, angle), project(-1, 1, 0, angle),
  ];
  const c = (x: number, y: number) => project(x, y, 0, angle);
  const outline = corners.map(p => p.x.toFixed(1) + "," + p.y.toFixed(1)).join(" ");

  return (
    <section aria-label="Evolution Meadow simulation" className="flex h-full min-h-0 flex-col bg-[#101417]">
      <style>{"@keyframes meadowBloom {0%{opacity:0;transform:scale(.2)}100%{opacity:1;transform:scale(1)}}" +
        "@keyframes meadowSow{0%{stroke-dashoffset:52;opacity:0}45%{opacity:.36}100%{stroke-dashoffset:0;opacity:.21}}" +
        ".meadow-bloom{transform-box:fill-box;transform-origin:center;animation:meadowBloom 600ms ease-out both}" +
        ".meadow-sow{animation:meadowSow 850ms ease-out both}"}</style>
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[var(--line)] px-4 py-3">
        <div>
          <h2 className="text-[14px] font-semibold tracking-[0.02em]">Evolution Meadow</h2>
          <p className="mt-1 text-[11px] text-[var(--muted)]">
            Sensitivität × TTP × Nettorendite · <span className="text-[#e0b870]">SIMULATION – keine echten Trades</span>
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button type="button" onClick={grow}
            className="rounded border border-[#6d947b] bg-[#234134] px-3 py-2 text-[11px] font-semibold hover:bg-[#315543]">
            Neue Generation aussäen
          </button>
          <button type="button" onClick={reset}
            className="rounded border border-[var(--line)] px-3 py-2 text-[11px] text-[var(--muted)] hover:text-white">
            Zurücksetzen
          </button>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2 border-b border-[var(--line)] px-4 py-2 text-[11px] sm:grid-cols-4">
        <div><span className="text-[var(--muted)]">Generation</span><div className="font-mono text-[16px]">{generation}</div></div>
        <div><span className="text-[var(--muted)]">Pflanzen</span><div className="font-mono text-[16px]">{shown.length}</div></div>
        <div><span className="text-[var(--muted)]">Fortpflanzungsfähig</span><div className="font-mono text-[16px]">{active}</div></div>
        <div><span className="text-[var(--muted)]">Max. Nettorendite (sim.)</span>
          <div className="font-mono text-[16px]">{shown.length ? fmt(Math.max(...shown.map(f => f.roi))) : "—"}</div>
        </div>
      </div>
      <div className="flex min-h-0 flex-1 flex-col xl:flex-row">
        <div className="relative flex min-h-[290px] min-w-0 flex-1 items-center justify-center overflow-hidden">
          <svg viewBox="0 0 900 540" preserveAspectRatio="xMidYMid meet" className="h-full w-full"
            role="img" aria-label="Räumliche Parameterlandschaft mit Gewinngipfeln aus Blumen">
            <defs>
              <radialGradient id="meadowSoil" cx="45%" cy="45%" r="75%">
                <stop offset="0" stopColor="#1a3030"/>
                <stop offset="1" stopColor="#132023"/>
              </radialGradient>
            </defs>
            <polygon points={outline} fill="url(#meadowSoil)" fillOpacity=".85"
              stroke="#6e9187" strokeWidth="1.1"/>
            {Array.from({ length: 11 }, (_, k) => {
              const t = -1 + .2 * k;
              const a = c(t, -1), b = c(t, 1), u = c(-1, t), v = c(1, t);
              return <g key={"grid-" + k} stroke="#6b8b80" strokeWidth=".7" opacity=".19">
                <line x1={a.x} y1={a.y} x2={b.x} y2={b.y}/>
                <line x1={u.x} y1={u.y} x2={v.x} y2={v.y}/>
              </g>;
            })}
            <text x="450" y="528" fontSize="12" textAnchor="middle" fill="#b4b8a8">
              Parameterebene · Sensitivität (12–25) / TTP (0,25–1,65 %)
            </text>
            <text x="34" y="40" fill="#a4cbb3" fontSize="12">↑ Profit</text>
            <line x1="48" y1="118" x2="48" y2="58" stroke="#a4cbb3" strokeWidth="1.6"/>
            <path d="M43 64 L48 56 L53 64" fill="none" stroke="#a4cbb3" strokeWidth="1.6"/>
            <text x={corners[0].x - 17} y={corners[0].y + 20} fontSize="11" fill="#9fada9">12 / 0.25</text>
            <text x={corners[2].x - 10} y={corners[2].y + 20} fontSize="11" fill="#9fada9">25 / 1.65</text>
            {showSeeds && shown.filter(f => f.parentId && f.generation === generation).slice(0, 160).map(f => {
              const parent = byId.get(f.parentId || "");
              if (!parent) return null;
              const p = world(parent), ground = world(f);
              const a = project(p.x, p.y, p.z, angle), b = c(ground.x, ground.y);
              const arch = Math.min(a.y, b.y) - 35;
              const d = "M " + a.x + " " + a.y + " Q " +
                ((a.x + b.x) / 2) + " " + arch + " " + b.x + " " + b.y;
              return <path key={"seedline-" + f.id} className="meadow-sow" d={d}
                fill="none" stroke={ROOT_COLORS[f.rootId]} opacity=".2"
                strokeDasharray="3 6" strokeWidth="1.1"/>;
            })}
            {sorted.map(f => {
              const w = world(f), foot = c(w.x, w.y);
              const head = project(w.x, w.y, w.z, angle);
              const positive = f.roi > 0, selectedFlower = selectedId === f.id;
              const color = ROOT_COLORS[f.rootId] || "#acc19b";
              const radius = clamp(3 + Math.sqrt(Math.max(0, f.score)) * 1.4, 3, 12);
              return <g key={f.id} role="button" tabIndex={0}
                aria-label={f.id + ": simulated return " + fmt(f.roi)}
                onClick={() => setSelectedId(f.id)} onKeyDown={e => press(e, f.id)}
                className="cursor-pointer">
                <circle cx={foot.x} cy={foot.y} r={positive ? 3.3 : 4.1} fill="#a6a49a" opacity={positive ? ".3" : ".55"}/>
                {positive && <line x1={foot.x} y1={foot.y} x2={head.x} y2={head.y}
                  stroke={color} strokeWidth={selectedFlower ? 2.6 : 1.7}
                  opacity={selectedFlower ? 1 : .67}/>}
                {positive ? <g className="meadow-bloom">
                  {Array.from({length: 6}, (_, j) => {
                    const theta = 2 * Math.PI * j / 6;
                    const px = head.x + radius * Math.cos(theta);
                    const py = head.y + radius * Math.sin(theta);
                    return <ellipse key={j} cx={px} cy={py} rx={radius * .72} ry={radius * .59}
                      transform={"rotate(" + (j * 60) + " " + px + " " + py + ")"}
                      fill={color} opacity={selectedFlower ? ".97" : ".72"}/>;
                  })}
                  <circle cx={head.x} cy={head.y} r={radius * .58} fill="#e4ddae"/>
                  {selectedFlower && <circle cx={head.x} cy={head.y} r={radius * 2.15}
                    stroke="#f7f4d8" strokeWidth="1.2" fill="none" opacity=".83"/>}
                </g> : <circle cx={foot.x} cy={foot.y} r={selectedFlower ? 8 : 5}
                  fill="#596361" stroke={selectedFlower ? "#d6d8c0" : "#78837e"} strokeWidth="1.5"/>}
                <title>{f.id + " | Sensitivität " + f.sensitivity.toFixed(2) +
                  " | TTP " + f.ttp.toFixed(3) + "% | (sim.) " + fmt(f.roi)}</title>
              </g>;
            })}
          </svg>
          {flowers.length >= 450 && <div className="absolute bottom-3 right-4 text-[10px] text-[#e0b870]">
            Anzeige auf 450 Simulationspflanzen begrenzt, kein Live-Bot-Limit
          </div>}
        </div>
        <aside className="w-full shrink-0 space-y-3 border-t border-[var(--line)] p-4 text-[11px] xl:w-[265px] xl:border-l xl:border-t-0">
          <div className="space-y-1">
            <label htmlFor="meadow-family" className="text-[var(--muted)]">Familie</label>
            <select id="meadow-family" value={family} onChange={e => setFamily(e.target.value)}
              className="w-full rounded border border-[var(--line)] bg-[#171e20] px-2 py-2 text-[12px]">
              {GROUPS.map(([id, name]) => <option key={id} value={id}>{name}</option>)}
            </select>
          </div>
          <div className="space-y-1">
            <label htmlFor="meadow-rotation" className="text-[var(--muted)]">Ansicht drehen · {angle}°</label>
            <input id="meadow-rotation" type="range" min="0" max="360" step="5"
              className="w-full accent-[#85c99f]" value={angle}
              onChange={e => setAngle(+e.target.value)}/>
          </div>
          <label className="flex items-center gap-2 text-[var(--muted)]">
            <input type="checkbox" checked={showSeeds} onChange={e => setShowSeeds(e.target.checked)}/>
            Saatflug anzeigen
          </label>
          <div className="border-t border-[var(--line)] pt-3">
            <h3 className="mb-2 font-semibold uppercase tracking-[0.08em] text-[var(--muted)]">Ausgewählte Blume</h3>
            {selected ? <>
              <div className="break-all text-[12px] font-semibold text-[#d4e0ce]">{selected.id}</div>
              <div className="mt-1 text-[var(--muted)]">Generation {selected.generation} · {selected.rootId}</div>
              <dl className="mt-3 grid grid-cols-2 gap-y-2">
                <dt>Sensitivität</dt><dd className="text-right font-mono">{selected.sensitivity.toFixed(3)}</dd>
                <dt>TTP</dt><dd className="text-right font-mono">{selected.ttp.toFixed(3)} %</dd>
                <dt>Gewinn*</dt><dd className="text-right font-mono">{fmt(selected.roi)}</dd>
                <dt>Drawdown*</dt><dd className="text-right font-mono">{fmt(selected.drawdown)}</dd>
                <dt>Fitness*</dt><dd className="text-right font-mono">{selected.score.toFixed(2)}</dd>
                <dt>Trades*</dt><dd className="text-right font-mono">{selected.trades}</dd>
                <dt>Samen / Zyklus*</dt><dd className="text-right font-mono">{seedCount(selected)}</dd>
              </dl>
            </> : <p>Eine Blume wählen.</p>}
          </div>
          <p className="border-t border-[var(--line)] pt-3 text-[10px] leading-relaxed text-[var(--muted)]">
            *Synthetische Beispieldaten. Live-Parameter, echte Equity und Fills müssen
            später aus Registry und Ledger kommen. Rein visuelle Evolution:
            <strong className="block pt-1 font-semibold text-[#e0b870]">Keine Orderausführung.</strong>
          </p>
        </aside>
      </div>
    </section>
  );
}
