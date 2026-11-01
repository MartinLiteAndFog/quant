import {useMemo, useState} from "react";
import {
  focusTile, getWorldStats, listTiles, proposeCrossTileKnowledge,
  tileWorldPosition, type TileWorld,
} from "../lib/tileWorld";
import {TILE_WORLD_DEMO} from "../lib/tileWorldDemo";

/** Read-only tile selection. Underlying paper evolution is independent of zoom. */
export function TileWorldOverview({world = TILE_WORLD_DEMO}: {world?: TileWorld}) {
  const [regionId, setRegionId] = useState<string | null>(null);
  const [tileId, setTileId] = useState<string | null>(null);
  const stats = useMemo(() => getWorldStats(world), [world]);
  const selected = useMemo(() => tileId ? focusTile(world, tileId) : null, [world, tileId]);
  const tiles = useMemo(() => listTiles(world, regionId ?? undefined), [world, regionId]);
  const assetById = useMemo(() => new Map(world.assets.map(a => [a.id, a])), [world]);
  const edges = useMemo(
    () => world.mycelium.map(e => proposeCrossTileKnowledge(world, e.id)), [world]
  );
  const focusRegion = (next: string | null) => {setRegionId(next);setTileId(null);};
  const selectedRegion = world.regions.find(r => r.id === regionId);
  return (
    <section className="h-full w-full overflow-y-auto p-5 text-[var(--text)]"
      style={{background:"radial-gradient(ellipse at 50% 0%, rgba(64,109,71,.17), transparent 72%)"}}>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold">Evolution · Kachelwelt</h2>
          <p className="mt-1 text-xs opacity-70">
            {world.synthetic ? "SYNTHETISCHE DEMO · KEINE ECHTEN TRADES" : "Validierter Forschungssnapshot"}
            {" · "}{stats.regionCount} Regionen · {stats.tileCount} Wiesen · {stats.population} unterschiedliche Strategien
          </p>
        </div>
        <button className="rounded-lg border border-white/25 px-3 py-2 text-sm hover:bg-white/10"
          onClick={() => focusRegion(null)}>Gesamte Welt</button>
      </div>
      <nav className="mb-4 flex flex-wrap gap-2" aria-label="Asset-Regionen">
        {world.regions.map(r => (
          <button key={r.id} type="button" onClick={() => focusRegion(r.id)}
            aria-pressed={regionId === r.id}
            className="rounded-lg px-3 py-1.5 text-sm"
            style={{background:regionId===r.id ? "rgba(87,152,103,.35)" : "rgba(255,255,255,.08)",
              border:"1px solid rgba(200,220,200,.25)"}}>
            {r.label}
          </button>
        ))}
      </nav>
      {selected && (
        <div className="mb-4 rounded-xl border border-white/20 bg-black/10 p-4">
          <div className="mb-3 flex items-start justify-between gap-4">
            <div>
              <h3 className="text-base font-semibold">{selected.tile.label} · Parameterwiese</h3>
              <p className="text-xs opacity-70">
                {selected.tile.axisX.label} × {selected.tile.axisY.label} ·
                {" "}{selected.homePopulation} Heimatpflanzen, {selected.visiblePopulation} sichtbare Genome
              </p>
            </div>
            <button className="rounded-md border border-white/20 px-3 py-1 text-xs"
              onClick={() => setTileId(null)}>Zurück zu Kacheln</button>
          </div>
          <div className="relative mx-auto max-w-[500px] rounded-xl border border-emerald-300/20"
            aria-label={`Parameterfläche ${selected.tile.label}`}
            style={{height:260,background:"repeating-linear-gradient(0deg,rgba(106,141,85,.08),rgba(106,141,85,.08) 1px,transparent 1px,transparent 26px),linear-gradient(125deg,rgba(48,95,64,.25),rgba(123,119,60,.12))"}}>
            {selected.plants.map(p => (
              <button type="button" key={p.strategyId} title={`${p.strategyId} · Vitalität ${Math.round(p.vitality*100)}%${p.home ? "" : " · Zweitansicht"}`}
                className="absolute -translate-x-1/2 translate-y-1/2 rounded-full border-2 border-white/60 shadow-md focus:outline focus:outline-2"
                style={{left:`${Math.max(2,Math.min(98,p.x*100))}%`,bottom:`${Math.max(2,Math.min(98,p.y*100))}%`,
                  width:p.home?17:12,height:p.home?17:12,
                  background:p.home?"#83bc86":"#c3ab6e",opacity:.4+.6*p.vitality}}
                aria-label={p.strategyId}/>
            ))}
          </div>
          <div className="mt-2 flex justify-between text-xs opacity-75">
            <span>X: {selected.tile.axisX.label} ({selected.tile.axisX.min}–{selected.tile.axisX.max})</span>
            <span>Y: {selected.tile.axisY.label} ({selected.tile.axisY.min}–{selected.tile.axisY.max})</span>
          </div>
          <p className="mt-2 text-xs opacity-65">Positionskoordinaten entsprechen echten Genwerten. Goldene Punkte sind reine Zweitansichten ohne zusätzliches Kapital.</p>
        </div>
      )}
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3" aria-label="Wiesen-Kacheln">
        {tiles.slice().sort((a,b) => {
          const aa = tileWorldPosition(world,a.id), bb = tileWorldPosition(world,b.id);
          return aa.y-bb.y || aa.x-bb.x;
        }).map(t => {
          const asset = assetById.get(t.assetId)!;
          const count = stats.homePopulationByTile[t.id];
          const active = tileId === t.id;
          return (
            <button key={t.id} onClick={() => {setRegionId(t.regionId);setTileId(t.id);}}
              aria-pressed={active}
              className="group min-h-[144px] rounded-xl border p-4 text-left transition hover:brightness-110"
              style={{borderColor:active?"#9ad4a6":"rgba(174,212,180,.22)",
                background:"linear-gradient(135deg,rgba(50,104,69,.28),rgba(109,101,59,.18))"}}>
              <div className="flex items-start justify-between gap-2">
                <strong className="text-sm">{t.label}</strong>
                <span className="rounded-full bg-white/10 px-2 py-1 text-[10px]">{asset.assetClass}</span>
              </div>
              <p className="mt-2 text-xs opacity-80">{t.axisX.label} × {t.axisY.label}</p>
              <div className="mt-5 flex items-end gap-1.5" aria-hidden="true">
                {Array.from({length:Math.min(16, Math.max(3,count*3))},(_,i) => (
                  <span key={i} className="w-1.5 rounded-t bg-emerald-300/65"
                    style={{height:8+(i*17+t.id.length)%24}}/>
                ))}
              </div>
              <p className="mt-2 text-xs opacity-65">{count} Heimatpflanzen · {asset.dataStatus === "synthetic"?"Demo-Daten":"Geprüfte Daten"}</p>
            </button>
          );
        })}
      </div>
      <div className="mt-5 grid gap-3 md:grid-cols-2">
        <aside className="rounded-lg border border-white/15 p-3">
          <h4 className="text-sm font-medium">Unterirdisches Myzel · Wissensbrücken</h4>
          <p className="mt-1 text-xs opacity-70">Nur Forschungsinformationen; keine Kapitaltransfers oder automatische Strategieveränderung.</p>
          {edges.map(e => <p key={e.researchId} className="mt-2 text-xs opacity-80">
            {e.sourceTileId} → {e.targetTileId}: {e.geneId} · {Math.round(e.confidence*100)}% Evidenzwert · OOS-Prüfung nötig
          </p>)}
        </aside>
        <aside className="rounded-lg border border-white/15 p-3">
          <h4 className="text-sm font-medium">Noch nicht zugeordnete Forschung</h4>
          <p className="mt-1 text-xs opacity-70">
            {stats.unassigned.length} Genome ohne passende zweidimensionale Heimatwiese.
            Neues echtes Genpaar / Kategorie erforderlich.
          </p>
          {stats.unassigned.map(id => <p className="mt-2 text-xs" key={id}>{id}</p>)}
          {selectedRegion && <p className="mt-2 text-xs opacity-65">Region: {selectedRegion.label}</p>}
        </aside>
      </div>
    </section>
  );
}
