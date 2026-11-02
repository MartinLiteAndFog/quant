import { useId, type KeyboardEvent, type CSSProperties } from 'react';
import { STARTING_NAV_USDT, type GardenPlant, type SimulationWorld } from './simulation';
import './isometric.css';

type Point3 = readonly [number, number, number];
type Point2 = readonly [number, number];

/** Orthographic isometry: both ground axes sit at 30°, elevation stays vertical. */
function project([x, y, z]: Point3): Point2 {
  return [(x - y) * Math.sqrt(3) / 2, (x + y) / 2 - z];
}

function polygon(points: readonly Point3[]): string {
  return points.map((point) => project(point).map((value) => value.toFixed(2)).join(',')).join(' ');
}

/** Small faceted solids, projected with the exact same camera as the terrain. */
function Crown({ center, radius, height, tone }: { center: Point3; radius: number; height: number; tone: number }) {
  const [x, y, z] = center;
  const ring: Point3[] = Array.from({ length: 6 }, (_, index) => {
    const angle = Math.PI * index / 3;
    return [x + Math.cos(angle) * radius, y + Math.sin(angle) * radius, z];
  });
  return <g>{ring.map((point, index) => <polygon key={index}
    points={polygon([point, ring[(index + 1) % 6], [x, y, z + height]])}
    fill={`hsl(${tone} 22% ${[51, 44, 39, 43, 56, 61][index]}%)`} />)}</g>;
}

function Stem({ x = 0, y = 0, z = 0, height, radius = 6 }: { x?: number; y?: number; z?: number; height: number; radius?: number }) {
  return <g>
    <polygon points={polygon([[x - radius, y, z], [x, y + radius, z], [x, y + radius, z + height], [x - radius, y, z + height]])} fill="#5d7b59" />
    <polygon points={polygon([[x, y + radius, z], [x + radius, y, z], [x + radius, y, z + height], [x, y + radius, z + height]])} fill="#73996a" />
    <polygon points={polygon([[x - radius, y, z + height], [x, y - radius, z + height], [x + radius, y, z + height], [x, y + radius, z + height]])} fill="#a3b78b" />
  </g>;
}

function PlantMesh({ plant }: { plant: GardenPlant }) {
  const height = 35 + Math.min(100, Math.max(0, plant.biomass)) * .64;
  const tone = 95 + Math.round(plant.genes.water * 28);
  if (plant.phenotype === 'Cactus') return <g>
    <Stem height={height} radius={8 + plant.genes.root * 4} />
    <Stem x={-17} z={height * .38} height={height * .32} radius={5} />
    <Stem x={17} z={height * .2} height={height * .31} radius={5} />
    <polygon points={polygon([[-17, 0, height * .4], [0, 0, height * .34], [0, 0, height * .45], [-17, 0, height * .51]])} fill="#719466" />
    <polygon points={polygon([[0, 0, height * .23], [17, 0, height * .21], [17, 0, height * .31], [0, 0, height * .34]])} fill="#5d7b59" />
    <path d={`M -3 ${-height + 10} v 9 M 3 ${-height + 27} v 8`} stroke="#dce4ba" strokeWidth="1.4" />
  </g>;
  if (plant.phenotype === 'Bush') return <g>
    <Stem height={height * .62} radius={3} />
    <Crown center={[-14, 0, height * .36]} radius={21 + plant.genes.leaf * 6} height={height * .43} tone={tone} />
    <Crown center={[13, 2, height * .42]} radius={22} height={height * .43} tone={tone + 5} />
    <Crown center={[0, 0, height * .64]} radius={23} height={height * .37} tone={tone} />
  </g>;
  return <g>{[-2, -1, 0, 1, 2].map((index) => {
    const tip: Point3 = [index * 10, Math.abs(index) * 2, height * (1 - Math.abs(index) * .12)];
    return <g key={index}>
      <polygon points={polygon([[-4 + index * 3, 0, 0], tip, [index * 3 + 2, 0, height * .25]])} fill={index < 0 ? '#5d7d55' : '#91a66d'} />
      <polygon points={polygon([[index * 3 + 2, 0, height * .25], tip, [7 + index * 3, 2, 0]])} fill={index < 0 ? '#87a06a' : '#64865a'} />
    </g>;
  })}</g>;
}

export type IsometricGardenProps = {
  world: SimulationWorld;
  selectedId: string;
  onSelect: (id: string) => void;
  running: boolean;
  motionEnabled: boolean;
  plane?: GardenPlane;
  onPlaneChange?: (plane: GardenPlane) => void;
  /** Starting capital plus net contributions; external deposits are not returns. */
  capitalBaselineUsdt?: number;
};

export type GardenPlane = 'root-leaf' | 'water-sun' | 'trading';

const PLANES = {
  'root-leaf': { name: 'Wurzel / Blatt', x: 'Wurzelgen', y: 'Blattgen', xMin: 0, xMax: 1, yMin: 0, yMax: 1 },
  'water-sun': { name: 'Wasser / Sonne', x: 'Wassergen', y: 'Sonnengen', xMin: 0, xMax: 1, yMin: 0, yMax: 1 },
  trading: { name: 'Ökologische Handelsgene', x: 'Ökologie-Lookback', y: 'Ökologie-Einstieg (bps)', xMin: 1, xMax: 120, yMin: 1, yMax: 3000 },
} as const;

/** Unmodified trait values and affine coordinates: crown separation never moves roots. */
export function gardenPlaneCoordinates(plant: GardenPlant, plane: GardenPlane): { xValue: number; yValue: number; x: number; y: number } {
  const axis = PLANES[plane];
  const [xValue, yValue] = plane === 'root-leaf' ? [plant.genes.root, plant.genes.leaf]
    : plane === 'water-sun' ? [plant.genes.water, plant.genes.sun]
      : [plant.tradingTraits.lookbackBars, plant.tradingTraits.entryThresholdBps];
  return { xValue, yValue, x: (xValue - axis.xMin) / (axis.xMax - axis.xMin) * 420, y: (yValue - axis.yMin) / (axis.yMax - axis.yMin) * 420 };
}

/** Spatial game view; all values and visible specimens come from the world. */
export function IsometricGarden({ world, selectedId, onSelect, running, motionEnabled, plane = 'root-leaf', onPlaneChange, capitalBaselineUsdt = STARTING_NAV_USDT }: IsometricGardenProps) {
  const descriptionId = useId();
  const living = world.plants.filter((plant) => plant.status === 'active');
  const selected = living.find((plant) => plant.id === selectedId);
  const side = 5;
  const cell = 84;
  const scale = 5 / side;
  const plot = side * cell;
  // The shared NAV moves one groundwater level. It is not a second capital pool.
  const baseline = Number.isFinite(capitalBaselineUsdt) && capitalBaselineUsdt > 0 ? capitalBaselineUsdt : STARTING_NAV_USDT;
  const waterDepth = Math.max(12, Math.min(104, 28 + 76 * (1 - Math.max(0, world.capital.navUsdt / baseline))));
  const axis = PLANES[plane];
  const meshScale = living.length > 100 ? .42 : living.length > 50 ? .55 : living.length > 25 ? .7 : 1;
  const occupied: Point2[] = [];
  const positioned = [...living].sort((a, b) => a.id.localeCompare(b.id)).map((plant) => {
    const point = gardenPlaneCoordinates(plant, plane);
    const [px, py] = project([point.x, point.y, 0]);
    const centerY = py - 65 * meshScale;
    let dx = 0, dy = 0;
    // Deterministic outward phyllotaxis separates crowns; exact root anchors remain untouched.
    const outward = Math.atan2(centerY - plot / 2, px);
    const separation = 47 * meshScale;
    for (let attempt = 0; attempt < 2000; attempt += 1) {
      const radius = attempt === 0 ? 0 : separation * .6 * Math.sqrt(attempt);
      const angle = outward + attempt * Math.PI * (3 - Math.sqrt(5));
      dx = Math.cos(angle) * radius;
      dy = Math.sin(angle) * radius;
      const insideView = px + dx >= -440 && px + dx <= 440 && centerY + dy >= -80 && centerY + dy <= 500;
      if (insideView && occupied.every(([ox, oy]) => Math.hypot(px + dx - ox, centerY + dy - oy) >= separation)) break;
    }
    occupied.push([px + dx, centerY + dy]);
    return { plant, ...point, dx, dy };
  }).sort((a, b) => a.x + a.y - b.x - b.y);
  const selectedPosition = positioned.find(({ plant }) => plant.id === selectedId);
  const recentBirths = world.events.filter((event) => event.type === 'conception' && event.day >= world.day - 1 && event.day <= world.day);
  const hive: Point2 = project([-45, 45, 0]);
  const activate = (event: KeyboardEvent<SVGGElement>, id: string) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onSelect(id);
    }
  };
  return <section className="iso-garden" data-motion={motionEnabled ? 'on' : 'off'} data-running={running} aria-label="Isometrischer Garten">
    <div className="iso-scene-heading">
      <div><strong>Dein Strategie-Garten</strong><span>{living.length} lebende Pflanzen · Tag {world.day}</span></div>
      <span className="iso-clock">{running ? 'Simulation läuft' : 'Simulation pausiert'}</span>
    </div>
    <div className="iso-plane-control">{onPlaneChange ? <label>Koordinatenebene <select value={plane} onChange={(event) => onPlaneChange(event.target.value as GardenPlane)}>{Object.entries(PLANES).map(([key, value]) => <option key={key} value={key}>{value.name}</option>)}</select></label> : <span>Koordinatenebene: {axis.name}</span>}<span>Anker = exakte Gene · Kronen weichen bei Nähe aus</span></div>
    <p id={descriptionId} className="iso-sr-only">{living.length} lebende Pflanzen auf isometrischer Fläche. Eine Pflanze anklicken oder mit Tab auswählen und mit Enter öffnen. Höhe zeigt Biomasse, der Balken Vitalität. Archivierte Pflanzen werden nicht dargestellt.</p>
    <svg className="iso-stage" viewBox="0 0 1000 760" aria-describedby={descriptionId} role="group" aria-label="Pflanzen auf dem Gartenfeld">
      <g aria-hidden="true" transform={`translate(500 135) scale(${scale})`}>
        <polygon points={polygon([[0, 0, -120], [plot, 0, -120], [plot, plot, -120], [0, plot, -120]])} transform="translate(12 12)" fill="#405838" opacity=".09" />
        <polygon points={polygon([[plot, 0, 0], [plot, plot, 0], [plot, plot, -120], [plot, 0, -120]])} fill="#b3a17e" />
        <polygon points={polygon([[0, plot, 0], [plot, plot, 0], [plot, plot, -120], [0, plot, -120]])} fill="#c2b08a" />
        {[35, 80].map((depth) => <polyline key={depth} points={polygon([[0, plot, -depth], [plot, plot, -depth], [plot, 0, -depth]])} fill="none" stroke="#d1c29d" strokeWidth="2" opacity=".7" />)}
        <polygon className="iso-groundwater" points={polygon([[0, 0, -waterDepth], [plot, 0, -waterDepth], [plot, plot, -waterDepth], [0, plot, -waterDepth]])} fill="#7da9aa" opacity=".8" />
        {positioned.map(({ plant, x, y }) => {
          const depth = 18 + plant.genes.root * 90;
          return <g key={plant.id} fill="none" stroke={plant.id === selectedId ? '#5b583a' : '#86774e'} strokeWidth={plant.id === selectedId ? 2.4 : 1.4} opacity={plant.id === selectedId ? 1 : .6}>
            <polyline points={polygon([[x, y, 0], [x + 3, y, -depth * .5], [x - 2, y + 3, -depth]])} />
            <polyline points={polygon([[x + 3, y, -depth * .5], [x - 9, y + 5, -depth * .8]])} />
            <polyline points={polygon([[x + 2, y, -depth * .62], [x + 11, y - 3, -depth * .92]])} />
          </g>;
        })}
        {Array.from({ length: side * side }, (_, index) => {
          const x = index % side * cell, y = Math.floor(index / side) * cell;
          return <polygon key={index} points={polygon([[x, y, 0], [x + cell, y, 0], [x + cell, y + cell, 0], [x, y + cell, 0]])}
            fill={['#d9e2c3', '#dce5c9', '#d5dfbc', '#dfe7cc'][(index + Math.floor(index / side)) % 4]} stroke="#c1ceb0" strokeWidth="1" />;
        })}
        <polyline points={polygon([[0, plot, 1], [0, 0, 1], [plot, 0, 1]])} fill="none" stroke="#ecf0de" strokeWidth="3" />
        <g transform={`translate(${hive[0]} ${hive[1]})`}>
          <polygon points="-19,-8 0,2 19,-8 0,-19" fill="#d5b96e" />
          <polygon points="-19,-8 0,2 0,-30 -19,-40" fill="#b89b55" />
          <polygon points="0,2 19,-8 19,-40 0,-30" fill="#c6aa63" />
          <polygon points="-22,-40 0,-27 22,-40 0,-53" fill="#dfcc8f" />
          <path d="M -14 -16 l 10 5 M -14 -26 l 10 5" stroke="#8f793e" strokeWidth="2" /><ellipse cx="8" cy="-10" rx="4" ry="5" fill="#655b38" />
          <text x="0" y="22" textAnchor="middle" fontSize="12" fill="#4c6142">Gen-Bibliothek</text>
        </g>
      </g>
      <g transform={`translate(500 135) scale(${scale})`}>
        {positioned.map(({ plant, x, y, dx, dy, xValue, yValue }) => {
          const [px, py] = project([x, y, 0]);
          const isSelected = plant.id === selectedId;
          const name = plant.role === 'parent' ? plant.name : `S${world.plants.indexOf(plant) - 2}`;
          return <g key={plant.id} className={`iso-plant${isSelected ? ' is-selected' : ''}${plant.physiology === 'dormant' ? ' is-dormant' : ''}`}
            data-plane-x={xValue} data-plane-y={yValue} data-anchor-x={x} data-anchor-y={y}
            data-crown-center-x={px + dx} data-crown-center-y={py - 65 * meshScale + dy}
            transform={`translate(${px} ${py})`} role="button" tabIndex={0} aria-pressed={isSelected}
            aria-label={`${plant.name}, ${plant.phenotype}, ${plant.physiology === 'dormant' ? 'ruhend, ' : ''}Vitalität ${plant.vigor.toFixed(0)} Prozent, Kapital ${plant.allocationUsdt.toFixed(2)} USDT. Informationen öffnen.`}
            onClick={() => onSelect(plant.id)} onKeyDown={(event) => activate(event, plant.id)}>
            <title>{`${plant.name} · ${plant.phenotype} · ${plant.allocationUsdt.toFixed(2)} USDT`}</title>
            <ellipse cx={dx + 8} cy={dy + 3} rx={30 * meshScale} ry={12 * meshScale} fill="#4b6245" opacity=".12" />
            <ellipse className="iso-selection" cx="0" cy="0" rx="9" ry="4.5" fill="none" stroke="#486d4b" strokeWidth="2.5" />
            <path d={`M 0 0 Q ${dx * .35} ${dy * .35 - 12} ${dx} ${dy}`} fill="none" stroke={isSelected ? '#3c613d' : '#668158'} strokeWidth={isSelected ? 2 : 1.2} opacity=".8" />
            <circle cx="0" cy="0" r={isSelected ? 3.5 : 2} fill={isSelected ? '#314e34' : '#71855e'} />
            <rect className="iso-hit" x={dx - 30 * meshScale} y={dy - 108 * meshScale} width={60 * meshScale} height={125 * meshScale} rx="8" fill="transparent" />
            <g className="iso-growth" style={{ transform: `translate(${dx}px, ${dy}px) scale(${meshScale * (.85 + Math.min(100, Math.max(0, plant.biomass)) * .003)})` }} aria-hidden="true"><PlantMesh plant={plant} /></g>
            {(isSelected || (living.length <= 25 && plant.role === 'parent')) && <g className="iso-label" transform={`translate(${dx} ${dy})`} aria-hidden="true">
              <rect x="-46" y="19" width="92" height="28" rx="6" fill={isSelected ? '#37583f' : '#f5f7ec'} />
              <text x="0" y="31" textAnchor="middle" fill={isSelected ? '#ffffff' : '#354c39'} fontSize="10" fontWeight="600">{name.length > 16 ? `${name.slice(0, 15)}…` : name}</text>
              <rect x="-31" y="36" width="62" height="3" rx="1.5" fill={isSelected ? '#638167' : '#d6dfca'} />
              <rect x="-31" y="36" width={62 * Math.max(0, Math.min(100, plant.vigor)) / 100} height="3" rx="1.5" fill={isSelected ? '#c8d9a9' : '#698654'} />
            </g>}
          </g>;
        })}
        {recentBirths.map((event) => {
          const child = positioned.find(({ plant }) => plant.id === event.subject);
          if (!child) return null;
          const [cx, cy] = project([child.x, child.y, 30]);
          const mode = String(event.detail.mode);
          const parents = mode === 'local' || mode === 'hybrid' ? child.plant.parents?.map((id) => positioned.find(({ plant }) => plant.id === id)).filter((entry) => entry !== undefined) ?? [] : [];
          const sources: Point2[] = parents.map((parent) => project([parent.x, parent.y, 45]));
          if (mode === 'hive' || mode === 'hybrid') sources.push([hive[0], hive[1] - 35]);
          return <g key={event.id} className="iso-pollination" aria-hidden="true">
            {sources.map(([sx, sy], index) => {
              const path = `M ${sx} ${sy} Q ${(sx + cx) / 2} ${Math.min(sy, cy) - 40} ${cx} ${cy}`;
              return <g key={index}>
                <path d={path} fill="none" stroke="#9b803d" strokeWidth="1.5" strokeDasharray="4 5" opacity=".75" />
                <ellipse className="iso-flying-bee" rx="5" ry="3.5" fill="#b28d35" style={{ offsetPath: `path('${path}')`, offsetRotate: 'auto' } as CSSProperties} />
              </g>;
            })}
            <circle cx={cx} cy={cy} r="5" fill="#b28d35" stroke="#fcf6dc" strokeWidth="2" />
          </g>;
        })}
      </g>
      <g className="iso-axis" aria-hidden="true" transform="translate(500 135)">
        {[0, .5, 1].map((fraction) => {
          const [xx, xy] = project([plot * fraction, plot + 20, 0]);
          const [yx, yy] = project([-20, plot * fraction, 0]);
          return <g key={fraction}><text x={xx} y={xy + 16} textAnchor="middle">{(axis.xMin + fraction * (axis.xMax - axis.xMin)).toLocaleString('de-DE', { maximumFractionDigits: 1 })}</text><text x={yx - 12} y={yy + 4} textAnchor="end">{(axis.yMin + fraction * (axis.yMax - axis.yMin)).toLocaleString('de-DE', { maximumFractionDigits: 1 })}</text></g>;
        })}
        <text x="-170" y="390" textAnchor="middle">{axis.x}</text><text x="-235" y="55" textAnchor="end">{axis.y}</text>
      </g>
    </svg>
    <p className="iso-water-caption">Gemeinsames Grundwasser: {world.capital.navUsdt.toFixed(2)} USDT NAV / {baseline.toFixed(2)} USDT Kapitalbasis inklusive Einzahlungen. Der Wasserstand ist eine begrenzte schematische Darstellung, die Kapitalzahlen sind exakt. Wurzeln zeigen das Wurzelgen.</p>
    {living.length === 0 && <p className="iso-empty">Im Garten sind keine lebenden Pflanzen mehr. Starte einen neuen Garten, um wieder zu pflanzen.</p>}
    <div className="iso-caption"><span><i />{selected ? `${selected.name} · Wurzel ${Math.round(selected.genes.root * 100)} % · ${18 + selected.genes.root * 90 >= waterDepth ? 'erreicht das Wasser' : 'über dem Wasser'}` : 'Wähle eine Pflanze für ihre Informationen'}</span><span>Höhe = Biomasse · Balken = Vitalität</span></div>
    {selectedPosition && <p className="iso-coordinate-caption">Exakter Anker: {axis.x} = {selectedPosition.xValue} · {axis.y} = {selectedPosition.yValue}{plane === 'trading' ? ' · Diese Ebene zeigt ökologische Handelsgene, keine Kraken-Lookback-Einstellung.' : ''}</p>}
    <div className={`iso-mobile-roster${living.length > 25 ? ' iso-dense-roster' : ''}`} aria-label="Pflanzen direkt auswählen">{living.map((plant) => <button type="button" key={plant.id} aria-pressed={plant.id === selectedId} onClick={() => onSelect(plant.id)}>{plant.name}</button>)}</div>
  </section>;
}
