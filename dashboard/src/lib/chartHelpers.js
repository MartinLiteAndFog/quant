/** Brick-mode chart time helpers (no DOM). */

export const BRICK_BASE_TS = 1704067200;

/**
 * @param {unknown} bars
 * @returns {{ map: Map<number, number>, timeAxis: number[] }}
 */
export function buildTimeMapFromBars(bars) {
  const m = new Map();
  const arr = [];
  if (!Array.isArray(bars)) return { map: m, timeAxis: arr };
  for (let i = 0; i < bars.length; i++) {
    const t = Number(bars[i].time);
    if (!Number.isFinite(t)) continue;
    arr.push(t);
    if (!m.has(t)) m.set(t, i);
  }
  return { map: m, timeAxis: arr };
}

/**
 * @param {unknown} t
 * @param {Map<number, number>|null|undefined} timeMap
 * @param {unknown} barsRaw
 * @param {number} [brickBaseTs]
 * @returns {number|null}
 */
export function mapTimeForChart(t, timeMap, barsRaw, brickBaseTs = BRICK_BASE_TS) {
  if (t == null) return null;
  const n = Number(t);
  if (!Number.isFinite(n)) return null;
  let idx = timeMap?.get(n);
  if (idx == null && Array.isArray(barsRaw) && barsRaw.length) {
    let lo = 0;
    let hi = barsRaw.length - 1;
    let best = null;
    while (lo <= hi) {
      const mid = Math.floor((lo + hi) / 2);
      const tv = Number(barsRaw[mid].time);
      if (tv <= n) {
        best = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    idx = best;
  }
  if (idx == null) return null;
  return brickBaseTs + idx * 60;
}

/**
 * @param {unknown} t
 * @param {number[]} timeAxis
 * @param {number} [brickBaseTs]
 * @returns {number|null}
 */
export function mapTimeAsOfForChart(t, timeAxis, brickBaseTs = BRICK_BASE_TS) {
  if (t == null) return null;
  const n = Number(t);
  if (!Number.isFinite(n)) return null;
  if (!Array.isArray(timeAxis) || timeAxis.length === 0) return null;
  let lo = 0;
  let hi = timeAxis.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (timeAxis[mid] <= n) {
      ans = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  if (ans < 0) return null;
  return brickBaseTs + ans * 60;
}

/**
 * @param {unknown} bars
 * @param {number} [brickBaseTs]
 * @returns {unknown[]}
 */
export function mapBarsForChart(bars, brickBaseTs = BRICK_BASE_TS) {
  if (!Array.isArray(bars)) return [];
  return bars.map((b, i) => ({ ...b, time: brickBaseTs + i * 60 }));
}

/**
 * @param {unknown} markers
 * @param {number[]} timeAxis
 * @param {number} [brickBaseTs]
 * @returns {unknown[]}
 */
export function mapMarkersForChart(markers, timeAxis, brickBaseTs = BRICK_BASE_TS) {
  if (!Array.isArray(markers)) return [];
  return markers
    .map((m) => {
      const mapped = mapTimeAsOfForChart(m.time, timeAxis, brickBaseTs);
      if (mapped == null) return null;
      return { ...m, time: mapped };
    })
    .filter(Boolean);
}

/**
 * @param {unknown} points
 * @param {Map<number, number>|null|undefined} timeMap
 * @param {unknown} barsRaw
 * @param {number} [brickBaseTs]
 * @returns {{ time: number, value: number }[]}
 */
export function mapLineForChart(points, timeMap, barsRaw, brickBaseTs = BRICK_BASE_TS) {
  if (!Array.isArray(points)) return [];
  return points
    .map((p) => {
      const mapped = mapTimeForChart(p.time, timeMap, barsRaw, brickBaseTs);
      if (mapped == null) return null;
      return { time: mapped, value: Number(p.value) };
    })
    .filter(Boolean);
}

/**
 * @param {{ from_time?: unknown, to_time?: unknown, from_price?: unknown, to_price?: unknown }} seg
 * @param {Map<number, number>|null|undefined} timeMap
 * @param {unknown} barsRaw
 * @param {number} [brickBaseTs]
 * @returns {{ time: number, value: number }[]}
 */
export function mapSegmentForChart(seg, timeMap, barsRaw, brickBaseTs = BRICK_BASE_TS) {
  const t0 = mapTimeForChart(seg.from_time, timeMap, barsRaw, brickBaseTs);
  const t1 = mapTimeForChart(seg.to_time, timeMap, barsRaw, brickBaseTs);
  if (t0 == null || t1 == null) return [];
  return [
    { time: t0, value: Number(seg.from_price) },
    { time: t1, value: Number(seg.to_price) },
  ];
}

/**
 * @param {unknown} bars
 * @param {unknown} levels
 * @param {Map<number, number>|null|undefined} timeMap
 * @param {unknown} barsRaw
 * @param {number} [brickBaseTs]
 * @returns {{ data: { time: number, value: number }[], mode: string }}
 */
export function buildUnifiedExitLine(bars, levels, timeMap, barsRaw, brickBaseTs = BRICK_BASE_TS) {
  if (!Array.isArray(bars) || !bars.length || !levels) return { data: [], mode: 'none' };
  const sl = Number(levels.sl);
  const ttp = Number(levels.ttp);
  const hasSl = Number.isFinite(sl);
  const hasTtp = Number.isFinite(ttp);
  if (!hasSl && !hasTtp) return { data: [], mode: 'none' };
  let entryTime = null;
  if (levels.entry_bar_ts != null) {
    entryTime = mapTimeForChart(Number(levels.entry_bar_ts), timeMap, barsRaw, brickBaseTs);
  }
  if (entryTime == null) {
    if (levels.side) {
      const startIdx = Math.max(0, bars.length - Math.round(bars.length * 0.2));
      entryTime = bars[startIdx].time;
    } else {
      entryTime = bars[0].time;
    }
  }
  const lastTime = bars[bars.length - 1].time;
  if (hasTtp) {
    const side = String(levels.side || '').toLowerCase();
    let exitVal = ttp;
    if (hasSl) {
      exitVal = side === 'short' ? Math.min(sl, ttp) : Math.max(sl, ttp);
    }
    return { data: [{ time: entryTime, value: exitVal }, { time: lastTime, value: exitVal }], mode: 'ttp' };
  }
  return { data: [{ time: entryTime, value: sl }, { time: lastTime, value: sl }], mode: 'sl' };
}

/**
 * @param {unknown} bars
 * @param {unknown} levels
 * @param {unknown} ttpTrailPct
 * @param {Map<number, number>|null|undefined} timeMap
 * @param {unknown} barsRaw
 * @param {number} [brickBaseTs]
 * @returns {{ time: number, value: number }[]}
 */
export function buildTTPTrail(bars, levels, ttpTrailPct, timeMap, barsRaw, brickBaseTs = BRICK_BASE_TS) {
  if (!Array.isArray(bars) || !bars.length || !levels) return [];
  const entryPx = Number(levels.entry_px);
  const sideStr = String(levels.side || '').toLowerCase();
  if (!Number.isFinite(entryPx) || !sideStr) return [];
  const isLong = sideStr === 'long' || sideStr === 'l' || sideStr === '1';
  const trail = Number.isFinite(Number(ttpTrailPct)) && Number(ttpTrailPct) > 0 ? Number(ttpTrailPct) : 0.012;
  let entryT = null;
  if (levels.entry_bar_ts != null) {
    entryT = mapTimeForChart(Number(levels.entry_bar_ts), timeMap, barsRaw, brickBaseTs);
  }
  if (entryT == null) return [];
  let startIdx = bars.length - 1;
  for (let i = 0; i < bars.length; i++) {
    if (Number(bars[i].time) >= Number(entryT)) {
      startIdx = i;
      break;
    }
  }
  let bestFav = entryPx;
  const points = [];
  for (let i = startIdx; i < bars.length; i++) {
    const h = Number(bars[i].high || bars[i].close);
    const l = Number(bars[i].low || bars[i].close);
    if (isLong) {
      bestFav = Math.max(bestFav, h);
      points.push({ time: bars[i].time, value: bestFav * (1 - trail) });
    } else {
      bestFav = Math.min(bestFav, l);
      points.push({ time: bars[i].time, value: bestFav * (1 + trail) });
    }
  }
  return points;
}

/**
 * @param {unknown} v
 * @returns {string}
 */
export function fmtNum(v) {
  if (v == null || Number.isNaN(Number(v))) return '-';
  return Number(v).toFixed(4);
}

/**
 * @param {unknown} lastBars
 * @param {unknown} level
 * @returns {{ time: number, value: number }[]}
 */
export function levelLineData(lastBars, level) {
  if (!Array.isArray(lastBars) || !lastBars.length || level == null || Number.isNaN(Number(level))) return [];
  const first = lastBars[0].time;
  const last = lastBars[lastBars.length - 1].time;
  return [
    { time: first, value: Number(level) },
    { time: last, value: Number(level) },
  ];
}

/**
 * @param {unknown} lastBars
 * @param {unknown} level
 * @param {unknown} levels
 * @param {Map<number, number>|null|undefined} timeMap
 * @param {unknown} barsRaw
 * @param {number} [brickBaseTs]
 * @returns {{ time: number, value: number }[]}
 */
export function levelLineFromEntry(lastBars, level, levels, timeMap, barsRaw, brickBaseTs = BRICK_BASE_TS) {
  if (!Array.isArray(lastBars) || !lastBars.length || level == null || Number.isNaN(Number(level))) return [];
  let first = lastBars[0].time;
  if (levels && levels.entry_bar_ts != null) {
    const mapped = mapTimeForChart(Number(levels.entry_bar_ts), timeMap, barsRaw, brickBaseTs);
    if (mapped != null) first = mapped;
  }
  const last = lastBars[lastBars.length - 1].time;
  return [
    { time: first, value: Number(level) },
    { time: last, value: Number(level) },
  ];
}

/**
 * @param {unknown} payload
 * @returns {number|null}
 */
export function liveRegimeScore(payload) {
  const gc = payload && payload.gate_confidence ? payload.gate_confidence : null;
  if (!gc) return null;
  const pTrend = Number(gc.selected_p_trend);
  if (!Number.isFinite(pTrend)) return null;
  return Math.max(-1, Math.min(1, 2 * pTrend - 1));
}
