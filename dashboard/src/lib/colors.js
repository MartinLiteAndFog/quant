/** Color utilities for the quant dashboard. */

/* ── Refined palette ───────────────────────────────────────────── */
export const palette = {
  bg:       '#181c24',
  bgCard:   '#1e2330',
  bgPanel:  '#232a38',
  border:   '#2a3040',
  borderLt: '#343e52',
  text:     '#d1d5e0',
  textDim:  '#8892a6',
  muted:    '#5a6478',

  green:    '#34d399',  // softer emerald
  red:      '#f87171',  // softer coral-red
  amber:    '#fbbf24',  // warm amber
  blue:     '#60a5fa',  // calm blue
  purple:   '#a78bfa',  // soft purple
  cyan:     '#22d3ee',  // cyan accent

  /* axis colors */
  axisX:    '#ff7755',  // drift orange
  axisY:    '#55bbff',  // elasticity blue
  axisZ:    '#ffcc44',  // instability yellow

  grid:     '#1e2530',
  gridLt:   '#262e3c',
};

/**
 * Map a continuous regime score (-1 to +1) to a smooth gradient color.
 *
 * -1  → red  (#f87171)
 *  0  → amber (#fbbf24)
 * +1  → green (#34d399)
 *
 * Uses cubic-smoothstep for perceptually smooth transitions.
 */
export function scoreToColor(score, alpha = 1.0) {
  const s = Math.max(-1, Math.min(1, score));
  const t = (s + 1.0) / 2.0;

  // cubic smoothstep for perceptually smoother blending
  const ts = t * t * (3 - 2 * t);

  let r, g, b;
  if (ts < 0.5) {
    // red → amber
    const u = ts / 0.5;
    r = Math.round(248 + (251 - 248) * u);
    g = Math.round(113 + (191 - 113) * u);
    b = Math.round(113 + (36 - 113) * u);
  } else {
    // amber → green
    const u = (ts - 0.5) / 0.5;
    r = Math.round(251 + (52 - 251) * u);
    g = Math.round(191 + (211 - 191) * u);
    b = Math.round(36 + (153 - 36) * u);
  }
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/**
 * Parse an rgba() or rgb() string into [r, g, b, a].
 */
export function parseRGBA(str) {
  const m = str.match(/[\d.]+/g);
  if (!m) return [0, 0, 0, 1];
  return [+m[0], +m[1], +m[2], m[3] != null ? +m[3] : 1];
}
