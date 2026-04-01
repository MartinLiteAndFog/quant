export function scoreToColor(score, alpha = 1.0) {
  const t = (Math.max(-1, Math.min(1, score)) + 1.0) / 2.0;
  let r, g, b;
  if (t < 0.5) {
    const u = t / 0.5;
    r = 247; g = Math.round(118 + 86 * u); b = Math.round(142 * (1 - u));
  } else {
    const u = (t - 0.5) / 0.5;
    r = Math.round(247 * (1 - u) + 46 * u); g = 204; b = Math.round(113 * u);
  }
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
