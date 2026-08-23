/** Normalize possibly-ms unix timestamps to seconds. */
export function asUnixSeconds(t: number): number {
  return t > 1e12 ? Math.floor(t / 1000) : Math.floor(t);
}

/**
 * Fixed chip windows (hours > 0) → absolute [t0, t1] in unix seconds.
 * Returns null for ``all`` (hours ≤ 0) so the chart can use API clock / data.
 *
 * Uses ``clockT1`` (API "now") when present so the window matches the
 * performance payload; otherwise ``nowSec``.
 */
export function rangeWindowUnix(
  hours: number,
  nowSec: number,
  clockT1?: number | null,
): { t0: number; t1: number } | null {
  if (!(hours > 0)) return null;
  const t1 =
    clockT1 != null && Number.isFinite(clockT1) && clockT1 > 0
      ? asUnixSeconds(clockT1)
      : Math.floor(nowSec);
  return { t0: t1 - hours * 3600, t1 };
}
