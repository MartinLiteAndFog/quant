import { KRAKEN_STARTER_PROFILES, type KrakenProfile } from "./krakenProfiles.ts";
import { validateSourceGenome, type SourceGenome } from "./sourceStrategies.ts";
import { validateKrakenEw } from "./krakenEw.ts";
import type { BeeMode } from "./simulation.ts";

/** Frozen birth profile: only these bounded source parameters change, never capital safety or execution policy. */
export const KRAKEN_INHERITABLE_FIELDS = ["channelLookback", "ttpTrailPct", "partialTpPct", "partialClosePct",
  "ctTp1Pct", "ctTp1ClosePct", "waitMinSlPct", "waitMaxSlPct", "swingLookback", "hardSlPct",
  "tfTp1Pct", "tfTp1ClosePct", "tfTp2Pct", "tfMinSlPct", "tfMaxSlPct"] as const;

export function validateEvolvedKrakenProfile(profile: KrakenProfile): void {
  if (profile.ew) validateKrakenEw(profile.ew);
  if (!profile || typeof profile.id !== "string" || !profile.id.trim() || profile.id.length > 180
    || !["CT", "TF"].includes(profile.mode) || profile.timeframeMinutes !== 5 || (!Number.isFinite(profile.renkoBoxUsd) || profile.renkoBoxUsd <= 0 || profile.renkoBoxUsd > 100000)
    || profile.paperOverrides?.maxNotionalMultiple !== 1
    || !Number.isFinite(profile.paperOverrides.mandatoryHardSlPct)
    || profile.paperOverrides.mandatoryHardSlPct <= 0 || profile.paperOverrides.mandatoryHardSlPct > 2.5
    || !Number.isFinite(profile.paperOverrides.slippageBps) || profile.paperOverrides.slippageBps < 0
    || !Number.isFinite(profile.commissionBps) || profile.commissionBps < 0) throw new RangeError("Unsafe evolved Kraken execution profile.");
  for (const key of KRAKEN_INHERITABLE_FIELDS) if (!Number.isFinite(profile[key])) throw new RangeError("Nonfinite Kraken gene.");
  if (!Number.isInteger(profile.channelLookback) || profile.channelLookback < 20 || profile.channelLookback > 500
    || !Number.isInteger(profile.swingLookback) || profile.swingLookback < 5 || profile.swingLookback > 50
    || profile.ttpTrailPct < .1 || profile.ttpTrailPct > 2.5
    || profile.partialTpPct < .5 || profile.partialTpPct > 6 || profile.partialClosePct < 10 || profile.partialClosePct > 100
    || profile.ctTp1Pct < 0 || profile.ctTp1Pct > 6 || profile.ctTp1ClosePct < 0 || profile.ctTp1ClosePct > 100
    || profile.waitMinSlPct < .5 || profile.waitMinSlPct > profile.waitMaxSlPct || profile.waitMaxSlPct > 5
    || profile.hardSlPct < .25 || profile.hardSlPct > 2.5
    || profile.tfTp1Pct < .5 || profile.tfTp1Pct > profile.tfTp2Pct || profile.tfTp2Pct > 10
    || profile.tfTp1ClosePct < 5 || profile.tfTp1ClosePct > 95
    || profile.tfMinSlPct < .5 || profile.tfMinSlPct > profile.tfMaxSlPct || profile.tfMaxSlPct > 8) {
    throw new RangeError("Evolved Kraken genes exceed birth bounds.");
  }
}

/** Deterministic bounded inheritance from actual parents (Local), source library (Hive), or both (Hybrid).
 * Call once at birth with parents known then; persist the returned profile with the child. Never re-resolve
 * from future performance or rerun on a later date. EW modules remain separate source genes, not an implied
 * live EW execution path in the Kraken state machine.
 */
export function inheritKrakenProfile(mode: BeeMode, parents: readonly KrakenProfile[], seed: string, childId: string, ewGenome?: SourceGenome): KrakenProfile {
  if (!["random", "local", "hive", "hybrid"].includes(mode) || !parents.length || typeof seed !== "string"
    || !childId?.trim() || childId.length > 140) throw new RangeError("Invalid frozen Kraken birth request.");
  let state = 2166136261;
  for (const char of `${seed}:${mode}:${childId}`) state = Math.imul(state ^ char.charCodeAt(0), 16777619) >>> 0;
  const random = () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296; };
  const donors = mode === "local" ? parents : mode === "hybrid" ? [...parents, ...KRAKEN_STARTER_PROFILES] : KRAKEN_STARTER_PROFILES;
  const base = donors[Math.floor(random() * donors.length)]!;
  const gene = (key: typeof KRAKEN_INHERITABLE_FIELDS[number], low: number, high: number, integer = false) => {
    const mean = donors.reduce((sum, parent) => sum + parent[key], 0) / donors.length;
    if (!Number.isFinite(mean)) throw new RangeError("Parent profile has nonfinite genes.");
    const value = mode === "random" ? low + random() * (high - low) : mean * (1 + (random() - .5) * .24);
    const clamped = Math.max(low, Math.min(high, value));
    return integer ? Math.round(clamped) : Math.round(clamped * 10000) / 10000;
  };
  const waitMin = gene("waitMinSlPct", .5, 3), tfMin = gene("tfMinSlPct", .5, 3), tp1 = gene("tfTp1Pct", .5, 6);
  const child: KrakenProfile = { ...base, id: `evolved:${childId}`, name: `${mode === "local" ? "Lokal" : mode === "hive" ? "Hive" : mode === "hybrid" ? "Hybrid" : "Zufall"} · ${childId}`,
    sourceFiles: [...new Set(donors.flatMap((parent) => parent.sourceFiles))],
    channelLookback: gene("channelLookback", 20, 500, true), ttpTrailPct: gene("ttpTrailPct", .1, 2.5),
    partialTpPct: gene("partialTpPct", .5, 6), partialClosePct: gene("partialClosePct", 10, 100),
    ctTp1Pct: base.ctTp1Pct === 0 ? 0 : gene("ctTp1Pct", .5, 6),
    ctTp1ClosePct: base.ctTp1ClosePct === 0 ? 0 : gene("ctTp1ClosePct", 10, 100),
    waitMinSlPct: waitMin, waitMaxSlPct: Math.max(waitMin, gene("waitMaxSlPct", .5, 5)),
    swingLookback: gene("swingLookback", 5, 50, true), hardSlPct: gene("hardSlPct", .25, 2.5),
    tfTp1Pct: tp1, tfTp2Pct: Math.max(tp1, gene("tfTp2Pct", .5, 10)), tfTp1ClosePct: gene("tfTp1ClosePct", 5, 95),
    tfMinSlPct: tfMin, tfMaxSlPct: Math.max(tfMin, gene("tfMaxSlPct", .5, 8)),
    // Every child keeps a hard stop, even if its legacy parent had none enabled in source.
    paperOverrides: { ...base.paperOverrides, maxNotionalMultiple: 1, mandatoryHardSlPct: 2.5 },
  };
  // Reflect the effective inherited lookback; retain exact original settings only on named starter profiles.
  const result: KrakenProfile = { ...child, originalSensitivity: child.channelLookback / 10, originalMultiplier: true };
  const frozen = ewGenome ? attachKrakenEwGenes(result, ewGenome) : result;
  validateEvolvedKrakenProfile(frozen); return frozen;
}

/** Attach only confirmed, single-timeframe executable EW modules from the frozen birth genome. */
export function attachKrakenEwGenes(profile: KrakenProfile, genome: SourceGenome): KrakenProfile {
  validateSourceGenome(genome);
  const ew = { filter: genome.ewFilter, exit: genome.ewExit, pivotLength: genome.pivotLength,
    atrLength: genome.atrLength, minMoveAtr: genome.ewMinMoveAtr, retraceMin: genome.ewRetraceMin,
    retraceMax: genome.ewRetraceMax };
  validateKrakenEw(ew);
  return { ...profile, ew };
}
