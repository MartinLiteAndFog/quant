/** Whitelisted strategy settings only. No webhook, account, token or alert payload fields. */
export type KrakenEwGenes = Readonly<{ filter: boolean; exit: boolean; pivotLength: number; atrLength: number; minMoveAtr: number; retraceMin: number; retraceMax: number }>;
export type KrakenProfile = Readonly<{
  ew?: KrakenEwGenes;
  id: string; name: string; mode: "CT" | "TF"; sourceFiles: readonly string[];
  originalSensitivity: number; originalMultiplier: boolean; channelLookback: number;
  timeframeMinutes: 5; renkoBoxUsd: number; bootstrap: boolean;
  ttpTrailPct: number; partialTpPct: number; partialClosePct: number;
  ctTp1Pct: number; ctTp1ClosePct: number; waitMinSlPct: number; waitMaxSlPct: number;
  swingLookback: number; reverseWaitSl: boolean; hardSlPct: number; reverseHardSl: boolean;
  armTtpAfterSlReverse: boolean; tfTp1Pct: number; tfTp1ClosePct: number; tfTp2Pct: number;
  tfMinSlPct: number; tfMaxSlPct: number; breakEvenAfterTp1: boolean; flipTrendOnOpposite: boolean;
  sourceSizing: Readonly<{ baseEquityPct: number; leverage: number; adaptiveKelly: boolean }>;
  sourceRecentWindowDays: number; commissionBps: number;
  paperOverrides: Readonly<{ maxNotionalMultiple: 1; mandatoryHardSlPct: number; slippageBps: number;
    archiveWindow: "full_archive"; signalPolicy: "confirmed_5m_close_renko_proxy" }>;
}>;
const source = "tradingview/sol_imba_strategy.pine";
const legacy: KrakenProfile = {
  id: "kraken_legacy", name: "Kraken Legacy · TTP", mode: "CT",
  sourceFiles: [source, "tradingview/imba_multibot_presets.json"],
  originalSensitivity: 20, originalMultiplier: true, channelLookback: 200,
  timeframeMinutes: 5, renkoBoxUsd: .1, bootstrap: true,
  ttpTrailPct: 1.2, partialTpPct: 3.5, partialClosePct: 90,
  ctTp1Pct: 0, ctTp1ClosePct: 0, waitMinSlPct: 1, waitMaxSlPct: 3,
  swingLookback: 50, reverseWaitSl: true, hardSlPct: 0, reverseHardSl: false,
  armTtpAfterSlReverse: false, tfTp1Pct: 3.2, tfTp1ClosePct: 10, tfTp2Pct: 6,
  tfMinSlPct: 3, tfMaxSlPct: 8, breakEvenAfterTp1: true, flipTrendOnOpposite: true,
  sourceSizing: { baseEquityPct: 95, leverage: 10, adaptiveKelly: true },
  sourceRecentWindowDays: 38, commissionBps: 6,
  paperOverrides: { maxNotionalMultiple: 1, mandatoryHardSlPct: 2.5, slippageBps: 2,
    archiveWindow: "full_archive", signalPolicy: "confirmed_5m_close_renko_proxy" },
};
const latest: KrakenProfile = {
  ...legacy, id: "kraken_17_2_close", name: "Kraken 17.2 · enger TTP · SL schließt",
  sourceFiles: [source, "Downloads/COLLECTION_KRAKEN_SOLUSD.PM_2026-11-01.xlsx"],
  originalSensitivity: 17.2, channelLookback: 172, ttpTrailPct: .5,
  partialTpPct: 2.8, partialClosePct: 50, ctTp1Pct: 3.15, ctTp1ClosePct: 95,
  waitMinSlPct: 1.3, waitMaxSlPct: 2, reverseWaitSl: false, hardSlPct: 2.5,
  tfTp1Pct: 4, tfTp1ClosePct: 50, tfTp2Pct: 5, tfMinSlPct: 1, tfMaxSlPct: 1.5,
  breakEvenAfterTp1: false, sourceSizing: { baseEquityPct: 90, leverage: 10, adaptiveKelly: false },
  sourceRecentWindowDays: 30,
};
export const KRAKEN_STARTER_PROFILES: readonly KrakenProfile[] = [
  legacy,
  { ...legacy, id: "countervariante", name: "Countervariante · TTP" },
  latest,
  { ...latest, id: "kraken_17_2_reverse", name: "Kraken 17.2 · enger TTP · SL dreht",
    reverseWaitSl: true, sourceFiles: [source, "Downloads/COLLECTION_KRAKEN_SOLUSD.PM_2026-11-01 (1).xlsx"] },
  { ...legacy, id: "imbatp", name: "Pure ImbaTP · TP1 / TP2", mode: "TF",
    originalSensitivity: 5, channelLookback: 50 },
];
export function getKrakenProfile(id: string): KrakenProfile {
  const profile = KRAKEN_STARTER_PROFILES.find((candidate) => candidate.id === id);
  if (!profile) throw new RangeError(`Unknown Kraken source profile: ${id}`);
  return profile;
}
