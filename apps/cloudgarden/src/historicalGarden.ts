import { normalizeMarketData, runPaperReplay, type MarketBarInput, type MarketDataset, type PaperFill } from "./paperReplay.ts";
import { advanceSimulationDay, assertSimulationInvariants, bookClosedPaperReplay, bookHistoricalMark, fundSimulationWorld, createSimulationWorld, setEcologyControls, GENTLE_ECOLOGY, type SimulationWorld, type GardenClimate } from "./simulation.ts";
import { generateSourceSignals, inheritSourceGenome, SOURCE_PARENT_GENOMES, validateSourceGenome, type SourceGenome } from "./sourceStrategies.ts";

import { createKrakenPaperState, advanceKrakenPaper, resumeKrakenPaperAfterGap, validateKrakenPaperState, type KrakenPaperState, type KrakenPaperFill } from "./krakenPaper.ts";
import { attachKrakenEwGenes, inheritKrakenProfile, validateEvolvedKrakenProfile } from "./krakenEvolution.ts";
import { getKrakenProfile, type KrakenProfile } from "./krakenProfiles.ts";
import { DEFAULT_FUNDING_SETTINGS, topUpWorldAfterDay, validateFundingSettings, type FundingMode, type FundingSettings } from "./fundingPolicy.ts";

export type ArchiveDay = Readonly<{ date: string; barCount: number; gapCount: number; missingMinutes: number; file: string; sha256: string }>;
export type SolArchive = Readonly<{
  schemaVersion: 1; source: string; sourceSha256: string; symbol: string; venue: string;
  firstDate: string; lastDate: string; totalBars: number; missingMinutes: number; duplicateCount: number;
  days: readonly ArchiveDay[];
}>;
export type PlantTradingRecord = Readonly<{ fills: number; netPnl: number; fees: number; rejected: number; opportunities: number; observedDays?: number; equityPeak?: number; maxDrawdownPct?: number; realizedPnl?: number; unrealizedPnl?: number; equityHistory?: readonly Readonly<{ day: number; equity: number; netPnl: number }>[]; fillReasons?: Readonly<Record<string, number>> }>;
export type HistoricalTrade = PaperFill & Readonly<{ plantId: string; plantName: string; family: string; reason?: string }>;
export type ArchiveLoop = Readonly<{
  cycle: number; archiveIndex: number; totalDays: number; totalBars: number; totalFills: number; totalFees: number; capitalHalted: boolean;
  roundStartDay?: number; roundStartNav?: number; roundStartFees?: number; roundStartFills?: number;
  completedRounds: readonly Readonly<{ cycle: number; nav: number; fees: number; fills: number; reason: string }>[];
}>;
export type HistoricalSession = Readonly<{
  schemaVersion: 1; sourceSha256: string; startDate: string; nextDate: string; world: SimulationWorld;
  loop: ArchiveLoop;
  funding?: HistoricalFunding;
  retiredRecords?: PlantTradingRecord;
  retiredPlantCount?: number;
  initialProfileIds?: readonly string[];
  krakenStates?: Readonly<Record<string, KrakenPaperState>>;
  krakenProfiles?: Readonly<Record<string, KrakenProfile>>;
  genomes: Readonly<Record<string, SourceGenome>>;
  records: Readonly<Record<string, PlantTradingRecord>>;
  history: readonly Readonly<{ date: string; nav: number; fills: number; close: number; observedBars?: number; skippedBars?: number; segments?: number; deposits?: number; netPnl?: number }>[];
  trades: readonly HistoricalTrade[];
}>;
/** Old snapshots without a mode retain their historical monthly setting. */
export type HistoricalFunding = Readonly<{
  monthlyEur: number; eurUsdt: number; totalEur: number; totalUsdt: number; lastMonthKey: string;
  mode?: FundingMode; topUpTargetEur?: number; lastTopUpDay?: number;
}>;
export const HISTORICAL_SAVE_KEY = "cloudgarden.historical.v1";
const DAY = 86_400_000;
const emptyRecord = (): PlantTradingRecord => ({ fills: 0, netPnl: 0, fees: 0, rejected: 0, opportunities: 0 });
const rounded = (n: number) => Math.round(n * 100_000_000) / 100_000_000;

/** Legacy saved sessions with unspecified mode continue as monthly; new games default to top-up. */
export function fundingSettings(funding?: HistoricalFunding): FundingSettings {
  return validateFundingSettings(funding ? {
    mode: funding.mode ?? "monthly",
    monthlyEur: funding.monthlyEur,
    topUpTargetEur: funding.topUpTargetEur ?? 100,
    eurUsdt: funding.eurUsdt,
  } : DEFAULT_FUNDING_SETTINGS);
}

/** Change future contributions only: no retroactive funding or rewritten PnL. */
export function configureHistoricalFunding(session: HistoricalSession, settings: FundingSettings): HistoricalSession {
  if (!session.funding) return session;
  const valid = validateFundingSettings(settings);
  const previous = fundingSettings(session.funding);
  return {
    ...session,
    funding: {
      ...session.funding, ...valid,
      lastMonthKey: valid.mode === "monthly" && previous.mode !== "monthly"
        ? `${session.loop.cycle}:${session.nextDate.slice(0, 7)}`
        : session.funding.lastMonthKey,
    },
  };
}

/** Exactly one end-of-observed-day top-up, after the account is marked. */
export function applyEndOfDayFunding(session: HistoricalSession): HistoricalSession {
  const funding = session.funding;
  if (!funding || fundingSettings(funding).mode !== "topup") return session;
  if (funding.lastTopUpDay !== undefined && funding.lastTopUpDay >= session.world.day) return session;
  const applied = topUpWorldAfterDay(session.world, fundingSettings(funding));
  return {
    ...session,
    world: applied.world,
    funding: {
      ...funding,
      totalEur: rounded(funding.totalEur + applied.depositEur),
      totalUsdt: rounded(funding.totalUsdt + applied.depositUsdt),
      lastTopUpDay: session.world.day,
    },
    ...(applied.depositUsdt > 0 ? { loop: { ...session.loop, capitalHalted: false } } : {}),
  };
}

export function nextUtcDate(date: string, offset = 1): string {
  return new Date(Date.parse(`${date}T00:00:00Z`) + offset * DAY).toISOString().slice(0, 10);
}

export function validateSolArchive(value: unknown): SolArchive {
  const a = value as SolArchive;
  if (!a || a.schemaVersion !== 1 || !a.source?.trim() || !["SOL-USDT", "BTC-USDT"].includes(a.symbol)
    || !/^[a-f0-9]{64}$/.test(a.sourceSha256) || !Array.isArray(a.days) || !a.days.length
    || !Number.isInteger(a.totalBars) || a.totalBars <= 0 || a.duplicateCount !== 0
    || !Number.isInteger(a.missingMinutes) || a.missingMinutes < 0) {
    throw new Error("The local SOL archive manifest is invalid.");
  }
  const seen = new Set<string>();
  let count = 0;
  for (const d of a.days) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(d.date) || !Number.isFinite(Date.parse(`${d.date}T00:00:00Z`))
      || d.file !== `${d.date}.json` || seen.has(d.date) || !/^[a-f0-9]{64}$/.test(d.sha256)
      || !Number.isInteger(d.barCount) || d.barCount < 1 || d.barCount > 1440
      || !Number.isInteger(d.missingMinutes) || d.missingMinutes < 0
      || !Number.isInteger(d.gapCount) || d.gapCount < 0) throw new Error("An archive day failed its contract.");
    if (seen.size && d.date <= a.days[seen.size - 1]!.date) throw new Error("Archive days must be ordered.");
    seen.add(d.date); count += d.barCount;
  }
  if (count !== a.totalBars || a.firstDate !== a.days[0]!.date || a.lastDate !== a.days.at(-1)!.date) {
    throw new Error("Archive totals and date bounds do not reconcile.");
  }
  return a;
}

export function createHistoricalSession(archive: SolArchive, startDate = archive.firstDate, options: {
  engine?: "source" | "kraken"; profileIds?: readonly string[];
  monthlyEur?: number; eurUsdt?: number; fundingMode?: FundingMode; topUpTargetEur?: number;
} = {}): HistoricalSession {
  const archiveIndex = archive.days.findIndex((day) => day.date === startDate);
  if (archiveIndex < 0) throw new Error("Choose an observed UTC date from the SOL archive.");
  // Game default: no predators and no negative-vitality penalty. Old non-game worlds retain their own rules.
  const world = setEcologyControls(createSimulationWorld(`${archive.symbol}:${archive.sourceSha256.slice(0, 12)}:${startDate}`, { dynamicEcology: options.engine !== "source", maxLivingPlants: 50 }), GENTLE_ECOLOGY);
  const genomes = Object.fromEntries(world.plants.map((plant, i) => [plant.id, SOURCE_PARENT_GENOMES[i]!])) as Record<string, SourceGenome>;
  const profileIds = options.profileIds ?? ["kraken_legacy", "kraken_17_2_close", "kraken_17_2_reverse"];
  if (profileIds.length !== 3) throw new RangeError("Choose exactly three source profiles for the three parent plants.");
  for (const id of profileIds) getKrakenProfile(id);
  const fundingConfig = validateFundingSettings({
    ...DEFAULT_FUNDING_SETTINGS,
    mode: options.fundingMode ?? (options.monthlyEur !== undefined ? "monthly" : "topup"),
    monthlyEur: options.monthlyEur ?? DEFAULT_FUNDING_SETTINGS.monthlyEur,
    topUpTargetEur: options.topUpTargetEur ?? DEFAULT_FUNDING_SETTINGS.topUpTargetEur,
    eurUsdt: options.eurUsdt ?? DEFAULT_FUNDING_SETTINGS.eurUsdt,
  });
  return { schemaVersion: 1, sourceSha256: archive.sourceSha256, startDate, nextDate: startDate, world,
    ...(options.engine === "source" ? {} : { funding: { ...fundingConfig, totalEur: 0, totalUsdt: 0, lastMonthKey: `1:${startDate.slice(0, 7)}` }, initialProfileIds: [...profileIds],
      krakenProfiles: Object.fromEntries(world.plants.map((plant, i) => [plant.id, attachKrakenEwGenes(marketProfile(getKrakenProfile(profileIds[i]!), archive.symbol), genomes[plant.id]!)])), krakenStates: Object.fromEntries(world.plants.map((plant, i) =>
      [plant.id, createKrakenPaperState(profileIds[i]!, plant.allocationUsdt)])) }),
    loop: { cycle: 1, archiveIndex, totalDays: 0, totalBars: 0, totalFills: 0, totalFees: 0, capitalHalted: false, completedRounds: [], roundStartDay: 0, roundStartNav: 100, roundStartFees: 0, roundStartFills: 0 },
    genomes, records: {}, history: [], trades: [] };
}

/** Validate snapshots before using saved accounting or inherited strategy modules. */
export function validateHistoricalSession(value: unknown, archive: SolArchive): HistoricalSession {
  let s = value as HistoricalSession;
  let legacyEnded = false;
  if (!s || s.schemaVersion !== 1 || s.sourceSha256 !== archive.sourceSha256 || !s.genomes || !s.records
    || !Array.isArray(s.history) || !Array.isArray(s.trades) || s.trades.length > 400
    || !archive.days.some((d) => d.date === s.startDate)) throw new Error("Saved historical session is invalid.");
  assertSimulationInvariants(s.world);
  // Older v1 snapshots ended at most one linear archive pass. Migrate their counters without rerunning fills.
  if (!s.loop) {
    const startIndex = archive.days.findIndex((d) => d.date === s.startDate);
    const expected = archive.days[startIndex + s.world.day]?.date ?? nextUtcDate(archive.lastDate);
    if (s.nextDate !== expected || s.history.length !== s.world.day) throw new Error("Saved historical session is invalid.");
    s = { ...s, loop: { cycle: 1, archiveIndex: startIndex + s.world.day, totalDays: s.world.day,
      totalBars: archive.days.slice(startIndex, startIndex + s.world.day).reduce((sum, d) => sum + d.barCount, 0),
      totalFills: Object.values(s.records).reduce((sum, r) => sum + r.fills, 0), totalFees: s.world.capital.paperFeesUsdt,
      capitalHalted: false, completedRounds: [] } };
    legacyEnded = s.loop.archiveIndex === archive.days.length;
  }
  const loop = s.loop;
  if (![loop.cycle, loop.archiveIndex, loop.totalDays, loop.totalBars, loop.totalFills].every((n) => Number.isSafeInteger(n) && n >= 0)
    || loop.cycle < 1 || loop.archiveIndex > archive.days.length || (loop.archiveIndex === archive.days.length && !legacyEnded) || !Number.isFinite(loop.totalFees) || loop.totalFees < 0
    || typeof loop.capitalHalted !== "boolean" || !Array.isArray(loop.completedRounds) || loop.completedRounds.length > 64
    || s.nextDate !== (legacyEnded ? nextUtcDate(archive.lastDate) : archive.days[loop.archiveIndex]!.date)) throw new Error("Invalid archive loop counters.");
  if (s.history.length !== Math.min(s.world.day, 300) || (!s.world.dynamicEcology && s.world.day > archive.days.length) || loop.totalDays < s.world.day) {
    throw new Error("Historical clock and history differ.");
  }
  const firstHistoryIndex = loop.archiveIndex - s.history.length;
  for (const [i, row] of s.history.entries()) {
    const day = archive.days[s.world.dynamicEcology ? ((firstHistoryIndex + i) % archive.days.length + archive.days.length) % archive.days.length : firstHistoryIndex + i];
    if (!day || row.date !== day.date || ![row.nav, row.close].every((n) => Number.isFinite(n) && n > 0)
      || !Number.isInteger(row.fills) || row.fills < 0
      || (row.observedBars !== undefined && row.observedBars !== day.barCount)
      || (row.skippedBars !== undefined && (!Number.isInteger(row.skippedBars) || row.skippedBars < 0 || row.skippedBars > day.barCount))) {
      throw new Error("Invalid historical daily record.");
    }
  }
  if (s.history.length && Math.abs(s.history.at(-1)!.nav - s.world.capital.navUsdt) > .0001) throw new Error("Saved NAV does not reconcile.");
  for (const round of loop.completedRounds) {
    if (!Number.isSafeInteger(round.cycle) || round.cycle < 1 || round.cycle >= loop.cycle
      || !Number.isFinite(round.nav) || round.nav <= 0 || !Number.isFinite(round.fees) || round.fees < 0
      || !Number.isSafeInteger(round.fills) || round.fills < 0 || !["archive_end", "capital_floor"].includes(round.reason)) throw new Error("Invalid evaluation round summary.");
  }
  for (const plant of s.world.plants) {
    validateSourceGenome(s.genomes[plant.id]!);
    if (s.krakenStates) {
      const state = s.krakenStates[plant.id];
      if (!state) throw new Error("Missing persistent Kraken plant state.");
      const profile = s.krakenProfiles?.[plant.id] ?? getKrakenProfile(state.profileId);
      if (profile.id.startsWith("evolved:")) validateEvolvedKrakenProfile(profile);
      else if (JSON.stringify(profile) !== JSON.stringify(attachKrakenEwGenes(marketProfile(getKrakenProfile(profile.id), archive.symbol), s.genomes[plant.id]!))) {
        // Pre-EW v1 sessions may keep canonical parents without an EW attachment.
        const canonical = getKrakenProfile(profile.id);
        const oldBtcScale = archive.symbol === "BTC-USDT" && [.1, 100].includes(profile.renkoBoxUsd)
          && JSON.stringify(profile) === JSON.stringify({ ...attachKrakenEwGenes(canonical, s.genomes[plant.id]!), renkoBoxUsd: profile.renkoBoxUsd });
        if (!oldBtcScale && JSON.stringify(profile) !== JSON.stringify(canonical)) throw new Error("Saved parent profile differs from its frozen source genes.");
      }
      validateKrakenPaperState(state, profile);
      if (plant.status === "archived" && state.quantity !== 0) throw new Error("An open position cannot belong to an archived plant.");
    }
  }
  if (s.krakenStates && (!s.initialProfileIds || s.initialProfileIds.length !== 3)) throw new Error("Saved Kraken parent profile selection is missing.");
  if (s.initialProfileIds) for (const id of s.initialProfileIds) getKrakenProfile(id);
  for (const [id, record] of [...Object.entries(s.records), ...(s.retiredRecords ? [["$retired", s.retiredRecords] as const] : [])]) {
    if (id !== "$retired" && !s.world.plants.some((p) => p.id === id)) throw new Error("Unknown plant in trading records.");
    if ([record.equityPeak, record.maxDrawdownPct, record.realizedPnl, record.unrealizedPnl].some((n) => n !== undefined && !Number.isFinite(n))
      || (record.observedDays !== undefined && (!Number.isSafeInteger(record.observedDays) || record.observedDays < 0))) throw new Error("Invalid cumulative plant statistics.");
    if (record.fillReasons && Object.values(record.fillReasons).some((count) => !Number.isSafeInteger(count) || count < 0)) throw new Error("Invalid fill reason counters.");
    if (record.equityHistory && (!Array.isArray(record.equityHistory) || record.equityHistory.length > 300
      || record.equityHistory.some((row, i) => !Number.isSafeInteger(row.day) || row.day < 1 || row.day > s.world.day
        || ![row.equity, row.netPnl].every(Number.isFinite) || (i > 0 && row.day <= record.equityHistory![i - 1]!.day)))) throw new Error("Invalid plant equity history.");
    if (!record || ![record.netPnl, record.fees].every(Number.isFinite) || record.fees < 0
      || ![record.fills, record.rejected, record.opportunities].every((n) => Number.isInteger(n) && n >= 0)) throw new Error("Invalid saved trading record.");
  }
  const records = [...Object.values(s.records), ...(s.retiredRecords ? [s.retiredRecords] : [])];
  if (Math.abs(records.reduce((sum, r) => sum + r.netPnl, 0) - (s.world.capital.navUsdt - 100 - (s.world.capital.externalDepositsUsdt ?? 0))) > .0001
    || Math.abs(records.reduce((sum, r) => sum + r.fees, 0) - s.world.capital.paperFeesUsdt) > .0001
    || (s.world.day <= 300 && records.reduce((sum, r) => sum + r.fills, 0) !== s.history.reduce((sum, row) => sum + row.fills, 0))
    || loop.totalFills < records.reduce((sum, r) => sum + r.fills, 0) || loop.totalFees + .0001 < s.world.capital.paperFeesUsdt) {
    throw new Error("Plant trading records do not reconcile with historical NAV, fees or fills.");
  }
  if (s.funding && (![s.funding.monthlyEur, s.funding.eurUsdt, s.funding.totalEur, s.funding.totalUsdt].every(Number.isFinite)
    || s.funding.monthlyEur <= 0 || s.funding.monthlyEur > 10_000 || s.funding.eurUsdt <= 0 || s.funding.eurUsdt > 100
    || s.funding.totalEur < 0 || s.funding.totalUsdt < 0 || typeof s.funding.lastMonthKey !== "string" || !/^[1-9]\d*:\d{4}-(?:0[1-9]|1[0-2])$/.test(s.funding.lastMonthKey)
    || (s.funding.mode !== undefined && !["off", "monthly", "topup"].includes(s.funding.mode))
    || (s.funding.topUpTargetEur !== undefined && (!Number.isFinite(s.funding.topUpTargetEur) || s.funding.topUpTargetEur <= 0 || s.funding.topUpTargetEur > 10_000))
    || (s.funding.lastTopUpDay !== undefined && (!Number.isSafeInteger(s.funding.lastTopUpDay) || s.funding.lastTopUpDay < 0 || s.funding.lastTopUpDay > s.world.day))
    || Math.abs(s.funding.totalUsdt - (s.world.capital.externalDepositsUsdt ?? 0)) > .0001)) throw new Error("Invalid external contribution ledger.");
  const retainedKeys = new Set<string>();
  for (const fill of s.trades) {
    const decimal = (v: unknown) => typeof v === "string" && /^-?(?:0|[1-9]\d*)(?:\.\d{1,8})?$/.test(v) && Number.isFinite(Number(v));
    if (!fill || typeof fill.plantId !== "string" || !s.world.plants.some((p) => p.id === fill.plantId)
      || typeof fill.plantName !== "string" || fill.plantName.length > 200
      || fill.family !== (s.krakenStates?.[fill.plantId]?.profileId ?? s.genomes[fill.plantId]!.family)
      || (fill.reason !== undefined && (typeof fill.reason !== "string" || fill.reason.length > 100))
      || typeof fill.orderId !== "string" || !fill.orderId.trim()
      || fill.strategyId !== `${fill.plantId}:${fill.family}:${s.krakenStates ? "kraken-v1" : "source-v1"}`
      || (fill.side !== "buy" && fill.side !== "sell") || typeof fill.executedAt !== "string"
      || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z$/.test(fill.executedAt)
      || !Number.isFinite(Date.parse(fill.executedAt))
      || Date.parse(fill.executedAt) < Date.parse(`${s.world.dynamicEcology ? archive.firstDate : s.startDate}T00:00:00Z`)
      || Date.parse(fill.executedAt) >= Date.parse(`${s.world.dynamicEcology ? nextUtcDate(archive.lastDate) : s.nextDate}T00:00:00Z`)
      || ![fill.targetQuantity, fill.filledQuantity, fill.referencePrice, fill.price, fill.notional, fill.fee, fill.executionCost].every(decimal)
      || [fill.filledQuantity, fill.referencePrice, fill.price].some((v) => Number(v) <= 0)
      || [fill.notional, fill.fee, fill.executionCost].some((v) => Number(v) < 0)
      || !/^[a-f0-9]{64}$/.test(fill.datasetFingerprint)) throw new Error("Invalid saved historical fill.");
    const key = `${fill.plantId}:${fill.orderId}:${fill.executedAt}`;
    if (retainedKeys.has(key)) throw new Error("Duplicated saved historical fill.");
    retainedKeys.add(key);
  }
  if (s.trades.length > records.reduce((sum, r) => sum + r.fills, 0)) throw new Error("Saved fill history exceeds its ledger count.");
  return legacyEnded ? validateHistoricalSession(wrapHistoricalRound(s, archive), archive) : s;
}

export async function loadSolArchive(symbol: "SOL-USDT" | "BTC-USDT" = "SOL-USDT"): Promise<SolArchive> {
  const base = symbol === "BTC-USDT" ? "/market-btc" : "/market";
  const response = await fetch(`${base}/manifest.json`, { cache: "no-cache" });
  if (!response.ok) throw new Error("SOL archive is missing. Run the local history exporter first.");
  const archive = validateSolArchive(await response.json());
  if (archive.symbol !== symbol) throw new Error("Requested archive symbol does not match manifest.");
  return archive;
}

/** Hash the exact local export bytes; no random data, filling or price replacement. */
export async function loadArchiveBars(day: ArchiveDay, symbol: string = "SOL-USDT"): Promise<readonly MarketBarInput[]> {
  const response = await fetch(`${symbol === "BTC-USDT" ? "/market-btc" : "/market"}/${day.file}`);
  if (!response.ok) throw new Error(`SOL data unavailable for ${day.date}.`);
  const bytes = await response.arrayBuffer();
  const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))].map((n) => n.toString(16).padStart(2, "0")).join("");
  if (digest !== day.sha256) throw new Error(`SOL data checksum mismatch for ${day.date}.`);
  const rows: unknown = JSON.parse(new TextDecoder().decode(bytes));
  if (!Array.isArray(rows) || rows.length !== day.barCount) throw new Error("SOL day row count differs from its manifest.");
  return rows.map((row: unknown) => {
    if (!Array.isArray(row) || row.length !== 6 || !row.every((v) => typeof v === "string") || !row[0].startsWith(day.date)) {
      throw new Error("Invalid SOL archive row.");
    }
    const [timestamp, open, high, low, close, volume] = row as string[];
    return { timestamp: timestamp!, open: open!, high: high!, low: low!, close: close!, volume: volume! };
  });
}

export async function normalizeArchiveBars(rows: readonly MarketBarInput[], archive: SolArchive): Promise<MarketDataset> {
  return normalizeMarketData(rows, {
    source: `${archive.source}; source SHA256 ${archive.sourceSha256}; modeled 1x linear research position; venue/instrument unverified`,
    license: "unverified; local private use", symbol: archive.symbol, marketType: "linear_futures", contractMultiplier: "1",
    intervalSeconds: 60, synthetic: false, timezone: "UTC", timestampSemantics: "open",
  });
}

/** A data-derived visual mapping: actual volume, intraday ranges and daily direction drive ecology. */
export function observedSolClimate(dataset: MarketDataset): GardenClimate {
  const first = Number(dataset.bars[0]!.open), last = Number(dataset.bars.at(-1)!.close);
  const range = dataset.bars.reduce((sum, b) => sum + (Number(b.high) - Number(b.low)) / Number(b.open), 0) / dataset.bars.length;
  const volume = dataset.bars.reduce((sum, b) => sum + Number(b.volume), 0) / dataset.bars.length;
  const bound = (n: number) => Math.max(0, Math.min(1, n));
  return { water: rounded(bound(Math.log1p(volume) / 16)), sunlight: rounded(bound(.5 + (last / first - 1) * 4 - range * 8)),
    predatorPatrol: range >= .004 || last / first < .97 };
}

function wrapHistoricalRound(session: HistoricalSession, archive: SolArchive): HistoricalSession {
  const summary = { cycle: session.loop.cycle, nav: session.world.capital.navUsdt, fees: session.world.capital.paperFeesUsdt,
    fills: Object.values(session.records).reduce((sum, r) => sum + r.fills, 0), reason: session.loop.capitalHalted ? "capital_floor" : "archive_end" };
  if (session.krakenStates && Object.values(session.krakenStates).some((state) => state.quantity !== 0 || state.pending.length)) {
    throw new Error("Archive boundary still has outstanding Kraken exposure; fresh capital cannot replace an open account.");
  }
  if (session.world.dynamicEcology) {
    const fees = session.world.capital.paperFeesUsdt - (session.loop.roundStartFees ?? 0);
    const fills = session.loop.totalFills - (session.loop.roundStartFills ?? 0);
    const round = { ...summary, fees, fills };
    return { ...session, nextDate: archive.firstDate,
      krakenStates: session.krakenStates ? Object.fromEntries(Object.entries(session.krakenStates).map(([id, state]) => [id, resumeKrakenPaperAfterGap(state)])) : undefined,
      loop: { ...session.loop, cycle: session.loop.cycle + 1, archiveIndex: 0,
        roundStartDay: session.world.day, roundStartNav: session.world.capital.navUsdt, roundStartFees: session.world.capital.paperFeesUsdt,
        roundStartFills: session.loop.totalFills, completedRounds: [...session.loop.completedRounds, round].slice(-64) } };
  }
  const fresh = createHistoricalSession(archive, archive.firstDate,
    session.krakenStates ? { profileIds: session.initialProfileIds } : { engine: "source" });
  return { ...fresh, loop: { ...session.loop, cycle: session.loop.cycle + 1, archiveIndex: 0, capitalHalted: false,
    completedRounds: [...session.loop.completedRounds, summary].slice(-64) } };
}

/** Execute actual contiguous segments independently, close flat before every gap, then update ecology once. */
export async function executeHistoricalDay(session: HistoricalSession, archive: SolArchive, dataset: MarketDataset, warmup: MarketDataset): Promise<HistoricalSession> {
  session = validateHistoricalSession(session, archive);
  if (session.funding && fundingSettings(session.funding).mode === "monthly") {
    const key = `${session.loop.cycle}:${session.nextDate.slice(0, 7)}`;
    if (key !== session.funding.lastMonthKey) {
      const monthIndex = (date: string) => Number(date.slice(0, 4)) * 12 + Number(date.slice(5, 7)) - 1;
      const span = monthIndex(archive.lastDate) - monthIndex(archive.firstDate) + 1;
      const [previousCycle, previousMonth] = session.funding.lastMonthKey.split(":");
      const months = (session.loop.cycle - Number(previousCycle)) * span + monthIndex(session.nextDate) - monthIndex(previousMonth!);
      if (!Number.isSafeInteger(months) || months < 1) throw new Error("Monthly contribution clock moved backwards.");
      const euros = months * session.funding.monthlyEur;
      const amount = Math.round(euros * session.funding.eurUsdt * 10_000) / 10_000;
      session = { ...session, world: fundSimulationWorld(session.world, amount),
        funding: { ...session.funding, totalEur: rounded(session.funding.totalEur + euros),
          totalUsdt: rounded(session.funding.totalUsdt + amount), lastMonthKey: key }, loop: { ...session.loop, capitalHalted: false } };
    }
  }
  const day = archive.days[session.loop.archiveIndex]!;
  const start = Date.parse(`${session.nextDate}T00:00:00Z`);
  if (dataset.provenance.synthetic || dataset.report.synthetic || dataset.bars.length !== day.barCount
    || dataset.bars.some((bar) => bar.timestampMs < start || bar.timestampMs >= start + DAY)) {
    throw new Error("Historical execution requires exactly the observed real SOL day; gaps are never filled.");
  }
  const tail = warmup.bars.filter((bar) => bar.timestampMs >= start);
  if (warmup.provenance.synthetic || warmup.report.synthetic
    || warmup.provenance.source !== dataset.provenance.source
    || !dataset.provenance.source.includes(archive.sourceSha256)
    || warmup.report.endExclusive !== dataset.report.endExclusive || tail.length !== dataset.bars.length
    || tail.some((bar, i) => {
      const d = dataset.bars[i]!;
      return bar.timestamp !== d.timestamp || bar.open !== d.open || bar.high !== d.high || bar.low !== d.low
        || bar.close !== d.close || bar.volume !== d.volume;
    })) throw new Error("Signal warmup must contain the exact same historical day and source as execution.");
  if (session.krakenStates) return executeKrakenHistoricalDay(session, archive, dataset);
  const groups: (typeof dataset.bars[number])[][] = [];
  for (const bar of dataset.bars) {
    const last = groups.at(-1);
    if (!last || bar.timestampMs !== last.at(-1)!.timestampMs + 60_000) groups.push([bar]);
    else last.push(bar);
  }
  const tradableGroups = groups.filter((group) => group.length >= 3);
  const skippedBars = dataset.bars.length - tradableGroups.reduce((sum, group) => sum + group.length, 0);
  const segments = await Promise.all(tradableGroups.map(async (group) => {
    const segmentStart = group[0]!.timestampMs;
    const previous: typeof dataset.bars[number][] = [];
    // Only the uninterrupted preceding tail can warm up an indicator. A missing minute resets it.
    let expected = segmentStart - 60_000;
    for (let i = warmup.bars.findIndex((bar) => bar.timestampMs === segmentStart) - 1; i >= 0; i -= 1) {
      const bar = warmup.bars[i]!;
      if (bar.timestampMs !== expected) break;
      previous.unshift(bar); expected -= 60_000;
    }
    return { dataset: await normalizeArchiveBars(group, archive), warmup: await normalizeArchiveBars([...previous, ...group], archive) };
  }));
  let booked = { ...session.world, day: session.world.day + 1 };
  const records = { ...session.records };
  const trades = [...session.trades];
  let totalFills = 0;
  let capitalHalted = session.loop.capitalHalted;
  for (const segment of segments) {
    const dataset = segment.dataset, warmup = segment.warmup;
    const segmentWorld = booked;
    const segmentWasHalted = capitalHalted;
    const segmentStart = dataset.bars[0]!.timestampMs, segmentEnd = dataset.bars.at(-1)!.timestampMs + 60_000;
    for (const plant of session.world.plants.filter((p) => p.status === "active")) {
      const current = segmentWorld.plants.find((p) => p.id === plant.id)!;
      const headroom = segmentWorld.capital.navUsdt - segmentWorld.capital.reservedMarginUsdt / .9;
      if (headroom <= .05) capitalHalted = true;
      if (segmentWasHalted || headroom <= .05 || current.allocationUsdt < .001) continue;
      // A small share of headroom above protected entitlement capital bounds new exposure causally.
      const budget = Math.floor(Math.min(current.allocationUsdt, headroom * .2) * 10_000) / 10_000;
      if (budget < .001) { capitalHalted = true; continue; }
      const allocation = budget.toFixed(4);
      // Known session-open price only; future prices never size an earlier trade.
      const quantity = (Math.floor(budget * .5 / Number(dataset.bars[0]!.open) * 100_000_000) / 100_000_000).toFixed(8);
      if (Number(quantity) <= 0) continue;
      const strategyId = `${plant.id}:${session.genomes[plant.id]!.family}:source-v1`;
      const strategy = generateSourceSignals(warmup, session.genomes[plant.id]!, strategyId, quantity, segmentStart, segmentEnd);
      // A predeclared exit-only window retries flat intents on each later observed open.
      // Its boundaries use the fixed segment clock, never future volume or prices.
      const exitWindow = Math.min(60, Math.max(1, Math.floor(dataset.bars.length / 3)));
      const firstFlatIndex = Math.max(0, dataset.bars.length - exitWindow - 2);
      const flatAt = new Date(dataset.bars[firstFlatIndex]!.timestampMs + 60_000).toISOString();
      const flatSignals = dataset.bars.slice(firstFlatIndex, -2).map((bar, index) => {
        const at = new Date(bar.timestampMs + 60_000).toISOString();
        return { orderId: `${strategyId}:session-flat:${index}`, strategyId, targetQuantity: "0", signalAt: at, availableAt: at, receivedAt: at };
      });
      const signals = strategy.status === "trade" ? [...strategy.signals.filter((signal) => signal.signalAt < flatAt), ...flatSignals] : strategy.signals;
      const report = runPaperReplay({ dataset, signals, allocationLimit: allocation, startingCash: allocation, historicalResearch: true });
      if (!report.endedFlat) throw new Error("Historical exposure remains open; no settlement can be booked.");
      if (report.riskBreach) capitalHalted = true;
      const navBefore = booked.capital.navUsdt;
      const feesBefore = booked.capital.paperFeesUsdt;
      if (report.fills.length > 0 || strategy.status === "no_trade") booked = bookClosedPaperReplay(booked, plant.id, {
        startDay: session.world.day, days: 1, datasetFingerprint: report.dataset.fingerprint, synthetic: false,
        historicalResearch: true, riskSettlement: report.riskBreach, marketSource: dataset.provenance.source, endedFlat: report.endedFlat, riskBreach: report.riskBreach,
        fillCount: report.fills.length, signalStatus: strategy.status, strategyId,
        observations: strategy.observations, opportunities: strategy.opportunities, generatedSignals: strategy.generatedSignals,
        allocationLimitUsdt: Number(report.allocationLimit), startingCashUsdt: Number(report.startingCash), endingCashUsdt: Number(report.endingCash),
        maxMarginUsedUsdt: Number(report.maxMarginUsed), realizedPricePnlUsdt: Number(report.realizedPricePnl), feesUsdt: Number(report.fees),
        fundingUsdt: Number(report.funding), netPnlUsdt: Number(report.netPnl),
      });
      const before = records[plant.id] ?? emptyRecord();
      // Mirror the single ledger's four-decimal booking precision in plant totals.
      records[plant.id] = { fills: before.fills + report.fills.length, netPnl: rounded(before.netPnl + booked.capital.navUsdt - navBefore),
        fees: rounded(before.fees + booked.capital.paperFeesUsdt - feesBefore), rejected: before.rejected + report.rejectedSignals.length,
        opportunities: before.opportunities + strategy.opportunities };
      totalFills += report.fills.length;
      trades.push(...report.fills.map((fill) => ({ ...fill, plantId: plant.id, plantName: plant.name, family: session.genomes[plant.id]!.family })));
    }
  }
  // Intraday accounting precedes end-of-day growth and predation. Advance ecology exactly once.
  let world = advanceSimulationDay({ ...booked, day: session.world.day }, observedSolClimate(dataset));
  const genomes = { ...session.genomes };
  for (const plant of world.plants) {
    if (genomes[plant.id]) continue;
    const parents = plant.parents?.map((id) => genomes[id]!).filter(Boolean) ?? SOURCE_PARENT_GENOMES;
    genomes[plant.id] = inheritSourceGenome(plant.conceivedBy ?? "hive", parents.length ? parents : SOURCE_PARENT_GENOMES, world.seed, world.plants.indexOf(plant));
  }
  const funded = applyEndOfDayFunding({ ...session, world });
  world = funded.world;
  const history = [...session.history, { date: session.nextDate, nav: world.capital.navUsdt, fills: totalFills,
    close: Number(dataset.bars.at(-1)!.close), observedBars: dataset.bars.length, skippedBars, segments: groups.length }].slice(-300);
  const index = session.loop.archiveIndex + 1;
  const next: HistoricalSession = { ...session, funding: funded.funding, nextDate: archive.days[index]?.date ?? nextUtcDate(session.nextDate), world, genomes, records, history,
    loop: { ...session.loop, archiveIndex: index, totalDays: session.loop.totalDays + 1, totalBars: session.loop.totalBars + dataset.bars.length,
      totalFills: session.loop.totalFills + totalFills,
      totalFees: rounded(session.loop.totalFees + world.capital.paperFeesUsdt - session.world.capital.paperFeesUsdt), capitalHalted: funded.loop.capitalHalted },
    trades: trades.sort((a, b) => a.executedAt.localeCompare(b.executedAt) || a.orderId.localeCompare(b.orderId)).slice(-400) };
  return validateHistoricalSession(index === archive.days.length ? wrapHistoricalRound(compactHistoricalSession(next), archive) : compactHistoricalSession(next), archive);
}

function compactHistoricalSession(session: HistoricalSession): HistoricalSession {
  if (!session.world.dynamicEcology) return session;
  const archived = session.world.plants.filter((p) => p.status === "archived" && p.role !== "parent");
  const remove = new Set(archived.slice(0, Math.max(0, archived.length - 100)).map((p) => p.id));
  if (!remove.size) return session;
  const retired = { ...(session.retiredRecords ?? emptyRecord()) };
  for (const id of remove) {
    const record = session.records[id] ?? emptyRecord();
    for (const key of ["fills", "fees", "netPnl", "rejected", "opportunities"] as const) retired[key] = rounded(retired[key] + record[key]);
    const reasons = { ...retired.fillReasons };
    for (const [reason, count] of Object.entries(record.fillReasons ?? {})) reasons[reason] = (reasons[reason] ?? 0) + count;
    retired.fillReasons = reasons;
  }
  const keep = <T,>(map: Readonly<Record<string, T>>) => Object.fromEntries(Object.entries(map).filter(([id]) => !remove.has(id)));
  return { ...session, retiredRecords: retired, retiredPlantCount: (session.retiredPlantCount ?? 0) + remove.size,
    world: { ...session.world, plants: session.world.plants.filter((p) => !remove.has(p.id)),
      capital: { ...session.world.capital, seedlingEntitlements: session.world.capital.seedlingEntitlements.filter((slot) => !remove.has(slot.plantId ?? "")) } },
    genomes: keep(session.genomes), records: keep(session.records), krakenProfiles: session.krakenProfiles ? keep(session.krakenProfiles) : undefined, krakenStates: session.krakenStates ? keep(session.krakenStates) : undefined,
    trades: session.trades.filter((fill) => !remove.has(fill.plantId)) };
}

function krakenTrade(fill: KrakenPaperFill, plantId: string, plantName: string, quantity: number, dataset: MarketDataset): HistoricalTrade {
  const decimal = (n: number) => (Math.abs(n) < .000000005 ? 0 : n).toFixed(8);
  return { orderId: fill.id, plantId, plantName, family: fill.profileId, strategyId: `${plantId}:${fill.profileId}:kraken-v1`,
    side: fill.side, targetQuantity: decimal(quantity), filledQuantity: decimal(fill.quantity), referencePrice: decimal(fill.referencePrice),
    price: decimal(fill.price), notional: decimal(fill.quantity * fill.price), fee: decimal(fill.fee), executionCost: decimal(fill.executionCost),
    executedAt: fill.executedAt, reason: fill.reason, datasetFingerprint: dataset.fingerprint };
}

/** Preserve Kraken positions and next-open orders between dates. Equity marks never pretend to be closed trades. */
async function executeKrakenHistoricalDay(session: HistoricalSession, archive: SolArchive, dataset: MarketDataset): Promise<HistoricalSession> {
  const ranked = session.world.plants.filter((plant) => plant.status === "active" && (session.records[plant.id]?.observedDays ?? 0) >= 20)
    .sort((a, b) => {
      const score = (id: string) => { const record = session.records[id]!; return record.netPnl / Math.max(1, record.observedDays!) - (record.maxDrawdownPct ?? 0) * .001; };
      return score(b.id) - score(a.id) || a.id.localeCompare(b.id);
    }).slice(0, 8).map((plant) => plant.id);
  // Selection physiology uses only the previously completed trading record; today's prices cannot affect today's order sizing.
  const priorPlants = session.world.plants.map((plant) => {
    const record = session.records[plant.id];
    if (!record?.fills || (record.observedDays ?? 0) < 20) return plant;
    const scale = Math.max(1, record.equityPeak ?? plant.allocationUsdt);
    const score = Math.max(0, Math.min(100, 60 + record.netPnl / scale * 80 - (record.maxDrawdownPct ?? 0) * .3));
    return { ...plant, vigor: score, evidence: score };
  });
  let booked: SimulationWorld = { ...session.world, plants: priorPlants, day: session.world.day + 1, breedingCandidates: ranked };
  const profiles = { ...session.krakenProfiles };
  const states = { ...session.krakenStates! };
  const records = { ...session.records };
  const trades = [...session.trades];
  let capitalHalted = session.loop.capitalHalted;
  let totalFills = 0;
  const isArchiveEnd = session.loop.archiveIndex === archive.days.length - 1;
  const exitStart = dataset.bars.at(-1)!.timestampMs - 59 * 60_000;
  // All concurrent strategy accounts receive the same pre-day information set.
  // Booking an earlier array member's full-day result must never size a later member's earlier fills.
  const decisionWorld = session.world;
  const dayWasHalted = session.loop.capitalHalted;
  for (const plant of session.world.plants.filter((p) => p.status === "active")) {
    let state = states[plant.id]!;
    const before = state;
    const beforeMark = state.cashEquity + state.quantity * (state.lastMarkPrice - state.entryPrice);
    const profile = profiles[plant.id] ?? getKrakenProfile(state.profileId);
    const current = decisionWorld.plants.find((p) => p.id === plant.id)!;
    const headroom = decisionWorld.capital.navUsdt - decisionWorld.capital.reservedMarginUsdt / .9;
    if (headroom <= .05) capitalHalted = true;
    const previousRecord = session.records[plant.id];
    const evidenceWeight = (previousRecord?.observedDays ?? 0) < 20 ? 1 : Math.max(.25, Math.min(1,
      .75 + (previousRecord?.netPnl ?? 0) / Math.max(1, previousRecord?.equityPeak ?? current.allocationUsdt) * .25 - (previousRecord?.maxDrawdownPct ?? 0) / 400));
    const budget = Math.max(0, Math.min(current.allocationUsdt, headroom * .1)) * evidenceWeight;
    let cursor = 0;
    const fills: KrakenPaperFill[] = [];
    while (cursor < dataset.bars.length) {
      const first = dataset.bars[cursor]!;
      if (state.lastTimestampMs !== null && first.timestampMs !== state.lastTimestampMs + 60_000) state = resumeKrakenPaperAfterGap(state);
      const exitOnly = dayWasHalted || headroom <= .05 || (isArchiveEnd && first.timestampMs >= exitStart);
      let end = cursor + 1;
      while (end < dataset.bars.length && dataset.bars[end]!.timestampMs === dataset.bars[end - 1]!.timestampMs + 60_000
        && (dayWasHalted || headroom <= .05 || (isArchiveEnd && dataset.bars[end]!.timestampMs >= exitStart)) === exitOnly) end += 1;
      const result = advanceKrakenPaper(state, dataset.bars.slice(cursor, end), profile, budget, { exitOnly });
      state = result.state; fills.push(...result.fills); cursor = end;
    }
    states[plant.id] = state;
    const afterMark = state.cashEquity + state.quantity * (state.lastMarkPrice - state.entryPrice);
    if (afterMark <= 0) capitalHalted = true;
    const realized = state.realizedPnl - before.realizedPnl;
    const fees = state.feesPaid - before.feesPaid;
    const unrealizedBefore = before.quantity * (before.lastMarkPrice - before.entryPrice);
    const unrealizedAfter = state.quantity * (state.lastMarkPrice - state.entryPrice);
    const navBefore = booked.capital.navUsdt, feesBefore = booked.capital.paperFeesUsdt;
    booked = bookHistoricalMark(booked, plant.id, { datasetFingerprint: dataset.fingerprint, marketSource: dataset.provenance.source,
      fillCount: fills.length, strategyId: `${plant.id}:${profile.id}:kraken-v1`, netEquityDeltaUsdt: afterMark - beforeMark,
      realizedPnlUsdt: realized, unrealizedDeltaUsdt: unrealizedAfter - unrealizedBefore, feesUsdt: fees });
    const old = records[plant.id] ?? emptyRecord();
    const fillReasons = { ...old.fillReasons };
    for (const fill of fills) fillReasons[fill.reason] = (fillReasons[fill.reason] ?? 0) + 1;
    records[plant.id] = { ...old, fillReasons, fills: old.fills + fills.length,
      netPnl: rounded(old.netPnl + booked.capital.navUsdt - navBefore), fees: rounded(old.fees + booked.capital.paperFeesUsdt - feesBefore),
      observedDays: (old.observedDays ?? 0) + 1, equityPeak: Math.max(old.equityPeak ?? beforeMark, afterMark),
      maxDrawdownPct: Math.max(old.maxDrawdownPct ?? 0, Math.max(0, 1 - afterMark / Math.max(old.equityPeak ?? beforeMark, afterMark)) * 100),
      realizedPnl: state.realizedPnl, unrealizedPnl: unrealizedAfter,
      equityHistory: [...(old.equityHistory ?? []), { day: session.world.day + 1, equity: afterMark,
        netPnl: old.netPnl + booked.capital.navUsdt - navBefore }].slice(-300) };
    let quantity = before.quantity;
    trades.push(...fills.map((fill) => { quantity += fill.side === "buy" ? fill.quantity : -fill.quantity; return krakenTrade(fill, plant.id, plant.name, quantity, dataset); }));
    // Keep only the latest execution tail throughout the day, not after all plants.
    trades.sort((a,b)=>a.executedAt.localeCompare(b.executedAt)||a.orderId.localeCompare(b.orderId));
    if(trades.length>400)trades.splice(0,trades.length-400);
    totalFills += fills.length;
  }
  const protectedIds = Object.entries(states).filter(([, state]) => state.quantity !== 0 || state.pending.length > 0).map(([id]) => id);
  let world = advanceSimulationDay({ ...booked, day: session.world.day }, observedSolClimate(dataset), protectedIds);
  const genomes = { ...session.genomes };
  for (const plant of world.plants) {
    if (genomes[plant.id]) continue;

    const donors = plant.parents?.map((id) => genomes[id]!).filter(Boolean) ?? SOURCE_PARENT_GENOMES;
    genomes[plant.id] = inheritSourceGenome(plant.conceivedBy ?? "hive", donors.length ? donors : SOURCE_PARENT_GENOMES, world.seed, world.plants.indexOf(plant));
    const profileDonors = plant.parents?.map((id) => profiles[id] ?? (states[id] ? getKrakenProfile(states[id]!.profileId) : undefined)).filter((p): p is KrakenProfile => !!p)
      ?? session.initialProfileIds!.map(getKrakenProfile);
    profiles[plant.id] = inheritKrakenProfile(plant.conceivedBy ?? "hive", profileDonors.length ? profileDonors : session.initialProfileIds!.map(getKrakenProfile),
      world.seed, plant.id, genomes[plant.id]!);
    profiles[plant.id] = marketProfile(profiles[plant.id]!, archive.symbol);
    states[plant.id] = createKrakenPaperState(profiles[plant.id]!.id, plant.allocationUsdt);
  }
  const funded = applyEndOfDayFunding({ ...session, world });
  world = funded.world;
  const index = session.loop.archiveIndex + 1;
  const next: HistoricalSession = { ...session, funding: funded.funding, world, genomes, krakenStates: states, krakenProfiles: profiles, records,
    nextDate: archive.days[index]?.date ?? nextUtcDate(session.nextDate),
    history: [...session.history, { date: session.nextDate, nav: world.capital.navUsdt, fills: totalFills, close: Number(dataset.bars.at(-1)!.close),
      observedBars: dataset.bars.length, skippedBars: 0, segments: dataset.report.gapCount + 1,
      deposits: world.capital.externalDepositsUsdt ?? 0, netPnl: rounded(world.capital.navUsdt - 100 - (world.capital.externalDepositsUsdt ?? 0)) }].slice(-300),
    trades: trades.sort((a, b) => a.executedAt.localeCompare(b.executedAt) || a.orderId.localeCompare(b.orderId)).slice(-400),
    loop: { ...session.loop, archiveIndex: index, totalDays: session.loop.totalDays + 1, totalBars: session.loop.totalBars + dataset.bars.length,
      totalFills: session.loop.totalFills + totalFills, totalFees: rounded(session.loop.totalFees + world.capital.paperFeesUsdt - session.world.capital.paperFeesUsdt), capitalHalted: funded.loop.capitalHalted } };
  return validateHistoricalSession(index === archive.days.length ? wrapHistoricalRound(compactHistoricalSession(next), archive) : compactHistoricalSession(next), archive);
}

export async function advanceHistoricalSession(session: HistoricalSession, archive: SolArchive): Promise<HistoricalSession> {
  session = validateHistoricalSession(session, archive);
  const day = archive.days[session.loop.archiveIndex]!;
  const previous = archive.days.find((d) => d.date === nextUtcDate(day.date, -1));
  const [rows, prior] = await Promise.all([loadArchiveBars(day, archive.symbol), previous ? loadArchiveBars(previous, archive.symbol) : Promise.resolve([])]);
  const [dataset, warmup] = await Promise.all([normalizeArchiveBars(rows, archive), normalizeArchiveBars([...prior, ...rows], archive)]);
  return executeHistoricalDay(session, archive, dataset, warmup);
}

/** Fixed causal reference: SOL first known open 2020-08-11 06:00=2.85; BTC first open 2020-08-12=11392.09. Same relative brick threshold, not return-fitted. */
function marketProfile(profile: KrakenProfile, symbol: string): KrakenProfile {
  return symbol === "BTC-USDT" ? { ...profile, renkoBoxUsd: .1 * 11392.09 / 2.85 } : profile;
}
