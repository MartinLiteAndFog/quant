import { normalizeMarketData, runPaperReplay, type MarketBarInput, type MarketDataset, type PaperFill } from "./paperReplay.ts";
import { advanceSimulationDay, assertSimulationInvariants, bookClosedPaperReplay, createSimulationWorld, type SimulationWorld, type TradingDayOutcomes } from "./simulation.ts";
import { generateSourceSignals, inheritSourceGenome, SOURCE_PARENT_GENOMES, validateSourceGenome, type SourceGenome } from "./sourceStrategies.ts";

export type ArchiveDay = Readonly<{ date: string; barCount: number; gapCount: number; missingMinutes: number; file: string; sha256: string }>;
export type SolArchive = Readonly<{
  schemaVersion: 1; source: string; sourceSha256: string; symbol: string; venue: string;
  firstDate: string; lastDate: string; totalBars: number; missingMinutes: number; duplicateCount: number;
  days: readonly ArchiveDay[];
}>;
export type PlantTradingRecord = Readonly<{ fills: number; netPnl: number; fees: number; rejected: number; opportunities: number }>;
export type HistoricalTrade = PaperFill & Readonly<{ plantId: string; plantName: string; family: string }>;
export type HistoricalSession = Readonly<{
  schemaVersion: 4; sourceSha256: string; startDate: string; nextDate: string; world: SimulationWorld;
  genomes: Readonly<Record<string, SourceGenome>>;
  records: Readonly<Record<string, PlantTradingRecord>>;
  history: readonly Readonly<{ date: string; nav: number; fills: number; close: number }>[];
  trades: readonly HistoricalTrade[];
}>;
export const HISTORICAL_SAVE_KEY = "cloudgarden.historical.v4";
const DAY = 86_400_000;
const emptyRecord = (): PlantTradingRecord => ({ fills: 0, netPnl: 0, fees: 0, rejected: 0, opportunities: 0 });
const rounded = (n: number) => Math.round(n * 100_000_000) / 100_000_000;

export function nextUtcDate(date: string, offset = 1): string {
  return new Date(Date.parse(`${date}T00:00:00Z`) + offset * DAY).toISOString().slice(0, 10);
}

export function validateSolArchive(value: unknown): SolArchive {
  const a = value as SolArchive;
  if (!a || a.schemaVersion !== 1 || !a.source?.trim() || a.symbol !== "SOL-USDT"
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

export function createHistoricalSession(archive: SolArchive, startDate: string): HistoricalSession {
  if (!archive.days.some((day) => day.date === startDate && day.barCount === 1440 && day.missingMinutes === 0)) {
    throw new Error("Choose a complete UTC day from the SOL archive.");
  }
  const world = createSimulationWorld(`SOL:${archive.sourceSha256.slice(0, 12)}:${startDate}`);
  const genomes = Object.fromEntries(world.plants.map((plant, i) => [plant.id, SOURCE_PARENT_GENOMES[i]!])) as Record<string, SourceGenome>;
  return { schemaVersion: 4, sourceSha256: archive.sourceSha256, startDate, nextDate: startDate, world,
    genomes, records: {}, history: [], trades: [] };
}

/** Validate snapshots before using saved accounting or inherited strategy modules. */
export function validateHistoricalSession(value: unknown, archive: SolArchive): HistoricalSession {
  const s = value as HistoricalSession;
  if (!s || s.schemaVersion !== 4 || s.sourceSha256 !== archive.sourceSha256 || !s.genomes || !s.records
    || !Array.isArray(s.history) || !Array.isArray(s.trades) || s.trades.length > 400
    || !archive.days.some((d) => d.date === s.startDate)
    || s.nextDate !== nextUtcDate(s.startDate, s.world?.day)) throw new Error("Saved historical session is invalid.");
  assertSimulationInvariants(s.world);
  if (s.history.length !== s.world.day || s.history.length > archive.days.length) throw new Error("Historical clock and history differ.");
  for (const [i, row] of s.history.entries()) {
    if (row.date !== nextUtcDate(s.startDate, i) || ![row.nav, row.close].every((n) => Number.isFinite(n) && n > 0)
      || !Number.isInteger(row.fills) || row.fills < 0) throw new Error("Invalid historical daily record.");
  }
  if (s.history.length && Math.abs(s.history.at(-1)!.nav - s.world.capital.navUsdt) > .0001) throw new Error("Saved NAV does not reconcile.");
  for (const plant of s.world.plants) validateSourceGenome(s.genomes[plant.id]!);
  for (const [id, record] of Object.entries(s.records)) {
    if (!s.world.plants.some((p) => p.id === id)) throw new Error("Unknown plant in trading records.");
    if (!record || ![record.netPnl, record.fees].every(Number.isFinite) || record.fees < 0
      || ![record.fills, record.rejected, record.opportunities].every((n) => Number.isInteger(n) && n >= 0)) throw new Error("Invalid saved trading record.");
  }
  const records = Object.values(s.records);
  if (Math.abs(records.reduce((sum, r) => sum + r.netPnl, 0) - (s.world.capital.navUsdt - 100)) > .0001
    || Math.abs(records.reduce((sum, r) => sum + r.fees, 0) - s.world.capital.paperFeesUsdt) > .0001
    || records.reduce((sum, r) => sum + r.fills, 0) !== s.history.reduce((sum, row) => sum + row.fills, 0)) {
    throw new Error("Plant trading records do not reconcile with historical NAV, fees or fills.");
  }
  const retainedKeys = new Set<string>();
  for (const fill of s.trades) {
    const decimal = (v: unknown) => typeof v === "string" && /^-?(?:0|[1-9]\d*)(?:\.\d{1,8})?$/.test(v) && Number.isFinite(Number(v));
    if (!fill || typeof fill.plantId !== "string" || !s.world.plants.some((p) => p.id === fill.plantId)
      || typeof fill.plantName !== "string" || fill.plantName.length > 200
      || fill.family !== s.genomes[fill.plantId]!.family
      || typeof fill.orderId !== "string" || !fill.orderId.trim()
      || fill.strategyId !== `${fill.plantId}:${fill.family}:source-v1`
      || (fill.side !== "buy" && fill.side !== "sell") || typeof fill.executedAt !== "string"
      || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z$/.test(fill.executedAt)
      || !Number.isFinite(Date.parse(fill.executedAt))
      || Date.parse(fill.executedAt) < Date.parse(`${s.startDate}T00:00:00Z`)
      || Date.parse(fill.executedAt) >= Date.parse(`${s.nextDate}T00:00:00Z`)
      || ![fill.targetQuantity, fill.filledQuantity, fill.referencePrice, fill.price, fill.notional, fill.fee, fill.executionCost].every(decimal)
      || [fill.filledQuantity, fill.referencePrice, fill.price].some((v) => Number(v) <= 0)
      || [fill.notional, fill.fee, fill.executionCost].some((v) => Number(v) < 0)
      || !/^[a-f0-9]{64}$/.test(fill.datasetFingerprint)) throw new Error("Invalid saved historical fill.");
    const key = `${fill.plantId}:${fill.orderId}:${fill.executedAt}`;
    if (retainedKeys.has(key)) throw new Error("Duplicated saved historical fill.");
    retainedKeys.add(key);
  }
  if (s.trades.length > records.reduce((sum, r) => sum + r.fills, 0)) throw new Error("Saved fill history exceeds its ledger count.");
  return s;
}

export async function loadSolArchive(): Promise<SolArchive> {
  const response = await fetch("/market/manifest.json", { cache: "no-cache" });
  if (!response.ok) throw new Error("SOL archive is missing. Run the local history exporter first.");
  return validateSolArchive(await response.json());
}

/** Hash the exact local export bytes; no random data, filling or price replacement. */
export async function loadArchiveBars(day: ArchiveDay): Promise<readonly MarketBarInput[]> {
  const response = await fetch(`/market/${day.file}`);
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
    license: "unverified; local private use", symbol: "SOL-USDT", marketType: "linear_futures", contractMultiplier: "1",
    intervalSeconds: 60, synthetic: false, timezone: "UTC", timestampSemantics: "open",
  });
}

/** Execute all financed plants against the same actual UTC day before the daily ecology/predator tick. */
export function executeHistoricalDay(session: HistoricalSession, archive: SolArchive, dataset: MarketDataset, warmup: MarketDataset): HistoricalSession {
  validateHistoricalSession(session, archive);
  const start = Date.parse(`${session.nextDate}T00:00:00Z`);
  if (dataset.provenance.synthetic || dataset.report.start !== new Date(start).toISOString()
    || dataset.report.endExclusive !== new Date(start + DAY).toISOString()
    || dataset.bars.length !== 1440 || dataset.report.missingBars || warmup.report.missingBars) {
    throw new Error("Historical execution requires a complete real SOL day and continuous warmup; gaps are never filled.");
  }
  const tail = warmup.bars.filter((bar) => bar.timestampMs >= start);
  if (warmup.provenance.synthetic || warmup.report.synthetic || dataset.report.synthetic
    || warmup.provenance.source !== dataset.provenance.source
    || !dataset.provenance.source.includes(archive.sourceSha256)
    || warmup.report.endExclusive !== dataset.report.endExclusive || tail.length !== dataset.bars.length
    || tail.some((bar, i) => {
      const d = dataset.bars[i]!;
      return bar.timestamp !== d.timestamp || bar.open !== d.open || bar.high !== d.high || bar.low !== d.low
        || bar.close !== d.close || bar.volume !== d.volume;
    })) throw new Error("Signal warmup must contain the exact same historical day and source as execution.");
  let booked = { ...session.world, day: session.world.day + 1 };
  const records = { ...session.records };
  // Born from bookable trades, not from synthetic climate or a research verdict.
  // The denominator is each plant's immutable start-of-day allocation.
  const dayOutcomes: Record<string, { netPnlUsdt: number; startingAllocationUsdt: number; executedFills: number }> = {};
  const trades = [...session.trades];
  let totalFills = 0;
  for (const plant of session.world.plants.filter((p) => p.status === "active")) {
    const current = booked.plants.find((p) => p.id === plant.id)!;
    if (current.allocationUsdt < .001) continue;
    const allocation = current.allocationUsdt.toFixed(4);
    // Known session-open price only; future prices never size an earlier trade.
    const quantity = (Math.floor(current.allocationUsdt * .5 / Number(dataset.bars[0]!.open) * 100_000_000) / 100_000_000).toFixed(8);
    if (Number(quantity) <= 0) continue;
    const strategyId = `${plant.id}:${session.genomes[plant.id]!.family}:source-v1`;
    const strategy = generateSourceSignals(warmup, session.genomes[plant.id]!, strategyId, quantity, start, start + DAY);
    const report = runPaperReplay({ dataset, signals: strategy.signals, allocationLimit: allocation, startingCash: allocation, historicalResearch: true });
    if (!report.endedFlat || report.riskBreach) throw new Error("A historical plant result could not be reconciled flat.");
    const navBefore = booked.capital.navUsdt;
    const feesBefore = booked.capital.paperFeesUsdt;
    booked = bookClosedPaperReplay(booked, plant.id, {
      startDay: session.world.day, days: 1, datasetFingerprint: report.dataset.fingerprint, synthetic: false,
      historicalResearch: true, marketSource: dataset.provenance.source, endedFlat: report.endedFlat, riskBreach: report.riskBreach,
      fillCount: report.fills.length, signalStatus: strategy.status, strategyId,
      observations: strategy.observations, opportunities: strategy.opportunities, generatedSignals: strategy.generatedSignals,
      allocationLimitUsdt: Number(report.allocationLimit), startingCashUsdt: Number(report.startingCash), endingCashUsdt: Number(report.endingCash),
      maxMarginUsedUsdt: Number(report.maxMarginUsed), realizedPricePnlUsdt: Number(report.realizedPricePnl), feesUsdt: Number(report.fees),
      fundingUsdt: Number(report.funding), netPnlUsdt: Number(report.netPnl),
    });
    const before = records[plant.id] ?? emptyRecord();
    // Mirror the single ledger's four-decimal booking precision in plant totals.
    dayOutcomes[plant.id] = {
      netPnlUsdt: rounded(booked.capital.navUsdt - navBefore),
      startingAllocationUsdt: plant.allocationUsdt,
      executedFills: report.fills.length,
    };
    records[plant.id] = { fills: before.fills + report.fills.length, netPnl: rounded(before.netPnl + booked.capital.navUsdt - navBefore),
      fees: rounded(before.fees + booked.capital.paperFeesUsdt - feesBefore), rejected: before.rejected + report.rejectedSignals.length,
      opportunities: before.opportunities + strategy.opportunities };
    totalFills += report.fills.length;
    trades.push(...report.fills.map((fill) => ({ ...fill, plantId: plant.id, plantName: plant.name, family: session.genomes[plant.id]!.family })));
  }
  // Intraday accounting precedes end-of-day growth and predation. Advance ecology exactly once.
  const world = advanceSimulationDay({ ...booked, day: session.world.day }, dayOutcomes satisfies TradingDayOutcomes);
  const genomes = { ...session.genomes };
  for (const plant of world.plants) {
    if (genomes[plant.id]) continue;
    const parents = plant.parents?.map((id) => genomes[id]!).filter(Boolean) ?? SOURCE_PARENT_GENOMES;
    genomes[plant.id] = inheritSourceGenome(plant.conceivedBy ?? "hive", parents.length ? parents : SOURCE_PARENT_GENOMES, world.seed, world.plants.indexOf(plant));
  }
  const history = [...session.history, { date: session.nextDate, nav: world.capital.navUsdt, fills: totalFills, close: Number(dataset.bars.at(-1)!.close) }];
  const next = { ...session, nextDate: nextUtcDate(session.nextDate), world, genomes, records, history,
    trades: trades.sort((a, b) => a.executedAt.localeCompare(b.executedAt) || a.orderId.localeCompare(b.orderId)).slice(-400) };
  return validateHistoricalSession(next, archive);
}

export async function advanceHistoricalSession(session: HistoricalSession, archive: SolArchive): Promise<HistoricalSession> {
  const day = archive.days.find((d) => d.date === session.nextDate);
  if (!day) throw new Error(`End of local SOL archive reached (${archive.lastDate}). Choose another start date.`);
  if (day.barCount !== 1440 || day.missingMinutes) throw new Error(`${day.date} has missing minutes. Playback stopped; choose a complete date after the gap.`);
  const previous = archive.days.find((d) => d.date === nextUtcDate(day.date, -1));
  const [rows, prior] = await Promise.all([loadArchiveBars(day), previous ? loadArchiveBars(previous) : Promise.resolve([])]);
  const [dataset, warmup] = await Promise.all([normalizeArchiveBars(rows, archive), normalizeArchiveBars([...prior, ...rows], archive)]);
  return executeHistoricalDay(session, archive, dataset, warmup);
}
