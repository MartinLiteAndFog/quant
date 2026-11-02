import assert from "node:assert/strict";
import test from "node:test";
import { createHistoricalSession as createBaseHistoricalSession, executeHistoricalDay, normalizeArchiveBars, validateHistoricalSession, validateSolArchive, type SolArchive } from "./historicalGarden.ts";
import { assertSimulationInvariants } from "./simulation.ts";
import type { MarketBarInput } from "./paperReplay.ts";

const archive: SolArchive = { schemaVersion: 1, source: "Historical contract test fixture", sourceSha256: "a".repeat(64), symbol: "SOL-USDT", venue: "unverified",
  firstDate: "2025-01-01", lastDate: "2025-01-10", totalBars: 14400, missingMinutes: 0, duplicateCount: 0,
  days: Array.from({ length: 10 }, (_, i) => ({ date: `2025-01-${String(i + 1).padStart(2, "0")}`, barCount: 1440, gapCount: 0,
    missingMinutes: 0, file: `2025-01-${String(i + 1).padStart(2, "0")}.json`, sha256: "b".repeat(64) })) };

const createHistoricalSession = (source: SolArchive, date = source.firstDate) => createBaseHistoricalSession(source, date, { engine: "source" });

function bars(date: string): MarketBarInput[] {
  const start = Date.parse(`${date}T00:00:00Z`);
  return Array.from({ length: 1440 }, (_, i) => {
    const op = 100 + Math.sin(i / 25) * 5, close = 100 + Math.sin((i + 1) / 25) * 5;
    return { timestamp: new Date(start + i * 60_000).toISOString(), open: op.toFixed(5), close: close.toFixed(5),
      high: (Math.max(op, close) + .1).toFixed(5), low: (Math.min(op, close) - .1).toFixed(5), volume: "1000" };
  });
}

test("historical garden executes all parents on the same minute day, books real changes once, and remains deterministic", async () => {
  validateSolArchive(archive);
  const session = createHistoricalSession(archive, "2025-01-02");
  const input = JSON.stringify(session);
  const dataset = await normalizeArchiveBars(bars(session.nextDate), archive);
  const warmup = await normalizeArchiveBars([...bars("2025-01-01"), ...bars(session.nextDate)], archive);
  const next = await executeHistoricalDay(session, archive, dataset, warmup);
  assert.equal(JSON.stringify(session), input);
  assert.deepEqual(await executeHistoricalDay(session, archive, dataset, warmup), next);
  assert.equal(next.world.day, 1);
  assert.equal(next.nextDate, "2025-01-03");
  assert.ok(next.trades.length > 0);
  assert.notEqual(next.world.capital.navUsdt, 100);
  assert.ok(next.world.capital.paperFeesUsdt > 0);
  assert.equal(next.world.capital.paperReplayCount, 3);
  assert.ok(next.trades.every((fill) => fill.executedAt >= "2025-01-02T00:02:00.000Z" && fill.executedAt < "2025-01-03"));
  const events = next.world.events.filter((e) => e.type === "paper_replay");
  assert.equal(events.length, 3);
  assert.ok(events.every((e) => e.detail.synthetic === false && e.detail.historicalResearch === true));
  assert.ok(Math.abs(Object.values(next.records).reduce((sum, r) => sum + r.netPnl, 0) - (next.world.capital.navUsdt - 100)) < .0001);
  assert.ok(Math.abs(next.world.capital.protectedReserveUsdt - next.world.capital.navUsdt * .1) < .0001);
  assertSimulationInvariants(next.world);
  assert.deepEqual(validateHistoricalSession(JSON.parse(JSON.stringify(next)), archive), next);
});

test("seven historical days conceive an offspring with source genome and a single conserved ledger", async () => {
  let session = createHistoricalSession(archive, "2025-01-02");
  for (let i = 0; i < 7; i += 1) {
    const rows = bars(session.nextDate);
    const dataset = await normalizeArchiveBars(rows, archive);
    session = await executeHistoricalDay(session, archive, dataset, dataset);
  }
  assert.equal(session.world.day, 7);
  assert.equal(session.world.plants.length, 4);
  const child = session.world.plants.find((p) => p.role === "seedling")!;
  assert.ok(session.genomes[child.id]);
  assert.ok(session.genomes[child.id]!.stopLossBps >= 50);
  assert.equal(session.world.capital.seedlingEntitlements.filter((s) => s.status === "used").length, 1);
  assertSimulationInvariants(session.world);
});

test("historical snapshots reject changed source identity, altered NAV records or removed risk genes", () => {
  const session = createHistoricalSession(archive, "2025-01-02");
  assert.throws(() => validateHistoricalSession({ ...session, sourceSha256: "c".repeat(64) }, archive));
  assert.throws(() => validateHistoricalSession({ ...session, records: { "parent-01": { fills: 0, fees: 0, rejected: 0, opportunities: 0, netPnl: 1 } } }, archive));
  assert.throws(() => validateHistoricalSession({ ...session, genomes: {} }, archive));
  assert.throws(() => validateHistoricalSession({ ...session, trades: [{}] }, archive), /saved historical fill/);
  assert.throws(() => validateSolArchive({ ...archive, totalBars: 2 }));
  assert.throws(() => validateSolArchive({ ...archive, days: archive.days.map((d) => ({ ...d, file: "../bad.json" })) }));
});

test("a dataset whose observed bar count differs from its manifest is rejected before booking", async () => {
  const session = createHistoricalSession(archive, "2025-01-02");
  const rows = bars(session.nextDate); rows.splice(700, 1);
  const dataset = await normalizeArchiveBars(rows, archive);
  await assert.rejects(() => executeHistoricalDay(session, archive, dataset, dataset), /observed real SOL day/);
  assert.equal(session.world.day, 0);
  assert.equal(session.world.capital.navUsdt, 100);
  assert.equal(session.trades.length, 0);
});

test("historical decisions reject mismatched warmup prices or a different source", async () => {
  const session = createHistoricalSession(archive, "2025-01-02");
  const rows = bars(session.nextDate);
  const dataset = await normalizeArchiveBars(rows, archive);
  const changed = rows.map((bar, i) => i === 30 ? { ...bar, volume: "999" } : bar);
  const other = await normalizeArchiveBars(changed, archive);
  await assert.rejects(() => executeHistoricalDay(session, archive, dataset, other), /exact same historical day/);
  const wrongSource = { ...dataset, provenance: { ...dataset.provenance, source: "another archive" } };
  await assert.rejects(() => executeHistoricalDay(session, archive, dataset, wrongSource), /exact same historical day/);
});

test("partial dates and genuine gaps process only observed contiguous segments, never fabricate candles", async () => {
  const observed = [bars("2025-01-01").slice(1200), bars("2025-01-03").slice(0, 10), bars("2025-01-05").slice(0, 2)];
  observed[1]!.splice(4, 2);
  const days = observed.map((rows) => ({ date: rows[0]!.timestamp.slice(0, 10), barCount: rows.length, gapCount: 0,
    missingMinutes: 1440 - rows.length, file: `${rows[0]!.timestamp.slice(0, 10)}.json`, sha256: "b".repeat(64) }));
  const partial: SolArchive = { ...archive, firstDate: days[0]!.date, lastDate: days.at(-1)!.date, days,
    totalBars: observed.reduce((sum, rows) => sum + rows.length, 0) };
  validateSolArchive(partial);
  let session = createHistoricalSession(partial);
  for (let cycle = 1; cycle <= 3; cycle += 1) {
    for (let i = 0; i < days.length; i += 1) {
      const dataset = await normalizeArchiveBars(observed[i]!, partial);
      session = await executeHistoricalDay(session, partial, dataset, dataset);
      if (i === 0) assert.equal(session.history.at(-1)!.observedBars, 240);
      if (i === 1) { assert.equal(session.history.at(-1)!.segments, 2); assert.equal(session.nextDate, "2025-01-05"); }
    }
    assert.equal(session.loop.cycle, cycle + 1);
    assert.equal(session.loop.totalBars, cycle * partial.totalBars);
    assert.equal(session.loop.totalDays, cycle * 3);
    assert.equal(session.world.capital.navUsdt, 100);
    assert.equal(session.world.day, 0);
    assert.equal(session.nextDate, partial.firstDate);
    assert.equal(session.world.plants.length, 3);
    assert.equal(session.trades.length, 0);
    assert.equal(session.loop.completedRounds.length, cycle);
    assert.deepEqual(validateHistoricalSession(JSON.parse(JSON.stringify(session)), partial), session);
  }
  assert.equal(session.loop.totalFills, session.loop.completedRounds.reduce((sum, round) => sum + round.fills, 0));
  assert.ok(Math.abs(session.loop.totalFees - session.loop.completedRounds.reduce((sum, round) => sum + round.fees, 0)) < .0001);
});

test("old v1 sessions migrate counters without replaying fills and malformed loop snapshots are rejected", async () => {
  const initial = createHistoricalSession(archive);
  const data = await normalizeArchiveBars(bars(initial.nextDate), archive);
  const current = await executeHistoricalDay(initial, archive, data, data);
  const { loop: _loop, ...legacy } = current;
  const migrated = validateHistoricalSession(legacy, archive);
  assert.equal(migrated.loop.totalDays, 1);
  assert.equal(migrated.loop.totalBars, 1440);
  assert.equal(migrated.loop.totalFills, current.loop.totalFills);
  assert.deepEqual(migrated.records, current.records);
  assert.throws(() => validateHistoricalSession({ ...current, loop: { ...current.loop, archiveIndex: 7 } }, archive), /loop/);
  assert.throws(() => validateHistoricalSession({ ...current, loop: { ...current.loop, totalFills: 0 } }, archive), /reconcile/);
});

test("explicit capital halt continues actual observations and preserves NAV until the archive boundary resets the round", async () => {
  let session = createHistoricalSession(archive, "2025-01-09");
  session = { ...session, loop: { ...session.loop, capitalHalted: true } };
  const before = session.world.capital;
  const dataset = await normalizeArchiveBars(bars(session.nextDate), archive);
  session = await executeHistoricalDay(session, archive, dataset, dataset);
  assert.equal(session.loop.totalBars, 1440);
  assert.equal(session.loop.capitalHalted, true);
  assert.equal(session.world.capital.navUsdt, before.navUsdt);
  assert.equal(session.world.capital.paperReplayCount, 0);
  const last = await normalizeArchiveBars(bars(session.nextDate), archive);
  session = await executeHistoricalDay(session, archive, last, last);
  assert.equal(session.loop.cycle, 2);
  assert.equal(session.loop.completedRounds.at(-1)!.reason, "capital_floor");
  assert.equal(session.loop.capitalHalted, false);
  assert.equal(session.loop.totalBars, 2880);
});

test("observed SOL volume, ranges and direction determine ecological climate independently of random seed", async () => {
  const session = createHistoricalSession(archive);
  const rows = bars(session.nextDate);
  const dataset = await normalizeArchiveBars(rows, archive);
  const next = await executeHistoricalDay(session, archive, dataset, dataset);
  const other = await executeHistoricalDay({ ...session, world: { ...session.world, seed: "other habitat seed" } }, archive, dataset, dataset);
  assert.deepEqual(next.world.climate, other.world.climate);
  const quietRows = rows.map((bar) => ({ ...bar, open: "100", high: "100", low: "100", close: "100", volume: "0" }));
  const quiet = await normalizeArchiveBars(quietRows, archive);
  const low = await executeHistoricalDay(session, archive, quiet, quiet);
  assert.equal(low.world.climate.water, 0);
  assert.equal(low.world.climate.sunlight, .5);
  assert.notDeepEqual(next.world.climate, low.world.climate);
});

test("predeclared exit retries close on observed liquidity when the final candles have zero volume", async () => {
  const session = createHistoricalSession(archive);
  const rows = bars(session.nextDate).map((bar, i) => i >= 1430 ? { ...bar, volume: "0" } : bar);
  const dataset = await normalizeArchiveBars(rows, archive);
  const next = await executeHistoricalDay(session, archive, dataset, dataset);
  assert.ok(next.trades.length > 0);
  assert.ok(next.trades.every((fill) => Date.parse(fill.executedAt) < Date.parse(`${session.nextDate}T23:50:00Z`)));
  assert.equal(next.history[0]!.observedBars, 1440);
  assert.equal(next.history[0]!.skippedBars, 0);
  assertSimulationInvariants(next.world);
});

test("round summaries are bounded across more than sixty-four repeated archive passes", async () => {
  const rows = bars("2025-01-01").slice(0, 3);
  const tiny: SolArchive = { ...archive, lastDate: archive.firstDate, days: [{ ...archive.days[0]!, barCount: 3 }], totalBars: 3 };
  const dataset = await normalizeArchiveBars(rows, tiny);
  let session = createHistoricalSession(tiny);
  for (let i = 0; i < 67; i += 1) session = await executeHistoricalDay(session, tiny, dataset, dataset);
  assert.equal(session.loop.cycle, 68);
  assert.equal(session.loop.completedRounds.length, 64);
  assert.equal(session.loop.completedRounds[0]!.cycle, 4);
  assert.equal(session.loop.totalDays, 67);
  assert.equal(session.loop.totalBars, 201);
  assert.equal(session.history.length, 0);
  assert.equal(session.world.events.length, 0);
});

test("new historical sessions run three selected Kraken profiles with persistent open positions and separate equity marks", async () => {
  let session = createBaseHistoricalSession(archive, "2025-01-01", { profileIds: ["kraken_legacy", "countervariante", "kraken_17_2_close"] });
  assert.deepEqual(Object.values(session.krakenStates!).map((s) => s.profileId), ["kraken_legacy", "countervariante", "kraken_17_2_close"]);
  const dataset = await normalizeArchiveBars(bars(session.nextDate), archive);
  session = await executeHistoricalDay(session, archive, dataset, dataset);
  assert.ok(session.trades.length > 0);
  assert.ok(Object.values(session.krakenStates!).some((state) => state.quantity !== 0));
  assert.ok(session.world.capital.paperUnrealizedPnlUsdt !== undefined);
  assert.ok(session.world.events.some((event) => event.detail.markedAccount));
  const firstStates = session.krakenStates!;
  const nextData = await normalizeArchiveBars(bars(session.nextDate), archive);
  session = await executeHistoricalDay(session, archive, nextData, nextData);
  for (const [id, state] of Object.entries(session.krakenStates!)) {
    assert.ok(state.fillCount >= firstStates[id]!.fillCount);
    assert.equal(state.lastTimestampMs, nextData.bars.at(-1)!.timestampMs);
  }
  assert.deepEqual(validateHistoricalSession(JSON.parse(JSON.stringify(session)), archive), session);
  assertSimulationInvariants(session.world);
});

test("Kraken archive gaps preserve actual account marks and the terminal window settles before carrying capital into the next loop", async () => {
  const observed = [bars("2025-01-01"), bars("2025-01-03")];
  observed[1]!.splice(500, 3);
  const days = observed.map((rows) => ({ date: rows[0]!.timestamp.slice(0, 10), barCount: rows.length, gapCount: rows.length < 1440 ? 1 : 0,
    missingMinutes: 1440 - rows.length, file: `${rows[0]!.timestamp.slice(0, 10)}.json`, sha256: "b".repeat(64) }));
  const gapped = { ...archive, days, firstDate: days[0]!.date, lastDate: days.at(-1)!.date, totalBars: 2877 };
  let session = createBaseHistoricalSession(gapped);
  for (const rows of observed) {
    const dataset = await normalizeArchiveBars(rows, gapped);
    session = await executeHistoricalDay(session, gapped, dataset, dataset);
  }
  assert.equal(session.loop.cycle, 2);
  assert.equal(session.loop.totalBars, 2877);
  assert.ok(session.loop.totalFills > 0);
  assert.ok(Object.values(session.krakenStates!).every((state) => state.quantity === 0 && !state.pending.length));
  assert.equal(session.world.capital.navUsdt, session.loop.completedRounds[0]!.nav);
  assert.equal(session.world.day, 2);
  assert.equal(session.loop.completedRounds.length, 1);
});

test("monthly EUR contributions are explicit, idempotent after restore, include unobserved calendar months and never masquerade as PnL", async () => {
  const observed = [bars("2025-01-31").slice(0, 3), bars("2025-04-01").slice(0, 3), bars("2025-04-02").slice(0, 3)];
  const days = observed.map((rows) => ({ date: rows[0]!.timestamp.slice(0, 10), barCount: rows.length, gapCount: 0,
    missingMinutes: 0, file: `${rows[0]!.timestamp.slice(0, 10)}.json`, sha256: "b".repeat(64) }));
  const source = { ...archive, days, firstDate: days[0]!.date, lastDate: days.at(-1)!.date, totalBars: 9 };
  let session = createBaseHistoricalSession(source, source.firstDate, { monthlyEur: 100, eurUsdt: 1.05 });
  const first = await normalizeArchiveBars(observed[0]!, source);
  session = await executeHistoricalDay(session, source, first, first);
  const april = await normalizeArchiveBars(observed[1]!, source);
  session = await executeHistoricalDay(validateHistoricalSession(JSON.parse(JSON.stringify(session)), source), source, april, april);
  assert.equal(session.funding!.totalEur, 300);
  assert.equal(session.funding!.totalUsdt, 315);
  assert.equal(session.world.capital.externalDepositsUsdt, 315);
  assert.equal(session.world.capital.navUsdt, 415);
  assert.equal(session.world.capital.realizedPricePnlUsdt, 0);
  const final = await normalizeArchiveBars(observed[2]!, source);
  session = await executeHistoricalDay(validateHistoricalSession(JSON.parse(JSON.stringify(session)), source), source, final, final);
  assert.equal(session.funding!.totalUsdt, 315);
  assert.equal(session.world.capital.navUsdt, 415);
  assert.equal(session.world.day, 3);
  assert.equal(session.loop.cycle, 2);
});

test("concurrent plant fills are invariant to array order because all daily budgets use the frozen pre-day account", async () => {
  const original = createBaseHistoricalSession(archive);
  const dataset = await normalizeArchiveBars(bars(original.nextDate), archive);
  const reversed = { ...original, world: { ...original.world, plants: [...original.world.plants].reverse() } };
  const normalResult = await executeHistoricalDay(original, archive, dataset, dataset);
  const reverseResult = await executeHistoricalDay(reversed, archive, dataset, dataset);
  const fills = (session: typeof normalResult) => [...session.trades].sort((a, b) => a.plantId.localeCompare(b.plantId) || a.orderId.localeCompare(b.orderId));
  assert.deepEqual(fills(normalResult), fills(reverseResult));
  for (const id of Object.keys(original.krakenStates!)) assert.deepEqual(normalResult.krakenStates![id], reverseResult.krakenStates![id]);
});

test("executed offspring profiles are frozen at conception, retain actual EW genes and survive restore unchanged", async () => {
  let session = createBaseHistoricalSession(archive);
  for (let i = 0; i < 7; i += 1) {
    const dataset = await normalizeArchiveBars(bars(session.nextDate), archive);
    session = await executeHistoricalDay(session, archive, dataset, dataset);
  }
  const child = session.world.plants.find((plant) => plant.role === "seedling")!;
  const profile = session.krakenProfiles![child.id]!;
  assert.equal(profile.id, `evolved:${child.id}`);
  assert.equal(profile.ew!.filter, session.genomes[child.id]!.ewFilter);
  assert.equal(profile.ew!.exit, session.genomes[child.id]!.ewExit);
  assert.equal(profile.ew!.pivotLength, session.genomes[child.id]!.pivotLength);
  const fingerprint = JSON.stringify(profile);
  const stateBefore = session.krakenStates![child.id]!;
  const dataset = await normalizeArchiveBars(bars(session.nextDate), archive);
  const next = await executeHistoricalDay(validateHistoricalSession(JSON.parse(JSON.stringify(session)), archive), archive, dataset, dataset);
  assert.equal(JSON.stringify(next.krakenProfiles![child.id]!), fingerprint);
  assert.equal(next.krakenStates![child.id]!.profileId, profile.id);
  assert.equal(next.krakenStates![child.id]!.lastTimestampMs, dataset.bars.at(-1)!.timestampMs);
  assert.ok(next.krakenStates![child.id]!.fillCount >= stateBefore.fillCount);
  assert.deepEqual(validateHistoricalSession(JSON.parse(JSON.stringify(next)), archive).krakenProfiles![child.id], profile);
});
