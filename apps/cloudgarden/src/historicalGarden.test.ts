import assert from "node:assert/strict";
import test from "node:test";
import { createHistoricalSession, executeHistoricalDay, normalizeArchiveBars, validateHistoricalSession, validateSolArchive, type SolArchive } from "./historicalGarden.ts";
import { assertSimulationInvariants } from "./simulation.ts";
import type { MarketBarInput } from "./paperReplay.ts";

const archive: SolArchive = { schemaVersion: 1, source: "Historical contract test fixture", sourceSha256: "a".repeat(64), symbol: "SOL-USDT", venue: "unverified",
  firstDate: "2025-01-01", lastDate: "2025-01-10", totalBars: 14400, missingMinutes: 0, duplicateCount: 0,
  days: Array.from({ length: 10 }, (_, i) => ({ date: `2025-01-${String(i + 1).padStart(2, "0")}`, barCount: 1440, gapCount: 0,
    missingMinutes: 0, file: `2025-01-${String(i + 1).padStart(2, "0")}.json`, sha256: "b".repeat(64) })) };

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
  const next = executeHistoricalDay(session, archive, dataset, warmup);
  assert.equal(JSON.stringify(session), input);
  assert.deepEqual(executeHistoricalDay(session, archive, dataset, warmup), next);
  assert.equal(next.world.day, 1);
  assert.equal(next.nextDate, "2025-01-03");
  assert.ok(next.trades.length > 0);
  assert.notEqual(next.world.capital.navUsdt, 100);
  assert.ok(next.world.capital.paperFeesUsdt > 0);
  // Biological growth is proportional to each parent's booked net return,
  // not to synthetic climate fitness or any research/significance verdict.
  for (const parent of session.world.plants) {
    const dayNetPnl = next.records[parent.id]?.netPnl ?? 0;
    const factor = Math.max(0, 1 + dayNetPnl / parent.allocationUsdt);
    const bound = (value: number, min: number, max: number) =>
      Math.round(Math.max(min, Math.min(max, value)) * 10_000) / 10_000;
    const grown = next.world.plants.find((p) => p.id === parent.id)!;
    assert.equal(grown.biomass, bound(parent.biomass * factor, 5, 100));
    assert.equal(grown.vigor, bound(parent.vigor * factor, 0, 100));
    assert.equal(grown.evidence, parent.evidence, "synthetic evidence does not become an empirical gate");
  }
  assert.equal(next.world.events.find((e) => e.type === "daily_tick")?.detail.marketDrivenBiomass, true);
  for (const parent of session.world.plants) {
    const executedToday = next.records[parent.id]?.fills ?? 0;
    if (executedToday > 0) assert.equal(next.world.plants.find((p) => p.id === parent.id)?.inactiveDays, 0);
    else assert.equal(next.world.plants.find((p) => p.id === parent.id)?.inactiveDays, 1);
  }
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
    session = executeHistoricalDay(session, archive, dataset, dataset);
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
  assert.equal(session.schemaVersion, 4, "recycled birth accounting requires its own historical save version");
  assert.throws(() => validateHistoricalSession({ ...session, schemaVersion: 1 }, archive));
  assert.throws(() => validateHistoricalSession({ ...session, schemaVersion: 2 }, archive));
  assert.throws(() => validateHistoricalSession({ ...session, schemaVersion: 3 }, archive));
  assert.throws(() => validateHistoricalSession({ ...session, sourceSha256: "c".repeat(64) }, archive));
  assert.throws(() => validateHistoricalSession({ ...session, records: { "parent-01": { fills: 0, fees: 0, rejected: 0, opportunities: 0, netPnl: 1 } } }, archive));
  assert.throws(() => validateHistoricalSession({ ...session, genomes: {} }, archive));
  assert.throws(() => validateHistoricalSession({ ...session, trades: [{}] }, archive), /saved historical fill/);
  assert.throws(() => validateSolArchive({ ...archive, totalBars: 2 }));
  assert.throws(() => validateSolArchive({ ...archive, days: archive.days.map((d) => ({ ...d, file: "../bad.json" })) }));
});

test("missing historical minutes stop before any day, trades or capital can be committed", async () => {
  const session = createHistoricalSession(archive, "2025-01-02");
  const rows = bars(session.nextDate); rows.splice(700, 1);
  const dataset = await normalizeArchiveBars(rows, archive);
  assert.throws(() => executeHistoricalDay(session, archive, dataset, dataset), /complete real SOL day/);
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
  assert.throws(() => executeHistoricalDay(session, archive, dataset, other), /exact same historical day/);
  const wrongSource = { ...dataset, provenance: { ...dataset.provenance, source: "another archive" } };
  assert.throws(() => executeHistoricalDay(session, archive, dataset, wrongSource), /exact same historical day/);
});

test("game playback is deterministic and independent of external statistical verdicts", async () => {
  // The garden may play on its single available archive even when research
  // comparison reports no independent market evidence.
  const { runControlledExperiment } = await import("./experiments.ts");
  const session = createHistoricalSession(archive, "2025-01-02");
  const before = JSON.stringify(session);
  const dataset = await normalizeArchiveBars(bars("2025-01-02"), archive);
  const warmup = await normalizeArchiveBars([...bars("2025-01-01"), ...bars("2025-01-02")], archive);
  const gameBeforeResearch = executeHistoricalDay(session, archive, dataset, warmup);

  const research = runControlledExperiment({ seeds: ["same-market-1"], days: 7 });
  assert.equal(research.benchmarkConclusion, "not_comparable");
  assert.equal(JSON.stringify(session), before, "external research must never mutate the playable world");

  const gameAfterResearch = executeHistoricalDay(session, archive, dataset, warmup);
  assert.deepEqual(gameAfterResearch, gameBeforeResearch);
  assert.equal(gameAfterResearch.world.day, 1, "there is no research approval gate in the daily clock");
  assertSimulationInvariants(gameAfterResearch.world);
});
