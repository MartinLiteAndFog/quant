// Full-source integration evidence: verifies exported bytes and traverses every observed archive date repeatedly.
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { advanceHistoricalSession, createHistoricalSession, loadSolArchive, validateHistoricalSession } from "../src/historicalGarden.ts";
const rounds = Number(process.argv[2] ?? 2);
const output = process.argv[3];
if (!Number.isInteger(rounds) || rounds < 1 || rounds > 100) throw new Error("Choose 1..100 complete archive rounds.");
globalThis.fetch = async (input) => {
  const path = String(input);
  if (!/^\/market\/(?:manifest|\d{4}-\d{2}-\d{2})\.json$/.test(path)) throw new Error("Unexpected archive path.");
  return new Response(await readFile(fileURLToPath(new URL(`../public${path}`, import.meta.url))));
};
const archive = await loadSolArchive();
let session = createHistoricalSession(archive);
const started = Date.now();
let maxHistory = 0, maxTrades = 0, maxPlants = 0, maxLivingPlants = 0, maxAccountCurve = 0, observedPartial = 0, observedGaps = 0;
for (let cycle = 1; cycle <= rounds; cycle += 1) {
  for (let i = 0; i < archive.days.length; i += 1) {
    assert.equal(session.nextDate, archive.days[i].date);
    session = await advanceHistoricalSession(session, archive);
    maxHistory = Math.max(maxHistory, session.history.length);
    maxTrades = Math.max(maxTrades, session.trades.length);
    maxPlants = Math.max(maxPlants, session.world.plants.length);
    maxLivingPlants = Math.max(maxLivingPlants, session.world.plants.filter((p) => p.status === "active").length);
    maxAccountCurve = Math.max(maxAccountCurve, ...Object.values(session.records).map((r) => r.equityHistory?.length ?? 0));
    assert.ok(maxPlants <= 303 && maxLivingPlants <= 200 && maxAccountCurve <= 300);
    if (archive.days[i].barCount < 1440) observedPartial += 1;
    if (archive.days[i].gapCount) observedGaps += 1;
    if ((i + 1) % 100 === 0) console.log(JSON.stringify({ cycle, processedDates: i + 1, totalDates: archive.days.length, nav: session.world.capital.navUsdt,
      fills: session.loop.totalFills, capitalHalted: session.loop.capitalHalted, elapsedSeconds: Math.round((Date.now() - started) / 1000) }));
  }
  assert.equal(session.loop.cycle, cycle + 1);
  assert.equal(session.loop.totalDays, archive.days.length * cycle);
  assert.equal(session.loop.totalBars, archive.totalBars * cycle);
  assert.equal(session.nextDate, archive.firstDate);
  assert.equal(session.world.day, archive.days.length * cycle);
  assert.equal(session.world.capital.navUsdt, session.loop.completedRounds.at(-1).nav);
  assert.ok(session.world.plants.filter((p) => p.status === "active").length <= 200);
  assert.ok(session.world.plants.length <= 303);
  assert.ok(Object.values(session.krakenStates).every((state) => state.quantity === 0 && !state.pending.length));
  assert.deepEqual(validateHistoricalSession(JSON.parse(JSON.stringify(session)), archive), session);
}
assert.ok(session.loop.totalFills > 0);
assert.ok(session.loop.totalFees > 0);
assert.ok(maxHistory <= 300 && maxTrades <= 400);
assert.equal(session.loop.completedRounds.length, Math.min(rounds, 64));
const evidence = { sourceSha256: archive.sourceSha256, source: archive.source, firstDate: archive.firstDate, lastDate: archive.lastDate,
  archiveDates: archive.days.length, archiveMinutes: archive.totalBars, rounds, totalDays: session.loop.totalDays, totalObservedBars: session.loop.totalBars,
  totalFills: session.loop.totalFills, totalFees: session.loop.totalFees, externalDepositsUsdt: session.world.capital.externalDepositsUsdt,
  netTradingPnlUsdt: Number((session.world.capital.navUsdt - 100 - (session.world.capital.externalDepositsUsdt ?? 0)).toFixed(4)),
  livingPlants: session.world.plants.filter((p) => p.status === "active").length, lifetimeBirths: session.world.dynamicEcology.birthCount,
  retainedPlants: session.world.plants.length, retiredPlantCount: session.retiredPlantCount ?? 0, funding: session.funding, completedRounds: session.loop.completedRounds, maxRetainedHistory: maxHistory,
  maxRetainedTrades: maxTrades, maxRetainedPlants: maxPlants, maxLivingPlants, maxAccountCurve, partialDateVisits: observedPartial, gappedDateVisits: observedGaps, elapsedSeconds: Math.round((Date.now() - started) / 1000),
  benchmark: "cash_no_trade", verdict: "not_comparable", note: "Full-source execution and bounded loop evidence. Assumed costs and unverified venue; no OOS or alpha claim." };
if (output) await writeFile(output, `${JSON.stringify(evidence, null, 2)}\n`);
console.log(JSON.stringify(evidence, null, 2));
