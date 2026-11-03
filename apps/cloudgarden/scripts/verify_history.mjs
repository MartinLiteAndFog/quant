// Reproducible integration check over actual exported SOL candles, not generated fixtures.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { advanceHistoricalSession, createHistoricalSession, loadSolArchive } from "../src/historicalGarden.ts";
const days = Number(process.argv[2] ?? 7);
const origin = process.argv[3];
if (!Number.isInteger(days) || days < 1 || days > 180) throw new Error("Choose 1..180 verification days.");
const httpFetch = globalThis.fetch;
globalThis.fetch = async (input, init) => {
  const path = String(input);
  if (!/^\/market\/(?:manifest|\d{4}-\d{2}-\d{2})\.json$/.test(path)) throw new Error("Unexpected history request.");
  if (origin) return httpFetch(new URL(path, origin), init);
  return new Response(await readFile(fileURLToPath(new URL(`../public${path}`, import.meta.url))));
};
const archive = await loadSolArchive();
let session = createHistoricalSession(archive, "2026-01-01");
for (let i = 0; i < days; i += 1) {
  session = await advanceHistoricalSession(session, archive);
  if ((i + 1) % 14 === 0) console.log(`Verified ${i + 1} historical days; NAV ${session.world.capital.navUsdt}`);
}
const fills = Object.values(session.records).reduce((sum, r) => sum + r.fills, 0);
assert.ok(fills > 0);
assert.notEqual(session.world.capital.navUsdt, 100);
assert.ok(session.world.capital.paperFeesUsdt > 0);
assert.equal(session.world.day, days);
assert.ok(session.world.events.filter((e) => e.type === "paper_replay").every((e) => e.detail.synthetic === false));
console.log(JSON.stringify({ sourceSha256: archive.sourceSha256, archiveMinutes: archive.totalBars,
  firstDay: session.startDate, lastDay: session.history.at(-1).date, days, fills,
  navUsdt: session.world.capital.navUsdt, netPnlUsdt: Number((session.world.capital.navUsdt - 100).toFixed(4)),
  feesUsdt: session.world.capital.paperFeesUsdt, seedlingEntitlementsUsed: session.world.capital.seedlingEntitlements.filter((s) => s.status === "used").length,
  plants: session.world.plants.length, source: origin ? "HTTP archive assets" : "local archive assets",
  benchmark: "cash_no_trade", verdict: "not_comparable", note: "Integration evidence only; source-derived daily-flat variants, assumed costs, unverified venue, no funding model or OOS claim." }, null, 2));
