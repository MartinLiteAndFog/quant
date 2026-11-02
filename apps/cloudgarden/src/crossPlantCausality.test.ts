import assert from 'node:assert/strict';
import test from 'node:test';
import { createHistoricalSession, executeHistoricalDay, normalizeArchiveBars, type HistoricalSession, type SolArchive } from './historicalGarden.ts';
import type { MarketBarInput } from './paperReplay.ts';
import { getKrakenProfile } from './krakenProfiles.ts';

test('later prices in one plant cannot resize another plant\'s earlier Kraken fills', async () => {
  const dates = ['2025-01-01', '2025-01-02', '2025-01-03'];
  const archive: SolArchive = {
    schemaVersion: 1, source: 'Cross-plant causality contract fixture', sourceSha256: 'a'.repeat(64),
    symbol: 'SOL-USDT', venue: 'unverified', firstDate: dates[0]!, lastDate: dates[2]!,
    totalBars: 4320, missingMinutes: 0, duplicateCount: 0,
    days: dates.map((date) => ({ date, barCount: 1440, gapCount: 0, missingMinutes: 0,
      file: `${date}.json`, sha256: 'b'.repeat(64) })),
  };
  const start = Date.parse(`${dates[0]}T00:00:00Z`);
  const cutoff = start + 65 * 60_000;
  const rows = (tailPrice: number): MarketBarInput[] => Array.from({ length: 1440 }, (_, index) => {
    const price = String(index < 65 ? 100 + index * .02 : tailPrice);
    return { timestamp: new Date(start + index * 60_000).toISOString(),
      open: price, high: price, low: price, close: price, volume: '1000' };
  });
  const initial = createHistoricalSession(archive);
  // This fixture isolates cross-account sizing: every source profile must actually trade the prefix.
  // Use the supported pre-EW canonical parent profiles. EW causality has separate tests and
  // a valid EW filter can correctly suppress every entry in this deliberately monotonic prefix.
  const session: HistoricalSession = { ...initial, krakenProfiles: Object.fromEntries(
    Object.entries(initial.krakenProfiles!).map(([id, profile]) => [id, getKrakenProfile(profile.id)]),
  ) };
  const outcomes: HistoricalSession[] = [];
  for (const tail of [105, 90]) {
    const dataset = await normalizeArchiveBars(rows(tail), archive);
    outcomes.push(await executeHistoricalDay(session, archive, dataset, dataset));
  }
  // Ensure the suffix perturbation changes actual account outcomes, not just unused fixture data.
  assert.notEqual(outcomes[0]!.records['parent-01']!.netPnl, outcomes[1]!.records['parent-01']!.netPnl);
  const earlyFills = (index: number, plantId: string) => outcomes[index]!.trades
    .filter((fill) => fill.plantId === plantId && Date.parse(fill.executedAt) < cutoff)
    .map((fill) => ({ executedAt: fill.executedAt, side: fill.side, quantity: fill.filledQuantity,
      price: fill.price, fee: fill.fee, orderId: fill.orderId }));
  for (const plantId of ['parent-02', 'parent-03']) {
    assert.ok(earlyFills(0, plantId).length > 0, 'The shared prefix must contain observable executions.');
    assert.deepEqual(earlyFills(0, plantId), earlyFills(1, plantId),
      `${plantId}: full-day PnL from an earlier-processed plant must never affect earlier sizing.`);
  }
});
