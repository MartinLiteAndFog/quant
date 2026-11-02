import assert from "node:assert/strict";
import test from "node:test";
import { TRADINGVIEW_REFERENCE_SEEDS, auditTradingViewReferenceManifest } from "./tradingViewSeedCatalog.ts";

function syntheticManifest() {
  return {
    schemaVersion: 1, claim: "tradingview_trade_reference_only_not_signal_parity", secretsExcluded: true,
    seeds: TRADINGVIEW_REFERENCE_SEEDS.map(seed => ({
      id: seed.id, originalXlsxSha256: seed.originalXlsxSha256.padEnd(64, "0"),
      referenceTradeCount: seed.expectedClosedTrades, openPositionCount: seed.expectedOpenTrades,
      trades: {
        closed: Array.from({ length: seed.expectedClosedTrades }, (_, i) => ({
          number: i + 1, side: "long", status: "closed_tradingview_reference",
          entryUtc: "2026-09-07T00:00:00Z", exitUtc: "2026-09-07T00:05:00Z",
          entryPrice: 100, exitPrice: 101, quantity: 1,
          referenceCommissionUsdt: 0.12, referenceNetPnlUsdt: 0.88,
        })),
        open: Array.from({ length: seed.expectedOpenTrades }, (_, i) => ({
          number: seed.expectedClosedTrades + i + 1,
          side: "long", status: "open_not_booked", entryUtc: "2026-10-17T00:00:00Z",
          entryPrice: 100, quantity: 1,
        })),
      },
    })),
  };
}

test("all nine TradingView seed variants retain identity and stay non-executable", () => {
  assert.equal(TRADINGVIEW_REFERENCE_SEEDS.length, 9);
  assert.equal(new Set(TRADINGVIEW_REFERENCE_SEEDS.map(seed => seed.id)).size, 9);
  assert.equal(TRADINGVIEW_REFERENCE_SEEDS.reduce((n,seed) => n + seed.expectedClosedTrades,0),1193);
  assert.equal(TRADINGVIEW_REFERENCE_SEEDS.reduce((n,seed) => n + seed.expectedOpenTrades,0),4);
  for (const seed of TRADINGVIEW_REFERENCE_SEEDS) {
    assert.equal(seed.signalFormulaStatus,"external_source_missing");
    assert.equal(seed.executionMode,"reference_only_no_live_orders");
    assert.equal(seed.entryLookbackBars,null);
  }
});

test("reference trade audits count closed and open positions without synthetic gains", () => {
  const result = auditTradingViewReferenceManifest(syntheticManifest());
  assert.equal(result.candidates,9);
  assert.equal(result.closedTrades,1193);
  assert.equal(result.openPositions,4);
  assert.equal(result.liveTradingAllowed,false);
  assert.equal(result.claim,"reference_replay_only_not_independent_signal_parity");
});

test("fail closed on duplicate numbers, fake PnL or an unclosed trade booked as closed", () => {
  const duplicate = syntheticManifest();
  duplicate.seeds[0]!.trades.closed[1]!.number = 1;
  assert.throws(() => auditTradingViewReferenceManifest(duplicate),/duplicate trade/i);
  const pnl = syntheticManifest();
  pnl.seeds[0]!.trades.closed[0]!.referenceNetPnlUsdt = 500;
  assert.throws(() => auditTradingViewReferenceManifest(pnl),/price\/PnL/);
  const open = syntheticManifest();
  (open.seeds[0]!.trades.open[0]! as { referenceNetPnlUsdt?: number }).referenceNetPnlUsdt = 42;
  assert.throws(() => auditTradingViewReferenceManifest(open),/Open positions/);
});