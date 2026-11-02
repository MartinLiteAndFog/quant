/**
 * TradingView Strategy Builder references imported as candidate *seed varieties*.
 * They are not executable strategy formulas: "Elliott Wave Detector Pro" and its
 * MTF RSI filter are external opaque inputs. Historical trades can be audited,
 * never promoted into causal fitness or live orders.
 *
 * No webhook payloads, credentials or raw XLSX files belong in this module.
 */
export type TvReferenceSeed = Readonly<{
  id: `tvseed-${string}`;
  name: string;
  originalXlsxSha256: string;
  symbol: "BINANCE:SOLUSDT" | "KUCOIN:SOLUSDT" | "KUCOIN:BTCUSDT";
  chartType: "Renko";
  chartTimeframe: "5 minutes";
  entrySource: "Elliott Wave Detector Pro";
  longOperator: string;
  shortOperator: string;
  mtfRsiEnabled: boolean;
  longRsiThreshold: 55;
  shortRsiThreshold: 75;
  stopLossBps: number;
  trailingDeviationBps: number;
  leverage: number;
  orderSizePercent: number;
  initialCapitalUsdt: number;
  slippageTicks: number;
  expectedClosedTrades: number;
  expectedOpenTrades: 0 | 1;
  signalFormulaStatus: "external_source_missing";
  executionMode: "reference_only_no_live_orders";
  // The builder's HH/LL Lookback=20 is attached to INACTIVE exit modes;
  // it must not be mistaken for the EW entry lookback.
  entryLookbackBars: null;
  ewPivotLength: null;
}>;

const common = {
  chartType: "Renko", chartTimeframe: "5 minutes",
  entrySource: "Elliott Wave Detector Pro", longRsiThreshold: 55,
  shortRsiThreshold: 75, orderSizePercent: 90,
  signalFormulaStatus: "external_source_missing",
  executionMode: "reference_only_no_live_orders",
  entryLookbackBars: null, ewPivotLength: null,
} as const;

/** Nine distinct XLSX snapshots. The BTC 20x/40x pair has one underlying
 * trade timeline but two economic setups: never merge the reference records.
 */
export const TRADINGVIEW_REFERENCE_SEEDS: readonly TvReferenceSeed[] = [
  { ...common, id: "tvseed-e667c46efc4b", name: "TV Binance SOL · 17 Oct", originalXlsxSha256: "e667c46efc4b0815c6bc3e29f50bbdd993ef3fe7eb79a8cf8a4137aa5fd1cfc0", symbol: "BINANCE:SOLUSDT", longOperator: "PRICE =", shortOperator: "PRICE =", mtfRsiEnabled: true, stopLossBps: 90, trailingDeviationBps: 80, leverage: 10, initialCapitalUsdt: 100, slippageTicks: 0, expectedClosedTrades: 85, expectedOpenTrades: 1 },
  { ...common, id: "tvseed-7d1a0113a983", name: "TV Binance SOL · 18 Oct A", originalXlsxSha256: "7d1a0113a98331fa73911bd2f1238d4bea248587debadeae36bdf9cafba9efd0", symbol: "BINANCE:SOLUSDT", longOperator: "Price >", shortOperator: "Source <", mtfRsiEnabled: true, stopLossBps: 90, trailingDeviationBps: 80, leverage: 10, initialCapitalUsdt: 100, slippageTicks: 0, expectedClosedTrades: 197, expectedOpenTrades: 0 },
  { ...common, id: "tvseed-21ab4b55912e", name: "TV Binance SOL · 18 Oct B", originalXlsxSha256: "21ab4b55912e6be0bff10560041693c3ca52a2931392a9b5c660500c501161b7", symbol: "BINANCE:SOLUSDT", longOperator: "Price >", shortOperator: "Source <", mtfRsiEnabled: false, stopLossBps: 90, trailingDeviationBps: 80, leverage: 10, initialCapitalUsdt: 100, slippageTicks: 0, expectedClosedTrades: 30, expectedOpenTrades: 0 },
  { ...common, id: "tvseed-1f159166edd1", name: "TV Binance SOL · 18 Oct C", originalXlsxSha256: "1f159166edd1678759d5593e30531fd4f77e3be58a3fdbb9c8e25373d7336ddc", symbol: "BINANCE:SOLUSDT", longOperator: "Price >", shortOperator: "Source <", mtfRsiEnabled: true, stopLossBps: 90, trailingDeviationBps: 80, leverage: 10, initialCapitalUsdt: 100, slippageTicks: 0, expectedClosedTrades: 194, expectedOpenTrades: 0 },
  { ...common, id: "tvseed-acc3c1e067bc", name: "TV Binance SOL · 18 Oct D", originalXlsxSha256: "acc3c1e067bcd061389be0be3be596a626b51273de146542ac6b7fa5e481adb8", symbol: "BINANCE:SOLUSDT", longOperator: "Price >", shortOperator: "PRICE =", mtfRsiEnabled: true, stopLossBps: 90, trailingDeviationBps: 80, leverage: 10, initialCapitalUsdt: 100, slippageTicks: 0, expectedClosedTrades: 170, expectedOpenTrades: 0 },
  { ...common, id: "tvseed-cccf89c57ded", name: "TV Binance SOL · 18 Oct baseline", originalXlsxSha256: "cccf89c57ded315864a269eee69f1d7bf48be1bb1c1cb0e042e2451963d0615a", symbol: "BINANCE:SOLUSDT", longOperator: "PRICE =", shortOperator: "PRICE =", mtfRsiEnabled: true, stopLossBps: 90, trailingDeviationBps: 80, leverage: 10, initialCapitalUsdt: 100, slippageTicks: 0, expectedClosedTrades: 87, expectedOpenTrades: 0 },
  { ...common, id: "tvseed-a217c7084948", name: "TV KuCoin BTC · 40x", originalXlsxSha256: "a217c70849489d16f083722b32fc92750735b6cc54bb9ebd3eb07714772ae1fb", symbol: "KUCOIN:BTCUSDT", longOperator: "Price >", shortOperator: "Source >", mtfRsiEnabled: true, stopLossBps: 90, trailingDeviationBps: 50, leverage: 40, initialCapitalUsdt: 250, slippageTicks: 3, expectedClosedTrades: 161, expectedOpenTrades: 1 },
  { ...common, id: "tvseed-e814318c64ad", name: "TV KuCoin BTC · 20x", originalXlsxSha256: "e814318c64ad2884da18c01c806c84cb664b12f290a1335eb02b789f92743b00", symbol: "KUCOIN:BTCUSDT", longOperator: "Price >", shortOperator: "Source >", mtfRsiEnabled: true, stopLossBps: 90, trailingDeviationBps: 50, leverage: 20, initialCapitalUsdt: 100, slippageTicks: 0, expectedClosedTrades: 161, expectedOpenTrades: 1 },
  { ...common, id: "tvseed-72a83f5de466", name: "TV KuCoin SOL · 17 Oct", originalXlsxSha256: "72a83f5de4669701dc88734b4c28a52388c937ae99baba71f744e1121eacb53f", symbol: "KUCOIN:SOLUSDT", longOperator: "PRICE =", shortOperator: "PRICE =", mtfRsiEnabled: true, stopLossBps: 110, trailingDeviationBps: 80, leverage: 10, initialCapitalUsdt: 100, slippageTicks: 0, expectedClosedTrades: 108, expectedOpenTrades: 1 },
] as const;

export type TvReferenceTrade = Readonly<{
  number: number;
  side: "long" | "short";
  entryUtc: string;
  entryPrice: number;
  quantity: number;
  status: "closed_tradingview_reference" | "open_not_booked";
  exitUtc?: string;
  exitPrice?: number;
  referenceNetPnlUsdt?: number;
  referenceCommissionUsdt?: number;
  displayPrecisionPnlResidualUsdt?: number;
}>;

export type TvSourceManifest = Readonly<{
  schemaVersion: 1;
  claim: "tradingview_trade_reference_only_not_signal_parity";
  secretsExcluded: true;
  seeds: readonly Readonly<{
    id: string;
    originalXlsxSha256: string;
    referenceTradeCount: number;
    openPositionCount: number;
    trades: { closed: readonly TvReferenceTrade[]; open: readonly TvReferenceTrade[] };
  }>[];
}>;

export type TvReferenceAudit = Readonly<{
  candidates: number;
  closedTrades: number;
  openPositions: number;
  displayPnlResidualMaxUsdt: number;
  claim: "reference_replay_only_not_independent_signal_parity";
  liveTradingAllowed: false;
}>;

/** Fail-closed, read-only validation of the genuine XLSX-derived trade tape.
 * The repeated TV Entry/Exit spreadsheet rows represent ONE trade and ONE fee,
 * not two fills to book. Open rows remain UNBOOKED until a closing reference
 * exists. Check entry/exit price arithmetic only as a rounding diagnostic.
 */
export function auditTradingViewReferenceManifest(source: unknown): TvReferenceAudit {
  if (!source || typeof source !== "object") throw new RangeError("Reference manifest must be an object.");
  const data = source as TvSourceManifest;
  if (data.schemaVersion !== 1 || data.claim !== "tradingview_trade_reference_only_not_signal_parity"
    || data.secretsExcluded !== true || !Array.isArray(data.seeds)
    || data.seeds.length !== TRADINGVIEW_REFERENCE_SEEDS.length) {
    throw new RangeError("Reference manifest schema/size/claim mismatch.");
  }
  const catalog = new Map(TRADINGVIEW_REFERENCE_SEEDS.map(seed => [seed.id, seed]));
  const seen = new Set<string>();
  let closedTrades = 0;
  let openPositions = 0;
  let displayPnlResidualMaxUsdt = 0;
  for (const exported of data.seeds) {
    const expected = catalog.get(exported.id);
    if (!expected || seen.has(exported.id)
      || !/^[a-f0-9]{64}$/.test(exported.originalXlsxSha256)
      || exported.originalXlsxSha256 !== expected.originalXlsxSha256
      || !exported.trades || !Array.isArray(exported.trades.closed)
      || !Array.isArray(exported.trades.open)
      || exported.referenceTradeCount !== expected.expectedClosedTrades
      || exported.openPositionCount !== expected.expectedOpenTrades
      || exported.trades.closed.length !== expected.expectedClosedTrades
      || exported.trades.open.length !== expected.expectedOpenTrades) {
      throw new RangeError(`Unrecognized or inconsistent seed: ${exported.id}`);
    }
    seen.add(exported.id);
    const perSeedNumbers = new Set<number>();
    for (const trade of [...exported.trades.closed, ...exported.trades.open]) {
      if (!trade || !Number.isSafeInteger(trade.number) || trade.number <= 0
        || perSeedNumbers.has(trade.number) || !["long", "short"].includes(trade.side)
        || !Number.isFinite(Date.parse(trade.entryUtc))
        || !Number.isFinite(trade.entryPrice) || trade.entryPrice <= 0
        || !Number.isFinite(trade.quantity) || trade.quantity <= 0) {
        throw new RangeError(`Invalid or duplicate trade in ${exported.id}`);
      }
      perSeedNumbers.add(trade.number);
      if (trade.status === "open_not_booked") {
        if (!exported.trades.open.includes(trade) || trade.exitUtc !== undefined
          || trade.referenceNetPnlUsdt !== undefined) {
          throw new RangeError("Open positions cannot be booked as closed PnL.");
        }
        openPositions += 1;
        continue;
      }
      if (trade.status !== "closed_tradingview_reference" || !exported.trades.closed.includes(trade)
        || !Number.isFinite(Date.parse(trade.exitUtc ?? ""))
        || Date.parse(trade.exitUtc!) < Date.parse(trade.entryUtc)
        || !Number.isFinite(trade.exitPrice) || trade.exitPrice! <= 0
        || !Number.isFinite(trade.referenceCommissionUsdt) || trade.referenceCommissionUsdt! < 0
        || !Number.isFinite(trade.referenceNetPnlUsdt)) {
        throw new RangeError(`Invalid closed trade in ${exported.id}`);
      }
      const sign = trade.side === "long" ? 1 : -1;
      const estimate = (trade.exitPrice! - trade.entryPrice) * trade.quantity * sign
        - trade.referenceCommissionUsdt!;
      const residual = Math.abs(estimate - trade.referenceNetPnlUsdt!);
      if (residual > .25) throw new RangeError(`Trade ${trade.number} price/PnL reconciliation exceeds display precision.`);
      displayPnlResidualMaxUsdt = Math.max(displayPnlResidualMaxUsdt, residual);
      closedTrades += 1;
    }
  }
  if (seen.size !== catalog.size) throw new RangeError("Incomplete seed catalog.");
  return { candidates: seen.size, closedTrades, openPositions, displayPnlResidualMaxUsdt,
    claim: "reference_replay_only_not_independent_signal_parity", liveTradingAllowed: false };
}