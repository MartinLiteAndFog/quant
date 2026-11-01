import type { PaperReplayReport } from "./paperReplay.ts";
import type { StrategySignalReport } from "./strategySignals.ts";

export type SyntheticReplayEvidence = Readonly<{
  schemaVersion: 1;
  claim: "synthetic_descriptive_only";
  benchmarkConclusion: "not_comparable";
  statisticalStatus: "insufficient_sample";
  independentOpportunityCount: null;
  datasetFingerprint: string;
  strategyId: string;
  strategyStatus: "trade" | "no_trade";
  observations: number;
  executableOpportunityObservations: number;
  generatedTargets: number;
  fills: number;
  rejectedSignals: number;
  dailyMarkCount: number;
  maxDailyDrawdownUsdt: number | null;
  maxDailyDrawdownPct: number | null;
  realizedPricePnlUsdt: number;
  feesUsdt: number;
  fundingUsdt: number;
  netPnlUsdt: number;
  netPnlResidualUsdt: number;
  cashResidualUsdt: number;
}>;

/**
 * Summarize a synthetic run without turning its short fixture into a fitness,
 * confidence, Sharpe, or opportunity-independence claim.
 */
export function buildSyntheticReplayEvidence(
  strategy: StrategySignalReport,
  execution: PaperReplayReport,
): SyntheticReplayEvidence {
  if (strategy.schemaVersion !== 1 || execution.schemaVersion !== 1
    || strategy.claim !== "synthetic_causal_fixture_not_alpha"
    || strategy.benchmarkConclusion !== "not_comparable"
    || execution.benchmarkConclusion !== "not_comparable"
    || !/^[a-f0-9]{64}$/.test(strategy.datasetFingerprint)
    || strategy.datasetFingerprint !== execution.dataset.fingerprint
    || strategy.strategyId.length === 0
    || !Number.isInteger(strategy.observations) || strategy.observations < 0
    || !Number.isInteger(strategy.opportunities) || strategy.opportunities < 0
    || strategy.opportunities > strategy.observations
    || !Number.isInteger(strategy.generatedSignals) || strategy.generatedSignals < 0
    || (strategy.status !== "trade" && strategy.status !== "no_trade")) {
    throw new RangeError("Replay evidence requires a valid, fingerprint-matched synthetic strategy report.");
  }
  if ((strategy.status === "no_trade" && (strategy.opportunities !== 0 || strategy.generatedSignals !== 0))
    || (strategy.status === "trade" && strategy.generatedSignals === 0)) {
    throw new RangeError("Strategy activity status does not reconcile to its opportunity and target counts.");
  }

  const realizedPricePnlUsdt = finite(execution.realizedPricePnl, "realized price PnL");
  const feesUsdt = finite(execution.fees, "fees");
  const fundingUsdt = finite(execution.funding, "funding");
  const netPnlUsdt = finite(execution.netPnl, "net PnL");
  const startingCashUsdt = finite(execution.startingCash, "starting cash");
  const endingCashUsdt = finite(execution.endingCash, "ending cash");
  if (!execution.endedFlat || finite(execution.finalQuantity, "final quantity") !== 0 || execution.riskBreach) {
    throw new RangeError("Replay evidence requires a flat execution without a risk breach.");
  }
  const netPnlResidualUsdt = realizedPricePnlUsdt - feesUsdt + fundingUsdt - netPnlUsdt;
  const cashResidualUsdt = startingCashUsdt + netPnlUsdt - endingCashUsdt;
  if (feesUsdt < 0 || Math.abs(netPnlResidualUsdt) > 0.0001 || Math.abs(cashResidualUsdt) > 0.0001) {
    throw new RangeError("Replay evidence cannot be reported when PnL, costs, or ending cash do not reconcile.");
  }
  if (strategy.status === "no_trade"
    && (execution.fills.length !== 0 || execution.rejectedSignals.length !== 0
      || realizedPricePnlUsdt !== 0 || feesUsdt !== 0 || fundingUsdt !== 0 || netPnlUsdt !== 0
      || startingCashUsdt !== endingCashUsdt || finite(execution.maxMarginUsed, "max margin") !== 0)) {
    throw new RangeError("A no-trade strategy report must have no fills, costs, PnL, or margin use.");
  }

  let peak = startingCashUsdt;
  let maxDailyDrawdownUsdt = 0;
  let maxDailyDrawdownPct = 0;
  let previousDate = "";
  let finalMarkedEquity: number | null = null;
  for (const mark of execution.dailyMarks) {
    const equity = finite(mark.equity, "daily equity");
    const openQuantity = finite(mark.openQuantity, "daily open quantity");
    if (!Number.isInteger(mark.fillCount) || mark.fillCount < 0) {
      throw new RangeError("Daily mark fill counts must be non-negative integers.");
    }
    if (!/^\d{4}-\d{2}-\d{2}$/.test(mark.dateUtc) || mark.dateUtc <= previousDate) {
      throw new RangeError("Daily marks must have strictly increasing UTC dates.");
    }
    previousDate = mark.dateUtc;
    finalMarkedEquity = equity;
    if (strategy.status === "no_trade"
      && (Math.abs(equity - startingCashUsdt) > 0.0001 || openQuantity !== 0 || mark.fillCount !== 0)) {
      throw new RangeError("No-trade daily marks must preserve starting cash with no open position or fills.");
    }
    peak = Math.max(peak, equity);
    const drawdown = Math.max(0, peak - equity);
    maxDailyDrawdownUsdt = Math.max(maxDailyDrawdownUsdt, drawdown);
    if (peak > 0) maxDailyDrawdownPct = Math.max(maxDailyDrawdownPct, drawdown / peak);
  }
  if (finalMarkedEquity !== null && Math.abs(finalMarkedEquity - endingCashUsdt) > 0.0001) {
    throw new RangeError("The final flat daily mark must reconcile to ending cash.");
  }

  return {
    schemaVersion: 1,
    claim: "synthetic_descriptive_only",
    benchmarkConclusion: "not_comparable",
    statisticalStatus: "insufficient_sample",
    independentOpportunityCount: null,
    datasetFingerprint: strategy.datasetFingerprint,
    strategyId: strategy.strategyId,
    strategyStatus: strategy.status,
    observations: strategy.observations,
    executableOpportunityObservations: strategy.opportunities,
    generatedTargets: strategy.generatedSignals,
    fills: execution.fills.length,
    rejectedSignals: execution.rejectedSignals.length,
    dailyMarkCount: execution.dailyMarks.length,
    maxDailyDrawdownUsdt: execution.dailyMarks.length > 0 ? maxDailyDrawdownUsdt : null,
    maxDailyDrawdownPct: execution.dailyMarks.length > 0 ? maxDailyDrawdownPct : null,
    realizedPricePnlUsdt,
    feesUsdt,
    fundingUsdt,
    netPnlUsdt,
    netPnlResidualUsdt,
    cashResidualUsdt,
  };
}

function finite(value: string, label: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new RangeError(`${label} must be finite.`);
  return parsed;
}
