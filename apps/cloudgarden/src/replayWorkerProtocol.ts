import type { SyntheticReplayCycleResult } from "./replayCycle.ts";

/** Validate the untrusted worker boundary before a replay result reaches the ledger. */
export function readSyntheticReplayWorkerResult(
  message: unknown,
  expectedId: string,
  expectedDays: number,
  expectedStart: string,
): SyntheticReplayCycleResult {
  if (!expectedId || !Number.isInteger(expectedDays) || expectedDays < 1 || expectedDays > 90
    || typeof expectedStart !== "string" || !Number.isFinite(Date.parse(expectedStart))) {
    throw new RangeError("Expected replay identity and horizon must be valid.");
  }
  const startMs = Date.parse(expectedStart);
  const endExclusiveMs = startMs + expectedDays * 86_400_000;
  const canonicalStart = new Date(startMs).toISOString();
  const canonicalEnd = new Date(endExclusiveMs).toISOString();
  const envelope = record(message);
  if (!envelope || envelope.id !== expectedId) throw new RangeError("Replay worker response ID does not match the active run.");
  const hasResult = Object.hasOwn(envelope, "result");
  const hasError = Object.hasOwn(envelope, "error");
  if (hasResult === hasError) throw new RangeError("Replay worker response must contain exactly one result or error.");
  if (hasError) {
    if (typeof envelope.error !== "string" || envelope.error.length === 0) {
      throw new RangeError("Replay worker returned an invalid error response.");
    }
    throw new Error(envelope.error);
  }

  const result = record(envelope.result);
  const execution = record(result?.execution);
  const dataset = record(execution?.dataset);
  const strategy = record(result?.strategy);
  const genome = record(result?.genome);
  const evidence = record(result?.evidence);
  if (!result || !execution || !dataset || !strategy || !genome || !evidence) {
    throw new RangeError("Replay worker result is missing required reports.");
  }

  const fingerprint = dataset.fingerprint;
  const fills = execution.fills;
  const rejectedSignals = execution.rejectedSignals;
  const dailyMarks = execution.dailyMarks;
  const fundingCashflows = execution.fundingCashflows;
  const days = execution.days;
  const startingCash = finite(execution.startingCash);
  const endingCash = finite(execution.endingCash);
  const netPnl = finite(execution.netPnl);
  const realizedPricePnl = finite(execution.realizedPricePnl);
  const fees = finite(execution.fees);
  const funding = finite(execution.funding);
  const allocationLimit = finite(execution.allocationLimit);
  const maxMargin = finite(execution.maxMarginUsed);
  const finalQuantity = execution.finalQuantity;
  const fillStream = Array.isArray(fills) ? fills : [];
  const fundingStream = Array.isArray(fundingCashflows) ? fundingCashflows : [];
  const fillReconciliation = reconcileFills(fillStream, strategy.strategyId, fingerprint);
  const outcomeIds = [...fillStream, ...(Array.isArray(rejectedSignals) ? rejectedSignals : [])]
    .map((outcome) => record(outcome)?.orderId);
  const fundingTotal = sumDecimals(fundingStream.map((cashflow) => record(cashflow)?.amount));
  const marksHaveExpectedDates = Array.isArray(dailyMarks) && dailyMarks.every((mark, index) => {
    const dayMark = record(mark);
    const expectedDate = new Date(startMs + index * 86_400_000).toISOString().slice(0, 10);
    return dayMark?.dateUtc === expectedDate;
  });
  if (days !== expectedDays || !Array.isArray(dailyMarks)
    || dailyMarks.length !== expectedDays || !dailyMarks.every(isDailyMark)
    || !marksHaveExpectedDates
    || dataset.start !== canonicalStart || dataset.endExclusive !== canonicalEnd
    || !Array.isArray(fills) || !fills.every((fill) => isPaperFill(fill, fingerprint, strategy.strategyId, startMs, endExclusiveMs))
    || !Array.isArray(rejectedSignals) || !rejectedSignals.every(isRejectedSignal)
    || outcomeIds.some((orderId, index) => typeof orderId !== "string" || outcomeIds.indexOf(orderId) !== index)
    || !Array.isArray(fundingCashflows) || !fundingCashflows.every(isFundingCashflow)
    || dataset.schemaVersion !== 1 || dataset.barCount !== expectedDays * 24 * 60
    || dataset.gapCount !== 0 || dataset.missingBars !== 0
    || dataset.synthetic !== true || dataset.eligibility !== "synthetic_demo_only"
    || typeof fingerprint !== "string" || !/^[a-f0-9]{64}$/.test(fingerprint)
    || strategy.datasetFingerprint !== fingerprint || evidence.datasetFingerprint !== fingerprint
    || execution.endedFlat !== true || finalQuantity !== "0" || execution.riskBreach !== false
    || execution.schemaVersion !== 1 || execution.benchmark !== "cash_no_trade"
    || execution.benchmarkConclusion !== "not_comparable"
    || !isNonnegativeInteger(strategy.observations) || !isNonnegativeInteger(strategy.opportunities)
    || strategy.opportunities > strategy.observations || !isNonnegativeInteger(strategy.generatedSignals)
    || typeof strategy.strategyId !== "string" || strategy.strategyId.length === 0
    || strategy.claim !== "synthetic_causal_fixture_not_alpha" || strategy.benchmarkConclusion !== "not_comparable"
    || genome.strategyId !== strategy.strategyId || evidence.strategyId !== strategy.strategyId
    || genome.schemaVersion !== 1 || !isNonnegativeInteger(genome.lookbackBars) || genome.lookbackBars < 1 || genome.lookbackBars > 120
    || !isNonnegativeInteger(genome.entryThresholdBps) || genome.entryThresholdBps < 1 || genome.entryThresholdBps > 3_000
    || !isNonnegativeInteger(genome.exitThresholdBps) || genome.exitThresholdBps < 0 || genome.exitThresholdBps > genome.entryThresholdBps
    || !isFiniteDecimal(genome.targetQuantity) || Number(genome.targetQuantity) <= 0
    || (strategy.status !== "trade" && strategy.status !== "no_trade")
    || evidence.strategyStatus !== strategy.status
    || evidence.schemaVersion !== 1 || evidence.claim !== "synthetic_descriptive_only"
    || evidence.benchmarkConclusion !== "not_comparable"
    || evidence.statisticalStatus !== "insufficient_sample"
    || evidence.independentOpportunityCount !== null
    || evidence.observations !== strategy.observations
    || evidence.executableOpportunityObservations !== strategy.opportunities
    || evidence.generatedTargets !== strategy.generatedSignals
    || evidence.fills !== fills.length || evidence.rejectedSignals !== rejectedSignals.length
    || fills.length + rejectedSignals.length !== strategy.generatedSignals
    || evidence.dailyMarkCount !== expectedDays
    || !isNonnegativeInteger(evidence.observations) || !isNonnegativeInteger(evidence.executableOpportunityObservations)
    || !isNonnegativeInteger(evidence.generatedTargets) || !isNonnegativeInteger(evidence.fills)
    || !isNonnegativeInteger(evidence.rejectedSignals)
    || !Number.isFinite(startingCash) || !Number.isFinite(endingCash) || !Number.isFinite(netPnl)
    || !Number.isFinite(realizedPricePnl) || !Number.isFinite(fees) || !Number.isFinite(funding)
    || !Number.isFinite(allocationLimit) || !Number.isFinite(maxMargin)
    || startingCash <= 0 || allocationLimit <= 0 || allocationLimit > startingCash
    || fees < 0 || maxMargin < 0 || maxMargin > allocationLimit + 0.0001
    || Math.abs(realizedPricePnl - fees + funding - netPnl) > 0.0001
    || !fillReconciliation.valid || Math.abs(fillReconciliation.realizedPricePnl - realizedPricePnl) > 0.000001
    || Math.abs(fillReconciliation.fees - fees) > 0.000001
    || !Number.isFinite(fundingTotal) || Math.abs(fundingTotal - funding) > 0.000001
    || funding !== 0 || fundingStream.length !== 0
    || Math.abs(startingCash + netPnl - endingCash) > 0.0001
    || !isFiniteNumber(evidence.realizedPricePnlUsdt) || Math.abs(evidence.realizedPricePnlUsdt - realizedPricePnl) > 0.0001
    || !isFiniteNumber(evidence.feesUsdt) || Math.abs(evidence.feesUsdt - fees) > 0.0001
    || !isFiniteNumber(evidence.fundingUsdt) || Math.abs(evidence.fundingUsdt - funding) > 0.0001
    || !isFiniteNumber(evidence.netPnlUsdt) || Math.abs(evidence.netPnlUsdt - netPnl) > 0.0001
    || !isFiniteNumber(evidence.maxDailyDrawdownUsdt) || evidence.maxDailyDrawdownUsdt < 0
    || !isFiniteNumber(evidence.maxDailyDrawdownPct) || evidence.maxDailyDrawdownPct < 0 || evidence.maxDailyDrawdownPct > 1
    || typeof evidence.cashResidualUsdt !== "number" || !Number.isFinite(evidence.cashResidualUsdt) || Math.abs(evidence.cashResidualUsdt) > 0.0001
    || typeof evidence.netPnlResidualUsdt !== "number" || !Number.isFinite(evidence.netPnlResidualUsdt) || Math.abs(evidence.netPnlResidualUsdt) > 0.0001) {
    throw new RangeError("Replay worker result does not reconcile to the active synthetic run.");
  }
  const dates = dailyMarks.map((mark) => (mark as Record<string, unknown>).dateUtc as string);
  if (dates.some((date, index) => index > 0 && date <= (dates[index - 1] ?? ""))
    || finite((dailyMarks.at(-1) as Record<string, unknown>).equity) !== endingCash
    || (dailyMarks.at(-1) as Record<string, unknown>).openQuantity !== "0"
    || dailyMarks.reduce((total, mark) => total + ((mark as Record<string, unknown>).fillCount as number), 0) !== fills.length) {
    throw new RangeError("Replay worker daily marks do not reconcile to the closed result.");
  }
  return result as unknown as SyntheticReplayCycleResult;
}

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null ? value as Record<string, unknown> : null;
}

function finite(value: unknown): number {
  return isFiniteDecimal(value) ? Number(value) : Number.NaN;
}

function isFiniteDecimal(value: unknown): value is string {
  return typeof value === "string" && /^-?(?:0|[1-9]\d*)(?:\.\d{1,8})?$/.test(value) && Number.isFinite(Number(value));
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isNonnegativeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0;
}

function isDailyMark(value: unknown): boolean {
  const mark = record(value);
  if (!mark || typeof mark.dateUtc !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(mark.dateUtc)) return false;
  const parsedDate = new Date(`${mark.dateUtc}T00:00:00Z`);
  return !!mark
    && Number.isFinite(parsedDate.getTime())
    && parsedDate.toISOString().slice(0, 10) === mark.dateUtc
    && isFiniteDecimal(mark.equity) && isFiniteDecimal(mark.openQuantity)
    && isNonnegativeInteger(mark.fillCount);
}

function isPaperFill(
  value: unknown,
  expectedFingerprint: unknown,
  expectedStrategyId: unknown,
  startMs: number,
  endExclusiveMs: number,
): boolean {
  const fill = record(value);
  if (!fill || typeof fill.orderId !== "string" || fill.orderId.length === 0
    || typeof fill.strategyId !== "string" || fill.strategyId !== expectedStrategyId
    || (fill.side !== "buy" && fill.side !== "sell")
    || ![fill.targetQuantity, fill.filledQuantity, fill.referencePrice, fill.price, fill.notional, fill.fee, fill.executionCost]
      .every(isFiniteDecimal)
    || typeof fill.executedAt !== "string" || !Number.isFinite(Date.parse(fill.executedAt))
    || Date.parse(fill.executedAt) < startMs || Date.parse(fill.executedAt) >= endExclusiveMs
    || typeof fill.datasetFingerprint !== "string" || fill.datasetFingerprint !== expectedFingerprint) return false;
  return Number(fill.filledQuantity) > 0 && Number(fill.referencePrice) > 0 && Number(fill.price) > 0
    && Number(fill.notional) > 0 && Number(fill.fee) >= 0 && Number(fill.executionCost) >= 0;
}

function sumDecimals(values: unknown[]): number {
  let total = 0;
  for (const value of values) {
    if (!isFiniteDecimal(value)) return Number.NaN;
    total += Number(value);
  }
  return total;
}

function reconcileFills(fills: unknown[], strategyId: unknown, fingerprint: unknown): {
  valid: boolean;
  realizedPricePnl: number;
  fees: number;
} {
  let position = 0;
  let averageEntry = 0;
  let realizedPricePnl = 0;
  let fees = 0;
  let previousExecutionMs = -Infinity;
  const orderIds = new Set<string>();
  for (const value of fills) {
    const fill = record(value);
    if (!fill || !isPaperFill(fill, fingerprint, strategyId, -Infinity, Infinity)
      || orderIds.has(fill.orderId as string)) return { valid: false, realizedPricePnl, fees };
    orderIds.add(fill.orderId as string);
    const target = Number(fill.targetQuantity);
    const quantity = Number(fill.filledQuantity);
    const referencePrice = Number(fill.referencePrice);
    const price = Number(fill.price);
    const notional = Number(fill.notional);
    const fee = Number(fill.fee);
    const executionCost = Number(fill.executionCost);
    const executedAt = Date.parse(fill.executedAt as string);
    const delta = target - position;
    const expectedSide = delta > 0 ? "buy" : "sell";
    const expectedUnitCost = rounded8(referencePrice * 0.0002);
    const expectedPrice = rounded8(referencePrice + (fill.side === "buy" ? expectedUnitCost : -expectedUnitCost));
    const expectedNotional = rounded8(quantity * price);
    const expectedFee = rounded8(expectedNotional * 0.0006);
    const expectedExecutionCost = rounded8(quantity * expectedUnitCost);
    if (Math.abs(Math.abs(delta) - quantity) > 0.00000001 || fill.side !== expectedSide
      || executedAt <= previousExecutionMs
      || Math.abs(price - expectedPrice) > 0.00000001
      || Math.abs(notional - expectedNotional) > 0.00000001
      || Math.abs(fee - expectedFee) > 0.00000001
      || Math.abs(executionCost - expectedExecutionCost) > 0.00000001) {
      return { valid: false, realizedPricePnl, fees };
    }
    previousExecutionMs = executedAt;
    const closingQuantity = Math.sign(position) === Math.sign(delta) ? 0 : Math.min(Math.abs(position), quantity);
    if (closingQuantity > 0) realizedPricePnl += closingQuantity * (price - averageEntry) * Math.sign(position);
    if (target === 0) averageEntry = 0;
    else if (position === 0 || Math.sign(target) !== Math.sign(position)) averageEntry = price;
    else if (Math.sign(position) === Math.sign(delta)) {
      averageEntry = (Math.abs(position) * averageEntry + quantity * price) / Math.abs(target);
    }
    position = target;
    fees += fee;
  }
  return { valid: Math.abs(position) <= 0.00000001, realizedPricePnl, fees };
}

function rounded8(value: number): number {
  return Number(value.toFixed(8));
}

function isRejectedSignal(value: unknown): boolean {
  const signal = record(value);
  return !!signal && typeof signal.orderId === "string" && signal.orderId.length > 0
    && (signal.reason === "no_timely_bar" || signal.reason === "zero_volume"
      || signal.reason === "same_bar_conflict" || signal.reason === "allocation_limit");
}

function isFundingCashflow(value: unknown): boolean {
  const cashflow = record(value);
  return !!cashflow && typeof cashflow.settlementId === "string" && cashflow.settlementId.length > 0
    && typeof cashflow.timestamp === "string" && Number.isFinite(Date.parse(cashflow.timestamp))
    && isFiniteDecimal(cashflow.amount);
}
