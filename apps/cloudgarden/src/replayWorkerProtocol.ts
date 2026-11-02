import type { SyntheticReplayCycleResult } from "./replayCycle.ts";

/** Validate the untrusted worker boundary before a replay result reaches the ledger. */
export function readSyntheticReplayWorkerResult(
  message: unknown,
  expectedId: string,
  expectedDays: number,
): SyntheticReplayCycleResult {
  if (!expectedId || !Number.isInteger(expectedDays) || expectedDays < 1 || expectedDays > 90) {
    throw new RangeError("Expected replay identity and horizon must be valid.");
  }
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
  const days = execution.days;
  const startingCash = finite(execution.startingCash);
  const endingCash = finite(execution.endingCash);
  const netPnl = finite(execution.netPnl);
  const finalQuantity = execution.finalQuantity;
  if (days !== expectedDays || !Array.isArray(execution.dailyMarks)
    || execution.dailyMarks.length !== expectedDays || dataset.barCount !== expectedDays * 24 * 60
    || dataset.synthetic !== true || dataset.eligibility !== "synthetic_demo_only"
    || typeof fingerprint !== "string" || !/^[a-f0-9]{64}$/.test(fingerprint)
    || strategy.datasetFingerprint !== fingerprint || evidence.datasetFingerprint !== fingerprint
    || execution.endedFlat !== true || finalQuantity !== "0" || execution.riskBreach !== false
    || !Array.isArray(fills) || !Array.isArray(rejectedSignals)
    || typeof strategy.strategyId !== "string" || strategy.strategyId.length === 0
    || genome.strategyId !== strategy.strategyId || evidence.strategyId !== strategy.strategyId
    || (strategy.status !== "trade" && strategy.status !== "no_trade")
    || evidence.strategyStatus !== strategy.status
    || evidence.benchmarkConclusion !== "not_comparable"
    || evidence.statisticalStatus !== "insufficient_sample"
    || evidence.independentOpportunityCount !== null
    || evidence.observations !== strategy.observations
    || evidence.executableOpportunityObservations !== strategy.opportunities
    || evidence.generatedTargets !== strategy.generatedSignals
    || evidence.fills !== fills.length || evidence.rejectedSignals !== rejectedSignals.length
    || !Number.isFinite(startingCash) || !Number.isFinite(endingCash) || !Number.isFinite(netPnl)
    || Math.abs(startingCash + netPnl - endingCash) > 0.0001
    || typeof evidence.cashResidualUsdt !== "number" || !Number.isFinite(evidence.cashResidualUsdt) || Math.abs(evidence.cashResidualUsdt) > 0.0001
    || typeof evidence.netPnlResidualUsdt !== "number" || !Number.isFinite(evidence.netPnlResidualUsdt) || Math.abs(evidence.netPnlResidualUsdt) > 0.0001) {
    throw new RangeError("Replay worker result does not reconcile to the active synthetic run.");
  }
  return result as unknown as SyntheticReplayCycleResult;
}

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null ? value as Record<string, unknown> : null;
}

function finite(value: unknown): number {
  const parsed = typeof value === "string" || typeof value === "number" ? Number(value) : Number.NaN;
  return parsed;
}
