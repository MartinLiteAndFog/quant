import type { MarketDataset, TargetPositionSignal } from "./paperReplay.ts";

/** A game-independent, explicitly synthetic signal recipe. It is not ecology Genome. */
export type TradingSignalGenome = Readonly<{
  schemaVersion: 1;
  strategyId: string;
  lookbackBars: number;
  entryThresholdBps: number;
  exitThresholdBps: number;
  targetQuantity: string;
}>;

export type StrategySignalReport = Readonly<{
  schemaVersion: 1;
  strategyId: string;
  datasetFingerprint: string;
  claim: "synthetic_causal_fixture_not_alpha";
  benchmark: "cash_no_trade";
  benchmarkConclusion: "not_comparable";
  status: "trade" | "no_trade";
  observations: number;
  opportunities: number;
  generatedSignals: number;
  endedFlat: true;
}>;

export type StrategySignalResult = Readonly<{
  signals: readonly TargetPositionSignal[];
  report: StrategySignalReport;
}>;

/**
 * Generate a causal close-to-close threshold stream for synthetic fixtures only.
 * Each decision uses closes available at that bar's close and is executed later
 * by the replay's strict next-open rule. The final signal closes any position.
 */
export function generateSyntheticMomentumSignals(
  dataset: MarketDataset,
  genome: TradingSignalGenome,
): StrategySignalResult {
  validateDataset(dataset);
  validateGenome(genome);
  const bars = dataset.bars;
  const intervalMs = dataset.provenance.intervalSeconds * 1_000;
  if (bars.length < genome.lookbackBars + 3) {
    throw new RangeError("Dataset must leave enough bars for lookback and a strict next-open flat close.");
  }

  const signals: TargetPositionSignal[] = [];
  let target = 0;
  let observations = 0;
  let opportunities = 0;
  const decisionEnd = bars.length - 3;
  for (let index = genome.lookbackBars; index <= decisionEnd; index += 1) {
    const current = bars[index];
    const prior = bars[index - genome.lookbackBars];
    if (!current || !prior) throw new RangeError("Dataset bars are incomplete.");
    const start = Number(prior.close);
    const end = Number(current.close);
    if (!(Number.isFinite(start) && start > 0 && Number.isFinite(end) && end > 0)) {
      throw new RangeError("Close prices must be finite and positive.");
    }
    const returnBps = ((end - start) / start) * 10_000;
    observations += 1;
    const magnitude = Math.abs(returnBps);
    // The final eligible observation is reserved for forced flattening, so it
    // cannot open a position and must not count as an executable opportunity.
    if (index < decisionEnd && magnitude >= genome.entryThresholdBps) opportunities += 1;

    let nextTarget = target;
    if (returnBps >= genome.entryThresholdBps) nextTarget = 1;
    else if (returnBps <= -genome.entryThresholdBps) nextTarget = -1;
    else if (magnitude < genome.exitThresholdBps) nextTarget = 0;

    // Reserve the final eligible decision for a deterministic forced flat.
    if (index === decisionEnd) nextTarget = 0;
    if (nextTarget !== target) {
      const decisionAt = new Date(current.timestampMs + intervalMs).toISOString();
      signals.push({
        orderId: `${genome.strategyId}-signal-${signals.length + 1}`,
        strategyId: genome.strategyId,
        targetQuantity: nextTarget === 0 ? "0" : nextTarget > 0 ? genome.targetQuantity : `-${genome.targetQuantity}`,
        signalAt: decisionAt,
        availableAt: decisionAt,
        receivedAt: decisionAt,
      });
      target = nextTarget;
    }
  }

  if (target !== 0) throw new RangeError("Signal stream failed to end flat.");
  return {
    signals,
    report: {
      schemaVersion: 1,
      strategyId: genome.strategyId,
      datasetFingerprint: dataset.fingerprint,
      claim: "synthetic_causal_fixture_not_alpha",
      benchmark: "cash_no_trade",
      benchmarkConclusion: "not_comparable",
      status: signals.some((signal) => signal.targetQuantity !== "0") ? "trade" : "no_trade",
      observations,
      opportunities,
      generatedSignals: signals.length,
      endedFlat: true,
    },
  };
}

function validateGenome(genome: TradingSignalGenome): void {
  if (!genome || genome.schemaVersion !== 1 || typeof genome.strategyId !== "string" || !genome.strategyId.trim()) {
    throw new RangeError("A version 1 genome with a non-empty strategy ID is required.");
  }
  if (!Number.isInteger(genome.lookbackBars) || genome.lookbackBars < 1 || genome.lookbackBars > 10_000) {
    throw new RangeError("lookbackBars must be a whole number from 1 to 10000.");
  }
  if (!Number.isFinite(genome.entryThresholdBps) || genome.entryThresholdBps <= 0 || genome.entryThresholdBps > 100_000
    || !Number.isFinite(genome.exitThresholdBps) || genome.exitThresholdBps < 0
    || genome.exitThresholdBps > genome.entryThresholdBps) {
    throw new RangeError("Thresholds must be finite, bounded, and exit must not exceed entry.");
  }
  if (typeof genome.targetQuantity !== "string" || !/^(?:0\.[0-9]{1,8}|[1-9][0-9]*(?:\.[0-9]{1,8})?)$/.test(genome.targetQuantity)
    || Number(genome.targetQuantity) <= 0) {
    throw new RangeError("targetQuantity must be a positive decimal string with at most 8 places.");
  }
}

function validateDataset(dataset: MarketDataset): void {
  if (!dataset || !Array.isArray(dataset.bars) || dataset.bars.length === 0
    || dataset.provenance?.synthetic !== true || dataset.report?.synthetic !== true
    || dataset.report?.eligibility !== "synthetic_demo_only"
    || !/^[a-f0-9]{64}$/.test(dataset.fingerprint)
    || dataset.report.fingerprint !== dataset.fingerprint) {
    throw new RangeError("Signal generation requires a fingerprint-matched synthetic-only dataset.");
  }
  if (dataset.provenance.marketType !== "linear_futures" || dataset.provenance.timestampSemantics !== "open"
    || dataset.provenance.timezone !== "UTC" || !Number.isInteger(dataset.provenance.intervalSeconds)
    || dataset.provenance.intervalSeconds <= 0) {
    throw new RangeError("Dataset provenance does not match the supported UTC linear-futures bar contract.");
  }
}
