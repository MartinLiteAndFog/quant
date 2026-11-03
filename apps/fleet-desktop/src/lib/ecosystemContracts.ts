/** Canonical ECO-01 JSON wire types. Money must never be converted to JS numbers. */
export type DecimalString = string;
export const ECOSYSTEM_SCHEMA_VERSION = "1.0" as const;
export const TICK_PHASES = ["execution", "metrics", "capital", "ecology", "breeding", "lifecycle", "snapshot"] as const;
export type TickPhase = typeof TICK_PHASES[number];
export interface StrategyGenome {
  readonly genome_id: string;
  readonly version: number;
  readonly strategy_code_version: string;
  readonly genes: Record<string, boolean | number>;
  readonly policies: ReadonlyArray<GenePolicy>;
  readonly parent_genome_ids: ReadonlyArray<string>;
  readonly complexity_penalty_weight: number;
}
export interface GenePolicy {
  readonly gene_id: string;
  readonly kind: "immutable_safety_policy" | "required_bounded_financial" | "optional_strategy_gene";
  readonly minimum: number | null;
  readonly maximum: number | null;
}
export interface StrategyInstance {
  readonly strategy_id: string;
  readonly genome_id: string;
  readonly family_id: string;
  readonly account_id: string;
  readonly created_at: string;
  readonly lifecycle: "proposed" | "screened" | "backtested" | "out_of_sample_verified" | "paper_forward" | "eligible_for_sowing" | "capitalized" | "dormant" | "closing" | "archived";
}
export interface DailyPlantState {
  readonly strategy_id: string;
  readonly as_of: string;
  readonly genome_id: string;
  readonly lifecycle: "proposed" | "screened" | "backtested" | "out_of_sample_verified" | "paper_forward" | "eligible_for_sowing" | "capitalized" | "dormant" | "closing" | "archived";
  readonly biomass: number;
  readonly root_depth: number;
  readonly root_spread: number;
  readonly leaf_area: number;
  readonly bark_strength: number;
  readonly fitness: MetricWindow;
  readonly vitality: number;
  readonly evidence: EvidenceState;
}
export interface RootNetwork {
  readonly strategy_id: string;
  readonly neighbors: ReadonlyArray<string>;
  readonly depth: number;
  readonly spread: number;
  readonly access_weight: number;
}
export interface ExposureIntent {
  readonly strategy_id: string;
  readonly signal_at: string;
  readonly available_at: string;
  readonly execute_after: string;
  readonly margin: string;
  readonly leverage: string;
  readonly notional: string;
}
export interface GroundwaterState {
  readonly as_of: string;
  readonly currency: string;
  readonly accounts: ReadonlyArray<AccountNAV>;
  readonly equity: string;
  readonly reserve: string;
  readonly distributable: string;
  readonly entitlements: ReadonlyArray<CapitalEntitlement>;
  readonly depth: number;
  readonly risk_breach: boolean;
  readonly allocations_frozen: boolean;
}
export interface CapitalEntitlement {
  readonly strategy_id: string;
  readonly account_id: string;
  readonly amount: string;
  readonly used_margin: string;
  readonly reserved_order_margin: string;
  readonly realizable_margin: string;
  readonly notional_exposure: string;
}
export interface AccountNAV {
  readonly account_id: string;
  readonly currency: string;
  readonly as_of: string;
  readonly equity: string;
  readonly free_margin: string;
}
export interface MetricWindow {
  readonly start: string;
  readonly end: string;
  readonly score: number | null;
  readonly net_return: number | null;
  readonly max_drawdown: number;
  readonly observations: number;
  readonly out_of_sample: boolean;
  readonly formula_version: string;
}
export interface EvidenceState {
  readonly effective_trades: number;
  readonly observed_days: number;
  readonly confidence: number;
  readonly data_quality: number;
  readonly opportunities: number;
  readonly out_of_sample: boolean;
}
export interface BeeAction {
  readonly action_id: string;
  readonly as_of: string;
  readonly bee_id: string;
  readonly mode: "random" | "local" | "hive" | "hybrid";
  readonly algorithm: "mixer" | "mosaic" | "explorer" | "collector" | "pruner";
  readonly parent_genome_ids: ReadonlyArray<string>;
  readonly child_genome_id: string;
  readonly seed: number;
  readonly validated_oos_honey: number;
}
export interface PredatorEvent {
  readonly strategy_id: string;
  readonly as_of: string;
  readonly reason: string;
  readonly positions_closed_confirmed: boolean;
  readonly released_budget: string;
}
export interface ResearchCandidate {
  readonly candidate_id: string;
  readonly genome: StrategyGenome;
  readonly stage: "proposed" | "screened" | "backtested" | "out_of_sample_verified" | "paper_forward" | "eligible_for_sowing" | "capitalized" | "dormant" | "closing" | "archived";
  readonly benchmark_id: string;
  readonly verdict: "beats" | "does_not_beat" | "not_comparable";
  readonly evidence: EvidenceState;
}
export interface MarketRegime {
  readonly as_of: string;
  readonly label: string;
  readonly sun_multiplier: number;
  readonly source: "synthetic" | "observed";
}
export interface MarketSnapshot {
  readonly symbol: string;
  readonly observed_at: string;
  readonly available_at: string;
  readonly price: string;
  readonly source: string;
  readonly synthetic: boolean;
}
export interface TradeFill {
  readonly fill_id: string;
  readonly strategy_id: string;
  readonly symbol: string;
  readonly signal_at: string;
  readonly available_at: string;
  readonly filled_at: string;
  readonly side: "buy" | "sell";
  readonly quantity: string;
  readonly price: string;
  readonly fee: string;
  readonly funding: string;
  readonly currency: string;
  readonly synthetic: boolean;
}
export interface SimulationEvent {
  readonly schema_version: "1.0";
  readonly event_id: string;
  readonly simulation_id: string;
  readonly day_index: number;
  readonly sequence: number;
  readonly phase: TickPhase;
  readonly occurred_at: string;
  readonly known_at: string;
  readonly kind: string;
  readonly payload: Record<string, unknown>;
}
export interface SimulationSnapshot {
  readonly schema_version: "1.0";
  readonly simulation_id: string;
  readonly as_of: string;
  readonly seed: number;
  readonly rng_state: string;
  readonly day_index: number;
  readonly plants: ReadonlyArray<DailyPlantState>;
  readonly genomes: ReadonlyArray<StrategyGenome>;
  readonly groundwater: GroundwaterState;
  readonly events: ReadonlyArray<SimulationEvent>;
  readonly provenance: "synthetic" | "historical" | "shadow";
}
export type GroundwaterAllocation = CapitalEntitlement;
export type PredationEvent = PredatorEvent;
