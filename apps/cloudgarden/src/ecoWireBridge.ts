/**
 * ECO-01 <-> Cloudgarden v1 read-only bridge. No exchange or account side effects.
 * Money crosses the wire as exact decimal strings; Cloudgarden's numeric ledger
 * is accepted only at its own four-decimal precision.
 */
import { assertSimulationInvariants, STARTING_NAV_USDT, type Genome, type TradingTraits, type SimulationWorld } from "./simulation.ts";
import { validateSourceGenome, type SourceGenome } from "./sourceStrategies.ts";
import type { HistoricalSession } from "./historicalGarden.ts";

export const BRIDGE_VERSION = "eco-cloudgarden/1.0" as const;
export type WireSource = "synthetic" | "historical";
export type WireEntitlement = Readonly<{
  strategy_id: string; account_id: string; amount: string; used_margin: string;
  reserved_order_margin: string; realizable_margin: string; notional_exposure: string;
}>;
export type WirePlant = Readonly<{
  strategy_id: string; status: "active" | "archived"; allocation: string;
  genome_version: 1; strategy_code_version: string;
  genome_kind: "source_candle_research" | "synthetic_demo";
  source_genome: SourceGenome | null; visual_genes: Genome; trading_traits: TradingTraits;
  lineage_parent_ids: readonly string[];
  habitat: Readonly<{ fitness: number; vigor: number; evidence: number; evidence_days: number }>;
  empirical: null;
}>;
export type CloudgardenWire = Readonly<{
  schema_version: typeof BRIDGE_VERSION; ecosystem_schema_version: "1.0";
  simulation_id: string; dataset_fingerprint: string; source_kind: WireSource;
  capital_writer: "cloudgarden-paper"; access: "read_only";
  account_id: string; currency: "USDT"; as_of: string; known_at: string;
  day_index: number; event_id: string;
  booked: Readonly<{
    opening_nav: string; cashflow: string; price_pnl: string; fees: string;
    funding: string; net_pnl: string; reconciliation_delta: string;
  }>;
  groundwater: Readonly<{
    as_of: string; currency: "USDT";
    accounts: readonly Readonly<{ account_id: string; currency: "USDT"; as_of: string; equity: string; free_margin: string }>[];
    equity: string; reserve: string; distributable: string;
    entitlements: readonly WireEntitlement[]; depth: number;
    risk_breach: boolean; allocations_frozen: boolean;
  }>;
  capital_accounting: Readonly<{
    deployed_allocation: string; seed_reserved_allocation: string; free_allocation: string;
    used_margin: string; reserved_order_margin: string; free_margin: string; rounding_holdback: string;
  }>;
  seed_entitlements: readonly Readonly<{ id: number; status: "available" | "used"; plant_id: string | null }>[];
  plants: readonly WirePlant[];
}>;

const MAX_AMOUNT = Number.MAX_SAFE_INTEGER;
const DECIMAL = /^(-?)(0|[1-9]\d*)(?:\.(\d{1,5}))?$/;
const ZERO = "0.0000";
const FINGERPRINT = /^[0-9a-f]{64}$/;

/** Return an exact integer in 1/100000 USDT units; never parse money as float. */
function units5(value: string): bigint {
  if (typeof value !== "string") throw new Error("Wire money must be a decimal string.");
  const parts = DECIMAL.exec(value);
  if (!parts) throw new Error("Invalid wire decimal string.");
  const n = BigInt(parts[2]!) * 100000n + BigInt((parts[3] || "").padEnd(5, "0"));
  return parts[1] === "-" ? -n : n;
}
function absolute(n: bigint): bigint { return n < 0n ? -n : n; }
function format5(n: bigint): string {
  const a = absolute(n);
  return (n < 0n ? "-" : "") + (a / 100000n).toString() + "." + (a % 100000n).toString().padStart(5, "0");
}
function exact4(value: number): string {
  if (!Number.isFinite(value)) throw new Error("Non-finite Cloudgarden money.");
  const scaled = Math.round(value * 10000);
  if (Math.abs(scaled) > MAX_AMOUNT || Math.abs(value * 10000 - scaled) > 0.0000001) {
    throw new Error("Cloudgarden money cannot be losslessly represented at 0.0001 USDT.");
  }
  const a = Math.abs(scaled);
  return (scaled < 0 ? "-" : "") + Math.floor(a / 10000).toString() + "." + String(a % 10000).padStart(4, "0");
}
function canonical(value: unknown): string {
  if (Array.isArray(value)) return "[" + value.map(canonical).join(",") + "]";
  if (value !== null && typeof value === "object") {
    return "{" + Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b))
      .map(([k, v]) => JSON.stringify(k) + ":" + canonical(v)).join(",") + "}";
  }
  return JSON.stringify(value);
}
function requireCondition(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}
function validTimestamp(value: string): boolean {
  return typeof value === "string"
    && /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{3})?Z$/.test(value)
    && !Number.isNaN(Date.parse(value))
    && new Date(Date.parse(value)).toISOString() === new Date(value).toISOString();
}

/** Read-only, fail-closed projection. A Cloudgarden day never books into ECO-01. */
export function exportWorldWire(
  world: SimulationWorld,
  options: Readonly<{
    datasetFingerprint: string; sourceKind: WireSource; asOf: string;
    sourceGenomes?: Readonly<Record<string, SourceGenome>>;
  }>,
): CloudgardenWire {
  assertSimulationInvariants(world);
  requireCondition(FINGERPRINT.test(options.datasetFingerprint), "Dataset hash is required.");
  requireCondition(options.sourceKind === "synthetic" || options.sourceKind === "historical", "Invalid source kind.");
  requireCondition(options.sourceKind !== "historical" || options.datasetFingerprint !== "0".repeat(64), "Historical source hash is missing.");
  requireCondition(validTimestamp(options.asOf), "as_of must be an explicit UTC timestamp.");
  const accountId = "cloudgarden-paper:" + world.seed;
  const nav = units5(exact4(world.capital.navUsdt));
  const positive = nav > 0n ? nav : 0n;
  const reserve = positive / 10n; // nav is at 4 dp, so 1/10 is exact at 5 dp
  const distributable = positive - reserve;
  const plants: WirePlant[] = world.plants.map((plant) => {
    const genome = options.sourceGenomes?.[plant.id] ?? null;
    if (options.sourceKind === "historical") {
      requireCondition(genome !== null, "Missing historical source genome.");
      validateSourceGenome(genome);
    }
    return {
      strategy_id: plant.id, status: plant.status, allocation: exact4(plant.status === "active" ? plant.allocationUsdt : 0),
      genome_version: 1, strategy_code_version: genome ? "cloudgarden-source-v1" : "cloudgarden-demo-v1",
      genome_kind: genome ? "source_candle_research" : "synthetic_demo",
      source_genome: genome, visual_genes: plant.genes, trading_traits: plant.tradingTraits,
      lineage_parent_ids: plant.parents ?? [],
      habitat: { fitness: plant.fitness, vigor: plant.vigor, evidence: plant.evidence, evidence_days: plant.evidenceDays },
      empirical: null, // Never promote habitat fitness or days to TWR/OOS evidence.
    };
  });
  const entitlements: WireEntitlement[] = plants.filter((p) => p.status === "active").map((p) => ({
    strategy_id: p.strategy_id, account_id: accountId, amount: p.allocation,
    used_margin: ZERO, reserved_order_margin: ZERO, realizable_margin: p.allocation, notional_exposure: ZERO,
  }));
  for (const slot of world.capital.seedlingEntitlements) {
    if (slot.status !== "available") continue;
    entitlements.push({
      strategy_id: "seedbank:" + String(slot.id).padStart(2, "0"), account_id: accountId,
      amount: exact4(1.5), used_margin: ZERO, reserved_order_margin: ZERO,
      realizable_margin: exact4(1.5), notional_exposure: ZERO,
    });
  }
  const entitlementSum = entitlements.reduce((s, e) => s + units5(e.amount), 0n);
  requireCondition(entitlementSum <= distributable, "Cloudgarden 4dp allocation rounds above exact ECO-01 90% ceiling; fail closed.");
  requireCondition(entitlementSum === units5(exact4(world.capital.deployedCapitalUsdt))
    + units5(exact4(world.capital.reservedMarginUsdt)), "Bridge entitlements do not reconcile.");
  requireCondition(absolute(units5(exact4(world.capital.protectedReserveUsdt)) - reserve) <= 5n,
    "Cloudgarden reserve disagrees with exact reserve by more than rounding.");
  const price = units5(exact4(world.capital.realizedPricePnlUsdt));
  const fees = units5(exact4(world.capital.paperFeesUsdt));
  const funding = units5(exact4(world.capital.paperFundingUsdt));
  const net = nav - units5(exact4(STARTING_NAV_USDT));
  const sourceFreeAllocation = units5(exact4(world.capital.freeCapitalUsdt));
  const safeFreeAllocation = sourceFreeAllocation < distributable - entitlementSum ? sourceFreeAllocation : distributable - entitlementSum;
  const roundingHoldback = sourceFreeAllocation - safeFreeAllocation;
  requireCondition(roundingHoldback >= 0n && roundingHoldback <= 5n, "Unrepresentable Cloudgarden ledger rounding drift.");
  const freeMargin = format5(positive);
  const wire: CloudgardenWire = {
    schema_version: BRIDGE_VERSION, ecosystem_schema_version: "1.0",
    simulation_id: world.seed, dataset_fingerprint: options.datasetFingerprint, source_kind: options.sourceKind,
    capital_writer: "cloudgarden-paper", access: "read_only", account_id: accountId, currency: "USDT",
    day_index: world.day, as_of: options.asOf, known_at: options.asOf,
    event_id: world.seed + ":d" + world.day + ":wire-snapshot",
    booked: {
      opening_nav: exact4(STARTING_NAV_USDT), cashflow: ZERO, price_pnl: format5(price), fees: format5(fees),
      funding: format5(funding), net_pnl: format5(net), reconciliation_delta: format5(net - price + fees - funding),
    },
    groundwater: {
      as_of: options.asOf, currency: "USDT",
      accounts: [{ account_id: accountId, currency: "USDT", as_of: options.asOf, equity: format5(nav), free_margin: freeMargin }],
      equity: format5(nav), reserve: format5(reserve), distributable: format5(distributable),
      entitlements, depth: 1, risk_breach: false, allocations_frozen: false,
    },
    capital_accounting: {
      deployed_allocation: exact4(world.capital.deployedCapitalUsdt),
      seed_reserved_allocation: exact4(world.capital.reservedMarginUsdt),
      free_allocation: format5(safeFreeAllocation), rounding_holdback: format5(roundingHoldback),
      used_margin: ZERO, reserved_order_margin: ZERO, free_margin: freeMargin,
    },
    seed_entitlements: world.capital.seedlingEntitlements.map((s) => ({
      id: s.id, status: s.status, plant_id: s.plantId ?? null,
    })),
    plants,
  };
  return validateWire(wire);
}

export function exportHistoricalWire(session: HistoricalSession): CloudgardenWire {
  return exportWorldWire(session.world, {
    datasetFingerprint: session.sourceSha256, sourceKind: "historical",
    asOf: session.nextDate + "T00:00:00.000Z", sourceGenomes: session.genomes,
  });
}

/** ECO-01-compatible money, entitlement, causal and provenance validation. */
export function validateWire(input: unknown): CloudgardenWire {
  const w = input as CloudgardenWire;
  requireCondition(w && w.schema_version === BRIDGE_VERSION && w.ecosystem_schema_version === "1.0"
    && w.capital_writer === "cloudgarden-paper" && w.access === "read_only"
    && w.currency === "USDT" && w.groundwater?.currency === w.currency
    && typeof w.simulation_id === "string" && w.simulation_id.length > 0
    && w.account_id === "cloudgarden-paper:" + w.simulation_id
    && FINGERPRINT.test(w.dataset_fingerprint)
    && (w.source_kind === "historical" || w.source_kind === "synthetic")
    && (w.source_kind === "synthetic" || w.dataset_fingerprint !== "0".repeat(64))
    && Number.isInteger(w.day_index) && w.day_index >= 0
    && w.event_id === w.simulation_id + ":d" + w.day_index + ":wire-snapshot"
    && validTimestamp(w.as_of) && validTimestamp(w.known_at) && w.known_at <= w.as_of
    && w.groundwater.as_of === w.as_of && Array.isArray(w.groundwater.accounts)
    && Array.isArray(w.groundwater.entitlements) && Array.isArray(w.plants)
    && Array.isArray(w.seed_entitlements), "Malformed or non-causal ECO-Cloudgarden bridge.");
  const water = w.groundwater;
  requireCondition(water.accounts.length === 1 && water.accounts[0]?.account_id === w.account_id
    && water.accounts[0]?.as_of === w.as_of && water.accounts[0]?.currency === "USDT", "Bridge must have exactly one authoritative account.");
  const nav = units5(water.equity);
  const positive = nav > 0n ? nav : 0n;
  requireCondition(nav === units5(water.accounts[0]!.equity)
    && units5(water.accounts[0]!.free_margin) === units5(w.capital_accounting.free_margin)
    && units5(water.accounts[0]!.free_margin) >= 0n
    && units5(water.accounts[0]!.free_margin) <= positive
    && positive % 10n === 0n, "Account precision/free margin mismatch.");
  requireCondition(units5(water.reserve) === positive / 10n
    && units5(water.distributable) === positive - positive / 10n
    && water.depth > 0 && Number.isFinite(water.depth), "Exact ECO-01 90/10 reserve violated.");
  requireCondition(water.risk_breach === water.allocations_frozen
    && (nav >= 0n || (water.risk_breach && water.allocations_frozen)), "Negative NAV must freeze new allocations.");
  const entIds = new Set<string>();
  let amount = 0n, used = 0n, reserved = 0n;
  for (const e of water.entitlements) {
    requireCondition(e.account_id === w.account_id && e.strategy_id.length > 0 && !entIds.has(e.strategy_id),
      "Duplicate or foreign entitlement.");
    entIds.add(e.strategy_id);
    const allocation = units5(e.amount), real = units5(e.realizable_margin);
    const u = units5(e.used_margin), r = units5(e.reserved_order_margin);
    requireCondition(allocation >= 0n && real >= 0n && real <= allocation
      && u >= 0n && r >= 0n && units5(e.notional_exposure) >= 0n, "Invalid entitlement/margin.");
    if (u + r > allocation) requireCondition(water.risk_breach && water.allocations_frozen, "Unfrozen margin breach.");
    amount += allocation; used += u; reserved += r;
  }
  requireCondition(amount <= positive - positive / 10n, "90% budget oversubscribed.");
  const cash = w.capital_accounting;
  requireCondition(units5(cash.deployed_allocation) >= 0n && units5(cash.seed_reserved_allocation) >= 0n
    && units5(cash.free_allocation) >= 0n
    && units5(cash.rounding_holdback) >= 0n && units5(cash.rounding_holdback) <= 5n
    && amount === units5(cash.deployed_allocation) + units5(cash.seed_reserved_allocation)
    && amount + units5(cash.free_allocation) <= units5(water.distributable)
    && used === units5(cash.used_margin) && reserved === units5(cash.reserved_order_margin),
    "Budget transfer, used/reserved/free margins do not conserve.");
  const plants = new Map<string, WirePlant>();
  for (const p of w.plants) {
    requireCondition(p.strategy_id.length > 0 && !plants.has(p.strategy_id)
      && p.genome_version === 1 && (p.status === "active" || p.status === "archived")
      && Array.isArray(p.lineage_parent_ids)
      && p.lineage_parent_ids.every((id) => typeof id === "string")
      && p.genome_kind === (p.source_genome ? "source_candle_research" : "synthetic_demo")
      && p.strategy_code_version === (p.source_genome ? "cloudgarden-source-v1" : "cloudgarden-demo-v1")
      && p.empirical === null && p.habitat && Number.isInteger(p.habitat.evidence_days)
      && p.habitat.evidence_days >= 0 && [p.habitat.fitness, p.habitat.vigor, p.habitat.evidence]
        .every((x) => Number.isFinite(x) && x >= 0 && x <= 100)
      && units5(p.allocation) >= 0n, "Invalid genome, habitat or fabricated empirical evidence.");
    if (p.source_genome) validateSourceGenome(p.source_genome);
    if (w.source_kind === "historical") requireCondition(p.source_genome !== null, "Historical plant lacks source genome.");
    if (p.status === "active") requireCondition(entIds.has(p.strategy_id), "Active plant is not funded.");
    else requireCondition(units5(p.allocation) === 0n && !entIds.has(p.strategy_id), "Archived plant still funded.");
    plants.set(p.strategy_id, p);
  }
  const slotIds = new Set<number>(), planted = new Set<string>();
  for (const slot of w.seed_entitlements) {
    requireCondition(Number.isInteger(slot.id) && slot.id > 0 && !slotIds.has(slot.id)
      && (slot.status === "available" || slot.status === "used"), "Invalid or duplicate seed slot.");
    slotIds.add(slot.id);
    const key = "seedbank:" + String(slot.id).padStart(2, "0");
    if (slot.status === "available") requireCondition(slot.plant_id === null && entIds.has(key), "Missing reserved seed budget.");
    else {
      requireCondition(typeof slot.plant_id === "string" && !planted.has(slot.plant_id)
        && plants.has(slot.plant_id) && !entIds.has(key), "Seed funding counted twice.");
      planted.add(slot.plant_id);
    }
  }
  requireCondition(plants.size > 0 && slotIds.size > 0
    && [...plants.values()].filter((p) => p.status === "active").reduce((sum, p) => sum + units5(p.allocation), 0n)
      === units5(cash.deployed_allocation), "Plant/seed allocation inconsistency.");
  const booked = w.booked;
  const price = units5(booked.price_pnl), fees = units5(booked.fees), funding = units5(booked.funding);
  const net = units5(booked.net_pnl);
  requireCondition(fees >= 0n && units5(booked.opening_nav) + units5(booked.cashflow) + net === nav
    && price - fees + funding + units5(booked.reconciliation_delta) === net,
    "PnL, cashflow and NAV reconciliation failed.");
  return w;
}

/** Observer journal only: no capital mutations, no authority transfer. */
export class CloudgardenWireJournal {
  private readonly seen = new Map<string, string>();
  private latest: CloudgardenWire | null = null;

  observe(input: unknown): "observed" | "duplicate" {
    const wire = validateWire(input);
    const fingerprint = canonical(wire);
    const old = this.seen.get(wire.event_id);
    if (old !== undefined) {
      if (old !== fingerprint) throw new Error("Conflicting payload reuses a previously booked event_id.");
      return "duplicate";
    }
    if (this.latest && (wire.simulation_id !== this.latest.simulation_id
      || wire.account_id !== this.latest.account_id
      || wire.dataset_fingerprint !== this.latest.dataset_fingerprint
      || wire.source_kind !== this.latest.source_kind
      || wire.day_index <= this.latest.day_index
      || wire.known_at < this.latest.known_at)) {
      throw new Error("Different capital writer, stale replay or data source switch.");
    }
    this.seen.set(wire.event_id, fingerprint);
    this.latest = wire;
    return "observed";
  }
  current(): CloudgardenWire | null { return this.latest; }
}
