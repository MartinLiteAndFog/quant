/**
 * Pure, deterministic simulation rules for Cloudgarden.
 *
 * This is a synthetic garden model. Capital is a conserved accounting unit used
 * to teach allocation and margin; it is never connected to an exchange, market,
 * account, or order system.
 */

export const STARTING_NAV_USDT = 100;
export const MAX_ALLOCATION_RATE = 0.9;
export const PROTECTED_RESERVE_RATE = 0.1;
export const PARENT_COUNT = 3;
export const SEEDLING_ENTITLEMENTS = 20;
export const PARENT_ALLOCATION_USDT = 20;
export const SEEDLING_ALLOCATION_USDT = 1.5;
export const MIN_OBSERVATION_DAYS = 80;
export const PREDATION_VIGOR_THRESHOLD = 70;
export const PREDATION_EVIDENCE_THRESHOLD = 80;
export const MAX_EVENT_HISTORY = 2_000;

export type Genome = Readonly<{
  root: number;
  leaf: number;
  water: number;
  sun: number;
}>;

export type Phenotype = "Grass" | "Bush" | "Cactus";
export type PlantRole = "parent" | "seedling";
export type PlantStatus = "active" | "archived";
export type PlantPhysiology = "awake" | "dormant";
export type BeeMode = "random" | "local" | "hive" | "hybrid";
export type ConceptionPolicy = BeeMode | "mixed";

export type GardenPlant = Readonly<{
  id: string;
  name: string;
  role: PlantRole;
  status: PlantStatus;
  /** Dormancy is reversible physiology; it does not remove visibility or capital. */
  physiology: PlantPhysiology;
  x: number;
  genes: Genome;
  phenotype: Phenotype;
  biomass: number;
  /** Momentary fit between this genome and today's synthetic environment. */
  fitness: number;
  vigor: number;
  /** Simulated evidence quality, not financial or market evidence. */
  evidence: number;
  evidenceDays: number;
  /** Original synthetic allocation; only active plants count as deployed. */
  allocationUsdt: number;
  conceivedBy?: BeeMode;
  parents?: readonly [string, string];
}>;

export type SeedEntitlement = Readonly<{
  id: number;
  status: "available" | "used";
  plantId?: string;
}>;

export type CapitalLedger = Readonly<{
  navUsdt: number;
  allocationCeilingUsdt: number;
  protectedReserveUsdt: number;
  freeCapitalUsdt: number;
  reservedMarginUsdt: number;
  deployedCapitalUsdt: number;
  realizedPricePnlUsdt: number;
  paperFeesUsdt: number;
  paperFundingUsdt: number;
  paperReplayCount: number;
  seedlingEntitlements: readonly SeedEntitlement[];
}>;

export type ClosedPaperReplayBooking = Readonly<{
  startDay: number;
  days: number;
  datasetFingerprint: string;
  synthetic: boolean;
  endedFlat: boolean;
  riskBreach: boolean;
  fillCount: number;
  signalStatus?: "trade" | "no_trade";
  strategyId?: string;
  signalGenome?: Readonly<{
    schemaVersion: 1;
    lookbackBars: number;
    entryThresholdBps: number;
    exitThresholdBps: number;
    targetQuantity: string;
  }>;
  observations?: number;
  opportunities?: number;
  generatedSignals?: number;
  allocationLimitUsdt: number;
  startingCashUsdt: number;
  endingCashUsdt: number;
  maxMarginUsedUsdt: number;
  realizedPricePnlUsdt: number;
  feesUsdt: number;
  fundingUsdt: number;
  netPnlUsdt: number;
}>;

export type GardenEvent = Readonly<{
  id: string;
  day: number;
  type: "daily_tick" | "conception" | "predation" | "archive" | "dormancy" | "awakening" | "paper_replay";
  actor?: string;
  subject?: string;
  detail: Readonly<Record<string, string | number | boolean>>;
}>;

export type GardenClimate = Readonly<{
  water: number;
  sunlight: number;
  predatorPatrol: boolean;
}>;

export type SimulationWorld = Readonly<{
  seed: string;
  day: number;
  conceptionPolicy: ConceptionPolicy;
  climate: GardenClimate;
  plants: readonly GardenPlant[];
  capital: CapitalLedger;
  events: readonly GardenEvent[];
}>;

const clamp = (value: number, min: number, max: number): number =>
  Math.max(min, Math.min(max, value));
const round1 = (value: number): number => Math.round(value * 10) / 10;
const round4 = (value: number): number => Math.round(value * 10_000) / 10_000;

function appendEvents(history: readonly GardenEvent[], additions: readonly GardenEvent[]): readonly GardenEvent[] {
  return [...history, ...additions].slice(-MAX_EVENT_HISTORY);
}

const PARENT_BLUEPRINTS: readonly Omit<GardenPlant, "phenotype" | "status" | "role" | "physiology" | "evidenceDays" | "allocationUsdt">[] = [
  { id: "parent-01", name: "Cloud fern", x: 18, genes: { root: 0.42, leaf: 0.82, water: 0.58, sun: 0.74 }, biomass: 59, fitness: 73, vigor: 82, evidence: 76 },
  { id: "parent-02", name: "Blue reed", x: 50, genes: { root: 0.84, leaf: 0.38, water: 0.78, sun: 0.46 }, biomass: 52, fitness: 68, vigor: 78, evidence: 74 },
  { id: "parent-03", name: "Stone sage", x: 82, genes: { root: 0.76, leaf: 0.44, water: 0.82, sun: 0.68 }, biomass: 43, fitness: 77, vigor: 84, evidence: 79 },
];

/** Classifies one of three Cloudgarden forms from the heritable root/leaf traits. */
export function derivePhenotype(genes: Genome): Phenotype {
  if (genes.root >= 0.68 && genes.leaf < 0.68) return "Cactus";
  if (genes.leaf >= 0.68) return "Bush";
  return "Grass";
}

/** Visible population intentionally excludes archived plants. */
export function visiblePopulation(world: SimulationWorld): readonly GardenPlant[] {
  return world.plants.filter((plant) => plant.status === "active");
}

export function createSimulationWorld(
  seed = "CLOUD-071",
  options: { conceptionPolicy?: ConceptionPolicy } = {},
): SimulationWorld {
  if (!seed.trim()) throw new RangeError("A non-empty deterministic world seed is required.");
  const plants: GardenPlant[] = PARENT_BLUEPRINTS.map((plant) => ({
    ...plant,
    genes: { ...plant.genes },
    role: "parent",
    status: "active",
    physiology: "awake",
    phenotype: derivePhenotype(plant.genes),
    evidenceDays: 0,
    allocationUsdt: PARENT_ALLOCATION_USDT,
  }));
  const entitlements: SeedEntitlement[] = Array.from({ length: SEEDLING_ENTITLEMENTS }, (_, index) => ({
    id: index + 1,
    status: "available",
  }));

  const world: SimulationWorld = {
    seed,
    day: 0,
    conceptionPolicy: options.conceptionPolicy ?? "mixed",
    climate: { water: 0.55, sunlight: 0.68, predatorPatrol: false },
    plants,
    capital: {
      navUsdt: STARTING_NAV_USDT,
      allocationCeilingUsdt: round4(STARTING_NAV_USDT * MAX_ALLOCATION_RATE),
      protectedReserveUsdt: round4(STARTING_NAV_USDT * PROTECTED_RESERVE_RATE),
      freeCapitalUsdt: 0,
      reservedMarginUsdt: round4(SEEDLING_ENTITLEMENTS * SEEDLING_ALLOCATION_USDT),
      deployedCapitalUsdt: round4(PARENT_COUNT * PARENT_ALLOCATION_USDT),
      realizedPricePnlUsdt: 0,
      paperFeesUsdt: 0,
      paperFundingUsdt: 0,
      paperReplayCount: 0,
      seedlingEntitlements: entitlements,
    },
    events: [],
  };
  assertSimulationInvariants(world);
  return world;
}

/** Return an immutable world snapshot with a selected bee/seed conception policy. */
export function setConceptionPolicy(world: SimulationWorld, conceptionPolicy: ConceptionPolicy): SimulationWorld {
  const next = { ...world, conceptionPolicy };
  assertSimulationInvariants(next);
  return next;
}

/** Book a completed, flat synthetic replay into the garden's single NAV ledger. */
export function bookClosedPaperReplay(
  world: SimulationWorld,
  plantId: string,
  booking: ClosedPaperReplayBooking,
): SimulationWorld {
  assertSimulationInvariants(world);
  const plant = world.plants.find((candidate) => candidate.id === plantId && candidate.status === "active");
  if (!plant) throw new RangeError("A paper replay can only be booked to a living plant.");
  const signalStatus = booking.signalStatus ?? "trade";
  const noTradeObservation = signalStatus === "no_trade"
    && booking.fillCount === 0
    && booking.generatedSignals === 0
    && booking.opportunities === 0
    && (booking.observations ?? 0) > 0
    && booking.realizedPricePnlUsdt === 0
    && booking.feesUsdt === 0
    && booking.fundingUsdt === 0
    && booking.netPnlUsdt === 0
    && booking.maxMarginUsedUsdt === 0
    && Math.abs(booking.endingCashUsdt - booking.startingCashUsdt) <= 0.0001;
  if (!booking.synthetic || !booking.endedFlat || booking.riskBreach
    || !/^[0-9a-f]{64}$/.test(booking.datasetFingerprint)
    || !Number.isInteger(booking.days) || booking.days <= 0
    || !Number.isInteger(booking.startDay) || world.day !== booking.startDay + booking.days
    || !Number.isInteger(booking.fillCount) || booking.fillCount < 0
    || (signalStatus === "no_trade" ? !noTradeObservation : booking.fillCount <= 0)
    || (booking.signalStatus !== undefined && booking.signalStatus !== "trade" && booking.signalStatus !== "no_trade")
    || [booking.observations, booking.opportunities, booking.generatedSignals].some((count) =>
      count !== undefined && (!Number.isInteger(count) || count < 0))) {
    throw new RangeError("Only a flat synthetic replay aligned to the garden clock can be booked.");
  }
  const amounts = [booking.realizedPricePnlUsdt, booking.feesUsdt, booking.fundingUsdt, booking.netPnlUsdt];
  const accountAmounts = [booking.allocationLimitUsdt, booking.startingCashUsdt, booking.endingCashUsdt, booking.maxMarginUsedUsdt];
  if ([...amounts, ...accountAmounts].some((amount) => !Number.isFinite(amount))
    || booking.feesUsdt < 0 || booking.maxMarginUsedUsdt < 0
    || booking.allocationLimitUsdt <= 0 || booking.allocationLimitUsdt > plant.allocationUsdt
    || booking.startingCashUsdt <= 0 || booking.startingCashUsdt > plant.allocationUsdt
    || booking.maxMarginUsedUsdt > booking.allocationLimitUsdt + 0.0001
    || Math.abs(booking.endingCashUsdt - (booking.startingCashUsdt + booking.netPnlUsdt)) > 0.0001) {
    throw new RangeError("Paper replay amounts must be finite and fees non-negative.");
  }
  const expectedNet = booking.realizedPricePnlUsdt - booking.feesUsdt + booking.fundingUsdt;
  if (Math.abs(expectedNet - booking.netPnlUsdt) > 0.0001) {
    throw new RangeError("Paper replay net PnL does not reconcile to price PnL, fees, and funding.");
  }
  const capital = world.capital;
  const navUsdt = round4(capital.navUsdt + booking.netPnlUsdt);
  if (navUsdt <= 0) throw new RangeError("Paper replay would exhaust NAV; booking rejected.");
  const protectedReserveUsdt = round4(navUsdt * PROTECTED_RESERVE_RATE);
  const allocationCeilingUsdt = round4(navUsdt * MAX_ALLOCATION_RATE);
  const availableForPlants = round4(allocationCeilingUsdt - capital.reservedMarginUsdt);
  if (availableForPlants < 0) throw new RangeError("Replay loss would breach reserved seed entitlements; booking rejected.");
  let deployedCapitalUsdt = capital.deployedCapitalUsdt;
  let freeCapitalUsdt = round4(availableForPlants - deployedCapitalUsdt);
  let plants = world.plants;
  if (freeCapitalUsdt < 0) {
    const reduction = round4(-freeCapitalUsdt);
    if (reduction >= plant.allocationUsdt) throw new RangeError("Replay loss exceeds the selected plant's remaining allocation.");
    plants = world.plants.map((candidate) => candidate.id === plantId
      ? { ...candidate, allocationUsdt: round4(candidate.allocationUsdt - reduction) }
      : candidate);
    deployedCapitalUsdt = round4(deployedCapitalUsdt - reduction);
    freeCapitalUsdt = 0;
  }
  const next: SimulationWorld = {
    ...world,
    plants,
    capital: {
      ...capital,
      navUsdt,
      protectedReserveUsdt,
      allocationCeilingUsdt,
      freeCapitalUsdt,
      deployedCapitalUsdt,
      realizedPricePnlUsdt: round4(capital.realizedPricePnlUsdt + booking.realizedPricePnlUsdt),
      paperFeesUsdt: round4(capital.paperFeesUsdt + booking.feesUsdt),
      paperFundingUsdt: round4(capital.paperFundingUsdt + booking.fundingUsdt),
      paperReplayCount: capital.paperReplayCount + 1,
    },
    events: appendEvents(world.events, [{
      id: `${world.seed}:d${world.day}:paper-replay:${capital.paperReplayCount + 1}`,
      day: world.day,
      type: "paper_replay",
      actor: "synthetic-paper-engine",
      subject: plantId,
      detail: {
        days: booking.days,
        fillCount: booking.fillCount,
        signalStatus,
        ...(booking.strategyId ? { strategyId: booking.strategyId } : {}),
        ...(booking.signalGenome ? {
          genomeVersion: booking.signalGenome.schemaVersion,
          lookbackBars: booking.signalGenome.lookbackBars,
          entryThresholdBps: booking.signalGenome.entryThresholdBps,
          exitThresholdBps: booking.signalGenome.exitThresholdBps,
          targetQuantity: booking.signalGenome.targetQuantity,
        } : {}),
        ...(booking.observations !== undefined ? { observations: booking.observations } : {}),
        ...(booking.opportunities !== undefined ? { opportunities: booking.opportunities } : {}),
        ...(booking.generatedSignals !== undefined ? { generatedSignals: booking.generatedSignals } : {}),
        realizedPricePnlUsdt: round4(booking.realizedPricePnlUsdt),
        feesUsdt: round4(booking.feesUsdt),
        fundingUsdt: round4(booking.fundingUsdt),
        netPnlUsdt: round4(booking.netPnlUsdt),
        datasetFingerprint: booking.datasetFingerprint,
        synthetic: true,
      },
    }]),
  };
  assertSimulationInvariants(next);
  return next;
}

function hash32(input: string): number {
  let hash = 2166136261;
  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function unit(seed: string, label: string): number {
  return hash32(`${seed}|${label}`) / 0xffff_ffff;
}

function climateForDay(seed: string, day: number): GardenClimate {
  const phase = hash32(seed) % 997 / 997;
  const water = clamp(0.54 + Math.sin(day * 0.11 + phase * Math.PI * 2) * 0.16, 0.28, 0.8);
  const sunlight = clamp(0.61 + Math.sin(day * 0.17 + phase * Math.PI) * 0.23, 0.24, 0.92);
  return { water: round4(water), sunlight: round4(sunlight), predatorPatrol: day % 5 === 0 };
}

function shadeAt(plant: GardenPlant, plants: readonly GardenPlant[]): number {
  return plants.reduce((shade, neighbor) => {
    if (neighbor.id === plant.id || neighbor.status !== "active") return shade;
    const distance = Math.abs(neighbor.x - plant.x) / 100;
    return shade + Math.max(0, 0.17 - distance) * neighbor.genes.leaf * 0.55;
  }, 0);
}

/** Purely synthetic root, leaf, water, and sunlight response for one plant. */
function growPlant(plant: GardenPlant, plants: readonly GardenPlant[], climate: GardenClimate): GardenPlant {
  if (plant.status === "archived") return plant;
  const entersDormancy = plant.physiology === "awake"
    && plant.vigor <= 55
    && (climate.water <= 0.4 || climate.sunlight <= 0.4);
  const recovers = plant.physiology === "dormant"
    && climate.water >= 0.45
    && climate.sunlight >= 0.42;
  const physiology: PlantPhysiology = entersDormancy
    ? "dormant"
    : recovers
      ? "awake"
      : plant.physiology;
  const rootAccess = clamp(0.18 + plant.genes.root * climate.water * 0.72 + plant.genes.root * 0.12, 0.12, 1);
  const leafCapture = clamp(plant.genes.leaf * climate.sunlight * 1.22 - shadeAt(plant, plants) + 0.12, 0.08, 1);
  const waterResponse = clamp(plant.genes.water * climate.water + (1 - plant.genes.water) * (1 - Math.abs(0.5 - climate.water)), 0.08, 1);
  const sunResponse = clamp(plant.genes.sun * climate.sunlight + (1 - plant.genes.sun) * (1 - Math.abs(0.5 - climate.sunlight)), 0.08, 1);
  const growth = (rootAccess * 0.28 + leafCapture * 0.28 + waterResponse * 0.22 + sunResponse * 0.22 - 0.49) * 2.4;
  // Root and leaf growth share finite resources: exceeding a combined trait
  // budget has an explicit biomass cost, making the root/leaf trade-off visible.
  const rootLeafTradeoff = Math.max(0, plant.genes.root + plant.genes.leaf - 1) * 1.25;
  const potentialGrowth = growth - rootLeafTradeoff;
  // Fitness is a current-day environmental match, not stored performance.
  // Vigor is accumulated physiology and evidence is the observation series.
  const fitness = round1(clamp(
    (rootAccess + leafCapture + waterResponse + sunResponse) * 25 - rootLeafTradeoff * 12,
    0,
    100,
  ));
  // Dormant plants conserve themselves: they do not grow, and environmental
  // damage is damped while they wait for both water and light to recover.
  const netGrowth = physiology === "dormant" ? Math.min(0, potentialGrowth) * 0.1 : potentialGrowth;
  const potentialVigorDelta = (rootAccess * 0.25 + leafCapture * 0.25 + waterResponse * 0.25 + sunResponse * 0.25 - 0.54) * 4;
  const vigorDelta = physiology === "dormant"
    ? Math.max(-0.1, Math.min(0, potentialVigorDelta * 0.1))
    : potentialVigorDelta;
  const evidenceDelta = physiology === "dormant" ? 0 : (netGrowth * 0.65 + vigorDelta * 0.25);
  return {
    ...plant,
    physiology,
    biomass: round1(clamp(plant.biomass + netGrowth, 5, 100)),
    fitness,
    vigor: round1(clamp(plant.vigor + vigorDelta, 0, 100)),
    evidence: round1(clamp(plant.evidence + evidenceDelta, 0, 100)),
    evidenceDays: plant.evidenceDays + 1,
  };
}

function hiveGenome(seed: string, conceptionIndex: number): Genome {
  const value = (trait: keyof Genome) => round4(0.2 + unit(seed, `hive-${conceptionIndex}-${trait}`) * 0.72);
  return { root: value("root"), leaf: value("leaf"), water: value("water"), sun: value("sun") };
}

function randomGenome(seed: string, day: number, conceptionIndex: number): Genome {
  const value = (trait: keyof Genome) => round4(0.05 + unit(seed, `random-${day}-${conceptionIndex}-${trait}`) * 0.9);
  return { root: value("root"), leaf: value("leaf"), water: value("water"), sun: value("sun") };
}

function makeConception(
  seed: string,
  day: number,
  index: number,
  plants: readonly GardenPlant[],
  mode: BeeMode,
): { plant: GardenPlant; entitlementId: number } | undefined {
  const entitlementId = index + 1;
  // Parent records remain a stable, auditable genetic source even if an
  // individual parent is later archived from the visible meadow.
  const parents = plants.filter((plant) => plant.role === "parent");
  if (parents.length < PARENT_COUNT) return undefined;
  const mother = parents[index % parents.length];
  const father = parents[(index + 1 + Math.floor(unit(seed, `mate-${index}`) * (parents.length - 1))) % parents.length];
  const hive = hiveGenome(seed, index);
  const genes = mode === "random" ? randomGenome(seed, day, index) : Object.fromEntries((Object.keys(mother.genes) as (keyof Genome)[]).map((trait) => {
    const local = (mother.genes[trait] + father.genes[trait]) / 2;
    const mutation = (unit(seed, `mutation-${day}-${trait}`) - 0.5) * 0.08;
    const value = mode === "local"
      ? local + mutation
      : mode === "hive"
        ? hive[trait]
        : (local + hive[trait]) / 2 + mutation / 2;
    return [trait, round4(clamp(value, 0.05, 0.95))];
  })) as unknown as Genome;
  const id = `seedling-${String(index + 1).padStart(2, "0")}`;
  const plant: GardenPlant = {
    id,
    name: `Seedling ${String(index + 1).padStart(2, "0")}`,
    role: "seedling",
    status: "active",
    physiology: "awake",
    x: round1(8 + unit(seed, `position-${index}`) * 84),
    genes,
    phenotype: derivePhenotype(genes),
    biomass: 8,
    fitness: 50,
    vigor: 60,
    evidence: 50,
    evidenceDays: 0,
    allocationUsdt: SEEDLING_ALLOCATION_USDT,
    conceivedBy: mode,
    ...(mode === "local" || mode === "hybrid" ? { parents: [mother.id, father.id] as const } : {}),
  };
  return { plant, entitlementId };
}

function allocateSeedling(world: SimulationWorld, plant: GardenPlant, entitlementId: number): SimulationWorld {
  const slots = world.capital.seedlingEntitlements.map((slot) => slot.id === entitlementId
    ? { ...slot, status: "used" as const, plantId: plant.id }
    : slot);
  return {
    ...world,
    plants: [...world.plants, plant],
    capital: {
      ...world.capital,
      reservedMarginUsdt: round4(world.capital.reservedMarginUsdt - SEEDLING_ALLOCATION_USDT),
      deployedCapitalUsdt: round4(world.capital.deployedCapitalUsdt + SEEDLING_ALLOCATION_USDT),
      seedlingEntitlements: slots,
    },
  };
}

function isPredationEligible(plant: GardenPlant): boolean {
  // The first 80 days are observation-protected. After a longer growth season,
  // mature bark adds a second defense: predators only select severely stressed
  // plants, rather than treating age alone as evidence of weakness.
  const barkMature = plant.evidenceDays >= 120;
  const vigorLimit = barkMature ? 50 : PREDATION_VIGOR_THRESHOLD;
  const evidenceLimit = barkMature ? 60 : PREDATION_EVIDENCE_THRESHOLD;
  return plant.status === "active"
    && plant.physiology === "awake"
    && plant.evidenceDays >= MIN_OBSERVATION_DAYS
    && plant.vigor < vigorLimit
    && plant.evidence < evidenceLimit;
}

function applyPredation(world: SimulationWorld, day: number): SimulationWorld {
  if (!world.climate.predatorPatrol) return world;
  const target = world.plants
    .filter(isPredationEligible)
    .sort((left, right) => left.vigor - right.vigor || left.evidence - right.evidence || left.id.localeCompare(right.id))[0];
  if (!target) return world;
  const plants = world.plants.map((plant) => plant.id === target.id ? { ...plant, status: "archived" as const } : plant);
  const events = appendEvents(world.events, [{
    id: `${world.seed}:d${day}:predation:${target.id}`,
    day,
    type: "predation",
    actor: "garden-predator",
    subject: target.id,
    detail: {
      vigor: target.vigor,
      evidence: target.evidence,
      evidenceDays: target.evidenceDays,
      barkProtected: target.evidenceDays >= 120,
      allocationReleasedUsdt: target.allocationUsdt,
    },
  }, {
    id: `${world.seed}:d${day}:archive:${target.id}`,
    day,
    type: "archive",
    actor: "garden-predator",
    subject: target.id,
    detail: { reason: "predation", visible: false },
  }]);
  return {
    ...world,
    plants,
    capital: {
      ...world.capital,
      deployedCapitalUsdt: round4(world.capital.deployedCapitalUsdt - target.allocationUsdt),
      freeCapitalUsdt: round4(world.capital.freeCapitalUsdt + target.allocationUsdt),
    },
    events,
  };
}

/** Advance exactly one deterministic synthetic day without mutating the input. */
export function advanceSimulationDay(world: SimulationWorld): SimulationWorld {
  assertSimulationInvariants(world);
  const day = world.day + 1;
  const climate = climateForDay(world.seed, day);
  const plants = world.plants.map((plant) => growPlant(plant, world.plants, climate));
  const physiologyEvents: GardenEvent[] = plants.flatMap((plant, index) => {
    const before = world.plants[index];
    if (!before || before.physiology === plant.physiology) return [];
    const waking = plant.physiology === "awake";
    return [{
      id: `${world.seed}:d${day}:${waking ? "awakening" : "dormancy"}:${plant.id}`,
      day,
      type: waking ? "awakening" as const : "dormancy" as const,
      actor: "climate-physiology",
      subject: plant.id,
      detail: { from: before.physiology, to: plant.physiology, water: climate.water, sunlight: climate.sunlight },
    }];
  });
  let next: SimulationWorld = {
    ...world,
    day,
    climate,
    plants,
    events: appendEvents(world.events, [{
      id: `${world.seed}:d${day}:tick`,
      day,
      type: "daily_tick",
      detail: { water: climate.water, sunlight: climate.sunlight, predatorPatrol: climate.predatorPatrol },
    }, ...physiologyEvents]),
  };

  // Every seventh day, one synthetic bee mode gets a conception opportunity.
  // The 20 entitlement slots are single-use; archived seedlings do not restore one.
  if (day % 7 === 0) {
    const used = next.capital.seedlingEntitlements.filter((slot) => slot.status === "used").length;
    const available = next.capital.seedlingEntitlements.length - used;
    const index = used;
    const mode: BeeMode = next.conceptionPolicy === "mixed"
      ? (["local", "hive", "hybrid"] as const)[index % 3]
      : next.conceptionPolicy;
    const fundingAvailable = next.capital.reservedMarginUsdt >= SEEDLING_ALLOCATION_USDT;
    const conception = available > 0 && fundingAvailable
      ? makeConception(next.seed, day, index, next.plants, mode)
      : undefined;
    if (conception) {
      next = allocateSeedling(next, conception.plant, conception.entitlementId);
      next = {
        ...next,
        events: appendEvents(next.events, [{
          id: `${next.seed}:d${day}:conception:${conception.plant.id}`,
          day,
          type: "conception",
          actor: `${mode}-bee`,
          subject: conception.plant.id,
          detail: { mode, entitlementId: conception.entitlementId, allocationUsdt: conception.plant.allocationUsdt, parents: conception.plant.parents?.join(",") ?? "" },
        }]),
      };
    }
  }
  next = applyPredation(next, day);
  assertSimulationInvariants(next);
  return next;
}

export function advanceSimulationDays(world: SimulationWorld, count: number): SimulationWorld {
  if (!Number.isInteger(count) || count < 0 || count > 365) {
    throw new RangeError("Cloudgarden advances in whole days, up to 365 at a time.");
  }
  let current = world;
  for (let index = 0; index < count; index += 1) current = advanceSimulationDay(current);
  return current;
}

/** Throws when the NAV, reserve, margin, entitlements, or plant population drift. */
export function assertSimulationInvariants(world: SimulationWorld): void {
  const { capital } = world;
  if (world.events.length > MAX_EVENT_HISTORY) throw new Error("Event history exceeded its bounded retention window.");
  const balances = [capital.navUsdt, capital.allocationCeilingUsdt, capital.protectedReserveUsdt,
    capital.freeCapitalUsdt, capital.reservedMarginUsdt, capital.deployedCapitalUsdt, capital.paperFeesUsdt];
  const signedTotals = [capital.realizedPricePnlUsdt, capital.paperFundingUsdt];
  if ([...balances, ...signedTotals].some((value) => !Number.isFinite(value))
    || balances.some((value) => value < -0.0001)) {
    throw new Error("Capital ledger must contain finite values and non-negative balances.");
  }
  if (!Number.isInteger(capital.paperReplayCount) || capital.paperReplayCount < 0) {
    throw new Error("Paper replay count must be a non-negative integer.");
  }
  if (capital.navUsdt <= 0) throw new Error("Synthetic NAV must remain positive.");
  if (Math.abs(capital.protectedReserveUsdt - capital.navUsdt * PROTECTED_RESERVE_RATE) > 0.0001) {
    throw new Error("The 10% protected reserve cannot be allocated.");
  }
  if (Math.abs(capital.allocationCeilingUsdt - capital.navUsdt * MAX_ALLOCATION_RATE) > 0.0001) {
    throw new Error("The deployable allocation ceiling must remain 90% of NAV.");
  }
  const allocated = capital.freeCapitalUsdt + capital.reservedMarginUsdt + capital.deployedCapitalUsdt;
  if (Math.abs(allocated - capital.allocationCeilingUsdt) > 0.0001) {
    throw new Error("Free, margin-reserved, and deployed funds must conserve the allocation ceiling.");
  }
  if (Math.abs(capital.protectedReserveUsdt + allocated - capital.navUsdt) > 0.0001) {
    throw new Error("Protected reserve plus deployable capital must reconcile to NAV.");
  }
  const parents = world.plants.filter((plant) => plant.role === "parent");
  if (parents.length !== PARENT_COUNT) throw new Error("The world must retain exactly three parent records.");
  const ids = new Set(world.plants.map((plant) => plant.id));
  if (ids.size !== world.plants.length) throw new Error("Plant IDs must be unique.");
  const deployedFromPlants = world.plants.filter((plant) => plant.status === "active")
    .reduce((total, plant) => total + plant.allocationUsdt, 0);
  if (Math.abs(deployedFromPlants - capital.deployedCapitalUsdt) > 0.0001) {
    throw new Error("Deployed capital must match active plant allocations exactly.");
  }
  if (capital.seedlingEntitlements.length !== SEEDLING_ENTITLEMENTS) {
    throw new Error("The seed bank must contain exactly 20 single-use entitlements.");
  }
  const usedSlots = capital.seedlingEntitlements.filter((slot) => slot.status === "used");
  const seedlings = world.plants.filter((plant) => plant.role === "seedling");
  if (usedSlots.length !== seedlings.length || usedSlots.length > SEEDLING_ENTITLEMENTS) {
    throw new Error("Each seedling must consume exactly one unique entitlement.");
  }
  const fundedIds = new Set(usedSlots.map((slot) => slot.plantId));
  if (fundedIds.size !== usedSlots.length || seedlings.some((plant) => !fundedIds.has(plant.id))) {
    throw new Error("Entitlements cannot fund a plant more than once or leave seedlings unfunded.");
  }
  if (Math.abs(capital.reservedMarginUsdt - (SEEDLING_ENTITLEMENTS - usedSlots.length) * SEEDLING_ALLOCATION_USDT) > 0.0001) {
    throw new Error("Reserved margin must equal the unfunded seedling entitlement value.");
  }
  for (const plant of world.plants) {
    if (plant.physiology !== "awake" && plant.physiology !== "dormant") {
      throw new Error(`Plant ${plant.id} has an unknown physiology state.`);
    }
    if (plant.phenotype !== derivePhenotype(plant.genes)) throw new Error(`Plant ${plant.id} has a stale phenotype.`);
    if (![plant.fitness, plant.vigor, plant.evidence].every((value) => Number.isFinite(value) && value >= 0 && value <= 100)) {
      throw new Error(`Plant ${plant.id} has an out-of-range fitness, vigor, or evidence value.`);
    }
    if (plant.status === "archived" && visiblePopulation(world).some((visible) => visible.id === plant.id)) {
      throw new Error("Archived plants must not appear in visible population.");
    }
  }
}
