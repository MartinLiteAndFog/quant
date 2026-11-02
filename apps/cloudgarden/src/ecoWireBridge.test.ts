import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { CloudgardenWireJournal, exportHistoricalWire, exportWorldWire, validateWire } from "./ecoWireBridge.ts";
import { createHistoricalSession, type SolArchive } from "./historicalGarden.ts";
import { advanceSimulationDays, createSimulationWorld } from "./simulation.ts";

const golden = JSON.parse(readFileSync(new URL("../../../tests/fixtures/ecosystem/cloudgarden-bridge-v1.json", import.meta.url), "utf8")) as Record<string, any>;
const clone = (): Record<string, any> => JSON.parse(JSON.stringify(golden));
const atNextDay = (v: Record<string, any>): void => {
  v.day_index = 1;
  v.event_id = "golden-demo:d1:wire-snapshot";
  v.as_of = "2026-01-02T00:00:00.000Z";
  v.known_at = v.as_of;
  v.groundwater.as_of = v.as_of;
  v.groundwater.accounts[0].as_of = v.as_of;
};

test("golden ECO-01/Cloudgarden JSON survives TS validation and serialization", () => {
  assert.deepEqual(validateWire(JSON.parse(JSON.stringify(validateWire(golden)))), golden);
  assert.equal(validateWire(golden).groundwater.reserve, "10.00000");
  assert.equal(validateWire(golden).plants[0]!.empirical, null);
});

test("historical session becomes a single, read-only account snapshot with real genomes", () => {
  const archive: SolArchive = {
    schemaVersion: 1, source: "private fixture", sourceSha256: "a".repeat(64), symbol: "SOL-USDT", venue: "unverified",
    firstDate: "2026-01-01", lastDate: "2026-01-01", totalBars: 1440, missingMinutes: 0, duplicateCount: 0,
    days: [{ date: "2026-01-01", barCount: 1440, gapCount: 0, missingMinutes: 0, file: "2026-01-01.json", sha256: "b".repeat(64) }],
  };
  const session = createHistoricalSession(archive, "2026-01-01");
  const before = JSON.stringify(session);
  const wire = exportHistoricalWire(session);
  assert.equal(JSON.stringify(session), before);
  assert.equal(wire.dataset_fingerprint, archive.sourceSha256);
  assert.equal(wire.groundwater.entitlements.length, 23);
  assert.equal(wire.capital_accounting.deployed_allocation, "60.0000");
  assert.equal(wire.capital_accounting.seed_reserved_allocation, "30.0000");
  assert.ok(wire.plants.every((p) => p.source_genome?.stopLossBps === 250 && p.empirical === null));
  assert.equal(wire.groundwater.entitlements.reduce((s, e) => s + Number(e.amount), 0), 90);
});

test("seed transfer moves, not duplicates, the entitlement and preserves the 90/10 reserve", () => {
  const v = clone();
  atNextDay(v);
  v.seed_entitlements[0] = { id: 1, status: "used", plant_id: "seedling-01" };
  v.groundwater.entitlements = v.groundwater.entitlements.filter((e: { strategy_id: string }) => e.strategy_id !== "seedbank:01");
  const child = { ...v.plants[0], strategy_id: "seedling-01", allocation: "1.5000", lineage_parent_ids: ["parent-01", "parent-01"] };
  v.plants.push(child);
  v.groundwater.entitlements.push({ ...v.groundwater.entitlements[0], strategy_id: "seedling-01", amount: "1.5000", realizable_margin: "1.5000" });
  v.capital_accounting.deployed_allocation = "21.5000";
  v.capital_accounting.seed_reserved_allocation = "0.0000";
  validateWire(v);
  const journal = new CloudgardenWireJournal();
  assert.equal(journal.observe(golden), "observed");
  assert.equal(journal.observe(v), "observed");
  assert.equal(journal.observe(v), "duplicate");
  assert.equal(journal.current()?.day_index, 1);
  assert.throws(() => journal.observe({ ...v, booked: { ...v.booked, price_pnl: "0.00001", reconciliation_delta: "-0.00001" } }), /Conflicting/);
});

test("capital balance changes must explicitly identify cashflow", () => {
  const v = clone();
  atNextDay(v);
  v.booked.cashflow = "10.0000";
  v.groundwater.accounts[0].equity = "110.00000";
  v.groundwater.accounts[0].free_margin = "110.00000";
  v.groundwater.equity = "110.00000";
  v.groundwater.reserve = "11.00000";
  v.groundwater.distributable = "99.00000";
  v.capital_accounting.free_margin = "110.00000";
  v.capital_accounting.free_allocation = "77.5000";
  assert.equal(validateWire(v).booked.cashflow, "10.0000");
  assert.throws(() => validateWire({ ...v, booked: { ...v.booked, cashflow: "0.0000" } }), /reconciliation/);
});

test("1/10000-USDT rounding never spends protected reserve", () => {
  const w = createSimulationWorld("rounding-test");
  const next = { ...w, capital: { ...w.capital, navUsdt: 100.0001, allocationCeilingUsdt: 90.0001,
    freeCapitalUsdt: 0.0001 } };
  const bridge = exportWorldWire(next, { datasetFingerprint: "0".repeat(64), sourceKind: "synthetic", asOf: "2026-01-01T00:00:00.000Z" });
  assert.equal(bridge.groundwater.reserve, "10.00001");
  assert.equal(bridge.groundwater.distributable, "90.00009");
  assert.equal(bridge.capital_accounting.free_allocation, "0.00009");
  assert.equal(bridge.capital_accounting.rounding_holdback, "0.00001");
  const overfunded = { ...next, plants: next.plants.map((p, i) => i === 0 ? { ...p, allocationUsdt: 20.0001 } : p),
    capital: { ...next.capital, deployedCapitalUsdt: 60.0001, freeCapitalUsdt: 0 } };
  assert.throws(() => exportWorldWire(overfunded, { datasetFingerprint: "0".repeat(64), sourceKind: "synthetic", asOf: "2026-01-01T00:00:00.000Z" }), /rounds above exact/);
});

test("evolution without historical market evidence is never OOS validated", () => {
  const w = advanceSimulationDays(createSimulationWorld("paper-ecology"), 7);
  const bridge = exportWorldWire(w, { datasetFingerprint: "0".repeat(64), sourceKind: "synthetic", asOf: "2026-01-08T00:00:00.000Z" });
  assert.ok(bridge.plants.every((p) => p.empirical === null));
  assert.ok(bridge.plants.every((p) => p.genome_kind === "synthetic_demo"));
  assert.throws(() => validateWire({ ...bridge, plants: bridge.plants.map((p) => ({
    ...p, empirical: { twr: 1, oos_credibility: 1 },
  })) }), /fabricated empirical/);
  assert.throws(() => validateWire({ ...bridge, dataset_fingerprint: "b".repeat(64), source_kind: "historical" }), /source genome/);
});

test("negative NAV can be inspected only as a frozen risk breach", () => {
  const v = clone();
  v.groundwater.equity = "-2.00000";
  v.groundwater.reserve = "0.00000";
  v.groundwater.distributable = "0.00000";
  v.groundwater.accounts[0].equity = "-2.00000";
  v.groundwater.accounts[0].free_margin = "0.00000";
  v.groundwater.entitlements = [{ ...v.groundwater.entitlements[0], amount: "0.0000", used_margin: "10.0000", realizable_margin: "0.0000" }];
  v.groundwater.risk_breach = true;
  v.groundwater.allocations_frozen = true;
  v.plants[0].allocation = "0.0000";
  v.seed_entitlements[0] = { id: 1, status: "used", plant_id: "parent-01" };
  v.capital_accounting = { deployed_allocation: "0.0000", seed_reserved_allocation: "0.0000", free_allocation: "0.0000",
    used_margin: "10.0000", reserved_order_margin: "0.0000", free_margin: "0.00000", rounding_holdback: "0.00000" };
  v.booked.net_pnl = "-102.00000";
  v.booked.reconciliation_delta = "-102.00000";
  assert.equal(validateWire(v).groundwater.equity, "-2.00000");
  assert.throws(() => validateWire({ ...v, groundwater: { ...v.groundwater, allocations_frozen: false } }), /freeze/);
});

test("reject false empirical claims, duplicate capital, future timestamps, stale replays and numeric money", () => {
  const mutants = [
    (v: Record<string, any>) => { v.plants[0].empirical = { twr: 0.5 }; },
    (v: Record<string, any>) => { v.groundwater.entitlements.push(v.groundwater.entitlements[0]); },
    (v: Record<string, any>) => { v.groundwater.reserve = "9.00000"; },
    (v: Record<string, any>) => { v.groundwater.equity = 100; },
    (v: Record<string, any>) => { v.known_at = "2026-01-02T00:00:00.000Z"; },
    (v: Record<string, any>) => { v.seed_entitlements[0].status = "used"; },
    (v: Record<string, any>) => { v.groundwater.entitlements[0].used_margin = "20.0001"; },
    (v: Record<string, any>) => { v.groundwater.entitlements[0].reserved_order_margin = "1.0000"; },
    (v: Record<string, any>) => { v.capital_writer = "python-ecosystem"; },
  ];
  for (const mutate of mutants) {
    const v = clone();
    mutate(v);
    assert.throws(() => validateWire(v));
  }
  const journal = new CloudgardenWireJournal();
  journal.observe(golden);
  const stale = clone();
  atNextDay(stale);
  stale.event_id = "golden-demo:d0:wire-snapshot";
  assert.throws(() => journal.observe(stale));
});
