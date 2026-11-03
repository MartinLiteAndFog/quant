import assert from "node:assert/strict";
import test from "node:test";
import { runSyntheticReplayCycle } from "./replayCycle.ts";
import { advanceSimulationDays, createSimulationWorld } from "./simulation.ts";
import { CLOUDGARDEN_SAVE_KEY, CLOUDGARDEN_SAVE_SCHEMA_VERSION, loadCloudgardenSave, saveCloudgardenSave, type StorageLike } from "./saveGame.ts";

class MemoryStorage implements StorageLike {
  readonly values = new Map<string, string>();
  getItem(key: string): string | null { return this.values.get(key) ?? null; }
  setItem(key: string, value: string): void { this.values.set(key, value); }
}

test("versioned browser save restores the same deterministic world and view state", () => {
  const storage = new MemoryStorage();
  const world = advanceSimulationDays(createSimulationWorld("save-roundtrip"), 23);
  saveCloudgardenSave(storage, {
    schemaVersion: CLOUDGARDEN_SAVE_SCHEMA_VERSION,
    world,
    selectedId: "parent-02",
    view: "understory",
    axisX: "water",
    axisY: "sun",
    experimentInput: { seeds: ["CLOUD-071", "CLOUD-113"], days: 140 },
    paperReplay: null,
    paperReplayConfig: null,
  });

  const restored = loadCloudgardenSave(storage);
  assert.equal(restored.status, "restored");
  assert.deepEqual(restored.save?.world, world);
  assert.equal(restored.save?.selectedId, "parent-02");
  assert.equal(restored.save?.view, "understory");
  assert.deepEqual(restored.save?.experimentInput, { seeds: ["CLOUD-071", "CLOUD-113"], days: 140 });
  assert.equal(storage.values.has(CLOUDGARDEN_SAVE_KEY), true);
});

test("invalid or unreconciled local saves fail closed without crashing the game", () => {
  const storage = new MemoryStorage();
  storage.values.set(CLOUDGARDEN_SAVE_KEY, "not-json");
  assert.equal(loadCloudgardenSave(storage).status, "invalid");

  storage.values.set(CLOUDGARDEN_SAVE_KEY, JSON.stringify({ schemaVersion: 1, world: {}, selectedId: "parent-01" }));
  assert.equal(loadCloudgardenSave(storage).status, "invalid");
});

test("saved replay records are restored only when their report and original controls reconcile", async () => {
  const storage = new MemoryStorage();
  const result = await runSyntheticReplayCycle({
    days: 7,
    start: "2024-01-01T00:00:00Z",
    gardenSeed: "save-replay",
    weekIndex: 0,
    plantId: "parent-01",
    allocationUsdt: 20,
    tradingTraits: { lookbackBars: 5, entryThresholdBps: 40, exitThresholdBps: 10 },
  });
  const save = {
    schemaVersion: CLOUDGARDEN_SAVE_SCHEMA_VERSION,
    world: createSimulationWorld("save-replay"),
    selectedId: "parent-01",
    view: "understory" as const,
    axisX: "root" as const,
    axisY: "leaf" as const,
    experimentInput: null,
    paperReplay: result,
    paperReplayConfig: {
      plantId: "parent-01",
      days: 30 as const,
      tradingTraits: { lookbackBars: 5, entryThresholdBps: 40, exitThresholdBps: 10 },
    },
  };
  assert.throws(() => saveCloudgardenSave(storage, save), /versioned save contract/);

  const validSave = { ...save, paperReplayConfig: { ...save.paperReplayConfig, days: 7 as 7 } };
  saveCloudgardenSave(storage, validSave);
  const restored = loadCloudgardenSave(storage);
  assert.equal(restored.status, "restored");
  assert.equal(restored.save?.paperReplay?.execution.dataset.fingerprint, result.execution.dataset.fingerprint);
  assert.equal(restored.save?.paperReplayConfig?.tradingTraits.entryThresholdBps, 40);

  const legacyWorld = { ...validSave.world, plants: validSave.world.plants.map(({ tradingTraits: _traits, ...plant }) => plant) };
  const { tradingTraits, ...legacyConfig } = validSave.paperReplayConfig;
  storage.values.set(CLOUDGARDEN_SAVE_KEY, JSON.stringify({
    ...validSave,
    schemaVersion: 1,
    world: legacyWorld,
    paperReplayConfig: { ...legacyConfig, controls: tradingTraits },
  }));
  const migrated = loadCloudgardenSave(storage);
  assert.equal(migrated.status, "restored");
  assert.equal(migrated.save?.paperReplay?.execution.dataset.fingerprint, result.execution.dataset.fingerprint);
  assert.deepEqual(migrated.save?.paperReplayConfig?.tradingTraits, tradingTraits);
});

test("first-version local gardens migrate trading genes without losing their world", () => {
  const storage = new MemoryStorage();
  const world = createSimulationWorld("LEGACY-SAVE");
  const legacy = {
    schemaVersion: 1,
    world: { ...world, plants: world.plants.map(({ tradingTraits: _traits, ...plant }) => plant) },
    selectedId: "parent-01",
    view: "meadow",
    axisX: "root",
    axisY: "leaf",
    experimentInput: null,
    paperReplay: null,
    paperReplayConfig: null,
  };
  storage.values.set(CLOUDGARDEN_SAVE_KEY, JSON.stringify(legacy));

  const restored = loadCloudgardenSave(storage);
  assert.equal(restored.status, "restored");
  assert.equal(restored.save?.schemaVersion, CLOUDGARDEN_SAVE_SCHEMA_VERSION);
  assert.deepEqual(restored.save?.world.plants[0]?.tradingTraits, { lookbackBars: 8, entryThresholdBps: 94, exitThresholdBps: 14 });
  assert.equal(restored.save?.world.seed, "LEGACY-SAVE");
});
