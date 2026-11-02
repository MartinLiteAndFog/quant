/**
 * Read-only canonical physiology projection for UE5 tile rendering.
 * This is data export ONLY; no simulation ticks, economic accounting,
 * Unreal writes or trade execution. A caller supplies the versioned genome
 * revision from its own canonical strategy store (never invented here).
 */
import { assertSimulationInvariants, type SimulationWorld, INACTIVITY_GRACE_DAYS } from "./simulation.ts";

export type VitalitySnapshotV1 = Readonly<{
  schemaVersion: 1;
  kind: "cloudgarden_vitality_snapshot";
  authority: "canonical_readonly_export";
  sceneRef: Readonly<{ seed: string; dayIndex: number }>;
  sourceContract: string;
  plants: readonly Readonly<{
    plantId: string;
    revision: string;
    status: "active" | "archived";
    vigor: number;
    patience: number;
    inactiveDays: number;
  }>[];
  policy: Readonly<{
    oneDailyUpdatePerCanonicalPlant: true;
    individualTileVitalityState: false;
    readOnly: true;
    liveTrading: false;
  }>;
}>;

export function exportVitalitySnapshot(
  world: SimulationWorld,
  revisionByPlantId: Readonly<Record<string, string>>,
): VitalitySnapshotV1 {
  assertSimulationInvariants(world);
  if (!revisionByPlantId || typeof revisionByPlantId !== "object"
    || Array.isArray(revisionByPlantId)
    || Object.keys(revisionByPlantId).length !== world.plants.length
    || world.plants.some((plant) => !Object.hasOwn(revisionByPlantId, plant.id))) {
    throw new RangeError("Supply exactly one known canonical revision per plant.");
  }
  const plants = world.plants.map((plant) => {
    const revision = revisionByPlantId[plant.id];
    if (typeof revision !== "string" || !revision.trim() || revision.length > 256
      || plant.genes.patience === undefined
      || plant.inactiveDays === undefined) {
      throw new RangeError("Missing genome revision or G2 vitality fields: refuse inferred values.");
    }
    if (plant.inactiveDays > world.day) {
      throw new RangeError("Idle days cannot exceed the canonical world clock.");
    }
    return {
      plantId: plant.id,
      revision,
      status: plant.status,
      vigor: plant.vigor,
      patience: plant.genes.patience,
      inactiveDays: plant.inactiveDays,
    };
  });
  return {
    schemaVersion: 1,
    kind: "cloudgarden_vitality_snapshot",
    authority: "canonical_readonly_export",
    sceneRef: { seed: world.seed, dayIndex: world.day },
    sourceContract: `Cloudgarden G2 biological fields; no UE-side ticks; inactivity grace ${INACTIVITY_GRACE_DAYS} days`,
    plants,
    policy: {
      oneDailyUpdatePerCanonicalPlant: true,
      individualTileVitalityState: false,
      readOnly: true,
      liveTrading: false,
    },
  };
}
