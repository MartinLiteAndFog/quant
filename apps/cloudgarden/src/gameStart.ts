import type { SolArchive } from "./historicalGarden";

export type StartPreset = "first" | "last30" | "last90" | "last365";

/** Starting late means starting a FRESH account at a genuine observed date.
 * It must never jump a live/recorded account forward without replaying trades. */
export function observedStartDate(archive: Pick<SolArchive, "days">, preset: StartPreset): string {
  if (!archive.days.length) throw new RangeError("A verified data archive with days is required.");
  if (preset === "first") return archive.days[0]!.date;
  const n = { last30: 30, last90: 90, last365: 365 }[preset];
  const index = Math.max(0, archive.days.length - n);
  return archive.days[index]!.date;
}

/** A date picker can display gaps, but the simulation starts only on observed UTC days. */
export function requireObservedDate(archive: Pick<SolArchive, "days">, date: string): string {
  if (!archive.days.some((day) => day.date === date)) throw new RangeError("Dieses Datum fehlt im geladenen Archiv.");
  return date;
}
