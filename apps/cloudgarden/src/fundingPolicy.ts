import { fundSimulationWorld, type SimulationWorld } from "./simulation.ts";

/** External research capital only. These transfers are never trading PnL. */
export type FundingMode = "off" | "monthly" | "topup";
export type FundingSettings = Readonly<{
  mode: FundingMode;
  monthlyEur: number;
  topUpTargetEur: number;
  /** Fixed research conversion; not a historical EUR/USDT FX series. */
  eurUsdt: number;
}>;

export const DEFAULT_FUNDING_SETTINGS: FundingSettings = Object.freeze({
  mode: "topup",
  monthlyEur: 100,
  topUpTargetEur: 100,
  eurUsdt: 1,
});

const money4 = (amount: number): number => Math.round(amount * 10_000) / 10_000;

export function validateFundingSettings(settings: FundingSettings): FundingSettings {
  if (!settings || !["off", "monthly", "topup"].includes(settings.mode)
    || ![settings.monthlyEur, settings.topUpTargetEur, settings.eurUsdt].every(Number.isFinite)
    || settings.monthlyEur <= 0 || settings.monthlyEur > 10_000
    || settings.topUpTargetEur <= 0 || settings.topUpTargetEur > 10_000
    || settings.eurUsdt <= 0 || settings.eurUsdt > 100) {
    throw new RangeError("Ungültige externe Kapitalregel.");
  }
  return { ...settings };
}

/** Called only after a full observed trading day, on total marked NAV.
 * If NAV is 93 and target is 100 at 1:1, contribute 7, not 100.
 * At most one application per game day is enforced by the session controller.
 */
export function topUpWorldAfterDay(world: SimulationWorld, settings: FundingSettings):
  Readonly<{ world: SimulationWorld; depositEur: number; depositUsdt: number }> {
  validateFundingSettings(settings);
  if (settings.mode !== "topup") return { world, depositEur: 0, depositUsdt: 0 };
  const targetUsdt = money4(settings.topUpTargetEur * settings.eurUsdt);
  const depositUsdt = money4(Math.max(0, targetUsdt - world.capital.navUsdt));
  if (!depositUsdt) return { world, depositEur: 0, depositUsdt: 0 };
  return {
    world: fundSimulationWorld(world, depositUsdt),
    depositEur: money4(depositUsdt / settings.eurUsdt),
    depositUsdt,
  };
}
