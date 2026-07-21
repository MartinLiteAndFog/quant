import type { BotSeries } from "../../types";
import { HeroChart } from "../HeroChart";

interface Props {
  series: BotSeries[];
  visibleIds: Set<string>;
  isolatedId: string | null;
}

export function AccountEquityDrawer({ series, visibleIds, isolatedId }: Props) {
  return (
    <div className="flex h-full min-h-[360px] flex-col gap-3">
      <p className="text-[12px] leading-relaxed text-[var(--muted)]">
        Account equity % rebased to 0 at window start (snapshots + live stitch).
        Prefer the main <span className="text-[var(--text)]">Equity $</span> mode
        for absolute balances.
      </p>
      <div className="min-h-[320px] flex-1 border border-[var(--line)] bg-black/20">
        <HeroChart
          series={series}
          visibleIds={visibleIds}
          mode="account"
          isolatedId={isolatedId}
          showMaxDd={false}
        />
      </div>
    </div>
  );
}
