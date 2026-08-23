import { useMemo, useState } from "react";
import type { BotSeries } from "../../types";
import { downloadCsv } from "../../lib/csv";

type SortKey = "display_name" | "return_pct" | "max_drawdown_pct" | "trade_count" | "win_rate" | "profit_factor";

interface Props {
  series: BotSeries[];
}

export function StatsDrawer({ series }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>("return_pct");
  const [asc, setAsc] = useState(false);

  const rows = useMemo(() => {
    const copy = [...series];
    copy.sort((a, b) => {
      const av =
        sortKey === "display_name"
          ? a.display_name
          : sortKey === "win_rate" || sortKey === "profit_factor"
            ? a.stats[sortKey] ?? -Infinity
            : a.stats[sortKey];
      const bv =
        sortKey === "display_name"
          ? b.display_name
          : sortKey === "win_rate" || sortKey === "profit_factor"
            ? b.stats[sortKey] ?? -Infinity
            : b.stats[sortKey];
      if (typeof av === "string" && typeof bv === "string") {
        return asc ? av.localeCompare(bv) : bv.localeCompare(av);
      }
      return asc ? Number(av) - Number(bv) : Number(bv) - Number(av);
    });
    return copy;
  }, [series, sortKey, asc]);

  const header = (key: SortKey, label: string) => (
    <th className="cursor-pointer py-2 pr-2 font-medium" onClick={() => {
      if (sortKey === key) setAsc(!asc);
      else {
        setSortKey(key);
        setAsc(false);
      }
    }}>
      {label}
      {sortKey === key ? (asc ? " ↑" : " ↓") : ""}
    </th>
  );

  return (
    <div className="space-y-3">
      <div className="flex justify-end">
        <button
          className="border border-[var(--line)] px-2 py-1 text-[11px] text-[var(--muted)] hover:text-[var(--text)]"
          onClick={() =>
            downloadCsv(
              "fleet-stats.csv",
              rows.map((r) => ({
                bot: r.display_name,
                instance: r.strategy_instance,
                price_move_bps: r.price_move_meta?.return_bps ?? r.stats.return_pct * 100,
                price_move_max_dd_bps: r.stats.max_drawdown_pct * 100,
                strategy_return_pct: r.strategy_meta?.available
                  ? r.strategy_meta.return_pct
                  : null,
                trades: r.stats.trade_count,
                win_rate: r.stats.win_rate,
                profit_factor: r.stats.profit_factor,
              })),
            )
          }
        >
          Export CSV
        </button>
      </div>
      <table className="w-full border-collapse text-left text-[11px]">
        <thead className="text-[var(--muted)]">
          <tr className="border-b border-[var(--line)]">
            {header("display_name", "Bot")}
            {header("return_pct", "Performance · BPS")}
            {header("max_drawdown_pct", "Max DD · BPS")}
            {header("trade_count", "Trades")}
            {header("win_rate", "Win")}
            {header("profit_factor", "PF")}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id} className="border-b border-[var(--line)]/60">
              <td className="py-2 pr-2 text-[var(--text)]">{r.display_name}</td>
              <td className="py-2 pr-2" style={{ color: r.stats.return_pct >= 0 ? "var(--live)" : "var(--down)" }}>
                {(r.price_move_meta?.return_bps ?? r.stats.return_pct * 100).toFixed(0)}
              </td>
              <td className="py-2 pr-2">{(r.stats.max_drawdown_pct * 100).toFixed(0)}</td>
              <td className="py-2 pr-2">{r.stats.trade_count}</td>
              <td className="py-2 pr-2">
                {r.stats.win_rate == null ? "—" : `${(r.stats.win_rate * 100).toFixed(0)}%`}
              </td>
              <td className="py-2">{r.stats.profit_factor == null ? "—" : r.stats.profit_factor.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
