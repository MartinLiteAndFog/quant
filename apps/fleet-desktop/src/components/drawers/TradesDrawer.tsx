import type { ClosedTrade } from "../../types";
import { downloadCsv } from "../../lib/csv";

interface Props {
  trades: ClosedTrade[];
  botLabel: string;
  loading?: boolean;
}

export function TradesDrawer({ trades, botLabel, loading }: Props) {
  if (loading) return <p className="text-[12px] text-[var(--muted)]">Loading trades…</p>;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-[12px] text-[var(--muted)]">{botLabel}</p>
        <button
          className="border border-[var(--line)] px-2 py-1 text-[11px] tracking-wide text-[var(--muted)] hover:text-[var(--text)]"
          onClick={() =>
            downloadCsv(
              `fleet-trades-${botLabel.replace(/\s+/g, "-").toLowerCase()}.csv`,
              trades as unknown as Array<Record<string, unknown>>,
            )
          }
          disabled={!trades.length}
        >
          Export CSV
        </button>
      </div>
      {!trades.length ? (
        <p className="text-[12px] text-[var(--muted)]">No closed trades.</p>
      ) : (
        <table className="w-full border-collapse text-left text-[11px]">
          <thead className="text-[var(--muted)]">
            <tr className="border-b border-[var(--line)]">
              <th className="py-2 font-medium">Exit</th>
              <th className="py-2 font-medium">Side</th>
              <th className="py-2 font-medium">PnL %</th>
              <th className="py-2 font-medium">Event</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((t) => (
              <tr key={t.trade_id} className="border-b border-[var(--line)]/60">
                <td className="py-2 pr-2 text-[var(--text)]">
                  {(t.exit_ts || "").replace("T", " ").slice(0, 16)}
                </td>
                <td className="py-2 pr-2">{t.side}</td>
                <td
                  className="py-2 pr-2"
                  style={{ color: (t.pnl_pct || 0) >= 0 ? "var(--live)" : "var(--down)" }}
                >
                  {t.pnl_pct == null ? "—" : `${t.pnl_pct.toFixed(3)}%`}
                </td>
                <td className="py-2 text-[var(--muted)]">{t.exit_event || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
