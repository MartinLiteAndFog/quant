import { useEffect, useMemo, useState } from "react";
import type { ActivityItem, ActivityKind, FleetBot } from "../../types";
import { downloadCsv } from "../../lib/csv";

type KindFilter = "all" | ActivityKind;
const PAGE_SIZE = 100;

interface Props {
  items: ActivityItem[];
  bots: Array<Pick<FleetBot, "id" | "display_name">>;
  botFilter: string | null;
  onBotFilter: (id: string | null) => void;
  loading?: boolean;
}

function fmtTs(raw?: string | null, len = 16): string {
  if (!raw) return "—";
  return raw.replace("T", " ").slice(0, len);
}

function fmtQty(q?: number | null): string {
  if (q == null || !Number.isFinite(q)) return "—";
  if (Math.abs(q) >= 100) return q.toFixed(2);
  if (Math.abs(q) >= 1) return q.toFixed(4);
  return q.toPrecision(3);
}

function fmtNumber(value?: number | null, digits = 4): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

function statusOrPnl(item: ActivityItem): { text: string; tone?: "up" | "down" } {
  if (item.kind === "fill" && item.pnl_pct != null && Number.isFinite(item.pnl_pct)) {
    const pct = item.pnl_pct;
    return {
      text: `${pct >= 0 ? "+" : ""}${pct.toFixed(3)}%`,
      tone: pct >= 0 ? "up" : "down",
    };
  }
  return { text: item.status || "—" };
}

function fmtSigned(value?: number | null, digits = 4): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

function activityTooltip(item: ActivityItem): string {
  const position = item.position_before != null || item.position_after != null
    ? `${fmtQty(item.position_before)} → ${fmtQty(item.position_after)}`
    : "—";
  const entry = item.entry_price != null
    ? `${item.side || "position"} @ ${fmtNumber(item.entry_price)}`
    : "not reported";
  const exit = item.exit_price != null
    ? `${item.side || "position"} @ ${fmtNumber(item.exit_price)}`
    : "open / not reported";
  return [
    `Time (UTC): ${fmtTs(item.ts, 24)}`,
    `Action: ${item.action}`,
    `Entry: ${entry}`,
    `Exit: ${exit}`,
    `Realized P/L: ${fmtSigned(item.realized_pnl)}`,
    `Fee: ${item.fee != null ? `${fmtSigned(item.fee)}${item.fee_currency ? ` ${item.fee_currency}` : ""}` : "not reported"}`,
    `Funding: ${fmtSigned(item.realized_funding)}`,
    `Position: ${position}`,
    item.execution_uid ? `Execution: ${item.execution_uid}` : "",
  ].filter(Boolean).join("\n");
}

export function ActivityDrawer({
  items,
  bots,
  botFilter,
  onBotFilter,
  loading,
}: Props) {
  const [kindFilter, setKindFilter] = useState<KindFilter>("all");
  const [page, setPage] = useState(0);

  const botScoped = useMemo(() => {
    if (!botFilter || botFilter === "__all__") return items;
    return items.filter(
      (e) => e.bot_id === botFilter || e.strategy_instance === botFilter,
    );
  }, [items, botFilter]);

  const fillCount = useMemo(
    () => botScoped.filter((i) => i.kind === "fill").length,
    [botScoped],
  );
  const eventCount = botScoped.length - fillCount;

  const filtered = useMemo(() => {
    if (kindFilter === "all") return botScoped;
    return botScoped.filter((e) => e.kind === kindFilter);
  }, [botScoped, kindFilter]);

  const botLabel =
    !botFilter || botFilter === "__all__"
      ? "All bots"
      : bots.find((b) => b.id === botFilter)?.display_name || botFilter;
  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const currentPage = Math.min(page, pageCount - 1);
  const pageItems = filtered.slice(currentPage * PAGE_SIZE, (currentPage + 1) * PAGE_SIZE);

  useEffect(() => setPage(0), [botFilter, kindFilter]);

  if (loading && !items.length) {
    return <p className="text-[12px] text-[var(--muted)]">Loading activity…</p>;
  }

  return (
    <div className="flex min-h-0 flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <div className="chip-group" role="tablist" aria-label="Activity kind">
          <button
            type="button"
            role="tab"
            className="chip"
            data-active={kindFilter === "all"}
            aria-selected={kindFilter === "all"}
            onClick={() => setKindFilter("all")}
          >
            All ({botScoped.length})
          </button>
          <button
            type="button"
            role="tab"
            className="chip"
            data-active={kindFilter === "fill"}
            aria-selected={kindFilter === "fill"}
            onClick={() => setKindFilter("fill")}
            title="Closed trades (fills with PnL)"
          >
            Fills ({fillCount})
          </button>
          <button
            type="button"
            role="tab"
            className="chip"
            data-active={kindFilter === "event"}
            aria-selected={kindFilter === "event"}
            onClick={() => setKindFilter("event")}
            title="Execution / webhook events"
          >
            Events ({eventCount})
          </button>
        </div>
        <select
          className="ml-auto max-w-[180px] border border-[var(--line)] bg-black/30 px-2 py-1.5 text-[11px] text-[var(--text)]"
          value={botFilter || "__all__"}
          onChange={(e) =>
            onBotFilter(e.target.value === "__all__" ? "__all__" : e.target.value)
          }
          aria-label="Filter by bot"
        >
          <option value="__all__">All bots</option>
          {bots.map((b) => (
            <option key={b.id} value={b.id}>
              {b.display_name}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="border border-[var(--line)] px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-[var(--muted)] hover:text-[var(--text)]"
          disabled={!filtered.length}
          onClick={() =>
            downloadCsv(
              `fleet-activity-${botLabel.replace(/\s+/g, "-").toLowerCase()}.csv`,
              filtered as unknown as Array<Record<string, unknown>>,
            )
          }
        >
          CSV
        </button>
      </div>

      <p className="text-[11px] text-[var(--muted)]">
        Exchange timeline · entries, reductions, closes, TP/SL, fees and funding ·
        exchange values are reported, never inferred · {botLabel} · range chip applies
        {loading ? " · refreshing…" : ""}
      </p>

      {!filtered.length ? (
        <p className="text-[12px] text-[var(--muted)]">No activity in this window.</p>
      ) : (
        <div className="flex min-h-0 flex-col gap-2">
          <div className="flex items-center justify-between text-[10px] text-[var(--muted)]">
            <span>{filtered.length} records · page {currentPage + 1} / {pageCount}</span>
            <span className="flex gap-1">
              <button type="button" className="chip" disabled={currentPage === 0} onClick={() => setPage((value) => Math.max(0, value - 1))}>Previous</button>
              <button type="button" className="chip" disabled={currentPage + 1 >= pageCount} onClick={() => setPage((value) => Math.min(pageCount - 1, value + 1))}>Next</button>
            </span>
          </div>
          <div className="overflow-x-auto">
          <table className="w-full min-w-[460px] border-collapse text-left text-[11px]">
            <thead className="text-[var(--muted)]">
              <tr className="border-b border-[var(--line)]">
                <th className="py-2 pr-2 font-medium">Time</th>
                {(!botFilter || botFilter === "__all__") && (
                  <th className="py-2 pr-2 font-medium">Bot</th>
                )}
                <th className="py-2 pr-2 font-medium">Action</th>
                <th className="py-2 pr-2 font-medium">Side</th>
                <th className="py-2 pr-2 font-medium">Qty @ price</th>
                <th className="py-2 pr-2 font-medium">Position</th>
                <th className="py-2 pr-2 font-medium">Fee / funding</th>
                <th className="py-2 font-medium">Status / PnL</th>
              </tr>
            </thead>
            <tbody>
              {pageItems.map((e) => {
                const result = statusOrPnl(e);
                return (
                  <tr key={e.id} className="border-b border-[var(--line)]/50" title={activityTooltip(e)}>
                    <td className="py-1.5 pr-2 whitespace-nowrap text-[var(--text)]">
                      {fmtTs(e.ts)}
                    </td>
                    {(!botFilter || botFilter === "__all__") && (
                      <td className="py-1.5 pr-2">
                        <span className="inline-flex items-center gap-1.5">
                          <span
                            className="inline-block h-1.5 w-1.5 shrink-0 rounded-full"
                            style={{ background: e.color || "var(--accent)" }}
                          />
                          {e.display_name || e.strategy_instance || "—"}
                        </span>
                      </td>
                    )}
                    <td className="py-1.5 pr-2 text-[var(--text)]">
                      <span className="text-[var(--muted)]">
                        {e.kind === "fill" ? "fill · " : ""}
                      </span>
                      {e.action}
                    </td>
                    <td className="py-1.5 pr-2 uppercase">{e.side || "—"}</td>
                    <td
                      className="py-1.5 pr-2 font-mono"
                      title={e.price != null ? `@ ${e.price}` : undefined}
                    >
                      {fmtQty(e.qty)} {e.price != null ? `@ ${fmtNumber(e.price)}` : ""}
                    </td>
                    <td className="py-1.5 pr-2 font-mono">
                      {e.position_before != null || e.position_after != null
                        ? `${fmtQty(e.position_before)} → ${fmtQty(e.position_after)}`
                        : "—"}
                    </td>
                    <td className="py-1.5 pr-2 font-mono" title={e.source || undefined}>
                      {e.fee != null
                        ? `fee ${fmtNumber(e.fee)}${e.fee_currency ? ` ${e.fee_currency}` : ""}`
                        : e.realized_funding != null
                          ? `fund ${fmtNumber(e.realized_funding)}`
                          : "—"}
                    </td>
                    <td
                      className="py-1.5 font-mono"
                      style={
                        result.tone === "up"
                          ? { color: "var(--live)" }
                          : result.tone === "down"
                            ? { color: "var(--down)" }
                            : { color: "var(--muted)" }
                      }
                    >
                      {result.text}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        </div>
      )}
    </div>
  );
}
