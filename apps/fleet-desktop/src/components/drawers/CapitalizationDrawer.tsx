import type { CapitalAccount } from "../../types";

interface Props {
  accounts: CapitalAccount[];
  loading?: boolean;
}

export function CapitalizationDrawer({ accounts, loading }: Props) {
  if (loading) return <p className="text-[12px] text-[var(--muted)]">Probing health…</p>;
  if (!accounts.length) {
    return <p className="text-[12px] text-[var(--muted)]">No capitalization data.</p>;
  }

  return (
    <div className="space-y-3">
      {accounts.map((a) => (
        <article key={a.id} className="border border-[var(--line)] px-3 py-3">
          <div className="mb-2 flex items-baseline justify-between gap-2">
            <h3 className="text-[13px] text-[var(--text)]">{a.display_name}</h3>
            <span className="text-[10px] uppercase tracking-[0.12em] text-[var(--muted)]">
              {a.status || "—"}
            </span>
          </div>
          <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px] text-[var(--muted)]">
            <dt>Equity</dt>
            <dd className="text-right text-[var(--text)]">
              {a.equity == null ? "—" : `${a.equity.toFixed(2)} ${a.currency || ""}`}
            </dd>
            <dt>Executor</dt>
            <dd className="text-right text-[var(--text)]">{a.executor_ready ? "ready" : "not ready"}</dd>
            <dt>Live</dt>
            <dd className="text-right text-[var(--text)]">
              {a.live_trading_enabled ? "on" : a.dry_run ? "dry-run" : "off"}
            </dd>
            <dt>Instance</dt>
            <dd className="truncate text-right text-[var(--text)]">{a.strategy_instance}</dd>
          </dl>
        </article>
      ))}
    </div>
  );
}
