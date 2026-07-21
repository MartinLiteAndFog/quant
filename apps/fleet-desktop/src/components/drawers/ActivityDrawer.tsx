import type { ActivityEvent } from "../../types";

interface Props {
  events: ActivityEvent[];
  loading?: boolean;
}

export function ActivityDrawer({ events, loading }: Props) {
  if (loading) {
    return <p className="text-[12px] text-[var(--muted)]">Loading activity…</p>;
  }
  if (!events.length) {
    return <p className="text-[12px] text-[var(--muted)]">No execution events in this window.</p>;
  }

  const lanes = new Map<string, ActivityEvent[]>();
  for (const e of events) {
    const key = e.display_name || e.strategy_instance;
    const list = lanes.get(key) || [];
    list.push(e);
    lanes.set(key, list);
  }

  return (
    <div className="space-y-5">
      {[...lanes.entries()].map(([name, rows]) => (
        <div key={name}>
          <div className="mb-2 flex items-center gap-2">
            <span
              className="h-2 w-2 rounded-full"
              style={{ background: rows[0]?.color || "var(--accent)" }}
            />
            <h3 className="text-[12px] tracking-wide text-[var(--text)]">{name}</h3>
            <span className="text-[11px] text-[var(--muted)]">{rows.length}</span>
          </div>
          <ul className="space-y-1 border-l border-[var(--line)] pl-3">
            {rows.slice(0, 40).map((e) => (
              <li key={`${e.event_id}-${e.ts}`} className="text-[11px] leading-snug text-[var(--muted)]">
                <span className="text-[var(--text)]">{(e.ts || "").replace("T", " ").slice(0, 19)}</span>
                {" · "}
                {e.stage || "event"}
                {e.side ? ` · ${e.side}` : ""}
                {e.qty != null ? ` · qty ${e.qty}` : ""}
                {e.status ? ` · ${e.status}` : ""}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
