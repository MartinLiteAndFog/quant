import type { BotStatus, FleetBot } from "../types";

const STATUS_COLOR: Record<BotStatus, string> = {
  live: "var(--live)",
  dry: "var(--dry)",
  up: "#5b8fad",
  down: "var(--down)",
};

interface Props {
  bots: FleetBot[];
  visibleIds: Set<string>;
  isolatedId: string | null;
  onToggle: (id: string) => void;
  onIsolate: (id: string | null) => void;
  onOpenBot: (id: string) => void;
}

function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
}

export function StatusRail({
  bots,
  visibleIds,
  isolatedId,
  onToggle,
  onIsolate,
  onOpenBot,
}: Props) {
  return (
    <aside className="flex w-[72px] shrink-0 flex-col items-center gap-2 border-r border-[var(--line)] bg-black/15 py-3">
      <p className="mb-1 px-1 text-center text-[9px] font-semibold tracking-[0.16em] uppercase text-[var(--muted)]">
        Fleet
      </p>
      {bots.map((bot) => {
        const status = (bot.status || "down") as BotStatus;
        const active = visibleIds.has(bot.id);
        const isolated = isolatedId === bot.id;
        return (
          <button
            key={bot.id}
            type="button"
            title={`${bot.display_name} · ${status} — click activity, dbl-click isolate, right-click toggle`}
            onClick={() => onOpenBot(bot.id)}
            onDoubleClick={() => onIsolate(isolated ? null : bot.id)}
            onContextMenu={(e) => {
              e.preventDefault();
              onToggle(bot.id);
            }}
            className="group relative flex w-[56px] flex-col items-center gap-1 rounded-sm border px-1 py-1.5"
            style={{
              opacity: active || isolated ? 1 : 0.4,
              borderColor: isolated ? bot.color || "var(--accent)" : "transparent",
              background: isolated ? "rgba(255,255,255,0.03)" : "transparent",
            }}
          >
            <span
              className="h-2 w-2 rounded-full"
              style={{
                background: STATUS_COLOR[status],
                boxShadow: `0 0 0 2px ${bot.color || "#333"}33`,
              }}
            />
            <span
              className="font-mono text-[9px] font-medium tracking-wide"
              style={{ color: bot.color || "var(--muted)" }}
            >
              {initials(bot.display_name)}
            </span>
            <span className="pointer-events-none absolute left-[62px] z-20 hidden whitespace-nowrap border border-[var(--line)] bg-[var(--bg-elevated)] px-2 py-1 text-[11px] text-[var(--text)] shadow-lg group-hover:block">
              {bot.display_name}
              <span className="ml-2 text-[var(--muted)]">{status}</span>
            </span>
          </button>
        );
      })}
    </aside>
  );
}
