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

export function StatusRail({
  bots,
  visibleIds,
  isolatedId,
  onToggle,
  onIsolate,
  onOpenBot,
}: Props) {
  return (
    <aside className="flex w-14 shrink-0 flex-col items-center gap-3 border-r border-[var(--line)] py-4">
      {bots.map((bot) => {
        const status = (bot.status || "down") as BotStatus;
        const active = visibleIds.has(bot.id);
        const isolated = isolatedId === bot.id;
        return (
          <button
            key={bot.id}
            title={`${bot.display_name} · ${status}`}
            onClick={() => onOpenBot(bot.id)}
            onDoubleClick={() => onIsolate(isolated ? null : bot.id)}
            onContextMenu={(e) => {
              e.preventDefault();
              onToggle(bot.id);
            }}
            className="group relative flex h-9 w-9 items-center justify-center rounded-none border border-transparent transition"
            style={{
              opacity: active || isolated ? 1 : 0.35,
              borderColor: isolated ? bot.color || "var(--accent)" : "transparent",
            }}
          >
            <span
              className="h-2.5 w-2.5 rounded-full"
              style={{ background: STATUS_COLOR[status], boxShadow: `0 0 0 3px ${bot.color || "#333"}22` }}
            />
            <span className="pointer-events-none absolute left-12 z-20 hidden whitespace-nowrap bg-[var(--bg-elevated)] px-2 py-1 text-[11px] tracking-wide text-[var(--text)] group-hover:block">
              {bot.display_name}
            </span>
          </button>
        );
      })}
    </aside>
  );
}
