import type { ReactNode } from "react";

export type DrawerId =
  | "account"
  | "activity"
  | "trades"
  | "stats"
  | "capital"
  | "settings"
  | null;

interface Props {
  open: DrawerId;
  onClose: () => void;
  title: string;
  children: ReactNode;
  widthClass?: string;
}

export function DrawerShell({ open, onClose, title, children, widthClass = "w-[420px]" }: Props) {
  if (!open) return null;
  return (
    <div className="absolute inset-y-0 right-0 z-30 flex">
      <button
        aria-label="Close drawer"
        className="flex-1 cursor-default bg-black/30"
        onClick={onClose}
      />
      <section
        className={`${widthClass} flex h-full flex-col border-l border-[var(--line)] bg-[var(--bg-panel)] shadow-[-24px_0_48px_rgba(0,0,0,0.35)]`}
      >
        <header className="flex items-center justify-between border-b border-[var(--line)] px-5 py-4">
          <h2 className="text-[13px] font-medium tracking-[0.14em] uppercase text-[var(--muted)]">
            {title}
          </h2>
          <button
            onClick={onClose}
            className="text-[var(--muted)] transition hover:text-[var(--text)]"
            aria-label="Close"
          >
            ✕
          </button>
        </header>
        <div className="min-h-0 flex-1 overflow-auto px-5 py-4">{children}</div>
      </section>
    </div>
  );
}
