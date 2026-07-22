import type { ReactNode } from "react";

export type DrawerId =
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

/** Slide-over panel only — never a second chart competing with the hero board. */
export function DrawerShell({
  open,
  onClose,
  title,
  children,
  widthClass = "w-[420px]",
}: Props) {
  if (!open) return null;
  return (
    <>
      <button
        type="button"
        aria-label="Close drawer"
        className="drawer-backdrop cursor-default border-0 p-0"
        onClick={onClose}
      />
      <section className={`drawer-panel ${widthClass}`} role="dialog" aria-modal="true">
        <header className="flex items-center justify-between border-b border-[var(--line)] px-5 py-4">
          <h2 className="text-[12px] font-semibold tracking-[0.12em] uppercase text-[var(--muted)]">
            {title}
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="px-2 py-1 text-[var(--muted)] hover:text-[var(--text)]"
            aria-label="Close"
          >
            ✕
          </button>
        </header>
        <div className="min-h-0 flex-1 overflow-auto px-5 py-4">{children}</div>
      </section>
    </>
  );
}
