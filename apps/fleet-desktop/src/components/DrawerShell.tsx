import {
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from "react";

export type DrawerId =
  | "activity"
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
  const [renderedOpen, setRenderedOpen] = useState<DrawerId>(open);
  const [visible, setVisible] = useState(Boolean(open));
  const panelRef = useRef<HTMLElement | null>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    let frame = 0;
    let dismissTimer = 0;
    if (open) {
      setRenderedOpen(open);
      frame = window.requestAnimationFrame(() => setVisible(true));
    } else {
      setVisible(false);
      dismissTimer = window.setTimeout(() => setRenderedOpen(null), 180);
    }
    return () => {
      window.cancelAnimationFrame(frame);
      window.clearTimeout(dismissTimer);
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    previousFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const frame = window.requestAnimationFrame(() => {
      panelRef.current?.querySelector<HTMLElement>("[data-drawer-close]")?.focus();
    });
    return () => {
      window.cancelAnimationFrame(frame);
      previousFocusRef.current?.focus();
    };
  }, [open]);

  useEffect(() => {
    if (!renderedOpen) return;
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [renderedOpen, onClose]);

  const keepFocusInDrawer = (event: KeyboardEvent<HTMLElement>) => {
    if (event.key !== "Tab") return;
    const focusable = [
      ...(panelRef.current?.querySelectorAll<HTMLElement>(
        'button:not(:disabled), select:not(:disabled), input:not(:disabled), [href], [tabindex]:not([tabindex="-1"])',
      ) || []),
    ].filter((element) => element.getClientRects().length > 0);
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  if (!renderedOpen) return null;
  const titleId = `drawer-title-${renderedOpen}`;
  return (
    <>
      <button
        type="button"
        aria-label="Close drawer"
        className="drawer-backdrop cursor-default border-0 p-0"
        data-visible={visible}
        onClick={onClose}
      />
      <section
        ref={panelRef}
        className={`drawer-panel ${widthClass}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        data-visible={visible}
        onKeyDown={keepFocusInDrawer}
      >
        <header className="flex items-center justify-between border-b border-[var(--line)] px-5 py-4">
          <h2
            id={titleId}
            className="text-[12px] font-semibold tracking-[0.12em] uppercase text-[var(--muted)]"
          >
            {title}
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="px-2 py-1 text-[var(--muted)] hover:text-[var(--text)]"
            aria-label="Close"
            data-drawer-close
          >
            ✕
          </button>
        </header>
        <div className="min-h-0 flex-1 overflow-auto px-5 py-4">{children}</div>
      </section>
    </>
  );
}
