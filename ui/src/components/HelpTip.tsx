import { useEffect, useId, useRef, useState } from "react";

/** Click/focus `?` popover — short jargon help, not a tour. */
export function HelpTip({ text, label = "More info" }: { text: string; label?: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const rootRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <span className={`help-tip ${open ? "open" : ""}`} ref={rootRef}>
      <button
        type="button"
        className="help-tip-btn"
        aria-label={label}
        aria-expanded={open}
        aria-controls={id}
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setOpen((v) => !v);
        }}
      >
        ?
      </button>
      {open && (
        <span className="help-tip-bubble" id={id} role="tooltip">
          {text}
        </span>
      )}
    </span>
  );
}
