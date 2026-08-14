import { useState } from "react";
import { api } from "../api";
import type { WriteReceipt } from "../types";
import { HelpTip } from "./HelpTip";

function todayISO(): string {
  return new Date().toISOString().slice(0, 10);
}

function formatCost(usd?: number | null): string {
  if (usd == null) return "";
  if (usd < 0.01) return `~$${usd.toFixed(4)}`;
  return `~$${usd.toFixed(3)}`;
}

export type LiveAddInfo = { date?: string; n_stored?: number };

export function LiveAdd({
  store,
  onAdded,
  compact = false,
}: {
  store: string;
  onAdded?: (info?: LiveAddInfo) => void;
  compact?: boolean;
}) {
  const [text, setText] = useState("");
  const [date, setDate] = useState(todayISO());
  const [busy, setBusy] = useState(false);
  const [receipt, setReceipt] = useState<WriteReceipt | null>(null);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    const content = text.trim();
    if (!content || busy) return;
    setBusy(true);
    setError(null);
    setReceipt(null);
    try {
      const r = await api.add(store, content, date || undefined);
      setReceipt(r);
      setText("");
      onAdded?.({ date: date || undefined, n_stored: r.n_stored });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const receiptLine = receipt
    ? [
        `Stored ${receipt.n_stored}`,
        receipt.superseded?.length ? `superseded ${receipt.superseded.length}` : null,
        formatCost(receipt.est_cost_usd) || null,
      ]
        .filter(Boolean)
        .join(" · ")
    : null;

  return (
    <div className={`live-add ${compact ? "live-add-compact" : ""}`}>
      {!compact && (
        <>
          <h4 className="section-label">
            Typed fact
            <HelpTip
              label="What is a typed fact?"
              text="One memory written with a statement date. Later facts can supersede earlier ones instead of deleting them."
            />
          </h4>
          <p className="muted live-add-lead">
            Add one dated memory without uploading a file.
          </p>
        </>
      )}
      <div className="live-add-form">
        <input
          value={text}
          placeholder={
            compact
              ? "e.g. Landlord raised rent to 1000€ from August"
              : "e.g. Landlord raised rent to 1000€ from August"
          }
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          disabled={busy}
        />
        <span className="live-add-date">
          <input
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            disabled={busy}
            aria-label="Statement date"
          />
          <HelpTip
            label="What is the statement date?"
            text="When this fact became true. Ask’s As of date decides whether it is visible."
          />
        </span>
        <button className="primary" onClick={submit} disabled={busy || !text.trim()}>
          {busy ? <span className="spinner" /> : "Add"}
        </button>
      </div>
      {error && <div className="error-banner">{error}</div>}
      {receiptLine && <div className="live-add-receipt muted">{receiptLine}</div>}
    </div>
  );
}
