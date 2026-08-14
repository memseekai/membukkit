import { useEffect, useState } from "react";
import { api } from "../api";
import type { SourceView } from "../types";

export function SourceModal({
  store,
  factId,
  onClose,
}: {
  store: string;
  factId: string;
  onClose: () => void;
}) {
  const [view, setView] = useState<SourceView | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setView(null);
    api
      .factSource(store, factId)
      .then(setView)
      .catch((e) => setError(String(e.message ?? e)));
  }, [store, factId]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        {error ? (
          <div className="error-banner">{error}</div>
        ) : !view ? (
          <div className="muted">
            <span className="spinner" /> loading source…
          </div>
        ) : (
          <>
            <h3>Memory provenance</h3>
            <div className="fact-row" style={{ cursor: "default" }}>
              <div>{view.fact.text}</div>
              <div className="fact-meta">
                <span className={`pill ${view.fact.kind}`}>{view.fact.kind}</span>
                {view.fact.timestamp && <span>{view.fact.timestamp.slice(0, 10)}</span>}
              </div>
            </div>

            {view.source?.turns ? (
              <>
                <h4 className="section-label" style={{ marginTop: 20 }}>
                  Verbatim source: {view.source.name}
                  {view.source.date ? ` — ${view.source.date}` : ""} (session{" "}
                  {view.source.session})
                </h4>
                {view.source.highlight != null && (
                  <div className="muted" style={{ fontSize: 12, marginBottom: 8 }}>
                    {view.source.highlight_kind === "lexical"
                      ? "highlighted: closest-matching turn (lexical match)"
                      : "highlighted: the turn this memory was distilled from"}
                  </div>
                )}
                {view.source.turns.map((t, i) => (
                  <div key={i} className={`turn ${i === view.source?.highlight ? "hl" : ""}`}>
                    <div className="role">{t.role}</div>
                    <div>{t.content}</div>
                  </div>
                ))}
              </>
            ) : (
              <div className="muted" style={{ marginTop: 14 }}>
                No raw source stored for this fact (ingested without document tracking).
              </div>
            )}
            <div style={{ marginTop: 16, textAlign: "right" }}>
              <button className="ghost" onClick={onClose}>
                Close
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
