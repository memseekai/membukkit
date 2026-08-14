import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type { PromptConfigDict, PromptsView } from "../types";

type Mode = "instructions" | "advanced";

const ADVANCED_FIELDS: { key: keyof PromptConfigDict; label: string }[] = [
  { key: "extraction", label: "extraction" },
  { key: "extraction_named", label: "extraction_named" },
  { key: "extraction_document", label: "extraction_document" },
  { key: "dated_reader", label: "dated_reader" },
  { key: "reasoning_reader", label: "reasoning_reader" },
  { key: "recommendation_reader", label: "recommendation_reader" },
];

function emptyDraft(): PromptConfigDict {
  return {
    extraction: "",
    extraction_named: "",
    extraction_document: "",
    dated_reader: "",
    recommendation_reader: "",
    reasoning_reader: "",
    abstain_gate: "",
    extraction_instructions: "",
    reader_instructions: "",
  };
}

function draftFromView(view: PromptsView): PromptConfigDict {
  const d = emptyDraft();
  for (const [k, v] of Object.entries(view.prompts)) {
    if (k in d && typeof v === "string") d[k as keyof PromptConfigDict] = v;
  }
  return d;
}

/** Strip empty strings so PUT only sends set fields. */
function draftPayload(draft: PromptConfigDict): PromptConfigDict {
  const out: PromptConfigDict = {};
  for (const [k, v] of Object.entries(draft)) {
    if (v != null && String(v).trim() !== "") out[k as keyof PromptConfigDict] = v;
  }
  return out;
}

export function PromptsPanel({ store }: { store: string }) {
  const [view, setView] = useState<PromptsView | null>(null);
  const [draft, setDraft] = useState<PromptConfigDict>(emptyDraft);
  const [mode, setMode] = useState<Mode>("instructions");
  const [openAdvanced, setOpenAdvanced] = useState<Record<string, boolean>>({});
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    const v = await api.getPrompts(store);
    setView(v);
    setDraft(draftFromView(v));
    return v;
  }, [store]);

  useEffect(() => {
    setError(null);
    setStatus(null);
    load().catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [load]);

  const setField = (key: keyof PromptConfigDict, value: string) => {
    setDraft((d) => ({ ...d, [key]: value }));
  };

  const applyView = (v: PromptsView, msg: string) => {
    setView(v);
    setDraft(draftFromView(v));
    setStatus(msg);
    setError(null);
  };

  const save = async () => {
    setBusy(true);
    setError(null);
    try {
      const v = await api.putPrompts(store, draftPayload(draft));
      applyView(v, "saved");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const reset = async () => {
    setBusy(true);
    setError(null);
    try {
      const v = await api.resetPrompts(store);
      applyView(v, "reset to defaults");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const applyPack = async (packId: string) => {
    setBusy(true);
    setError(null);
    try {
      const v = await api.applyPromptPack(store, packId);
      applyView(v, `applied pack “${packId}”`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  if (!view) {
    return (
      <div className="prompts">
        {error ? <div className="error-banner">{error}</div> : <div className="muted">loading prompts…</div>}
      </div>
    );
  }

  return (
    <div className="prompts">
      {error && <div className="error-banner">{error}</div>}

      <div className="prompts-header">
        <div>
          <h4 className="section-label">Prompt packs</h4>
          <p className="muted prompts-lead">
            Apply a use-case pack, or edit instructions / full templates for store{" "}
            <span className="accent">{store}</span>
            {view.pack_id && (
              <>
                {" "}
                · active pack <span className="badge">{view.pack_id}</span>
              </>
            )}
            {view.is_default && !view.pack_id && (
              <>
                {" "}
                · <span className="muted">defaults</span>
              </>
            )}
          </p>
        </div>
        <div className="prompts-actions">
          <button className="ghost" onClick={reset} disabled={busy}>
            reset
          </button>
          <button className="primary" onClick={save} disabled={busy}>
            {busy ? "…" : "save"}
          </button>
        </div>
      </div>

      {status && <div className="prompts-status">{status}</div>}

      <div className="pack-grid">
        {view.packs.map((p) => (
          <button
            key={p.id}
            type="button"
            className={`pack-card ${view.pack_id === p.id ? "selected" : ""}`}
            onClick={() => applyPack(p.id)}
            disabled={busy}
            title={p.description}
          >
            <span className="pack-title">{p.title}</span>
            <span className="pack-desc muted">{p.description}</span>
          </button>
        ))}
      </div>

      <div className="prompts-warn">
        Changing extraction prompts does not rewrite existing facts — run Extract atomic facts /
        re-ingest.
      </div>

      <div className="mode-toggle">
        <button
          type="button"
          className={mode === "instructions" ? "active" : ""}
          onClick={() => setMode("instructions")}
        >
          Instructions
        </button>
        <button
          type="button"
          className={mode === "advanced" ? "active" : ""}
          onClick={() => setMode("advanced")}
        >
          Advanced
        </button>
      </div>

      {mode === "instructions" ? (
        <div className="prompts-fields">
          <label className="prompt-field">
            <span className="section-label">extraction_instructions</span>
            <textarea
              rows={8}
              value={draft.extraction_instructions ?? ""}
              onChange={(e) => setField("extraction_instructions", e.target.value)}
              placeholder="Natural-language overlay appended into the extraction template…"
            />
          </label>
          <label className="prompt-field">
            <span className="section-label">reader_instructions</span>
            <textarea
              rows={8}
              value={draft.reader_instructions ?? ""}
              onChange={(e) => setField("reader_instructions", e.target.value)}
              placeholder="Natural-language overlay appended into stock reader templates…"
            />
          </label>
        </div>
      ) : (
        <div className="prompts-fields">
          {ADVANCED_FIELDS.map(({ key, label }) => {
            const open = openAdvanced[key] ?? false;
            return (
              <div key={key} className="prompt-collapse">
                <button
                  type="button"
                  className="prompt-collapse-head"
                  onClick={() => setOpenAdvanced((o) => ({ ...o, [key]: !open }))}
                >
                  <span className="section-label" style={{ margin: 0 }}>
                    {label}
                  </span>
                  <span className="muted mono-small">{open ? "▾" : "▸"}</span>
                </button>
                {open && (
                  <textarea
                    rows={12}
                    value={draft[key] ?? ""}
                    onChange={(e) => setField(key, e.target.value)}
                    placeholder={`Full ${label} template override (leave empty for built-in)…`}
                  />
                )}
              </div>
            );
          })}
        </div>
      )}

      <div className="placeholders">
        <h4 className="section-label">Placeholders</h4>
        <ul>
          {Object.entries(view.placeholders).map(([field, ph]) => (
            <li key={field}>
              <span className="ph-field">{field}</span>
              <span className="muted">{ph.join(" ")}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
