import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type { KeysStatus } from "../types";

type Props = {
  /** When true, open the editor even if keys are ready (user clicked Settings). */
  forceOpen?: boolean;
  onCloseForce?: () => void;
  onReadyChange?: (ready: boolean) => void;
};

export function KeysPanel({ forceOpen = false, onCloseForce, onReadyChange }: Props) {
  const [status, setStatus] = useState<KeysStatus | null>(null);
  const [open, setOpen] = useState(false);
  const [openai, setOpenai] = useState("");
  const [anthropic, setAnthropic] = useState("");
  const [gemini, setGemini] = useState("");
  const [persist, setPersist] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedMsg, setSavedMsg] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const s = await api.keysStatus();
      setStatus(s);
      onReadyChange?.(s.ready);
      return s;
    } catch {
      setStatus(null);
      return null;
    }
  }, [onReadyChange]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (forceOpen) {
      setOpen(true);
      setSavedMsg(null);
      setError(null);
    }
  }, [forceOpen]);

  useEffect(() => {
    if (status && !status.ready && !forceOpen) {
      setOpen(true);
    }
  }, [status, forceOpen]);

  const needs = status?.needs ?? "openai";
  const showBanner = status != null && !status.ready && !open;

  const save = async () => {
    setSaving(true);
    setError(null);
    setSavedMsg(null);
    try {
      const body: Parameters<typeof api.putKeys>[0] = { persist };
      if (openai.trim()) body.openai_api_key = openai.trim();
      if (anthropic.trim()) body.anthropic_api_key = anthropic.trim();
      if (gemini.trim()) body.gemini_api_key = gemini.trim();
      if (!body.openai_api_key && !body.anthropic_api_key && !body.gemini_api_key) {
        setError(`Paste a ${providerLabel} key first.`);
        setSaving(false);
        return;
      }
      const s = await api.putKeys(body);
      setStatus(s);
      onReadyChange?.(s.ready);
      setOpenai("");
      setAnthropic("");
      setGemini("");
      setSavedMsg(
        persist
          ? `Saved to ${s.persisted_to || s.credentials_path} (also applied for this session).`
          : "Applied for this session only (not written to disk).",
      );
      if (s.ready) {
        setOpen(false);
        onCloseForce?.();
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const dismiss = () => {
    setOpen(false);
    onCloseForce?.();
  };

  const providerLabel =
    needs === "anthropic"
      ? "Anthropic"
      : needs === "google"
        ? "Google / Gemini"
        : needs === "ollama"
          ? "Ollama"
          : "OpenAI";

  return (
    <>
      {showBanner && (
        <div className="keys-banner" role="status">
          <span>
            No {providerLabel} API key for <code>{status?.llm || "llm"}</code>. Paste one to
            distill and ask — or use a local model.
          </span>
          <button type="button" className="ghost" onClick={() => setOpen(true)}>
            add key
          </button>
        </div>
      )}

      {open && (
        <div className="modal-backdrop" onClick={dismiss}>
          <div className="modal keys-modal" onClick={(e) => e.stopPropagation()}>
            <h3>
              API keys <span className="accent">· local only</span>
            </h3>
            <p className="muted keys-lead">
              Keys stay on this machine
              {status?.credentials_path ? (
                <>
                  {" "}
                  (<code className="keys-path">{status.credentials_path}</code>)
                </>
              ) : null}
              . Never sent to MemBukkit servers. Shell <code>export</code> still wins if set.
            </p>

            {status?.ready ? (
              <p className="keys-ready">
                Ready for <code>{status.llm || "llm"}</code>
                {status.providers.openai.set && needs === "openai" ? (
                  <>
                    {" "}
                    · OpenAI {status.providers.openai.mask} ({status.providers.openai.source})
                  </>
                ) : null}
                {status.providers.anthropic.set && needs === "anthropic" ? (
                  <>
                    {" "}
                    · Anthropic {status.providers.anthropic.mask} (
                    {status.providers.anthropic.source})
                  </>
                ) : null}
                {status.providers.google.set && needs === "google" ? (
                  <>
                    {" "}
                    · Google {status.providers.google.mask} ({status.providers.google.source})
                  </>
                ) : null}
              </p>
            ) : (
              <p className="keys-needed">
                Current LLM needs a <strong>{providerLabel}</strong> key.
              </p>
            )}

            <label className="keys-field">
              <span>OpenAI</span>
              <input
                type="password"
                autoComplete="off"
                placeholder={
                  status?.providers.openai.set
                    ? `set (${status.providers.openai.mask}) — paste to replace`
                    : "sk-…"
                }
                value={openai}
                onChange={(e) => setOpenai(e.target.value)}
              />
            </label>
            <label className="keys-field">
              <span>Anthropic</span>
              <input
                type="password"
                autoComplete="off"
                placeholder={
                  status?.providers.anthropic.set
                    ? `set (${status.providers.anthropic.mask}) — paste to replace`
                    : "sk-ant-…"
                }
                value={anthropic}
                onChange={(e) => setAnthropic(e.target.value)}
              />
            </label>
            <label className="keys-field">
              <span>Gemini</span>
              <input
                type="password"
                autoComplete="off"
                placeholder={
                  status?.providers.google.set
                    ? `set (${status.providers.google.mask}) — paste to replace`
                    : "AI…"
                }
                value={gemini}
                onChange={(e) => setGemini(e.target.value)}
              />
            </label>

            <label className="keys-persist">
              <input
                type="checkbox"
                checked={persist}
                onChange={(e) => setPersist(e.target.checked)}
              />
              Remember on this machine (writes credentials.env, mode 0600)
            </label>

            {error && <div className="error-banner">{error}</div>}
            {savedMsg && <p className="muted keys-saved">{savedMsg}</p>}

            <div className="confirm-actions">
              <button type="button" className="ghost" onClick={dismiss} disabled={saving}>
                {status?.ready ? "close" : "later"}
              </button>
              <button type="button" onClick={save} disabled={saving}>
                {saving ? "saving…" : "save"}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
