import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { IngestReceipt, Overview, UploadResult } from "../types";
import { LiveAdd } from "./LiveAdd";
import { ProgressBar, type ProgressState } from "./ProgressBar";

function formatCost(usd?: number | null): string {
  if (usd == null) return "";
  if (usd === 0) return "$0";
  if (usd < 0.01) return `$${usd.toFixed(4)}`;
  if (usd < 1) return `$${usd.toFixed(3)}`;
  return `$${usd.toFixed(2)}`;
}

export function DropZone({
  store,
  onIngested,
  refreshKey,
}: {
  store: string;
  onIngested: () => void;
  refreshKey: number;
}) {
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<ProgressState | null>(null);
  const [feed, setFeed] = useState<UploadResult[]>([]);
  const [batchReceipt, setBatchReceipt] = useState<IngestReceipt | null>(null);
  const [overview, setOverview] = useState<Overview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirmDoc, setConfirmDoc] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.overview(store).then(setOverview).catch(() => setOverview(null));
  }, [store, refreshKey]);

  const empty = !overview || overview.n_facts === 0;

  const handleFiles = async (files: File[]) => {
    if (files.length === 0 || busy) return;
    setBusy(true);
    setProgress(null);
    setError(null);
    try {
      const res = await api.upload(store, files, setProgress);
      setFeed((f) => [...res.results, ...f]);
      if (res.receipt) setBatchReceipt(res.receipt);
      onIngested();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  const deleteDoc = async (docId: string) => {
    setConfirmDoc(null);
    setError(null);
    try {
      await api.deleteDocument(store, docId);
      onIngested();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div>
      {error && <div className="error-banner">{error}</div>}

      {empty && (
        <div className="byo-checklist">
          <h4 className="section-label">Bring your own</h4>
          <ol>
            <li>Export WhatsApp (no media), ChatGPT/Claude data ZIP, or grab PDFs / CRM CSV / notes.</li>
            <li>Drop them below (stays on this machine under ~/.membukkit).</li>
            <li>Open Ask — same question at two dates; receipts show tokens and ~$.</li>
          </ol>
          <p className="muted byo-note">No files handy? Load a demo from the sidebar.</p>
        </div>
      )}

      <section className="ingest-typed">
        <LiveAdd store={store} onAdded={onIngested} />
      </section>

      <h4 className="section-label ingest-files-label">Files</h4>
      <div
        className={`dropzone ${drag ? "drag" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          handleFiles(Array.from(e.dataTransfer.files));
        }}
        onClick={() => !busy && fileInput.current?.click()}
      >
        <div className="big">{busy ? "Ingesting…" : "Drop files here"}</div>
        {busy ? (
          <div style={{ marginTop: 16, textAlign: "left", maxWidth: 420, marginInline: "auto" }}>
            <ProgressBar progress={progress} fallbackLabel="starting…" />
          </div>
        ) : (
          <div className="hint">
            WhatsApp .txt · ChatGPT/Claude ZIP · PDF · CSV · md folder — or click to browse
          </div>
        )}
        <input
          ref={fileInput}
          type="file"
          multiple
          hidden
          accept=".txt,.md,.markdown,.pdf,.csv,.json,.jsonl,.zip"
          onChange={(e) => handleFiles(Array.from(e.target.files ?? []))}
        />
      </div>

      {batchReceipt && (
        <div className="ingest-receipt">
          <h4 className="section-label">Ingest receipt</h4>
          <p>
            {batchReceipt.files ?? 0} file(s) · +{batchReceipt.new_facts ?? 0} facts
            {batchReceipt.superseded
              ? ` · superseded ${batchReceipt.superseded}`
              : ""}
            {batchReceipt.est_cost_label || formatCost(batchReceipt.est_cost_usd)
              ? ` · ${batchReceipt.est_cost_label || formatCost(batchReceipt.est_cost_usd)}`
              : ""}
            {batchReceipt.usage
              ? ` · ${batchReceipt.usage.prompt_tokens?.toLocaleString() ?? 0} in / ${batchReceipt.usage.completion_tokens?.toLocaleString() ?? 0} out (${batchReceipt.usage.source || "est."})`
              : ""}
          </p>
          <p className="muted">{batchReceipt.note || "one-time index cost"}</p>
        </div>
      )}

      {feed.length > 0 && (
        <div className="feed">
          {feed.map((r, i) => (
            <div key={i} className="feed-item">
              <span>{r.file}</span>
              {r.error ? (
                <span className="err">{r.error}</span>
              ) : (
                <span className="ok">
                  +{r.new_facts} facts from {r.sessions} session{r.sessions === 1 ? "" : "s"}
                  {r.est_cost_usd != null ? ` · ${formatCost(r.est_cost_usd)}` : ""}
                </span>
              )}
            </div>
          ))}
        </div>
      )}

      {overview && overview.documents.length > 0 && (
        <div className="doc-list">
          <h4 className="section-label">
            {overview.n_facts} facts in memory ({overview.n_verbatim} verbatim, {overview.n_atomic}{" "}
            distilled)
            {overview.est_lifetime_cost_usd != null
              ? ` · ~${formatCost(overview.est_lifetime_cost_usd)} lifetime`
              : ""}
          </h4>
          <table>
            <thead>
              <tr>
                <th>Document</th>
                <th>Type</th>
                <th>Sessions</th>
                <th>Added</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {overview.documents.map((d) => (
                <tr key={d.doc_id}>
                  <td>{d.name}</td>
                  <td>{d.type}</td>
                  <td>{d.n_sessions}</td>
                  <td>{d.added_at?.slice(0, 10)}</td>
                  <td className="doc-actions">
                    <button
                      className={`x-btn ${confirmDoc === d.doc_id ? "confirm" : ""}`}
                      title={`delete “${d.name}” and all its facts`}
                      onClick={() => {
                        if (confirmDoc === d.doc_id) deleteDoc(d.doc_id);
                        else setConfirmDoc(d.doc_id);
                      }}
                      onMouseLeave={() => {
                        if (confirmDoc === d.doc_id) setConfirmDoc(null);
                      }}
                    >
                      {confirmDoc === d.doc_id ? "delete?" : "×"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
