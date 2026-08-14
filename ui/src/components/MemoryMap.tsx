import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { Fact, PartitionView } from "../types";
import { ProgressBar, type ProgressState } from "./ProgressBar";

type ViewMode = "truth" | "buckets";
type StatusFilter = "all" | "current" | "superseded";

const TOPIC_RULES: { id: string; label: string; re: RegExp }[] = [
  { id: "rent", label: "Rent & housing", re: /\b(rent|lease|landlord|apartment|flat)\b/i },
  { id: "gym", label: "Gym & fitness", re: /\b(gym|workout|fitness|yoga)\b/i },
  { id: "work", label: "Work & projects", re: /\b(meridian|project|job|employer|work)\b/i },
  { id: "money", label: "Money & budgets", re: /\b(budget|salary|price|€|\$|cost)\b/i },
  { id: "people", label: "People", re: /\b(lena|birthday|friend|family)\b/i },
  { id: "contract", label: "Contracts", re: /\b(breach|liability|msa|dpa|notify|uptime)\b/i },
];

function topicFor(text: string): string {
  for (const rule of TOPIC_RULES) {
    if (rule.re.test(text)) return rule.label;
  }
  return "Other";
}

function dateKey(ts: string | null): string {
  return ts ? ts.slice(0, 10) : "unknown";
}

export function MemoryMap({
  store,
  refreshKey,
  onFactClick,
  onChanged,
}: {
  store: string;
  refreshKey: number;
  onFactClick: (factId: string) => void;
  onChanged?: () => void;
}) {
  const [mode, setMode] = useState<ViewMode>("truth");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [allFacts, setAllFacts] = useState<Fact[]>([]);
  const [truthError, setTruthError] = useState<string | null>(null);
  const [truthLoading, setTruthLoading] = useState(true);

  const [part, setPart] = useState<PartitionView | null>(null);
  const [labeling, setLabeling] = useState(false);
  const [labelProgress, setLabelProgress] = useState<ProgressState | null>(null);
  const [extracting, setExtracting] = useState(false);
  const [extractProgress, setExtractProgress] = useState<ProgressState | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [bucketFacts, setBucketFacts] = useState<Fact[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [labelError, setLabelError] = useState<string | null>(null);
  const [confirmId, setConfirmId] = useState<string | null>(null);

  useEffect(() => {
    setSelected(null);
    setBucketFacts([]);
    setLabelError(null);
    setTruthLoading(true);
    setTruthError(null);
    api
      .facts(store, { limit: 200, kind: "atomic" })
      .then(async (page) => {
        let facts = page.facts;
        if (page.total === 0) {
          const v = await api.facts(store, { limit: 200, kind: "verbatim" });
          facts = v.facts;
        } else if (page.total > facts.length) {
          // Load remaining pages for small-to-medium stores.
          const more: Fact[] = [...facts];
          for (let offset = facts.length; offset < page.total && offset < 500; offset += 200) {
            const next = await api.facts(store, { offset, limit: 200, kind: "atomic" });
            more.push(...next.facts);
          }
          facts = more;
        }
        setAllFacts(facts);
      })
      .catch((e) => setTruthError(String(e.message ?? e)))
      .finally(() => setTruthLoading(false));

    api
      .partition(store)
      .then(setPart)
      .catch((e) => setError(String(e.message ?? e)));
  }, [store, refreshKey]);

  const lane = part?.lane;
  useEffect(() => {
    if (selected === null || mode !== "buckets") {
      setBucketFacts([]);
      return;
    }
    api
      .facts(store, { bucket: selected, kind: lane, limit: 50 })
      .then((p) => setBucketFacts(p.facts))
      .catch((e) => setLabelError(String(e.message ?? e)));
  }, [store, selected, lane, mode]);

  const filtered = useMemo(() => {
    let rows = allFacts;
    if (statusFilter !== "all") {
      rows = rows.filter((f) => (f.status || "current") === statusFilter);
    }
    return [...rows].sort((a, b) => dateKey(a.timestamp).localeCompare(dateKey(b.timestamp)));
  }, [allFacts, statusFilter]);

  const grouped = useMemo(() => {
    const map = new Map<string, Fact[]>();
    for (const f of filtered) {
      const key = topicFor(f.text);
      const list = map.get(key) || [];
      list.push(f);
      map.set(key, list);
    }
    return [...map.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [filtered]);

  const label = async (refresh: boolean) => {
    setLabeling(true);
    setLabelProgress(null);
    setLabelError(null);
    try {
      setPart(await api.partition(store, true, refresh, setLabelProgress));
    } catch (e) {
      setLabelError(`labeling failed: ${String((e as Error).message ?? e)}`);
    } finally {
      setLabeling(false);
      setLabelProgress(null);
    }
  };

  const deleteFact = async (id: string) => {
    setConfirmId(null);
    try {
      await api.deleteFact(store, id);
      setAllFacts((fs) => fs.filter((f) => f.id !== id));
      setBucketFacts((fs) => fs.filter((f) => f.id !== id));
      const p = await api.partition(store);
      setPart(p);
      if (selected !== null && selected >= p.k_eff) setSelected(null);
      onChanged?.();
    } catch (e) {
      setLabelError(`delete failed: ${String((e as Error).message ?? e)}`);
    }
  };

  const extract = async () => {
    setExtracting(true);
    setExtractProgress(null);
    setLabelError(null);
    try {
      const r = await api.distill(store, setExtractProgress);
      if (r.new_facts === 0) {
        setLabelError("extraction added no new facts — the LLM found nothing to distill");
      }
      setSelected(null);
      setPart(await api.partition(store));
      onChanged?.();
    } catch (e) {
      setLabelError(`extraction failed: ${String((e as Error).message ?? e)}`);
    } finally {
      setExtracting(false);
      setExtractProgress(null);
    }
  };

  const renderFactRow = (f: Fact) => (
    <div key={f.id} className="fact-row" onClick={() => onFactClick(f.id)}>
      <button
        className={`x-btn ${confirmId === f.id ? "confirm" : ""}`}
        title="delete this memory"
        onClick={(e) => {
          e.stopPropagation();
          if (confirmId === f.id) deleteFact(f.id);
          else setConfirmId(f.id);
        }}
        onMouseLeave={() => {
          if (confirmId === f.id) setConfirmId(null);
        }}
      >
        {confirmId === f.id ? "delete?" : "×"}
      </button>
      <div className="fact-text">{f.text}</div>
      <div className="fact-meta">
        <span className={`pill ${f.kind}`}>{f.kind}</span>
        {f.status && f.status !== "current" ? (
          <span className={`pill status-${f.status}`}>{f.status}</span>
        ) : (
          <span className="pill status-current">current</span>
        )}
        {f.timestamp && <span>{f.timestamp.slice(0, 10)}</span>}
        {f.doc_name && <span>from {f.doc_name}</span>}
        {f.entities.slice(0, 4).map((e) => (
          <span key={e} className="pill">
            {e}
          </span>
        ))}
      </div>
    </div>
  );

  return (
    <div>
      <div className="map-mode-toggle">
        <button
          className={`tab ${mode === "truth" ? "active" : ""}`}
          onClick={() => setMode("truth")}
        >
          Truth
        </button>
        <button
          className={`tab ${mode === "buckets" ? "active" : ""}`}
          onClick={() => setMode("buckets")}
        >
          Buckets
        </button>
        <span className="muted map-mode-hint">
          {mode === "truth"
            ? "Timeline of facts — current and superseded"
            : "How retrieval opens topic regions"}
        </span>
      </div>

      {mode === "truth" && (
        <>
          <div className="truth-filters">
            {(["all", "current", "superseded"] as StatusFilter[]).map((s) => (
              <button
                key={s}
                className={`suggest-chip ${statusFilter === s ? "active-chip" : ""}`}
                onClick={() => setStatusFilter(s)}
              >
                {s}
              </button>
            ))}
            <span className="muted">
              {filtered.length} shown
              {allFacts.length ? ` · ${allFacts.length} loaded` : ""}
            </span>
          </div>
          {truthError && <div className="error-banner">{truthError}</div>}
          {truthLoading && <div className="muted">loading facts…</div>}
          {!truthLoading && !filtered.length && (
            <div className="empty-state">No facts match this filter yet.</div>
          )}
          {grouped.map(([topic, facts]) => (
            <div key={topic} className="truth-group">
              <h4 className="section-label">{topic}</h4>
              <div className="facts-panel truth-facts">{facts.map(renderFactRow)}</div>
            </div>
          ))}
        </>
      )}

      {mode === "buckets" && (
        <>
          {error && <div className="error-banner">{error}</div>}
          {!part && <div className="muted">loading…</div>}
          {part && part.k_eff === 0 && (
            <div className="empty-state">Not enough facts to build topic buckets yet.</div>
          )}
          {part && part.k_eff > 0 && (
            <>
              <div className="map-header">
                <div>
                  <b>{part.k_eff}</b> topic buckets over <b>{part.n_facts}</b> facts
                  <span className="muted"> — click a bucket to browse its memories</span>
                </div>
                <div className="map-header-actions">
                  {labeling ? (
                    <div className="map-label-progress">
                      <ProgressBar progress={labelProgress} fallbackLabel="labeling buckets…" />
                    </div>
                  ) : (
                    <button
                      className="primary"
                      onClick={() => label(part.buckets.some((b) => b.label))}
                      disabled={labeling}
                    >
                      {part.buckets.some((b) => b.label)
                        ? "Re-label buckets"
                        : "Auto-label buckets"}
                    </button>
                  )}
                </div>
              </div>
              {labelError && <div className="error-banner">{labelError}</div>}

              {part.lane === "verbatim" && (
                <div className="lane-notice">
                  <div>
                    These buckets are over raw (verbatim) memories — this store has no distilled
                    facts yet.
                  </div>
                  {extracting ? (
                    <div style={{ minWidth: 220, flex: 1 }}>
                      <ProgressBar progress={extractProgress} fallbackLabel="distilling…" />
                    </div>
                  ) : (
                    <button className="primary" onClick={extract} disabled={extracting}>
                      Extract atomic facts
                    </button>
                  )}
                </div>
              )}

              <div className="bucket-grid">
                {part.buckets.map((b) => {
                  const maxSize = Math.max(...part.buckets.map((x) => x.size), 1);
                  return (
                    <div
                      key={b.bucket}
                      className={`bucket-card ${selected === b.bucket ? "selected" : ""}`}
                      onClick={() => setSelected(selected === b.bucket ? null : b.bucket)}
                    >
                      <div className="label">{b.label || `bucket ${b.bucket}`}</div>
                      <div className="size">{b.size} facts</div>
                      <div className="bucket-bar">
                        <div style={{ width: `${(100 * b.size) / maxSize}%` }} />
                      </div>
                      {b.exemplars.slice(0, 2).map((e, i) => (
                        <div key={i} className="exemplar">
                          {e}
                        </div>
                      ))}
                    </div>
                  );
                })}
              </div>

              {selected !== null && (
                <div className="facts-panel">
                  <h4 className="section-label">
                    facts in bucket {selected} — click one to see its source
                  </h4>
                  {bucketFacts.map(renderFactRow)}
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
