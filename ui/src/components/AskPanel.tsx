import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { AskResponse, DemoMeta, PromptPackMeta, ProveBeat } from "../types";
import { HelpTip } from "./HelpTip";
import { AskTip } from "./Onboarding";
import { LiveAdd, type LiveAddInfo } from "./LiveAdd";

interface QA {
  q: string;
  a?: AskResponse;
  error?: string;
  questionDate?: string;
  label?: string;
  /** True after memory changed; cached answer may not reflect new facts. */
  stale?: boolean;
}

/** Grounded chips — avoid open “summarize memory” prompts that invite hallucinated rankings. */
const GENERIC_STARTERS = [
  "What is the most recent dated fact, and what does it say?",
  "Which concrete numbers or amounts appear in memory?",
  "What older facts were replaced by newer ones?",
];

/** Benchmark-flavored abstain copy — soften when receipts still have evidence. */
const BENCHMARK_ABSTAIN =
  "Based on our past conversations, you never mentioned that, so I don't have any information about it.";

const SOFT_ABSTAIN = "Couldn't synthesize an answer from the retrieved memories.";

const READER_EXPLAIN: Record<string, string> = {
  dated: "Lookup-style reader — usually one or a few decisive facts.",
  reasoning: "Multi-fact synthesis — temporal, aggregation, or knowledge updates.",
  recommendation: "Preference / advice reader — what to do next from past taste.",
  search: "Evidence-only retrieval (no generated answer).",
  reason: "Multi-fact synthesis — temporal, aggregation, or knowledge updates.",
  rec: "Preference / advice reader — what to do next from past taste.",
};

/** Matches server `_FULL_SCAN_BELOW` — selective scan is not meaningful below this. */
const FULL_SCAN_BELOW = 500;

function todayISO(): string {
  return new Date().toISOString().slice(0, 10);
}

function bucketId(b: number | { bucket: number }): number {
  return typeof b === "number" ? b : b.bucket;
}

function routeProb(b: number | { bucket: number; route_prob?: number }): number | undefined {
  return typeof b === "number" ? undefined : b.route_prob;
}

function displayAnswer(a: AskResponse): string {
  const raw = (a.answer ?? "").trim();
  if (!raw) return "(no answer)";
  const isAbstain =
    raw === BENCHMARK_ABSTAIN ||
    raw.toUpperCase() === "N/I" ||
    raw.toUpperCase() === "NI";
  if (isAbstain && (a.evidence?.length ?? 0) > 0) {
    return SOFT_ABSTAIN;
  }
  return raw;
}

function formatUsd(usd?: number | null): string {
  if (usd == null) return "—";
  if (usd < 0.01) return `~$${usd.toFixed(4)}`;
  return `~$${usd.toFixed(3)}`;
}

export function AskPanel({
  store,
  onEvidenceClick,
  proveToken = 0,
  onMemoryChanged,
  onEditPrompts,
  refreshKey = 0,
}: {
  store: string;
  onEvidenceClick: (factId: string) => void;
  /** Increment after a demo load to auto-run prove beats / first question. */
  proveToken?: number;
  onMemoryChanged?: () => void;
  /** Jump to Advanced → prompts editor. */
  onEditPrompts?: () => void;
  /** Bump after ingest/add so chips reload from overview. */
  refreshKey?: number;
}) {
  const [question, setQuestion] = useState("");
  const [questionDate, setQuestionDate] = useState(todayISO());
  const [current, setCurrent] = useState<QA | null>(null);
  const [recent, setRecent] = useState<QA[]>([]);
  const [compare, setCompare] = useState<QA[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>(GENERIC_STARTERS);
  const [proves, setProves] = useState<string>("");
  const [demoHintDate, setDemoHintDate] = useState<string | null>(null);
  const [horizonHintDate, setHorizonHintDate] = useState<string | null>(null);
  const [packs, setPacks] = useState<PromptPackMeta[]>([]);
  const [packId, setPackId] = useState<string | null>(null);
  const [packBusy, setPackBusy] = useState(false);
  const lastProveToken = useRef(0);
  const busyRef = useRef(false);
  const metaRef = useRef<{
    chips: string[];
    asOf: string;
    beats: ProveBeat[];
  }>({
    chips: GENERIC_STARTERS,
    asOf: todayISO(),
    beats: [],
  });

  const runAsk = useCallback(
    async (q: string, asOf: string, label?: string): Promise<QA> => {
      const entry: QA = { q, questionDate: asOf, label, stale: false };
      setBusy(true);
      busyRef.current = true;
      setCurrent(entry);
      try {
        const a = await api.ask(store, q, asOf);
        const done = { ...entry, a, stale: false };
        setCurrent(done);
        setRecent((r) => {
          const rest = r.filter(
            (x) => !(x.q === done.q && x.questionDate === done.questionDate && !x.stale),
          );
          return [done, ...rest].slice(0, 12);
        });
        return done;
      } catch (e) {
        const failed = {
          ...entry,
          stale: false,
          error: e instanceof Error ? e.message : String(e),
        };
        setCurrent(failed);
        return failed;
      } finally {
        setBusy(false);
        busyRef.current = false;
      }
    },
    [store],
  );

  const markAnswersStale = useCallback(() => {
    setCurrent((c) => (c?.a || c?.error ? { ...c, stale: true } : c));
    setRecent((rs) => rs.map((r) => (r.a || r.error ? { ...r, stale: true } : r)));
    setCompare((cs) =>
      cs ? cs.map((c) => (c.a || c.error ? { ...c, stale: true } : c)) : cs,
    );
  }, []);

  /** Open a recent/compare item; re-ask when memory has moved on. */
  const openQa = async (qa: QA) => {
    if (busy) return;
    if (qa.stale && qa.q) {
      const asOf = questionDate.trim() || qa.questionDate || todayISO();
      setCompare(null);
      await runAsk(qa.q, asOf, qa.label);
      return;
    }
    setCurrent(qa);
  };

  useEffect(() => {
    setCurrent(null);
    setRecent([]);
    setCompare(null);
    setQuestion("");
    setSuggestions(GENERIC_STARTERS);
    setProves("");
    setDemoHintDate(null);
    setHorizonHintDate(null);
    setQuestionDate(todayISO());
    setPacks([]);
    setPackId(null);
    metaRef.current = { chips: GENERIC_STARTERS, asOf: todayISO(), beats: [] };

    let cancelled = false;
    Promise.all([
      api.demos().catch(() => ({ demos: [] as DemoMeta[] })),
      api.overview(store).catch(() => null),
      api.getPrompts(store).catch(() => null),
    ]).then(([{ demos }, overview, prompts]) => {
      if (cancelled) return;
      const match = demos.find((d: DemoMeta) => store === `demo-${d.id}`);
      const stored =
        overview?.suggested_questions?.filter((q) => typeof q === "string" && q.trim()) ??
        [];
      const chips = match?.questions?.length
        ? match.questions.slice(0, 5)
        : stored.length
          ? stored.slice(0, 5)
          : GENERIC_STARTERS;
      setSuggestions(chips);
      if (match?.proves) setProves(match.proves);

      let asOf = todayISO();
      if (match?.question_date) {
        asOf = match.question_date;
        setQuestionDate(match.question_date);
        setDemoHintDate(match.question_date);
      } else if (overview?.latest_fact_date) {
        asOf = overview.latest_fact_date;
        setQuestionDate(overview.latest_fact_date);
        setHorizonHintDate(overview.latest_fact_date);
      }
      metaRef.current = {
        chips,
        asOf,
        beats: match?.prove_beats?.length ? match.prove_beats : [],
      };

      if (prompts) {
        setPacks(prompts.packs ?? []);
        setPackId(prompts.pack_id);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [store]);

  /** Soft refresh after ingest/add: chips + advance as-of so new facts are visible. */
  useEffect(() => {
    if (!refreshKey) return;
    let cancelled = false;
    api
      .overview(store)
      .then((overview) => {
        if (cancelled) return;
        const stored =
          overview?.suggested_questions?.filter((q) => typeof q === "string" && q.trim()) ??
          [];
        if (stored.length && !store.startsWith("demo-")) {
          setSuggestions(stored.slice(0, 5));
          metaRef.current = { ...metaRef.current, chips: stored.slice(0, 5) };
        }
        // New memories often post-date the previous as-of; bump forward so Ask can see them.
        const latest = overview?.latest_fact_date;
        if (latest) {
          setHorizonHintDate(latest);
          setQuestionDate((prev) => (prev && prev >= latest ? prev : latest));
          setDemoHintDate(null);
          metaRef.current = {
            ...metaRef.current,
            asOf: latest > (metaRef.current.asOf || "") ? latest : metaRef.current.asOf,
          };
        }
        markAnswersStale();
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [refreshKey, store, markAnswersStale]);

  useEffect(() => {
    if (!proveToken || proveToken === lastProveToken.current) return;
    lastProveToken.current = proveToken;
    let cancelled = false;
    const kick = async () => {
      for (let i = 0; i < 20 && !cancelled; i++) {
        await new Promise((r) => setTimeout(r, 50));
        const { chips, asOf, beats } = metaRef.current;
        if (beats.length >= 2) {
          const results: QA[] = [];
          for (const beat of beats.slice(0, 2)) {
            if (cancelled) return;
            const qa = await runAsk(beat.question, beat.as_of, beat.label);
            results.push(qa);
          }
          if (!cancelled) {
            setCompare(results);
            setCurrent(results[results.length - 1] ?? null);
            setQuestionDate(beats[beats.length - 1]?.as_of || asOf);
          }
          return;
        }
        if (chips[0]) {
          await runAsk(chips[0], asOf);
          return;
        }
      }
    };
    void kick();
    return () => {
      cancelled = true;
    };
  }, [proveToken, store, runAsk]);

  const ask = async (raw?: string) => {
    const q = (raw ?? question).trim();
    if (!q || busy) return;
    const asOf = questionDate.trim() || todayISO();
    setQuestion("");
    setCompare(null);
    await runAsk(q, asOf);
  };

  const onStyleChange = async (value: string) => {
    if (packBusy) return;
    setPackBusy(true);
    try {
      if (value === "") {
        const v = await api.resetPrompts(store);
        setPackId(v.pack_id);
      } else {
        const v = await api.applyPromptPack(store, value);
        setPackId(v.pack_id);
      }
    } catch {
      /* keep prior packId; user can retry */
    } finally {
      setPackBusy(false);
    }
  };

  const trace = current?.a?.trace;

  const labels = trace?.bucket_labels ?? {};

  const formatBuckets = (
    buckets?: (number | { bucket: number; route_prob?: number })[],
  ) => {
    if (!buckets?.length) return "—";
    return buckets
      .map((b) => {
        const id = bucketId(b);
        const label = labels[String(id)];
        const prob = routeProb(b);
        const name = label ? `"${label}" (#${id})` : `#${id}`;
        return prob != null ? `${name} ${(prob * 100).toFixed(0)}%` : name;
      })
      .join(", ");
  };

  const readerBlurb = trace
    ? READER_EXPLAIN[trace.reader_type] ?? `Reader mode: ${trace.reader_type}`
    : "";

  const nFacts = trace?.n_facts ?? 0;
  const nScanned = trace?.n_scanned ?? 0;
  const scanPct = Math.round((trace?.scan_fraction ?? 0) * 100);
  const estTokens = trace?.est_reader_tokens ?? 0;
  const isLargeStore = nFacts >= FULL_SCAN_BELOW;
  const isSmallFullScan =
    !isLargeStore && nFacts > 0 && (trace?.scan_fraction ?? 0) >= 0.99;
  const tokensLabel = estTokens > 0 ? `~${estTokens.toLocaleString()}` : "—";
  const windowPct = (
    (trace?.window_fraction ?? (estTokens > 0 ? estTokens / 128000 : 0)) * 100
  ).toFixed(2);
  const activePackTitle =
    packId != null
      ? packs.find((p) => p.id === packId)?.title || packId
      : "Default";

  return (
    <div className="ask-layout">
      <div className="ask-main">
        <div className="ask-toolbar">
          <div className="ask-style">
            <label className="ask-style-label" htmlFor="ask-style-select">
              Style
              <HelpTip
                label="What is Style?"
                text="Prompt pack for this store — how facts are extracted and answers are phrased. Default is fine for most use."
              />
            </label>
            <select
              id="ask-style-select"
              value={packId ?? ""}
              disabled={packBusy}
              onChange={(e) => void onStyleChange(e.target.value)}
              title={`Active: ${activePackTitle}`}
            >
              <option value="">Default</option>
              {packs.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.title}
                </option>
              ))}
            </select>
            {onEditPrompts && (
              <button type="button" className="linkish" onClick={onEditPrompts}>
                Edit instructions…
              </button>
            )}
          </div>
          {proves ? <span className="ask-proves">Proves: {proves}</span> : null}
        </div>

        <div className="ask-form">
          <input
            value={question}
            placeholder={`Ask “${store}”…`}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask()}
          />
          <label className="ask-asof-inline">
            <span>
              As of
              <HelpTip
                label="What is As of?"
                text="Answer using only facts known by this date. Same memory, earlier date can change the answer."
              />
            </span>
            <input
              type="date"
              value={questionDate}
              onChange={(e) => setQuestionDate(e.target.value)}
            />
          </label>
          <button className="primary" onClick={() => ask()} disabled={busy}>
            {busy ? <span className="spinner" /> : "Ask"}
          </button>
        </div>

        {(demoHintDate || horizonHintDate) && (
          <p className="ask-asof-tip muted">
            {demoHintDate
              ? `Demo default as-of ${demoHintDate}`
              : `Defaults to latest memory date (${horizonHintDate})`}
          </p>
        )}

        <AskTip />

        <details className="ask-add-disclosure">
          <summary>Add a dated memory</summary>
          <LiveAdd
            store={store}
            compact
            onAdded={(info?: LiveAddInfo) => {
              // Advance as-of immediately so the next Ask can see the new fact.
              if (info?.date) {
                setQuestionDate((prev) => (prev && prev >= info.date! ? prev : info.date!));
                setHorizonHintDate(info.date);
                setDemoHintDate(null);
                metaRef.current = {
                  ...metaRef.current,
                  asOf:
                    info.date > (metaRef.current.asOf || "")
                      ? info.date
                      : metaRef.current.asOf,
                };
              }
              if (info?.n_stored) markAnswersStale();
              onMemoryChanged?.();
            }}
          />
        </details>

        {suggestions.length > 0 && (
          <div className="suggest-row">
            {suggestions.map((s) => (
              <button
                key={s}
                className={`suggest-chip ${current?.q === s ? "active-chip" : ""}`}
                disabled={busy}
                onClick={() => ask(s)}
              >
                {s}
              </button>
            ))}
          </div>
        )}

        {compare && compare.length >= 2 && (
          <div className="ask-compare">
            <h4 className="section-label">Same question, different as-of</h4>
            <div className="ask-compare-grid">
              {compare.map((qa) => (
                <button
                  key={`${qa.questionDate}-${qa.label}`}
                  type="button"
                  className={`ask-compare-card ${
                    current?.questionDate === qa.questionDate ? "active" : ""
                  } ${qa.stale ? "stale" : ""}`}
                  onClick={() => void openQa(qa)}
                  disabled={busy}
                >
                  <div className="ask-compare-label">
                    {qa.label || qa.questionDate}
                    {qa.stale ? <span className="stale-badge">stale</span> : null}
                  </div>
                  <div className="ask-compare-answer">
                    {qa.error
                      ? qa.error
                      : qa.a
                        ? displayAnswer(qa.a)
                        : "…"}
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="ask-result">
          {!current && !compare && (
            <div className="ask-empty muted">
              Pick a suggestion or type a question. Evidence and receipts appear on the right.
            </div>
          )}
          {current && (
            <>
              <div className="ask-question">
                {current.q}
                {current.questionDate ? (
                  <span className="ask-asof-badge muted"> as of {current.questionDate}</span>
                ) : null}
                {current.stale ? <span className="stale-badge">stale</span> : null}
              </div>
              {current.stale && (current.a || current.error) && (
                <div className="ask-stale-bar">
                  <span className="muted">Memory changed since this answer.</span>
                  <button
                    type="button"
                    className="ghost"
                    disabled={busy}
                    onClick={() => void openQa(current)}
                  >
                    {busy ? "…" : "re-ask"}
                  </button>
                </div>
              )}
              {current.error ? (
                <div className="ask-answer" style={{ color: "var(--bad)" }}>
                  {current.error}
                </div>
              ) : current.a ? (
                <div className={`ask-answer appear ${current.stale ? "is-stale" : ""}`}>
                  {displayAnswer(current.a)}
                </div>
              ) : (
                <div className="ask-answer muted">
                  <span className="spinner" /> thinking…
                </div>
              )}
            </>
          )}
        </div>

        {recent.length > 1 && (
          <div className="ask-recent">
            <h4 className="section-label">Recent</h4>
            <div className="ask-recent-list">
              {recent.slice(1).map((r, i) => (
                <button
                  key={`${r.q}-${r.questionDate}-${i}`}
                  className={`ask-recent-item ${
                    current?.q === r.q && current?.questionDate === r.questionDate
                      ? "active"
                      : ""
                  } ${r.stale ? "stale" : ""}`}
                  onClick={() => void openQa(r)}
                  disabled={busy}
                  title={
                    r.stale
                      ? "Memory changed — click to re-ask"
                      : "Show this answer and its receipts"
                  }
                >
                  <span className="ask-recent-text">{r.label || r.q}</span>
                  {r.stale ? <span className="stale-badge">stale</span> : null}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="ask-side">
        <h4 className="section-label">
          Receipts
          <HelpTip
            label="What are Receipts?"
            text="Tokens, estimated cost, and how much of the store was scanned for this answer — not a chat bill."
          />
        </h4>
        {!trace ? (
          <div className="muted receipts-idle">Ask to see cost, scan, and sources.</div>
        ) : (
          <>
            <div className="receipt-strip">
              <span className="receipt-strip-item">
                <strong>{tokensLabel}</strong> tokens
              </span>
              <span className="receipt-strip-sep">·</span>
              <span className="receipt-strip-item">{formatUsd(trace.est_cost_usd)}</span>
              <span className="receipt-strip-sep">·</span>
              <span className="receipt-strip-item">{windowPct}% of 128k</span>
              <span className="receipt-strip-sep">·</span>
              <span className="receipt-strip-item">
                {isLargeStore
                  ? `${scanPct}% scanned`
                  : isSmallFullScan
                    ? `${nFacts.toLocaleString()} facts`
                    : `${nScanned}/${nFacts} ranked`}
              </span>
              {trace.usage?.source ? (
                <span className={`usage-badge ${trace.usage.source}`}>
                  {trace.usage.source === "api" ? "metered" : trace.usage.source}
                </span>
              ) : (
                <span className="usage-badge estimate">est.</span>
              )}
            </div>

            <details className="receipt-details">
              <summary>Details</summary>
              <p className="trace-narrative muted">
                Tokens estimate what the reader sees (ranked memories + question). Opened topic
                buckets until the scan budget was met.
              </p>
              <div className="trace-stats">
                <div className="stat">
                  <div className="v">{nFacts.toLocaleString()}</div>
                  <div className="k">facts in store</div>
                </div>
                <div className="stat">
                  <div className="v">{trace.reader_type}</div>
                  <div className="k">reader</div>
                </div>
              </div>
              {readerBlurb && <p className="reader-explain muted">{readerBlurb}</p>}
              {isSmallFullScan ? (
                <p className="trace-hero-note muted">
                  Full scan — selective bucket scan kicks in above ~{FULL_SCAN_BELOW} facts.
                </p>
              ) : null}
              {Object.entries(trace.lanes ?? {}).map(([lane, info]) => (
                <div key={lane} className="lane">
                  <span className={`lane-name ${lane}`}>{lane}</span> lane: opened{" "}
                  {formatBuckets(info.buckets)} · scanned{" "}
                  {Math.round((info.scan_frac ?? 0) * 100)}%
                </div>
              ))}
              {!Object.keys(trace.lanes ?? {}).length && (
                <div className="lane muted">
                  opened{" "}
                  {formatBuckets(
                    (trace.opened_buckets as (number | { bucket: number })[]) ?? [],
                  )}
                </div>
              )}
            </details>

            <h4 className="section-label" style={{ marginTop: 16 }}>
              Memories used ({current?.a?.evidence.length ?? 0})
              <HelpTip
                label="What are Memories used?"
                text="Facts the reader saw. Click one to open the original source. Superseded means an older value kept for history."
              />
            </h4>
            {current?.a?.evidence.map((e) => (
              <div key={e.ref} className="evidence-item" onClick={() => onEvidenceClick(e.fact_id)}>
                <div className="evidence-top">
                  {e.kind ? <span className={`pill ${e.kind}`}>{e.kind}</span> : null}
                  {e.status && e.status !== "current" ? (
                    <span className={`pill status-${e.status}`}>{e.status}</span>
                  ) : e.status === "current" ? (
                    <span className="pill status-current">current</span>
                  ) : null}
                  {e.timestamp ? <span className="evidence-date muted">{e.timestamp}</span> : null}
                </div>
                <div>{e.fact}</div>
                <div className="src">
                  {e.doc_name ? `${e.doc_name} · ${e.source_ref}` : e.ref} — click for source
                </div>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}
