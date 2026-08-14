import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type { BenchRecipe, BenchRun } from "../types";
import { ProgressBar } from "./ProgressBar";

const POLL_MS = 3000;

function RunResult({ run, recipe }: { run: BenchRun; recipe?: BenchRecipe }) {
  if (run.status === "failed") {
    return <div className="bench-result fail">FAILED — see log below</div>;
  }
  if (run.status !== "done" || !run.result) return null;
  const r = run.result;
  const metric = recipe?.metric ?? "score";
  if (r.smoke) {
    return (
      <div className="bench-result smoke">
        smoke run — score not comparable to the frozen number
        {r.measured != null && (
          <>
            {" "}
            (measured {metric}={r.measured.toFixed(3)})
          </>
        )}
      </div>
    );
  }
  if (r.measured == null) {
    return <div className="bench-result fail">done, but no summary found in the output dir</div>;
  }
  return (
    <div className={`bench-result ${r.passed ? "pass" : "fail"}`}>
      {r.passed ? "PASS" : "FAIL"} — measured {metric}={r.measured.toFixed(3)} vs expected{" "}
      {r.expected.toFixed(3)} (±{r.tolerance.toFixed(2)})
    </div>
  );
}

export function BenchPanel() {
  const [recipes, setRecipes] = useState<BenchRecipe[]>([]);
  const [runs, setRuns] = useState<BenchRun[]>([]);
  const [activeRun, setActiveRun] = useState<BenchRun | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const [{ recipes }, { runs }] = await Promise.all([api.benchRecipes(), api.benchRuns()]);
    setRecipes(recipes);
    setRuns(runs);
    return runs;
  }, []);

  useEffect(() => {
    refresh()
      .then((rs) => {
        const live = rs.find((r) => r.status === "running");
        if (live) api.benchRun(live.run_id).then(setActiveRun);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [refresh]);

  useEffect(() => {
    if (!activeRun || activeRun.status !== "running") return;
    const t = setInterval(async () => {
      try {
        const run = await api.benchRun(activeRun.run_id);
        setActiveRun(run);
        if (run.status !== "running") await refresh();
      } catch {
        /* transient poll failure — keep the last state, retry next tick */
      }
    }, POLL_MS);
    return () => clearInterval(t);
  }, [activeRun, refresh]);

  const start = async (recipeId: string, lite: boolean) => {
    setError(null);
    setStarting(true);
    try {
      const { run_id } = await api.benchStart(recipeId, lite);
      setConfirming(null);
      setActiveRun(await api.benchRun(run_id));
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  };

  const copy = (r: BenchRecipe) => {
    navigator.clipboard.writeText(r.cli_command);
    setCopied(r.id);
    setTimeout(() => setCopied((c) => (c === r.id ? null : c)), 1500);
  };

  const liveRecipeIds = new Set(runs.filter((r) => r.status === "running").map((r) => r.recipe_id));
  const recipeOf = (id: string) => recipes.find((r) => r.id === id);
  const activeRecipe = activeRun ? recipeOf(activeRun.recipe_id) : undefined;

  return (
    <div className="bench">
      {error && <div className="error-banner">{error}</div>}

      {activeRun && (
        <div className="bench-run">
          <div className="bench-run-head">
            <span className="bench-title">{activeRecipe?.title ?? activeRun.recipe_id}</span>
            {activeRun.lite && <span className="badge smoke">lite</span>}
            <span className={`run-status ${activeRun.status}`}>
              {activeRun.status === "running" && <span className="spinner" />} {activeRun.status}
            </span>
            <span className="muted mono-small">{activeRun.started_at}</span>
            <button className="x-btn" title="dismiss" onClick={() => setActiveRun(null)}>
              ×
            </button>
          </div>
          <RunResult run={activeRun} recipe={activeRecipe} />
          {activeRun.status === "running" && (
            <div style={{ margin: "12px 0" }}>
              <ProgressBar
                progress={
                  activeRun.progress
                    ? {
                        phase: activeRun.progress.phase,
                        done: activeRun.progress.done,
                        total: activeRun.progress.total,
                        detail: activeRun.progress.detail,
                      }
                    : null
                }
                fallbackLabel="starting bench…"
              />
            </div>
          )}
          <pre className="log-tail">
            {(activeRun.log_tail ?? []).join("\n") || "waiting for output…"}
          </pre>
        </div>
      )}

      <h4 className="section-label">Repro recipes</h4>
      <div className="bench-grid">
        {recipes.map((r) => {
          const envOk = r.env.every((e) => e.set);
          const busy = liveRecipeIds.has(r.id);
          return (
            <div className="bench-card" key={r.id}>
              <div className="bench-card-head">
                <span className="bench-title">{r.title}</span>
                <span className="badge">{r.dataset}</span>
              </div>
              <div className="bench-desc muted">{r.description}</div>
              <div className="bench-expected">
                expected {r.metric}=<b>{r.expected.toFixed(3)}</b>{" "}
                <span className="muted">±{r.tolerance.toFixed(2)}</span>
              </div>
              <div className="bench-models">
                <div>
                  <span className="k">reader</span>
                  <span className="v">{r.reader}</span>
                </div>
                <div className="distiller-row">
                  <span className="k">distiller</span>
                  <span className="v">{r.distiller}</span>
                </div>
                <div>
                  <span className="k">judge</span>
                  <span className="v">{r.judge}</span>
                </div>
                <div>
                  <span className="k">encoder</span>
                  <span className="v">{r.encoder}</span>
                </div>
              </div>
              <div className="bench-env">
                {r.env.map((e) => (
                  <span key={e.name} className={`env-pill ${e.set ? "ok" : "missing"}`}>
                    {e.name}
                    {e.set ? "" : " MISSING"}
                  </span>
                ))}
              </div>
              <div className="bench-cost muted">{r.cost_estimate}</div>
              <div className="cli-cmd" title="click to copy" onClick={() => copy(r)}>
                <code>{r.cli_command}</code>
                <span className="copy-hint">{copied === r.id ? "copied ✓" : "copy"}</span>
              </div>
              {confirming === r.id ? (
                <div className="bench-confirm">
                  <div className="muted">full run: {r.cost_estimate}</div>
                  <div className="confirm-actions">
                    <button className="ghost" onClick={() => setConfirming(null)}>
                      cancel
                    </button>
                    <button className="danger" disabled={starting} onClick={() => start(r.id, false)}>
                      {starting ? "starting…" : "run full"}
                    </button>
                  </div>
                </div>
              ) : (
                <div className="bench-actions">
                  <button
                    className="ghost"
                    disabled={!envOk || busy || starting}
                    title={envOk ? "small cheap subset" : "set the missing env vars first"}
                    onClick={() => start(r.id, true)}
                  >
                    run lite
                  </button>
                  <button
                    className="primary"
                    disabled={!envOk || busy || starting}
                    title={envOk ? "full reproduction run" : "set the missing env vars first"}
                    onClick={() => setConfirming(r.id)}
                  >
                    run full
                  </button>
                  {busy && <span className="muted mono-small">running…</span>}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <h4 className="section-label" style={{ marginTop: 28 }}>
        Runs this session
      </h4>
      {runs.length === 0 ? (
        <div className="muted">none yet</div>
      ) : (
        <div className="bench-runs">
          {runs.map((r) => (
            <div
              key={r.run_id}
              className="bench-run-row"
              onClick={() => api.benchRun(r.run_id).then(setActiveRun)}
            >
              <span className={`run-status ${r.status}`}>{r.status}</span>
              <span className="bench-title">{recipeOf(r.recipe_id)?.title ?? r.recipe_id}</span>
              {r.lite && <span className="badge smoke">lite</span>}
              <span className="muted mono-small">{r.started_at}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
