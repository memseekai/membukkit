import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api";
import type { DemoMeta, StoreMeta } from "./types";
import { AskPanel } from "./components/AskPanel";
import { BenchPanel } from "./components/BenchPanel";
import { DropZone } from "./components/DropZone";
import { KeysPanel } from "./components/KeysPanel";
import { MemoryMap } from "./components/MemoryMap";
import {
  OnboardingWelcome,
  hasCompletedOnboarding,
  resetOnboarding,
} from "./components/Onboarding";
import { ProgressBar, type ProgressState } from "./components/ProgressBar";
import { PromptsPanel } from "./components/PromptsPanel";
import { SourceModal } from "./components/SourceModal";

type Tab = "demos" | "ingest" | "map" | "ask" | "prompts" | "bench";
type Theme = "dark" | "light";

const THEME_KEY = "membukkit-theme";
const ADVANCED_KEY = "membukkit-advanced-tabs";
const TABS: Tab[] = ["demos", "ingest", "map", "ask", "prompts", "bench"];

function initialTheme(): Theme {
  return localStorage.getItem(THEME_KEY) === "light" ? "light" : "dark";
}

/** Read deep-link query params once (first paint only). */
function readLaunchQuery(): { store: string | null; tab: Tab | null } {
  const params = new URLSearchParams(window.location.search);
  const store = params.get("store");
  const rawTab = params.get("tab");
  const tab = TABS.includes(rawTab as Tab) ? (rawTab as Tab) : store ? "ask" : null;
  return { store, tab };
}

export default function App() {
  const launchQuery = useMemo(readLaunchQuery, []);
  const [stores, setStores] = useState<StoreMeta[]>([]);
  const [demos, setDemos] = useState<DemoMeta[]>([]);
  const [active, setActive] = useState<string | null>(launchQuery.store);
  /** Prefer Ask; demos only when there are no stores yet. */
  const [tab, setTab] = useState<Tab>(launchQuery.tab ?? "ask");
  const [newName, setNewName] = useState("");
  const [sourceFactId, setSourceFactId] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [loadingDemo, setLoadingDemo] = useState<string | null>(null);
  const [demoProgress, setDemoProgress] = useState<ProgressState | null>(null);
  const [demoError, setDemoError] = useState<string | null>(null);
  const [navOpen, setNavOpen] = useState(false);
  const [proveToken, setProveToken] = useState(0);
  const [showAdvanced, setShowAdvanced] = useState(
    () => localStorage.getItem(ADVANCED_KEY) === "1",
  );
  const [keysForceOpen, setKeysForceOpen] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(
    () => !hasCompletedOnboarding() && !launchQuery.store,
  );

  useEffect(() => {
    if (theme === "light") {
      document.documentElement.setAttribute("data-theme", "light");
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  useEffect(() => {
    localStorage.setItem(ADVANCED_KEY, showAdvanced ? "1" : "0");
  }, [showAdvanced]);

  const loadStores = useCallback(async () => {
    const { stores } = await api.stores();
    setStores(stores);
    return stores;
  }, []);

  useEffect(() => {
    api.demos().then((r) => setDemos(r.demos)).catch(() => setDemos([]));
  }, []);

  useEffect(() => {
    loadStores().then((s) => {
      if (launchQuery.store) {
        setActive(launchQuery.store);
        setTab(launchQuery.tab ?? "ask");
        // Deep-link from `membukkit ui --demo` → prove the hero question once.
        if (launchQuery.store.startsWith("demo-")) {
          setProveToken((t) => t + 1);
        }
        return;
      }
      if (s.length > 0) {
        setActive((cur) => cur ?? s[0].name);
        setTab("ask");
      } else {
        setActive(null);
        setTab("demos");
      }
    });
  }, [loadStores, launchQuery]);

  const selectStore = (name: string) => {
    setActive(name);
    setTab("ask");
    setNavOpen(false);
  };

  const createStore = async () => {
    const name = newName.trim();
    if (!name) return;
    await api.createStore(name);
    setNewName("");
    await loadStores();
    setActive(name);
    setTab("ingest");
    setNavOpen(false);
  };

  const onIngested = async () => {
    await loadStores();
    setRefreshKey((k) => k + 1);
  };

  const deleteStore = async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      await api.deleteStore(deleteTarget);
      const remaining = await loadStores();
      if (active === deleteTarget) {
        if (remaining.length > 0) {
          setActive(remaining[0].name);
          setTab("ask");
        } else {
          setActive(null);
          setTab("demos");
        }
      }
      setDeleteTarget(null);
    } finally {
      setDeleting(false);
    }
  };

  /** Open a demo: reuse an existing store instantly; distill only on first load. */
  const openDemo = async (id: string) => {
    if (loadingDemo) return;
    const storeName = `demo-${id}`;
    const existing = stores.find((s) => s.name === storeName);
    if (existing && (existing.n_facts ?? 0) > 0) {
      setDemoError(null);
      setActive(storeName);
      setTab("ask");
      setProveToken((t) => t + 1);
      setNavOpen(false);
      return;
    }

    setLoadingDemo(id);
    setDemoProgress(null);
    setDemoError(null);
    try {
      const res = await api.loadDemo(id, setDemoProgress);
      await loadStores();
      setProveToken((t) => t + 1);
      setActive(res.store);
      setTab("ask");
      setNavOpen(false);
    } catch (e) {
      setDemoError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoadingDemo(null);
      setDemoProgress(null);
    }
  };

  const primaryTabs: [Tab, string][] = [
    ["ask", "ask"],
    ["ingest", "ingest"],
    ["map", "memory map"],
    ["demos", "demos"],
  ];
  const advancedTabs: [Tab, string][] = [
    ["prompts", "prompts"],
    ["bench", "bench"],
  ];

  // If advanced is hidden but user is on an advanced tab, fall back to ask.
  useEffect(() => {
    if (!showAdvanced && (tab === "prompts" || tab === "bench")) {
      setTab("ask");
    }
  }, [showAdvanced, tab]);

  const demoLauncher = (
    <div className="demo-launcher">
      <p className="demo-launcher-lead muted">
        First open builds the store (one-time distill). After that, click opens Ask immediately.
      </p>
      {loadingDemo && (
        <div style={{ marginBottom: 14 }}>
          <ProgressBar progress={demoProgress} fallbackLabel={`building ${loadingDemo}…`} />
        </div>
      )}
      <div className="demo-grid">
        {demos.map((d) => {
          const ready = stores.some(
            (s) => s.name === `demo-${d.id}` && (s.n_facts ?? 0) > 0,
          );
          return (
            <button
              key={d.id}
              className={`demo-card ${active === `demo-${d.id}` ? "active" : ""}`}
              disabled={!!loadingDemo}
              onClick={() => openDemo(d.id)}
            >
              <span className="demo-title">{d.title}</span>
              {d.proves ? <span className="demo-proves">Proves: {d.proves}</span> : null}
              <span className="demo-desc muted">{d.description}</span>
              <span className="demo-cta">
                {loadingDemo === d.id
                  ? "building…"
                  : ready
                    ? "open →"
                    : "build & open →"}
              </span>
            </button>
          );
        })}
      </div>
      {demoError && <div className="error-banner">{demoError}</div>}
    </div>
  );

  return (
    <>
      <button
        className="nav-toggle"
        aria-label="Toggle stores"
        onClick={() => setNavOpen((o) => !o)}
      >
        {navOpen ? "close" : "stores"}
      </button>
      {navOpen && <div className="nav-backdrop" onClick={() => setNavOpen(false)} />}

      <aside className={`sidebar ${navOpen ? "open" : ""}`}>
        <div className="logo">
          <img className="logo-mark" src="/logo.png" alt="" width={36} height={36} />
          <span>
            mem<span className="accent">bukkit</span>
          </span>
        </div>
        <p className="brand-tagline">memory with receipts</p>

        <h4 className="section-label">Stores</h4>
        {stores.map((s) => (
          <div
            key={s.name}
            className={`store-item ${s.name === active ? "active" : ""}`}
            onClick={() => selectStore(s.name)}
          >
            <span>{s.name}</span>
            <span className="store-item-right">
              <span className="count">{s.n_facts ?? 0}</span>
              {s.usage_totals?.est_cost_usd != null ? (
                <span className="store-cost muted" title="Lifetime est. LLM spend">
                  ~$
                  {s.usage_totals.est_cost_usd < 0.01
                    ? s.usage_totals.est_cost_usd.toFixed(4)
                    : s.usage_totals.est_cost_usd.toFixed(2)}
                </span>
              ) : null}
              <button
                className="x-btn"
                title={`delete store “${s.name}”`}
                onClick={(e) => {
                  e.stopPropagation();
                  setDeleteTarget(s.name);
                }}
              >
                ×
              </button>
            </span>
          </div>
        ))}
        {stores.length === 0 && (
          <div className="muted" style={{ padding: "0 8px" }}>
            none yet — open Demos
          </div>
        )}
        {launchQuery.store &&
          active === launchQuery.store &&
          !stores.some((s) => s.name === launchQuery.store) && (
            <div className="muted" style={{ padding: "8px" }}>
              waiting for store “{launchQuery.store}”…
            </div>
          )}

        {demos.length > 0 && (
          <div className="sidebar-demos">
            <h4 className="section-label">
              <button
                type="button"
                className={`section-label-btn ${tab === "demos" ? "active" : ""}`}
                onClick={() => {
                  setTab("demos");
                  setNavOpen(false);
                }}
              >
                Demos
              </button>
            </h4>
            {demos.slice(0, 4).map((d) => (
              <button
                key={d.id}
                className={`demo-link ${active === `demo-${d.id}` ? "active" : ""}`}
                disabled={!!loadingDemo}
                onClick={() => openDemo(d.id)}
                title={d.proves || d.description}
              >
                {loadingDemo === d.id ? "…" : d.title}
              </button>
            ))}
            {demos.length > 4 && (
              <button
                type="button"
                className="demo-link demo-link-more"
                onClick={() => {
                  setTab("demos");
                  setNavOpen(false);
                }}
              >
                all demos →
              </button>
            )}
          </div>
        )}

        <h4 className="section-label">New store</h4>
        <div className="new-store">
          <input
            value={newName}
            placeholder="name…"
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && createStore()}
          />
          <button className="ghost" onClick={createStore}>
            +
          </button>
        </div>
        <div className="sidebar-footer">
          <div className="status-line">
            <span className="dot">●</span>local · {stores.length} store
            {stores.length === 1 ? "" : "s"}
          </div>
          <div className="sidebar-footer-actions">
            <button className="ghost" onClick={() => setKeysForceOpen(true)} title="API keys">
              keys
            </button>
            <button
              className="ghost"
              onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
            >
              theme
            </button>
            <button
              className="ghost"
              title="Show first-run tips again"
              onClick={() => {
                resetOnboarding();
                setShowOnboarding(true);
                setTab("demos");
              }}
            >
              tips
            </button>
          </div>
        </div>
      </aside>

      <div className="main">
        {showOnboarding && (
          <OnboardingWelcome
            onDismiss={() => setShowOnboarding(false)}
            onTryDemo={() => setTab("demos")}
          />
        )}
        <KeysPanel
          forceOpen={keysForceOpen}
          onCloseForce={() => setKeysForceOpen(false)}
        />
        <nav className="tabs">
          {primaryTabs.map(([id, label]) => (
            <button
              key={id}
              className={`tab ${tab === id ? "active" : ""} ${id === "demos" ? "tab-demos" : ""}`}
              onClick={() => setTab(id)}
            >
              {label}
            </button>
          ))}
          <button
            className={`tab tab-advanced-toggle ${showAdvanced ? "active" : ""}`}
            onClick={() => setShowAdvanced((v) => !v)}
            title="Prompts editor and benchmark runner"
          >
            advanced {showAdvanced ? "▴" : "▾"}
          </button>
          {showAdvanced &&
            advancedTabs.map(([id, label]) => (
              <button
                key={id}
                className={`tab tab-secondary ${tab === id ? "active" : ""}`}
                onClick={() => setTab(id)}
              >
                {label}
              </button>
            ))}
        </nav>

        <div className="content" key={tab}>
          {tab === "bench" ? (
            <BenchPanel />
          ) : tab === "demos" ? (
            <div className="demos-page">
              <div className="title">Try a scene</div>
              <p className="empty-pitch">
                Load a curated store, jump to Ask, and see as-of answers with receipts. Or create
                your own store in the sidebar.
              </p>
              {demos.length > 0 ? (
                demoLauncher
              ) : (
                <div className="muted">No demos packaged in this install.</div>
              )}
            </div>
          ) : !active ? (
            <div className="empty-state">
              <div className="title">Pick a store</div>
              <p className="empty-pitch">
                Open <button type="button" className="linkish" onClick={() => setTab("demos")}>demos</button>{" "}
                or create a store in the sidebar, then ingest files.
              </p>
            </div>
          ) : tab === "ingest" ? (
            <DropZone store={active} onIngested={onIngested} refreshKey={refreshKey} />
          ) : tab === "map" ? (
            <MemoryMap
              store={active}
              refreshKey={refreshKey}
              onFactClick={setSourceFactId}
              onChanged={loadStores}
            />
          ) : tab === "prompts" ? (
            <PromptsPanel store={active} />
          ) : (
            <AskPanel
              store={active}
              onEvidenceClick={setSourceFactId}
              proveToken={proveToken}
              refreshKey={refreshKey}
              onMemoryChanged={() => {
                void loadStores();
                setRefreshKey((k) => k + 1);
              }}
              onEditPrompts={() => {
                setShowAdvanced(true);
                setTab("prompts");
              }}
            />
          )}
        </div>
      </div>

      {active && sourceFactId && (
        <SourceModal store={active} factId={sourceFactId} onClose={() => setSourceFactId(null)} />
      )}

      {deleteTarget && (
        <div className="modal-backdrop" onClick={() => !deleting && setDeleteTarget(null)}>
          <div className="modal confirm-modal" onClick={(e) => e.stopPropagation()}>
            <h3>
              delete store <span className="accent">{deleteTarget}</span>?
            </h3>
            <p className="muted">
              This permanently removes all of its memories, documents and raw sources. There is no
              undo.
            </p>
            <div className="confirm-actions">
              <button className="ghost" onClick={() => setDeleteTarget(null)} disabled={deleting}>
                cancel
              </button>
              <button className="danger" onClick={deleteStore} disabled={deleting}>
                {deleting ? "deleting…" : "delete store"}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
