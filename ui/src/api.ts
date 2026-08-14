import type {
  AskResponse,
  BenchRecipe,
  BenchRun,
  DemoLoadResult,
  DemoMeta,
  FactsPage,
  KeysStatus,
  Overview,
  PartitionView,
  PromptConfigDict,
  PromptsView,
  SourceView,
  StoreMeta,
  UploadResult,
  WriteReceipt,
} from "./types";
import type { ProgressState } from "./components/ProgressBar";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not json */
    }
    throw new Error(String(detail));
  }
  return res.json();
}

/** Consume an NDJSON progress stream; invoke onProgress for each tick. */
async function readProgressStream<T>(
  res: Response,
  onProgress?: (p: ProgressState) => void,
): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not json */
    }
    throw new Error(String(detail));
  }
  if (!res.body) throw new Error("empty response body");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let result: T | null = null;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const msg = JSON.parse(line) as {
        type: string;
        phase?: string;
        done?: number;
        total?: number;
        detail?: string;
        [k: string]: unknown;
      };
      if (msg.type === "progress") {
        onProgress?.({
          phase: String(msg.phase ?? "work"),
          done: Number(msg.done ?? 0),
          total: Number(msg.total ?? 0),
          detail: msg.detail ? String(msg.detail) : undefined,
        });
      } else if (msg.type === "error") {
        throw new Error(String(msg.detail ?? "stream failed"));
      } else if (msg.type === "result") {
        const { type: _t, ...rest } = msg;
        result = rest as T;
      }
    }
  }
  if (result == null) throw new Error("stream ended without a result");
  return result;
}

export const api = {
  stores: () =>
    fetch("/api/stores").then((r) => json<{ stores: StoreMeta[] }>(r)),

  createStore: (name: string) =>
    fetch(`/api/stores/${encodeURIComponent(name)}`, { method: "POST" }).then((r) => json(r)),

  deleteStore: (name: string) =>
    fetch(`/api/stores/${encodeURIComponent(name)}`, { method: "DELETE" }).then((r) => json(r)),

  overview: (store: string) =>
    fetch(`/api/stores/${encodeURIComponent(store)}/overview`).then((r) => json<Overview>(r)),

  upload: (
    store: string,
    files: File[],
    onProgress?: (p: ProgressState) => void,
  ) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return fetch(`/api/stores/${encodeURIComponent(store)}/upload?stream=1`, {
      method: "POST",
      body: form,
    }).then((r) =>
      readProgressStream<{
        results: UploadResult[];
        n_facts: number;
        suggested_questions?: string[];
        receipt?: import("./types").IngestReceipt;
      }>(r, onProgress),
    );
  },

  ask: (store: string, question: string, questionDate?: string) =>
    fetch(`/api/stores/${encodeURIComponent(store)}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        ...(questionDate ? { question_date: questionDate } : {}),
      }),
    }).then((r) => json<AskResponse>(r)),

  add: (store: string, content: string, date?: string, userId?: string) =>
    fetch(`/api/v1/${encodeURIComponent(store)}/add`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content,
        ...(date ? { date } : {}),
        ...(userId ? { user_id: userId } : {}),
      }),
    }).then((r) => json<WriteReceipt>(r)),

  facts: (store: string, opts: { offset?: number; limit?: number; kind?: string; bucket?: number }) => {
    const params = new URLSearchParams();
    if (opts.offset) params.set("offset", String(opts.offset));
    params.set("limit", String(opts.limit ?? 50));
    if (opts.kind) params.set("kind", opts.kind);
    if (opts.bucket !== undefined) params.set("bucket", String(opts.bucket));
    return fetch(`/api/stores/${encodeURIComponent(store)}/facts?${params}`).then((r) =>
      json<FactsPage>(r),
    );
  },

  deleteFact: (store: string, factId: string) =>
    fetch(`/api/stores/${encodeURIComponent(store)}/facts/${encodeURIComponent(factId)}`, {
      method: "DELETE",
    }).then((r) =>
      json<{ deleted: string; n_facts: number; n_verbatim: number; n_atomic: number }>(r),
    ),

  deleteDocument: (store: string, docId: string) =>
    fetch(`/api/stores/${encodeURIComponent(store)}/documents/${encodeURIComponent(docId)}`, {
      method: "DELETE",
    }).then((r) => json<{ deleted: string; facts_removed: number; n_facts: number }>(r)),

  factSource: (store: string, factId: string) =>
    fetch(`/api/stores/${encodeURIComponent(store)}/facts/${factId}/source`).then((r) =>
      json<SourceView>(r),
    ),

  distill: (store: string, onProgress?: (p: ProgressState) => void) =>
    fetch(`/api/stores/${encodeURIComponent(store)}/distill?stream=1`, { method: "POST" }).then(
      (r) =>
        readProgressStream<{ new_facts: number; n_facts: number; n_atomic: number }>(
          r,
          onProgress,
        ),
    ),

  partition: (
    store: string,
    label = false,
    refresh = false,
    onProgress?: (p: ProgressState) => void,
  ) => {
    const params = new URLSearchParams({
      label: String(label),
      refresh: String(refresh),
    });
    if (label && onProgress) {
      params.set("stream", "1");
      return fetch(
        `/api/stores/${encodeURIComponent(store)}/partition?${params}`,
      ).then((r) => readProgressStream<PartitionView>(r, onProgress));
    }
    return fetch(
      `/api/stores/${encodeURIComponent(store)}/partition?${params}`,
    ).then((r) => json<PartitionView>(r));
  },

  benchRecipes: () =>
    fetch("/api/bench/recipes").then((r) => json<{ recipes: BenchRecipe[] }>(r)),

  benchStart: (recipeId: string, lite: boolean) =>
    fetch("/api/bench/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ recipe_id: recipeId, lite }),
    }).then((r) => json<{ run_id: string }>(r)),

  benchRuns: () => fetch("/api/bench/runs").then((r) => json<{ runs: BenchRun[] }>(r)),

  benchRun: (runId: string) =>
    fetch(`/api/bench/runs/${encodeURIComponent(runId)}`).then((r) => json<BenchRun>(r)),

  getPrompts: (store: string) =>
    fetch(`/api/stores/${encodeURIComponent(store)}/prompts`).then((r) => json<PromptsView>(r)),

  putPrompts: (store: string, prompts: PromptConfigDict) =>
    fetch(`/api/stores/${encodeURIComponent(store)}/prompts`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(prompts),
    }).then((r) => json<PromptsView>(r)),

  applyPromptPack: (store: string, packId: string) =>
    fetch(`/api/stores/${encodeURIComponent(store)}/prompts/pack`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pack_id: packId }),
    }).then((r) => json<PromptsView>(r)),

  resetPrompts: (store: string) =>
    fetch(`/api/stores/${encodeURIComponent(store)}/prompts/reset`, { method: "POST" }).then((r) =>
      json<PromptsView>(r),
    ),

  demos: () => fetch("/api/demos").then((r) => json<{ demos: DemoMeta[] }>(r)),

  loadDemo: (name: string, onProgress?: (p: ProgressState) => void) =>
    fetch(`/api/demos/${encodeURIComponent(name)}?stream=1`, { method: "POST" }).then((r) =>
      readProgressStream<DemoLoadResult>(r, onProgress),
    ),

  keysStatus: () => fetch("/api/settings/keys").then((r) => json<KeysStatus>(r)),

  putKeys: (body: {
    openai_api_key?: string;
    anthropic_api_key?: string;
    gemini_api_key?: string;
    ollama_host?: string;
    persist?: boolean;
  }) =>
    fetch("/api/settings/keys", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => json<KeysStatus & { applied?: string[]; persisted_to?: string | null }>(r)),
};
