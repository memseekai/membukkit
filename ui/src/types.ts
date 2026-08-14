export interface StoreMeta {
  name: string;
  n_facts?: number;
  encoder?: string;
  updated_at?: string;
  usage_totals?: UsageTotals;
}

export interface UsageTotals {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  source?: string;
  calls?: number;
  est_cost_usd?: number;
  model?: string;
}

export interface TokenUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  source?: string;
  calls?: number;
}

export interface IngestReceipt {
  files?: number;
  new_facts?: number;
  superseded?: number;
  usage?: TokenUsage | null;
  est_cost_usd?: number | null;
  est_cost_label?: string | null;
  note?: string;
}

export interface UploadResult {
  file: string;
  doc_id?: string;
  doc_type?: string;
  sessions?: number;
  new_facts?: number;
  superseded?: number;
  usage?: TokenUsage | null;
  est_cost_usd?: number | null;
  error?: string;
  warning?: string;
}

export interface Fact {
  id: string;
  text: string;
  timestamp: string | null;
  kind: string;
  entities: string[];
  source_session: string;
  doc_id: string;
  doc_name: string;
  source_ref: string;
  status?: string;
  superseded_by?: string;
}

export interface ProveBeat {
  label: string;
  question: string;
  as_of: string;
}

export interface WriteReceipt {
  status: string;
  n_stored: number;
  n_verbatim?: number;
  n_atomic?: number;
  superseded?: { old_id: string; new_id: string }[];
  warnings?: string[];
  n_facts?: number;
  usage?: TokenUsage | null;
  est_cost_usd?: number | null;
  model?: string;
  suggested_questions?: string[];
}

export interface FactsPage {
  total: number;
  offset: number;
  facts: Fact[];
}

export interface Bucket {
  bucket: number;
  size: number;
  label: string;
  exemplars: string[];
}

export interface PartitionView {
  k_eff: number;
  lane?: string;
  n_facts: number;
  buckets: Bucket[];
}

export interface Evidence {
  ref: string;
  fact: string;
  text: string;
  timestamp: string | null;
  fact_id: string;
  doc_id: string;
  doc_name: string;
  source_ref: string;
  kind?: string;
  status?: string;
  superseded_by?: string;
}

export interface LaneTrace {
  buckets?: (number | { bucket: number; route_prob?: number; size?: number })[];
  scan_frac?: number;
  n_facts?: number;
  n_scanned?: number;
}

export interface AskResponse {
  answer: string | null;
  question_date?: string;
  trace: {
    scan_fraction: number;
    n_facts: number;
    n_scanned: number;
    /** Approx tokens in reader context (memory lines + question, chars/4). */
    est_reader_tokens?: number;
    reader_type: string;
    lanes: Record<string, LaneTrace>;
    opened_buckets: unknown[];
    bucket_labels?: Record<string, string>;
    bucket_labels_lane?: string | null;
    usage?: TokenUsage | null;
    est_cost_usd?: number | null;
    window_fraction?: number;
    model?: string;
  };
  evidence: Evidence[];
}

export interface SourceView {
  fact: Fact;
  source: {
    doc_id: string;
    name: string;
    session?: number;
    date?: string | null;
    turns?: { role: string; content: string }[];
    highlight?: number | null;
    highlight_kind?: "stored" | "lexical" | null;
  } | null;
}

export interface BenchEnvVar {
  name: string;
  set: boolean;
}

export interface BenchRecipe {
  id: string;
  title: string;
  dataset: string;
  description: string;
  reader: string;
  distiller: string;
  judge: string;
  encoder: string;
  expected: number;
  metric: string;
  tolerance: number;
  env: BenchEnvVar[];
  cost_estimate: string;
  cli_command: string;
}

export interface BenchRunResult {
  measured: number | null;
  expected: number;
  tolerance: number;
  passed: boolean | null;
  smoke: boolean;
}

export interface BenchProgress {
  phase: string;
  done: number;
  total: number;
  detail?: string;
  updated_at?: string;
}

export interface BenchRun {
  run_id: string;
  recipe_id: string;
  lite: boolean;
  status: "running" | "done" | "failed";
  started_at: string;
  log_tail?: string[];
  result?: BenchRunResult;
  progress?: BenchProgress;
}

export interface Overview {
  name: string;
  n_facts: number;
  n_verbatim: number;
  n_atomic: number;
  /** Newest fact statement date (YYYY-MM-DD), when known. */
  latest_fact_date?: string | null;
  documents: { doc_id: string; name: string; type: string; n_sessions: number; added_at: string }[];
  meta?: Record<string, unknown>;
  usage_totals?: UsageTotals | null;
  est_lifetime_cost_usd?: number | null;
  /** LLM-generated Ask chips from store facts (after ingest/add). */
  suggested_questions?: string[];
}

export interface AskCallout {
  match: string;
  title: string;
  body: string;
}

export interface DemoMeta {
  id: string;
  title: string;
  description: string;
  proves?: string;
  question_date?: string | null;
  prompt_pack?: string | null;
  questions: string[];
  ask_callouts?: AskCallout[];
  prove_beats?: ProveBeat[];
}

export interface DemoLoadResult {
  store: string;
  id: string;
  title: string;
  questions: string[];
  proves?: string;
  question_date?: string | null;
  ask_callouts?: AskCallout[];
  prove_beats?: ProveBeat[];
}

export interface PromptPackMeta {
  id: string;
  title: string;
  description: string;
}

export interface PromptConfigDict {
  extraction?: string;
  extraction_named?: string;
  extraction_document?: string;
  dated_reader?: string;
  recommendation_reader?: string;
  reasoning_reader?: string;
  abstain_gate?: string;
  extraction_instructions?: string;
  reader_instructions?: string;
}

export interface PromptsView {
  prompts: PromptConfigDict;
  pack_id: string | null;
  packs: PromptPackMeta[];
  placeholders: Record<string, string[]>;
  is_default: boolean;
}

export interface KeyProviderStatus {
  set: boolean;
  mask: string;
  source: "env" | "file" | "none" | string;
}

export interface KeysStatus {
  llm: string;
  needs: string;
  ready: boolean;
  credentials_path: string;
  providers: {
    openai: KeyProviderStatus;
    anthropic: KeyProviderStatus;
    google: KeyProviderStatus;
    ollama: KeyProviderStatus;
  };
  applied?: string[];
  persisted_to?: string | null;
}
