/**
 * Thin client for the FastAPI server. All paths are relative — Next.js
 * rewrites /api/* to the actual backend in next.config.ts.
 */

export interface Adapter {
  name: string;
  class: string;
}

export interface Dataset {
  name: string;
  class: string;
  description: string;
  // Per-dataset metadata surfaced by /api/registry. Older servers omit these.
  total_questions?: number | null;
  supported_tracks?: string[];
}

export interface TrackInfo {
  name: string;
  description: string;
}

export interface BrainModeInfo {
  name: string;
  description: string;
}

export interface Registry {
  adapters: Adapter[];
  brain_adapters: Adapter[];
  datasets: Dataset[];
  tracks: TrackInfo[];
  brain_modes: BrainModeInfo[];
}

// ---------------------------------------------------------------------------
// Track 2 (agent-oracle) and Track 3 (agent-e2e) summary shapes
// ---------------------------------------------------------------------------

export interface SystemSummary {
  system: string;
  n_questions: number;
  n_passed: number;
  pass_rate: number;
  // Bootstrap 95% CI on pass_rate; null for runs from before stats integration.
  pass_rate_ci_low?: number | null;
  pass_rate_ci_high?: number | null;
  total_cost_usd: number;
  cost_per_question_usd: number;
  cost_per_solved_usd: number | null;
  avg_latency_ms: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  // Brain-efficiency fields (Track 3 only; default 0/null/false for Track 2)
  mean_input_tokens_per_q: number;
  efficiency_ratio: number | null;
  tokens_unavailable: boolean;
}

export interface AgentRunSummary {
  run_id: string;
  dataset: string;
  track: string;
  systems: string[];
  judge_model: string;
  n_questions: number;
  started_at: string;
  finished_at: string | null;
  total_cost_usd: number;
  summaries: SystemSummary[];
  // Brain-efficiency narrative (Track 3 only; null for Track 2)
  efficiency_narrative: string | null;
}

// ---------------------------------------------------------------------------
// Track 1 (brain-only) summary shapes
// ---------------------------------------------------------------------------

export interface BrainSystemSummary {
  system: string;
  n_questions: number;
  mean_recall_at_10: number;
  mean_ndcg_at_10: number;
  mean_mrr: number;
  mean_hit_at_1: number;
  // Bootstrap 95% CIs on the headline retrieval metrics.
  recall_at_10_ci_low?: number | null;
  recall_at_10_ci_high?: number | null;
  hit_at_1_ci_low?: number | null;
  hit_at_1_ci_high?: number | null;
  total_cost_usd: number;
  avg_latency_ms: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
}

export interface BrainRunSummary {
  run_id: string;
  dataset: string;
  track: string;
  systems: string[];
  n_questions: number;
  started_at: string;
  finished_at: string | null;
  total_cost_usd: number;
  summaries: BrainSystemSummary[];
}

// ---------------------------------------------------------------------------
// Scale summary shapes (legacy track; read-only display of existing results)
// ---------------------------------------------------------------------------

export interface ScaleSystemSummary extends BrainSystemSummary {
  p99_latency_ms: number;
}

export interface ScaleRunSummary {
  run_id: string;
  dataset: string;
  track: string;
  systems: string[];
  n_questions: number;
  corpus_label: string;
  started_at: string;
  finished_at: string | null;
  total_cost_usd: number;
  summaries: ScaleSystemSummary[];
}

// Union type for any run summary shape
export type RunSummary = AgentRunSummary | BrainRunSummary | ScaleRunSummary;

// ---------------------------------------------------------------------------
// Run detail
// ---------------------------------------------------------------------------

export interface RunDetail {
  id: string;
  dataset: string;
  track: string;
  systems: string[];
  status: string;
  n_questions: number;
  judge_model: string;
  started_at: string;
  finished_at: string | null;
  summary: RunSummary | null;
  results: QuestionResult[] | null;
  error: string | null;
}

export interface AdapterResult {
  answer: string;
  cost_usd: number;
  latency_ms: number;
  raw: Record<string, unknown>;
  error: string | null;
}

export interface JudgeResult {
  score: number;
  passed: boolean;
  confidence: number;
  reasoning: string;
  cost_usd: number;
}

export interface QuestionResult {
  question_id: string;
  system: string;
  adapter: AdapterResult;
  judge: JudgeResult | null;
}

// ---------------------------------------------------------------------------
// Type guards for summary discrimination
// ---------------------------------------------------------------------------

export function isAgentSummary(s: RunSummary): s is AgentRunSummary {
  return s.track === "agent-oracle" || s.track === "agent-e2e";
}

export function isBrainSummary(s: RunSummary): s is BrainRunSummary {
  return s.track === "brain-only";
}

export function isScaleSummary(s: RunSummary): s is ScaleRunSummary {
  return s.track === "scale";
}

// Server-side fetch needs an absolute URL; browser fetch uses the relative
// path so the Next.js rewrite (next.config.ts) proxies to the FastAPI server.
// `typeof window === "undefined"` is the canonical SSR detector.
const API_BASE: string =
  typeof window === "undefined" ? (process.env.UNISON_EVALS_API ?? "http://localhost:8001") : "";

const apiUrl = (path: string): string => `${API_BASE}${path}`;

export async function getRegistry(): Promise<Registry> {
  const res = await fetch(apiUrl("/api/registry"));
  if (!res.ok) throw new Error(`Registry fetch failed: ${res.status}`);
  return res.json();
}

export interface StartRunBody {
  dataset: string;
  track: string;
  systems: string[];
  limit: number;
  judge_model?: string | null;
  pass_threshold?: number;
  corpus?: string | null;
  mode?: string | null;
}

export async function startRun(body: StartRunBody): Promise<{ run_id: string }> {
  const res = await fetch(apiUrl("/api/runs"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Start run failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function listRuns(): Promise<RunDetail[]> {
  const res = await fetch(apiUrl("/api/runs"));
  if (!res.ok) throw new Error(`List runs failed: ${res.status}`);
  const data = (await res.json()) as { runs: RunDetail[] };
  return data.runs;
}

export async function getRun(runId: string): Promise<RunDetail> {
  const res = await fetch(apiUrl(`/api/runs/${runId}`));
  if (!res.ok) throw new Error(`Get run failed: ${res.status}`);
  return res.json();
}

export async function cancelRun(runId: string): Promise<void> {
  const res = await fetch(apiUrl(`/api/runs/${runId}`), { method: "DELETE" });
  if (!res.ok && res.status !== 404) {
    const detail = await res.text();
    throw new Error(`Cancel failed (${res.status}): ${detail}`);
  }
}

