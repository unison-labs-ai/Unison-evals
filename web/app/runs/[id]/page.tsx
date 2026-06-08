"use client";

import { use, useEffect, useState } from "react";
import {
  cancelRun,
  getRun,
  isAgentSummary,
  isBrainSummary,
  isScaleSummary,
  type AgentRunSummary,
  type BrainRunSummary,
  type BrainSystemSummary,
  type QuestionResult,
  type RunDetail,
  type RunSummary,
  type ScaleRunSummary,
  type ScaleSystemSummary,
} from "@/lib/api";
import { ParetoChart } from "@/components/pareto-chart";

interface RunEvent {
  type: string;
  run_id: string;
  system?: string;
  question_id?: string;
  questions_total?: number;
  questions_done?: number;
  result?: QuestionResult;
  summary?: RunSummary;
  error?: string;
}

export default function RunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [run, setRun] = useState<RunDetail | null>(null);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [completedResults, setCompletedResults] = useState<QuestionResult[]>([]);
  const [finalSummary, setFinalSummary] = useState<RunSummary | null>(null);

  useEffect(() => {
    let cancelled = false;
    getRun(id)
      .then((r) => {
        if (!cancelled) {
          setRun(r);
          if (r.summary) setFinalSummary(r.summary);
          if (r.results) setCompletedResults(r.results);
        }
      })
      .catch((e) => setStreamError(e.message ?? String(e)));
    return () => {
      cancelled = true;
    };
  }, [id]);

  useEffect(() => {
    const es = new EventSource(`/api/runs/${id}/stream`);
    es.onmessage = (e) => {
      try {
        const data: RunEvent = JSON.parse(e.data);
        if (typeof data.questions_done === "number" && typeof data.questions_total === "number") {
          setProgress({ done: data.questions_done, total: data.questions_total });
        }
        if (data.type === "question_completed" && data.result) {
          setCompletedResults((prev) => [...prev, data.result!]);
        }
        if (data.type === "run_completed" && data.summary) {
          setFinalSummary(data.summary);
        }
        if (data.type === "run_failed") {
          setStreamError(data.error ?? "run failed");
        }
      } catch {
        // ignore parse errors
      }
    };
    es.onerror = () => {
      // Don't close — the browser's native EventSource auto-reconnects on
      // transport blips. Closing here permanently stops the stream and
      // forces the user to refresh to see anything new.
      // The stream naturally terminates when the server sends the _eof
      // sentinel inside the jobs runner.
    };
    return () => es.close();
  }, [id]);

  if (!run) {
    return (
      <div className="text-sm text-[color:var(--muted)]">
        Loading run {id}…
        {streamError && <p className="mt-2 text-red-400">{streamError}</p>}
      </div>
    );
  }

  const status = finalSummary
    ? "completed"
    : streamError
      ? "failed"
      : progress.total > 0
        ? "running"
        : run.status;

  return (
    <div className="space-y-6">
      <header>
        <p className="text-xs text-[color:var(--muted)]">Run</p>
        <h1 className="font-mono text-lg">{run.id}</h1>
        <p className="mt-1 text-sm text-[color:var(--muted)]">
          {run.dataset} · {run.track} · {run.systems.join(", ")}
          {run.track === "agent-oracle" || run.track === "agent-e2e"
            ? ` · judge: ${run.judge_model}`
            : ""}
        </p>
      </header>

      <div className="flex items-end justify-between gap-3">
        <div className="flex-1">
          <ProgressBar
            done={progress.done || completedResults.length}
            total={progress.total || run.n_questions * run.systems.length}
            status={status}
          />
        </div>
        {status === "running" && (
          <CancelButton runId={run.id} onError={(msg) => setStreamError(msg)} />
        )}
      </div>

      {finalSummary ? (
        <ResultsSection summary={finalSummary} results={completedResults} runId={run.id} />
      ) : (
        <LiveLog
          completedResults={completedResults}
          track={run.track}
        />
      )}

      {streamError && (
        <div className="rounded border border-red-700 bg-red-950/40 px-3 py-2 text-sm text-red-200">
          {streamError}
        </div>
      )}
    </div>
  );
}

function ProgressBar({
  done,
  total,
  status,
}: {
  done: number;
  total: number;
  status: string;
}) {
  const pct = total === 0 ? 0 : Math.min(100, (done / total) * 100);
  const bar =
    status === "completed"
      ? "bg-green-500"
      : status === "failed"
        ? "bg-red-500"
        : "bg-blue-500";
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs text-[color:var(--muted)]">
        <span>
          {status} · {done}/{total}
        </span>
        <span>{pct.toFixed(0)}%</span>
      </div>
      <div className="h-1.5 w-full rounded bg-[color:var(--card)]">
        <div className={`h-1.5 rounded ${bar}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function CancelButton({
  runId,
  onError,
}: {
  runId: string;
  onError: (msg: string) => void;
}) {
  const [submitting, setSubmitting] = useState(false);
  const [confirming, setConfirming] = useState(false);

  const onClick = async () => {
    if (!confirming) {
      setConfirming(true);
      return;
    }
    setSubmitting(true);
    try {
      await cancelRun(runId);
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
      setConfirming(false);
    }
  };

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={submitting}
      className="shrink-0 rounded border px-3 py-1.5 text-xs hover:opacity-80 disabled:opacity-40"
      style={{
        borderColor: confirming ? "#dc2626" : "var(--border)",
        color: confirming ? "#fca5a5" : "var(--fg)",
      }}
    >
      {submitting ? "Cancelling…" : confirming ? "Confirm cancel?" : "Cancel"}
    </button>
  );
}

function LiveLog({
  completedResults,
  track,
}: {
  completedResults: QuestionResult[];
  track: string;
}) {
  const isAgentTrack = track === "agent-oracle" || track === "agent-e2e";
  if (completedResults.length === 0) {
    return (
      <div className="rounded border px-4 py-6 text-center text-sm text-[color:var(--muted)]" style={{ borderColor: "var(--border)" }}>
        Waiting for first answers…
      </div>
    );
  }
  return (
    <div className="rounded border" style={{ borderColor: "var(--border)" }}>
      <div className="border-b px-3 py-2 text-xs uppercase tracking-wide text-[color:var(--muted)]" style={{ borderColor: "var(--border)" }}>
        Live · {completedResults.length} answers in
      </div>
      <ul className="divide-y" style={{ borderColor: "var(--border)" }}>
        {completedResults.slice(-20).map((r, i) => (
          <li key={i} className="flex items-center gap-3 px-3 py-2 text-sm font-mono">
            {isAgentTrack ? (
              <span>{r.judge?.passed ? "✓" : "✗"}</span>
            ) : (
              <span className="text-[color:var(--muted)]">·</span>
            )}
            <span className="w-32 truncate">{r.system}</span>
            <span className="w-32 truncate text-[color:var(--muted)]">{r.question_id}</span>
            {isAgentTrack && (
              <span className="w-20 text-right">${r.adapter.cost_usd.toFixed(4)}</span>
            )}
            <span className="w-20 text-right text-[color:var(--muted)]">
              {(r.adapter.latency_ms / 1000).toFixed(1)}s
            </span>
            {r.adapter.error && (
              <span className="truncate text-xs text-red-400">{r.adapter.error}</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function ResultsSection({
  summary,
  results,
  runId,
}: {
  summary: RunSummary;
  results: QuestionResult[];
  runId: string;
}) {
  if (isAgentSummary(summary)) {
    return <AgentResultsTable summary={summary} results={results} runId={runId} />;
  }
  if (isScaleSummary(summary)) {
    return <ScaleResultsTable summary={summary} runId={runId} />;
  }
  if (isBrainSummary(summary)) {
    return <BrainResultsTable summary={summary} runId={runId} />;
  }
  return (
    <div className="text-sm text-[color:var(--muted)]">
      Unknown result shape for track &quot;{(summary as RunSummary).track}&quot;.
    </div>
  );
}

function AgentResultsTable({
  summary,
  results,
  runId,
}: {
  summary: AgentRunSummary;
  results: QuestionResult[];
  runId: string;
}) {
  const isE2E = summary.track === "agent-e2e";
  const paretoPoints = summary.summaries.map((s) => ({
    system: s.system,
    x: s.p50_latency_ms,
    y: s.pass_rate * 100,
  }));

  return (
    <div className="space-y-4">
      {isE2E && summary.efficiency_narrative && (
        <div
          className="rounded border px-3 py-2 text-sm"
          style={{ borderColor: "var(--border)", background: "var(--card)" }}
        >
          <span className="font-medium text-green-400">Brain efficiency: </span>
          {summary.efficiency_narrative}
        </div>
      )}

      <div className="rounded border" style={{ borderColor: "var(--border)" }}>
        <table className="w-full text-sm">
          <thead className="bg-[color:var(--card)] text-[color:var(--muted)]">
            <tr>
              <th className="px-3 py-2 text-left font-normal">System</th>
              <th className="px-3 py-2 text-right font-normal">Pass</th>
              <th className="px-3 py-2 text-right font-normal">%</th>
              <th className="px-3 py-2 text-right font-normal">$/Q</th>
              <th className="px-3 py-2 text-right font-normal">$/solved</th>
              <th className="px-3 py-2 text-right font-normal">p50 ms</th>
              <th className="px-3 py-2 text-right font-normal">p95 ms</th>
              {isE2E && <th className="px-3 py-2 text-right font-normal">tokens/Q</th>}
              {isE2E && <th className="px-3 py-2 text-right font-normal">efficiency</th>}
            </tr>
          </thead>
          <tbody>
            {summary.summaries.map((s) => (
              <tr key={s.system} className="border-t" style={{ borderColor: "var(--border)" }}>
                <td className="px-3 py-2 font-mono">{s.system}</td>
                <td className="px-3 py-2 text-right">
                  {s.n_passed}/{s.n_questions}
                </td>
                <td className="px-3 py-2 text-right">
                  {(s.pass_rate * 100).toFixed(1)}%
                  {s.pass_rate_ci_low != null && s.pass_rate_ci_high != null && (
                    <span className="ml-1 text-[10px] text-[color:var(--muted)]">
                      [{(s.pass_rate_ci_low * 100).toFixed(0)}–
                      {(s.pass_rate_ci_high * 100).toFixed(0)}]
                    </span>
                  )}
                </td>
                <td className="px-3 py-2 text-right">${s.cost_per_question_usd.toFixed(4)}</td>
                <td className="px-3 py-2 text-right">
                  {s.cost_per_solved_usd != null ? `$${s.cost_per_solved_usd.toFixed(4)}` : "—"}
                </td>
                <td className="px-3 py-2 text-right">{s.p50_latency_ms.toFixed(0)}</td>
                <td className="px-3 py-2 text-right">{s.p95_latency_ms.toFixed(0)}</td>
                {isE2E && (
                  <td className="px-3 py-2 text-right text-[color:var(--muted)]">
                    {s.tokens_unavailable ? "n/a" : s.mean_input_tokens_per_q.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </td>
                )}
                {isE2E && (
                  <td className="px-3 py-2 text-right font-medium">
                    {s.tokens_unavailable
                      ? "n/a"
                      : s.efficiency_ratio != null
                        ? `${s.efficiency_ratio.toFixed(1)}×`
                        : "baseline"}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {paretoPoints.length > 1 && (
        <div className="rounded border px-4 py-4" style={{ borderColor: "var(--border)", background: "var(--card)" }}>
          <p className="text-xs uppercase tracking-wide text-[color:var(--muted)] mb-3">
            Pass rate vs. p50 latency — Pareto frontier (dashed)
          </p>
          <ParetoChart
            points={paretoPoints}
            xLabel="Latency p50 (ms)"
            yLabel="Pass rate (%)"
            invertX={true}
            width={560}
            height={280}
          />
        </div>
      )}

      <DownloadLink runId={runId} />
      <PerQuestionDetails results={results} track={isE2E ? "agent-e2e" : "agent"} />
    </div>
  );
}

function BrainResultsTable({
  summary,
  runId,
}: {
  summary: BrainRunSummary;
  runId: string;
}) {
  const paretoPoints = summary.summaries.map((s) => ({
    system: s.system,
    x: s.p50_latency_ms,
    y: s.mean_recall_at_10 * 100,
  }));

  return (
    <div className="space-y-4">
      <div className="rounded border" style={{ borderColor: "var(--border)" }}>
        <table className="w-full text-sm">
          <thead className="bg-[color:var(--card)] text-[color:var(--muted)]">
            <tr>
              <th className="px-3 py-2 text-left font-normal">System</th>
              <th className="px-3 py-2 text-right font-normal">Recall@10</th>
              <th className="px-3 py-2 text-right font-normal">nDCG@10</th>
              <th className="px-3 py-2 text-right font-normal">MRR</th>
              <th className="px-3 py-2 text-right font-normal">Hit@1</th>
              <th className="px-3 py-2 text-right font-normal">p50 ms</th>
              <th className="px-3 py-2 text-right font-normal">p95 ms</th>
            </tr>
          </thead>
          <tbody>
            {summary.summaries.map((s) => (
              <BrainSystemRow key={s.system} s={s} />
            ))}
          </tbody>
        </table>
      </div>

      {paretoPoints.length > 1 && (
        <div className="rounded border px-4 py-4" style={{ borderColor: "var(--border)", background: "var(--card)" }}>
          <p className="text-xs uppercase tracking-wide text-[color:var(--muted)] mb-3">
            Recall@10 vs. p50 latency — Pareto frontier (dashed)
          </p>
          <ParetoChart
            points={paretoPoints}
            xLabel="Latency p50 (ms)"
            yLabel="Recall@10 (%)"
            invertX={true}
            width={560}
            height={280}
          />
        </div>
      )}

      <DownloadLink runId={runId} />
    </div>
  );
}

function BrainSystemRow({ s }: { s: BrainSystemSummary }) {
  return (
    <tr className="border-t" style={{ borderColor: "var(--border)" }}>
      <td className="px-3 py-2 font-mono">{s.system}</td>
      <td className="px-3 py-2 text-right">
        {(s.mean_recall_at_10 * 100).toFixed(1)}%
        {s.recall_at_10_ci_low != null && s.recall_at_10_ci_high != null && (
          <span className="ml-1 text-[10px] text-[color:var(--muted)]">
            [{(s.recall_at_10_ci_low * 100).toFixed(0)}–
            {(s.recall_at_10_ci_high * 100).toFixed(0)}]
          </span>
        )}
      </td>
      <td className="px-3 py-2 text-right">{(s.mean_ndcg_at_10 * 100).toFixed(1)}%</td>
      <td className="px-3 py-2 text-right">{s.mean_mrr.toFixed(3)}</td>
      <td className="px-3 py-2 text-right">
        {(s.mean_hit_at_1 * 100).toFixed(1)}%
        {s.hit_at_1_ci_low != null && s.hit_at_1_ci_high != null && (
          <span className="ml-1 text-[10px] text-[color:var(--muted)]">
            [{(s.hit_at_1_ci_low * 100).toFixed(0)}–
            {(s.hit_at_1_ci_high * 100).toFixed(0)}]
          </span>
        )}
      </td>
      <td className="px-3 py-2 text-right">{s.p50_latency_ms.toFixed(0)}</td>
      <td className="px-3 py-2 text-right">{s.p95_latency_ms.toFixed(0)}</td>
    </tr>
  );
}

function ScaleResultsTable({
  summary,
  runId,
}: {
  summary: ScaleRunSummary;
  runId: string;
}) {
  const paretoPoints = summary.summaries.map((s) => ({
    system: s.system,
    x: s.p50_latency_ms,
    y: s.mean_recall_at_10 * 100,
  }));

  return (
    <div className="space-y-4">
      <div className="rounded border px-3 py-2 text-xs text-[color:var(--muted)]" style={{ borderColor: "var(--border)" }}>
        Corpus: <span className="font-mono">{summary.corpus_label}</span>
      </div>

      <div className="rounded border" style={{ borderColor: "var(--border)" }}>
        <table className="w-full text-sm">
          <thead className="bg-[color:var(--card)] text-[color:var(--muted)]">
            <tr>
              <th className="px-3 py-2 text-left font-normal">System</th>
              <th className="px-3 py-2 text-right font-normal">Recall@10</th>
              <th className="px-3 py-2 text-right font-normal">nDCG@10</th>
              <th className="px-3 py-2 text-right font-normal">MRR</th>
              <th className="px-3 py-2 text-right font-normal">Hit@1</th>
              <th className="px-3 py-2 text-right font-normal">p50 ms</th>
              <th className="px-3 py-2 text-right font-normal">p95 ms</th>
              <th className="px-3 py-2 text-right font-normal">p99 ms</th>
            </tr>
          </thead>
          <tbody>
            {summary.summaries.map((s) => (
              <ScaleSystemRow key={s.system} s={s} />
            ))}
          </tbody>
        </table>
      </div>

      {paretoPoints.length > 1 && (
        <div className="rounded border px-4 py-4" style={{ borderColor: "var(--border)", background: "var(--card)" }}>
          <p className="text-xs uppercase tracking-wide text-[color:var(--muted)] mb-3">
            Recall@10 vs. p50 latency — Pareto frontier (dashed)
          </p>
          <ParetoChart
            points={paretoPoints}
            xLabel="Latency p50 (ms)"
            yLabel="Recall@10 (%)"
            invertX={true}
            width={560}
            height={280}
          />
        </div>
      )}

      <DownloadLink runId={runId} />
    </div>
  );
}

function ScaleSystemRow({ s }: { s: ScaleSystemSummary }) {
  return (
    <tr className="border-t" style={{ borderColor: "var(--border)" }}>
      <td className="px-3 py-2 font-mono">{s.system}</td>
      <td className="px-3 py-2 text-right">{(s.mean_recall_at_10 * 100).toFixed(1)}%</td>
      <td className="px-3 py-2 text-right">{(s.mean_ndcg_at_10 * 100).toFixed(1)}%</td>
      <td className="px-3 py-2 text-right">{s.mean_mrr.toFixed(3)}</td>
      <td className="px-3 py-2 text-right">{(s.mean_hit_at_1 * 100).toFixed(1)}%</td>
      <td className="px-3 py-2 text-right">{s.p50_latency_ms.toFixed(0)}</td>
      <td className="px-3 py-2 text-right">{s.p95_latency_ms.toFixed(0)}</td>
      <td className="px-3 py-2 text-right">{s.p99_latency_ms.toFixed(0)}</td>
    </tr>
  );
}

function DownloadLink({ runId }: { runId: string }) {
  return (
    <a
      href={`/api/runs/${runId}`}
      download={`${runId}.json`}
      className="inline-block rounded border px-3 py-1.5 text-xs hover:opacity-80"
      style={{ borderColor: "var(--border)" }}
    >
      ↓ Download JSON
    </a>
  );
}

function corpusAnnotation(raw: Record<string, unknown>, track: string): string | null {
  if (track !== "agent-e2e") return null;
  const parts: string[] = [];
  if (typeof raw["seed_docs_count"] === "number") {
    parts.push(`docs=${raw["seed_docs_count"]}`);
  }
  if (typeof raw["seed_embed_ms"] === "number") {
    parts.push(`embed=${(raw["seed_embed_ms"] as number).toFixed(0)}ms`);
  }
  if (typeof raw["inlined_chars"] === "number") {
    parts.push(`inlined=${((raw["inlined_chars"] as number) / 1000).toFixed(1)}k chars`);
  }
  if (typeof raw["inlined_tokens"] === "number" && (raw["inlined_tokens"] as number) > 0) {
    parts.push(`tokens=${raw["inlined_tokens"]}`);
  }
  return parts.length > 0 ? parts.join(" · ") : null;
}

function PerQuestionDetails({
  results,
  track,
}: {
  results: QuestionResult[];
  track: string;
}) {
  const isAgent = track === "agent" || track === "agent-e2e";
  return (
    <details>
      <summary className="cursor-pointer text-sm text-[color:var(--muted)]">
        Per-question results ({results.length})
      </summary>
      <ul className="mt-2 divide-y rounded border" style={{ borderColor: "var(--border)" }}>
        {results.map((r, i) => {
          const annotation = corpusAnnotation(r.adapter.raw, track);
          return (
            <li key={i} className="px-3 py-2 text-xs font-mono">
              <div className="flex items-center gap-3">
                {isAgent ? (
                  <span>{r.judge?.passed ? "✓" : "✗"}</span>
                ) : (
                  <span className="text-[color:var(--muted)]">·</span>
                )}
                <span className="w-32 truncate">{r.system}</span>
                <span className="w-32 truncate">{r.question_id}</span>
                {isAgent && (
                  <span className="w-20 text-right">${r.adapter.cost_usd.toFixed(4)}</span>
                )}
                <span className="w-20 text-right text-[color:var(--muted)]">
                  {(r.adapter.latency_ms / 1000).toFixed(1)}s
                </span>
                {annotation && (
                  <span className="truncate text-[color:var(--muted)]">{annotation}</span>
                )}
              </div>
              {isAgent && r.judge?.reasoning && (
                <div className="mt-1 ml-6 text-[color:var(--muted)] truncate">
                  judge: {r.judge.reasoning}
                </div>
              )}
              {r.adapter.error && (
                <div className="mt-1 ml-6 text-red-400 truncate">err: {r.adapter.error}</div>
              )}
            </li>
          );
        })}
      </ul>
    </details>
  );
}
