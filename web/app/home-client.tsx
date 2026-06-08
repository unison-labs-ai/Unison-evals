"use client";

import Link from "next/link";
import { useSearchParams, useRouter } from "next/navigation";
import {
  isAgentSummary,
  isBrainSummary,
  isScaleSummary,
  type RunDetail,
} from "@/lib/api";

export function HomeClient({
  runs,
}: {
  runs: RunDetail[];
}) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const tab = searchParams.get("tab") ?? "runs";

  function setTab(t: string) {
    const url = new URL(window.location.href);
    url.searchParams.set("tab", t);
    router.push(url.pathname + url.search);
  }

  return (
    <div className="space-y-4">
      {/* Tab bar */}
      <div
        className="flex border-b"
        style={{ borderColor: "var(--border)" }}
      >
        <TabButton active={tab === "runs"} onClick={() => setTab("runs")}>
          Recent runs
        </TabButton>
        <TabButton
          active={tab === "leaderboard"}
          onClick={() => setTab("leaderboard")}
        >
          Leaderboard
        </TabButton>
      </div>

      {tab === "runs" && <RecentRuns runs={runs} />}
      {tab === "leaderboard" && <Leaderboard runs={runs} />}
    </div>
  );
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className="px-4 py-2 text-sm border-b-2 -mb-px transition-colors"
      style={{
        borderBottomColor: active ? "var(--accent)" : "transparent",
        color: active ? "var(--fg)" : "var(--muted)",
      }}
    >
      {children}
    </button>
  );
}

function RecentRuns({ runs }: { runs: RunDetail[] }) {
  if (runs.length === 0) {
    return (
      <div
        className="rounded border px-4 py-8 text-center"
        style={{ borderColor: "var(--border)" }}
      >
        <p className="text-[color:var(--muted)]">No runs yet.</p>
        <Link href="/runs/new" className="mt-2 inline-block text-sm">
          Start your first run →
        </Link>
      </div>
    );
  }

  return (
    <div
      className="overflow-hidden rounded border"
      style={{ borderColor: "var(--border)" }}
    >
      <table className="w-full text-sm">
        <thead className="bg-[color:var(--card)] text-[color:var(--muted)]">
          <tr>
            <th className="px-3 py-2 text-left font-normal">Run</th>
            <th className="px-3 py-2 text-left font-normal">Dataset</th>
            <th className="px-3 py-2 text-left font-normal">Systems</th>
            <th className="px-3 py-2 text-left font-normal">Status</th>
            <th className="px-3 py-2 text-left font-normal">Started</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.id} className="border-t" style={{ borderColor: "var(--border)" }}>
              <td className="px-3 py-2">
                <Link href={`/runs/${r.id}`} className="font-mono text-xs">
                  {r.id}
                </Link>
              </td>
              <td className="px-3 py-2">{r.dataset}</td>
              <td className="px-3 py-2 font-mono text-xs">{r.systems.join(", ")}</td>
              <td className="px-3 py-2">
                <StatusBadge status={r.status} />
              </td>
              <td className="px-3 py-2 text-[color:var(--muted)] text-xs">
                {new Date(r.started_at).toLocaleString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface LeaderboardRow {
  dataset: string;
  track: string;
  system: string;
  bestMetric: number;
  metricLabel: string;
  costPerSolved: number | null;
  p50LatencyMs: number;
  runId: string;
  runStartedAt: string;
}

function Leaderboard({ runs: allRuns }: { runs: RunDetail[] }) {
  // Aggregate: for each (dataset, track, system) pick best metric value (ties → most recent)
  const completedRuns = allRuns.filter((r) => r.summary != null);

  const best = new Map<string, LeaderboardRow>();

  for (const run of completedRuns) {
    const summary = run.summary!;
    for (const s of summary.summaries) {
      const key = `${run.dataset}::${run.track}::${s.system}`;
      const existing = best.get(key);

      let metric: number;
      let metricLabel: string;
      let costPerSolved: number | null = null;

      if (isAgentSummary(summary)) {
        const agentS = s as typeof summary.summaries[number];
        metric = (agentS as { pass_rate: number }).pass_rate;
        metricLabel = "pass rate";
        costPerSolved = (agentS as { cost_per_solved_usd: number | null }).cost_per_solved_usd;
      } else if (isBrainSummary(summary) || isScaleSummary(summary)) {
        metric = (s as { mean_recall_at_10: number }).mean_recall_at_10;
        metricLabel = "Recall@10";
      } else {
        metric = 0;
        metricLabel = "—";
      }

      const isBetter =
        !existing ||
        metric > existing.bestMetric ||
        (metric === existing.bestMetric && run.started_at > existing.runStartedAt);

      if (isBetter) {
        best.set(key, {
          dataset: run.dataset,
          track: run.track,
          system: s.system,
          bestMetric: metric,
          metricLabel,
          costPerSolved,
          p50LatencyMs: s.p50_latency_ms,
          runId: run.id,
          runStartedAt: run.started_at,
        });
      }
    }
  }

  const rows = [...best.values()].sort((a, b) => {
    if (a.dataset !== b.dataset) return a.dataset.localeCompare(b.dataset);
    if (a.track !== b.track) return a.track.localeCompare(b.track);
    return b.bestMetric - a.bestMetric;
  });

  if (rows.length === 0) {
    return (
      <div
        className="rounded border px-4 py-8 text-center"
        style={{ borderColor: "var(--border)" }}
      >
        <p className="text-[color:var(--muted)]">
          No completed runs yet. The leaderboard populates once runs finish.
        </p>
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded border" style={{ borderColor: "var(--border)" }}>
      <table className="w-full text-sm">
        <thead className="bg-[color:var(--card)] text-[color:var(--muted)]">
          <tr>
            <th className="px-3 py-2 text-left font-normal">Dataset</th>
            <th className="px-3 py-2 text-left font-normal">Track</th>
            <th className="px-3 py-2 text-left font-normal">System</th>
            <th className="px-3 py-2 text-right font-normal">Best metric</th>
            <th className="px-3 py-2 text-right font-normal">$/solved</th>
            <th className="px-3 py-2 text-right font-normal">p50 ms</th>
            <th className="px-3 py-2 text-left font-normal">Run</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={`${row.dataset}::${row.track}::${row.system}`}
              className="border-t"
              style={{ borderColor: "var(--border)" }}
            >
              <td className="px-3 py-2">{row.dataset}</td>
              <td className="px-3 py-2 text-xs text-[color:var(--muted)]">{row.track}</td>
              <td className="px-3 py-2 font-mono text-xs">{row.system}</td>
              <td className="px-3 py-2 text-right">
                {(row.bestMetric * 100).toFixed(1)}%{" "}
                <span className="text-xs text-[color:var(--muted)]">{row.metricLabel}</span>
              </td>
              <td className="px-3 py-2 text-right">
                {row.costPerSolved != null
                  ? `$${row.costPerSolved.toFixed(4)}`
                  : "—"}
              </td>
              <td className="px-3 py-2 text-right">
                {row.p50LatencyMs.toFixed(0)}
              </td>
              <td className="px-3 py-2">
                <Link
                  href={`/runs/${row.runId}`}
                  className="font-mono text-xs"
                >
                  {row.runId}
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    completed: "bg-green-900/40 text-green-300 border-green-800",
    running: "bg-blue-900/40 text-blue-300 border-blue-800",
    queued: "bg-zinc-800 text-zinc-300 border-zinc-700",
    failed: "bg-red-900/40 text-red-300 border-red-800",
    cancelled: "bg-zinc-800 text-zinc-400 border-zinc-700",
  };
  const style = styles[status] ?? "bg-zinc-800 text-zinc-300 border-zinc-700";
  return (
    <span className={`rounded border px-2 py-0.5 text-xs uppercase tracking-wide ${style}`}>
      {status}
    </span>
  );
}
