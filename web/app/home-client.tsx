"use client";

import Link from "next/link";
import { useSearchParams, useRouter } from "next/navigation";
import {
  isAgentSummary,
  isBrainSummary,
  isScaleSummary,
  type RunDetail,
  type ComboEntry,
  type ComprehensiveGroup,
} from "@/lib/api";

export function HomeClient({
  runs,
  comprehensiveGroups,
}: {
  runs: RunDetail[];
  comprehensiveGroups: ComprehensiveGroup[];
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
        <TabButton
          active={tab === "cross-dataset"}
          onClick={() => setTab("cross-dataset")}
        >
          Cross-dataset
        </TabButton>
      </div>

      {tab === "runs" && <RecentRuns runs={runs} />}
      {tab === "leaderboard" && <Leaderboard runs={runs} />}
      {tab === "cross-dataset" && (
        <CrossDatasetLeaderboard groups={comprehensiveGroups} />
      )}
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

// ---------------------------------------------------------------------------
// Cross-dataset aggregated leaderboard (populated from comprehensive runs)
// ---------------------------------------------------------------------------

type AggRow = {
  system: string;
  // Track 2 + 3 mean pass rate across all datasets where the system ran
  meanPassRate: number | null;
  // Track 1 mean Recall@10 across all brain datasets
  meanRecallAt10: number | null;
  // mean cost per solved task (Track 2+3)
  meanCostPerSolved: number | null;
  // mean p50 latency across all runs
  meanP50LatencyMs: number | null;
  // per-dataset breakdown: dataset -> headline metric
  byDataset: Record<string, { metric: number | null; metricLabel: string }>;
  tracksPresent: Set<string>;
};

function aggregateComprehensiveRuns(groups: ComprehensiveGroup[]): {
  rows: AggRow[];
  datasets: string[];
} {
  // Flatten all combos from all groups; for each (system, dataset, track) keep the
  // best combo (highest pass_rate or recall_at_10).
  const best = new Map<string, ComboEntry>();

  for (const group of groups) {
    for (const combo of group.combos) {
      if (combo.status !== "done") continue;
      const key = `${combo.system}||${combo.dataset}||${combo.track}`;
      const existing = best.get(key);
      const metric =
        combo.pass_rate ?? combo.recall_at_10 ?? -1;
      const existingMetric =
        existing
          ? (existing.pass_rate ?? existing.recall_at_10 ?? -1)
          : -Infinity;
      if (!existing || metric > existingMetric) {
        best.set(key, combo);
      }
    }
  }

  const combos = [...best.values()];

  // Gather all datasets present in any combo.
  const datasetSet = new Set<string>();
  for (const c of combos) datasetSet.add(c.dataset);
  const datasets = [...datasetSet].sort();

  // Group by system.
  const bySystem = new Map<string, ComboEntry[]>();
  for (const c of combos) {
    if (!bySystem.has(c.system)) bySystem.set(c.system, []);
    bySystem.get(c.system)!.push(c);
  }

  const rows: AggRow[] = [];

  for (const [system, systemCombos] of bySystem) {
    const agentCombos = systemCombos.filter(
      (c) => c.track === "agent" || c.track === "agent-oracle" || c.track === "together" || c.track === "agent-e2e"
    );
    const brainCombos = systemCombos.filter(
      (c) => c.track === "brain" || c.track === "brain-only"
    );

    const passRates = agentCombos
      .map((c) => c.pass_rate)
      .filter((v): v is number => v != null);
    const recalls = brainCombos
      .map((c) => c.recall_at_10)
      .filter((v): v is number => v != null);
    const costsPerSolved = agentCombos
      .map((c) => c.cost_per_solved_usd)
      .filter((v): v is number => v != null);
    const latencies = systemCombos
      .map((c) => c.p50_latency_ms)
      .filter((v): v is number => v != null);

    const avg = (arr: number[]) =>
      arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;

    // Per-dataset breakdown — pick the "best" combo per dataset for this system.
    const byDataset: AggRow["byDataset"] = {};
    for (const dataset of datasets) {
      const dsCombo = systemCombos.find((c) => c.dataset === dataset);
      if (!dsCombo) {
        byDataset[dataset] = { metric: null, metricLabel: "n/a" };
      } else if (dsCombo.pass_rate != null) {
        byDataset[dataset] = {
          metric: dsCombo.pass_rate,
          metricLabel: "pass rate",
        };
      } else if (dsCombo.recall_at_10 != null) {
        byDataset[dataset] = {
          metric: dsCombo.recall_at_10,
          metricLabel: "Recall@10",
        };
      } else {
        byDataset[dataset] = { metric: null, metricLabel: "n/a" };
      }
    }

    rows.push({
      system,
      meanPassRate: avg(passRates),
      meanRecallAt10: avg(recalls),
      meanCostPerSolved: avg(costsPerSolved),
      meanP50LatencyMs: avg(latencies),
      byDataset,
      tracksPresent: new Set(systemCombos.map((c) => c.track)),
    });
  }

  // Sort: highest mean pass rate first; brain-only systems (no pass rate) go below.
  rows.sort((a, b) => {
    const am = a.meanPassRate ?? a.meanRecallAt10 ?? -1;
    const bm = b.meanPassRate ?? b.meanRecallAt10 ?? -1;
    return bm - am;
  });

  return { rows, datasets };
}

function CrossDatasetLeaderboard({
  groups,
}: {
  groups: ComprehensiveGroup[];
}) {
  if (groups.length === 0) {
    return (
      <div
        className="rounded border px-4 py-8 text-center"
        style={{ borderColor: "var(--border)" }}
      >
        <p className="text-[color:var(--muted)]">
          No comprehensive runs found. Run{" "}
          <code className="font-mono text-xs">
            bash scripts/run_comprehensive.sh
          </code>{" "}
          to populate this view.
        </p>
      </div>
    );
  }

  const { rows, datasets } = aggregateComprehensiveRuns(groups);

  const mostRecent = groups[0];

  return (
    <div className="space-y-6">
      <div className="text-xs text-[color:var(--muted)]">
        {groups.length} comprehensive run{groups.length !== 1 ? "s" : ""} found.
        Most recent:{" "}
        <span className="font-mono">{mostRecent.comprehensive_id}</span>
        {mostRecent.limit != null && ` · limit=${mostRecent.limit}`}
        {mostRecent.judge && ` · judge=${mostRecent.judge}`}
      </div>

      {/* ---- Aggregated summary table ---- */}
      <section>
        <h2 className="mb-2 text-sm font-medium">
          Aggregated cross-dataset leaderboard
        </h2>
        <div
          className="overflow-x-auto rounded border"
          style={{ borderColor: "var(--border)" }}
        >
          <table className="w-full text-sm">
            <thead className="bg-[color:var(--card)] text-[color:var(--muted)]">
              <tr>
                <th className="px-3 py-2 text-left font-normal">System</th>
                <th className="px-3 py-2 text-right font-normal">
                  Mean pass rate
                  <span className="block text-xs font-normal opacity-60">
                    T2+T3
                  </span>
                </th>
                <th className="px-3 py-2 text-right font-normal">
                  Mean Recall@10
                  <span className="block text-xs font-normal opacity-60">
                    T1 brain
                  </span>
                </th>
                <th className="px-3 py-2 text-right font-normal">
                  Mean $/solved
                </th>
                <th className="px-3 py-2 text-right font-normal">
                  Mean p50 ms
                </th>
                <th className="px-3 py-2 text-left font-normal">Tracks</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr
                  key={row.system}
                  className="border-t"
                  style={{ borderColor: "var(--border)" }}
                >
                  <td className="px-3 py-2 font-mono text-xs">{row.system}</td>
                  <td className="px-3 py-2 text-right">
                    {row.meanPassRate != null
                      ? `${(row.meanPassRate * 100).toFixed(1)}%`
                      : <span className="text-[color:var(--muted)]">n/a</span>}
                  </td>
                  <td className="px-3 py-2 text-right">
                    {row.meanRecallAt10 != null
                      ? `${(row.meanRecallAt10 * 100).toFixed(1)}%`
                      : <span className="text-[color:var(--muted)]">n/a</span>}
                  </td>
                  <td className="px-3 py-2 text-right">
                    {row.meanCostPerSolved != null
                      ? `$${row.meanCostPerSolved.toFixed(4)}`
                      : <span className="text-[color:var(--muted)]">—</span>}
                  </td>
                  <td className="px-3 py-2 text-right">
                    {row.meanP50LatencyMs != null
                      ? row.meanP50LatencyMs.toFixed(0)
                      : <span className="text-[color:var(--muted)]">—</span>}
                  </td>
                  <td className="px-3 py-2 text-xs text-[color:var(--muted)]">
                    {[...row.tracksPresent].join(", ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* ---- Per-dataset breakdown ---- */}
      <section>
        <h2 className="mb-2 text-sm font-medium">
          Per-dataset breakdown
        </h2>
        <div
          className="overflow-x-auto rounded border"
          style={{ borderColor: "var(--border)" }}
        >
          <table className="w-full text-sm">
            <thead className="bg-[color:var(--card)] text-[color:var(--muted)]">
              <tr>
                <th className="px-3 py-2 text-left font-normal sticky left-0 bg-[color:var(--card)]">
                  System
                </th>
                {datasets.map((ds) => (
                  <th key={ds} className="px-3 py-2 text-right font-normal whitespace-nowrap">
                    {ds}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr
                  key={row.system}
                  className="border-t"
                  style={{ borderColor: "var(--border)" }}
                >
                  <td className="px-3 py-2 font-mono text-xs sticky left-0 bg-[color:var(--bg)]">
                    {row.system}
                  </td>
                  {datasets.map((ds) => {
                    const cell = row.byDataset[ds];
                    return (
                      <td key={ds} className="px-3 py-2 text-right">
                        {cell && cell.metric != null ? (
                          <>
                            {(cell.metric * 100).toFixed(1)}%
                            <span className="ml-1 text-xs text-[color:var(--muted)]">
                              {cell.metricLabel === "pass rate" ? "pass" : "R@10"}
                            </span>
                          </>
                        ) : (
                          <span className="text-[color:var(--muted)]">n/a</span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
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
