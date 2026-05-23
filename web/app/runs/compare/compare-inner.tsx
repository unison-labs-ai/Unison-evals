"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { getRun, isAgentSummary, listRuns, type RunDetail, type SystemSummary } from "@/lib/api";

export function CompareInner() {
  const searchParams = useSearchParams();
  const router = useRouter();

  const aParam = searchParams.get("a") ?? "";
  const bParam = searchParams.get("b") ?? "";

  const [runs, setRuns] = useState<RunDetail[]>([]);
  const [runA, setRunA] = useState<RunDetail | null>(null);
  const [runB, setRunB] = useState<RunDetail | null>(null);
  const [pickA, setPickA] = useState(aParam);
  const [pickB, setPickB] = useState(bParam);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    listRuns().then(setRuns).catch((e) => setLoadError(e.message ?? String(e)));
  }, []);

  const fetchBoth = useCallback(async (a: string, b: string) => {
    if (!a || !b) return;
    setLoading(true);
    setLoadError(null);
    try {
      const [ra, rb] = await Promise.all([getRun(a), getRun(b)]);
      setRunA(ra);
      setRunB(rb);
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (aParam && bParam) {
      setPickA(aParam);
      setPickB(bParam);
      fetchBoth(aParam, bParam);
    }
  }, [aParam, bParam, fetchBoth]);

  function handleCompare() {
    if (!pickA || !pickB) return;
    router.push(
      `/runs/compare?a=${encodeURIComponent(pickA)}&b=${encodeURIComponent(pickB)}`
    );
  }

  const showComparison = runA && runB && !loading;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Compare runs</h1>
        <p className="mt-1 text-sm text-[color:var(--muted)]">
          Select two completed runs to see a side-by-side diff.
        </p>
      </header>

      {/* Picker */}
      <div
        className="rounded border px-4 py-4 flex flex-wrap gap-4 items-end"
        style={{ borderColor: "var(--border)", background: "var(--card)" }}
      >
        <div className="flex flex-col gap-1 min-w-[240px]">
          <label className="text-xs text-[color:var(--muted)] uppercase tracking-wide">
            Run A
          </label>
          <RunSelect runs={runs} value={pickA} onChange={setPickA} exclude={pickB} />
        </div>
        <div className="flex flex-col gap-1 min-w-[240px]">
          <label className="text-xs text-[color:var(--muted)] uppercase tracking-wide">
            Run B
          </label>
          <RunSelect runs={runs} value={pickB} onChange={setPickB} exclude={pickA} />
        </div>
        <button
          onClick={handleCompare}
          disabled={!pickA || !pickB || pickA === pickB}
          className="rounded border px-4 py-1.5 text-sm hover:opacity-80 disabled:opacity-40 disabled:cursor-not-allowed"
          style={{ borderColor: "var(--border)" }}
        >
          Compare
        </button>
      </div>

      {loadError && (
        <div className="rounded border border-red-700 bg-red-950/40 px-4 py-3 text-sm text-red-200">
          {loadError}
        </div>
      )}

      {loading && (
        <p className="text-sm text-[color:var(--muted)]">Loading runs…</p>
      )}

      {showComparison && <ComparisonView runA={runA} runB={runB} />}
    </div>
  );
}

function RunSelect({
  runs,
  value,
  onChange,
  exclude,
}: {
  runs: RunDetail[];
  value: string;
  onChange: (v: string) => void;
  exclude: string;
}) {
  const filtered = runs.filter((r) => r.id !== exclude);
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="rounded border px-2 py-1.5 text-sm font-mono bg-[color:var(--bg)] text-[color:var(--fg)]"
      style={{ borderColor: "var(--border)" }}
    >
      <option value="">— select a run —</option>
      {filtered.map((r) => (
        <option key={r.id} value={r.id}>
          {r.id} · {r.dataset} · {r.status}
        </option>
      ))}
    </select>
  );
}

function ComparisonView({ runA, runB }: { runA: RunDetail; runB: RunDetail }) {
  // Only agent-oracle runs have pass_rate / cost_per_question_usd — show a
  // notice for non-agent tracks rather than crashing on missing fields.
  const bothAgent =
    runA.summary != null &&
    runB.summary != null &&
    isAgentSummary(runA.summary) &&
    isAgentSummary(runB.summary);

  const summA = bothAgent
    ? (runA.summary!.summaries as SystemSummary[])
    : [];
  const summB = bothAgent
    ? (runB.summary!.summaries as SystemSummary[])
    : [];

  const systemsA = new Set(summA.map((s) => s.system));
  const systemsB = new Set(summB.map((s) => s.system));
  const sharedSystems = [...systemsA].filter((s) => systemsB.has(s)).sort();

  const mapA = Object.fromEntries(summA.map((s) => [s.system, s]));
  const mapB = Object.fromEntries(summB.map((s) => [s.system, s]));

  const resultsA = runA.results ?? [];
  const resultsB = runB.results ?? [];

  type QKey = string;
  const passA = new Map<QKey, boolean>();
  const passB = new Map<QKey, boolean>();
  for (const r of resultsA)
    passA.set(`${r.system}::${r.question_id}`, r.judge?.passed ?? false);
  for (const r of resultsB)
    passB.set(`${r.system}::${r.question_id}`, r.judge?.passed ?? false);

  const disagreements: {
    system: string;
    questionId: string;
    aPass: boolean;
    bPass: boolean;
  }[] = [];
  for (const [key, aPass] of passA.entries()) {
    if (!passB.has(key)) continue;
    const bPass = passB.get(key)!;
    if (aPass !== bPass) {
      const [system, questionId] = key.split("::");
      if (sharedSystems.includes(system)) {
        disagreements.push({ system, questionId, aPass, bPass });
      }
    }
  }

  const onlyInA = [...systemsA].filter((s) => !systemsB.has(s));
  const onlyInB = [...systemsB].filter((s) => !systemsA.has(s));

  if (!bothAgent) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-2 gap-4">
          <RunCard label="Run A" run={runA} />
          <RunCard label="Run B" run={runB} />
        </div>
        <div
          className="rounded border px-4 py-4 text-sm text-[color:var(--muted)]"
          style={{ borderColor: "var(--border)" }}
        >
          Side-by-side metric comparison is only available for agent-oracle runs.
          Retrieval tracks (brain-only, scale) use different metric shapes.
          View each run individually for full results.
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Run headers */}
      <div className="grid grid-cols-2 gap-4">
        <RunCard label="Run A" run={runA} />
        <RunCard label="Run B" run={runB} />
      </div>

      {sharedSystems.length === 0 && (
        <div
          className="rounded border px-4 py-6 text-center text-sm text-[color:var(--muted)]"
          style={{ borderColor: "var(--border)" }}
        >
          No systems in common between these two runs.
          {onlyInA.length > 0 && <p className="mt-2">Run A only: {onlyInA.join(", ")}</p>}
          {onlyInB.length > 0 && <p className="mt-2">Run B only: {onlyInB.join(", ")}</p>}
        </div>
      )}

      {sharedSystems.length > 0 && (
        <>
          {/* Metrics table */}
          <div className="overflow-x-auto overflow-hidden rounded border" style={{ borderColor: "var(--border)" }}>
            <div
              className="border-b px-3 py-2 text-xs uppercase tracking-wide text-[color:var(--muted)]"
              style={{ borderColor: "var(--border)" }}
            >
              System metrics — shared systems ({sharedSystems.length})
            </div>
            <table className="w-full text-sm">
              <thead className="bg-[color:var(--card)] text-[color:var(--muted)]">
                <tr>
                  <th className="px-3 py-2 text-left font-normal" rowSpan={2}>
                    System
                  </th>
                  <th className="px-3 py-1 text-center font-normal border-l" style={{ borderColor: "var(--border)" }} colSpan={3}>
                    Pass rate
                  </th>
                  <th className="px-3 py-1 text-center font-normal border-l" style={{ borderColor: "var(--border)" }} colSpan={3}>
                    $/Q
                  </th>
                  <th className="px-3 py-1 text-center font-normal border-l" style={{ borderColor: "var(--border)" }} colSpan={3}>
                    p50 ms
                  </th>
                </tr>
                <tr className="text-xs text-[color:var(--muted)]">
                  <th className="px-3 py-1 text-center font-normal border-l" style={{ borderColor: "var(--border)" }}>A</th>
                  <th className="px-3 py-1 text-center font-normal">B</th>
                  <th className="px-3 py-1 text-center font-normal">B-A</th>
                  <th className="px-3 py-1 text-center font-normal border-l" style={{ borderColor: "var(--border)" }}>A</th>
                  <th className="px-3 py-1 text-center font-normal">B</th>
                  <th className="px-3 py-1 text-center font-normal">B-A</th>
                  <th className="px-3 py-1 text-center font-normal border-l" style={{ borderColor: "var(--border)" }}>A</th>
                  <th className="px-3 py-1 text-center font-normal">B</th>
                  <th className="px-3 py-1 text-center font-normal">B-A</th>
                </tr>
              </thead>
              <tbody>
                {sharedSystems.map((sys) => {
                  const a = mapA[sys];
                  const b = mapB[sys];
                  return (
                    <tr key={sys} className="border-t" style={{ borderColor: "var(--border)" }}>
                      <td className="px-3 py-2 font-mono text-xs">{sys}</td>
                      {/* Pass rate */}
                      <td className="px-3 py-2 text-center border-l text-xs" style={{ borderColor: "var(--border)" }}>
                        {(a.pass_rate * 100).toFixed(1)}%
                      </td>
                      <td className="px-3 py-2 text-center text-xs">
                        {(b.pass_rate * 100).toFixed(1)}%
                      </td>
                      <td className="px-3 py-2 text-center">
                        <Delta
                          v={(b.pass_rate - a.pass_rate) * 100}
                          fmt={(d) => `${d >= 0 ? "+" : ""}${d.toFixed(1)}%`}
                          higherIsBetter
                        />
                      </td>
                      {/* $/Q */}
                      <td className="px-3 py-2 text-center border-l text-xs" style={{ borderColor: "var(--border)" }}>
                        ${a.cost_per_question_usd.toFixed(4)}
                      </td>
                      <td className="px-3 py-2 text-center text-xs">
                        ${b.cost_per_question_usd.toFixed(4)}
                      </td>
                      <td className="px-3 py-2 text-center">
                        <Delta
                          v={b.cost_per_question_usd - a.cost_per_question_usd}
                          fmt={(d) =>
                            `${d >= 0 ? "+" : "-"}$${Math.abs(d).toFixed(4)}`
                          }
                          higherIsBetter={false}
                        />
                      </td>
                      {/* p50 latency */}
                      <td className="px-3 py-2 text-center border-l text-xs" style={{ borderColor: "var(--border)" }}>
                        {a.p50_latency_ms.toFixed(0)}
                      </td>
                      <td className="px-3 py-2 text-center text-xs">
                        {b.p50_latency_ms.toFixed(0)}
                      </td>
                      <td className="px-3 py-2 text-center">
                        <Delta
                          v={b.p50_latency_ms - a.p50_latency_ms}
                          fmt={(d) => `${d >= 0 ? "+" : ""}${d.toFixed(0)}ms`}
                          higherIsBetter={false}
                        />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {(onlyInA.length > 0 || onlyInB.length > 0) && (
            <p className="text-xs text-[color:var(--muted)]">
              {onlyInA.length > 0 && `Run A only: ${onlyInA.join(", ")}. `}
              {onlyInB.length > 0 && `Run B only: ${onlyInB.join(", ")}.`}
            </p>
          )}

          {/* Disagreements */}
          <div className="overflow-hidden rounded border" style={{ borderColor: "var(--border)" }}>
            <div
              className="border-b px-3 py-2 text-xs uppercase tracking-wide text-[color:var(--muted)]"
              style={{ borderColor: "var(--border)" }}
            >
              Per-question disagreements ({disagreements.length})
            </div>
            {disagreements.length === 0 ? (
              <div className="px-4 py-4 text-sm text-[color:var(--muted)]">
                A and B agree on every question for the shared systems.
              </div>
            ) : (
              <table className="w-full text-sm">
                <thead className="bg-[color:var(--card)] text-[color:var(--muted)]">
                  <tr>
                    <th className="px-3 py-2 text-left font-normal">System</th>
                    <th className="px-3 py-2 text-left font-normal">Question</th>
                    <th className="px-3 py-2 text-center font-normal">A</th>
                    <th className="px-3 py-2 text-center font-normal">B</th>
                  </tr>
                </thead>
                <tbody>
                  {disagreements.map((d, i) => (
                    <tr key={i} className="border-t" style={{ borderColor: "var(--border)" }}>
                      <td className="px-3 py-2 font-mono text-xs">{d.system}</td>
                      <td className="px-3 py-2 font-mono text-xs">{d.questionId}</td>
                      <td className="px-3 py-2 text-center">
                        <PassBadge passed={d.aPass} />
                      </td>
                      <td className="px-3 py-2 text-center">
                        <PassBadge passed={d.bPass} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function RunCard({ label, run }: { label: string; run: RunDetail }) {
  return (
    <div
      className="rounded border px-4 py-3"
      style={{ borderColor: "var(--border)", background: "var(--card)" }}
    >
      <p className="text-xs text-[color:var(--muted)] uppercase tracking-wide mb-1">{label}</p>
      <p className="font-mono text-sm">
        <Link href={`/runs/${run.id}`}>{run.id}</Link>
      </p>
      <p className="mt-1 text-xs text-[color:var(--muted)]">
        {run.dataset} · {run.track} · {run.status}
      </p>
      {run.summary && (
        <p className="mt-1 text-xs text-[color:var(--muted)]">
          {run.systems.length} system{run.systems.length !== 1 ? "s" : ""} · {run.summary.n_questions} Qs
        </p>
      )}
    </div>
  );
}

function Delta({
  v,
  fmt,
  higherIsBetter,
}: {
  v: number;
  fmt: (v: number) => string;
  higherIsBetter: boolean;
}) {
  if (Math.abs(v) < 0.0001) {
    return <span className="text-[color:var(--muted)] text-xs">—</span>;
  }
  const isGood = higherIsBetter ? v > 0 : v < 0;
  const color = isGood ? "text-green-400" : "text-red-400";
  return <span className={`text-xs font-mono ${color}`}>{fmt(v)}</span>;
}

function PassBadge({ passed }: { passed: boolean }) {
  return passed ? (
    <span className="rounded border border-green-800 bg-green-900/40 px-2 py-0.5 text-xs text-green-300">
      pass
    </span>
  ) : (
    <span className="rounded border border-red-800 bg-red-900/40 px-2 py-0.5 text-xs text-red-300">
      fail
    </span>
  );
}
