import { Suspense } from "react";
import Link from "next/link";
import {
  listRuns,
  type RunDetail,
} from "@/lib/api";
import { HomeClient } from "./home-client";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  let runs: RunDetail[] = [];
  let loadError: string | null = null;
  try {
    runs = await listRuns();
  } catch (e) {
    loadError = e instanceof Error ? e.message : String(e);
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">unison-evals</h1>
        <Link
          href="/runs/new"
          className="rounded border px-3 py-1.5 text-sm hover:opacity-80"
          style={{ borderColor: "var(--border)" }}
        >
          ▶ New run
        </Link>
      </div>

      {loadError && (
        <div className="rounded border border-red-700 bg-red-950/40 px-4 py-3 text-sm">
          <p className="font-medium text-red-300">Could not reach the eval server.</p>
          <p className="mt-1 text-red-200/80">
            Start it with <code className="font-mono">uv run unison-evals-server</code>. Error:{" "}
            {loadError}
          </p>
        </div>
      )}

      {!loadError && (
        <Suspense fallback={<p className="text-sm text-[color:var(--muted)]">Loading…</p>}>
          <HomeClient runs={runs} />
        </Suspense>
      )}
    </div>
  );
}
