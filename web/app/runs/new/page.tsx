"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getRegistry, startRun, type Registry, type BrainModeInfo } from "@/lib/api";

interface TrackDef {
  value: string;
  label: string;
  description: string;
  experimental?: boolean;
  comingSoon?: boolean;
}

const TRACKS: TrackDef[] = [
  {
    value: "brain-only",
    label: "Track 1 — Brain only",
    description: "Measures retrieval quality on small per-question corpora. No LLM judge needed.",
  },
  {
    value: "agent-oracle",
    label: "Track 2 — Agent oracle",
    description: "Agent reasons given gold context, no retrieval. Uses an LLM judge to score.",
  },
  {
    value: "agent-e2e",
    label: "Track 3 — Agent + brain E2E",
    description:
      "Given the same per-question corpus, who handles it best? Unison ingests into a brain + retrieves; raw models stuff everything into the prompt; Mem0 extracts memories. Same data, different consumption, measured side by side. Surfaces brain-efficiency ratio (Unison tokens vs baseline).",
  },
  {
    value: "scale",
    label: "Track 4 — Scale corpus",
    description:
      "Query a pre-loaded large corpus. Corpus must be loaded first via scripts/load_corpus_*.sh.",
  },
];

const LIMIT_OPTIONS = [
  { value: 3, label: "3 (smoke)" },
  { value: 10, label: "10" },
  { value: 25, label: "25" },
  { value: 50, label: "50" },
  { value: 100, label: "100" },
  { value: 500, label: "500 (full)" },
];

const JUDGE_OPTIONS = [
  { value: "claude-haiku-4-5", label: "Haiku 4.5 (cheap)" },
  { value: "claude-sonnet-4-5", label: "Sonnet 4.5" },
  { value: "claude-opus-4-7", label: "Opus 4.7 (best)" },
];

const AGENT_TRACKS = new Set(["agent-oracle", "agent-e2e"]);
const BRAIN_TRACKS = new Set(["brain-only", "scale"]);

export default function NewRunPage() {
  const router = useRouter();
  const [registry, setRegistry] = useState<Registry | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [dataset, setDataset] = useState("longmemeval");
  const [track, setTrack] = useState("agent-oracle");
  const [systems, setSystems] = useState<string[]>([]);
  const [limit, setLimit] = useState(3);
  const [judge, setJudge] = useState("claude-haiku-4-5");
  const [corpus, setCorpus] = useState("");
  const [brainMode, setBrainMode] = useState("cold");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  useEffect(() => {
    getRegistry()
      .then((r) => {
        setRegistry(r);
        // Set sensible defaults based on initial track
        setSystems(["unison-agent", "claude-code"].filter((s) =>
          r.adapters.some((a) => a.name === s)
        ));
      })
      .catch((e) => setLoadError(e.message ?? String(e)));
  }, []);

  // When track changes, reset systems to appropriate defaults
  const onTrackChange = (newTrack: string) => {
    setTrack(newTrack);
    setSubmitError(null);
    if (!registry) return;
    if (BRAIN_TRACKS.has(newTrack)) {
      const firstBrain = registry.brain_adapters[0]?.name;
      setSystems(firstBrain ? [firstBrain] : []);
    } else {
      setSystems(
        ["unison-agent", "claude-code"].filter((s) =>
          registry.adapters.some((a) => a.name === s)
        )
      );
    }
  };

  const availableSystems = registry
    ? BRAIN_TRACKS.has(track)
      ? registry.brain_adapters
      : registry.adapters
    : [];

  const onToggleSystem = (name: string) => {
    setSystems((prev) =>
      prev.includes(name) ? prev.filter((s) => s !== name) : [...prev, name],
    );
  };

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (systems.length === 0) {
      setSubmitError("Select at least one system.");
      return;
    }
    if (track === "scale" && !corpus.trim()) {
      setSubmitError("Corpus label is required for Track 4 (scale).");
      return;
    }
    setSubmitting(true);
    setSubmitError(null);
    try {
      const body: Parameters<typeof startRun>[0] = {
        dataset,
        track,
        systems,
        limit,
      };
      if (AGENT_TRACKS.has(track)) {
        body.judge_model = judge;
      }
      if (track === "scale") {
        body.corpus = corpus.trim();
      }
      if (track === "brain-only") {
        body.mode = brainMode;
      }
      const { run_id } = await startRun(body);
      router.push(`/runs/${run_id}`);
    } catch (e) {
      setSubmitting(false);
      setSubmitError(e instanceof Error ? e.message : String(e));
    }
  };

  const isScaleTrack = track === "scale";
  const isAgentTrack = AGENT_TRACKS.has(track);
  const isBrainOnlyTrack = track === "brain-only";
  const brainModes: BrainModeInfo[] = registry?.brain_modes ?? [
    { name: "cold", description: "Per-question reset → ingest → search (default)" },
    { name: "warm", description: "Corpus pre-loaded; skip reset+ingest" },
    { name: "bitemporal", description: "As-of temporal correctness scoring" },
    { name: "compaction", description: "LLM-judged wiki synthesis (unison-brain only)" },
  ];

  if (loadError) {
    return (
      <div className="rounded border border-red-700 bg-red-950/40 px-4 py-3 text-sm">
        <p className="font-medium text-red-300">Could not reach the eval server.</p>
        <p className="mt-1 text-red-200/80">
          Start it with <code className="font-mono">uv run unison-evals-server</code>. Error:{" "}
          {loadError}
        </p>
      </div>
    );
  }

  return (
    <div className="max-w-xl">
      <h1 className="mb-6 text-2xl font-semibold tracking-tight">New evaluation run</h1>

      <form onSubmit={onSubmit} className="space-y-5">
        <Field label="Track">
          <div className="space-y-2">
            {TRACKS.map((t) => {
              const datasetMeta = registry?.datasets.find((d) => d.name === dataset);
              const supported = datasetMeta?.supported_tracks;
              const unsupported = supported && !supported.includes(t.value);
              return (
                <label
                  key={t.value}
                  className="flex cursor-pointer items-start gap-3 rounded border px-3 py-2.5 text-sm transition-colors"
                  style={{
                    borderColor: track === t.value ? "var(--accent)" : "var(--border)",
                    background: track === t.value ? "var(--card)" : "transparent",
                    opacity: unsupported ? 0.4 : 1,
                    cursor: unsupported ? "not-allowed" : "pointer",
                  }}
                >
                  <input
                    type="radio"
                    name="track"
                    value={t.value}
                    checked={track === t.value}
                    onChange={() => onTrackChange(t.value)}
                    disabled={unsupported}
                    className="mt-0.5 shrink-0"
                  />
                  <div>
                    <span className="font-medium">{t.label}</span>
                    {t.experimental && (
                      <span
                        className="ml-2 rounded px-1.5 py-0.5 text-xs"
                        style={{ background: "var(--card)", color: "var(--muted)", border: "1px solid var(--border)" }}
                      >
                        experimental
                      </span>
                    )}
                    {unsupported && (
                      <span className="ml-2 text-xs text-[color:var(--muted)]">
                        (not available for {dataset})
                      </span>
                    )}
                    <p className="mt-0.5 text-[color:var(--muted)]">{t.description}</p>
                  </div>
                </label>
              );
            })}
          </div>
        </Field>

        <Field label="Dataset">
          <select
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
            className="input"
            disabled={!registry}
          >
            {registry?.datasets.map((d) => (
              <option key={d.name} value={d.name}>
                {d.name}
              </option>
            ))}
          </select>
          {(() => {
            const meta = registry?.datasets.find((d) => d.name === dataset);
            if (!meta) return null;
            return (
              <div className="mt-1 text-xs text-[color:var(--muted)]">
                {meta.description && <p>{meta.description}</p>}
                <p className="mt-0.5">
                  {meta.total_questions != null
                    ? `${meta.total_questions.toLocaleString()} total questions`
                    : "size: varies"}
                  {meta.supported_tracks && meta.supported_tracks.length > 0 && (
                    <span>
                      {" · supports: "}
                      <span className="font-mono">{meta.supported_tracks.join(", ")}</span>
                    </span>
                  )}
                </p>
                {meta.total_questions != null && limit > meta.total_questions && (
                  <p className="mt-0.5 text-yellow-400">
                    Note: limit={limit} exceeds dataset size — actual run capped at{" "}
                    {meta.total_questions}.
                  </p>
                )}
              </div>
            );
          })()}
        </Field>

        {isScaleTrack && (
          <div className="rounded border border-yellow-700/60 bg-yellow-950/20 px-3 py-2.5 text-sm">
            <p className="font-medium text-yellow-300">Corpus pre-load required</p>
            <p className="mt-1 text-yellow-200/80">
              Did you run <code className="font-mono">bash scripts/load_corpus_msmarco.sh</code>{" "}
              first?{" "}
              <a href="/scale-howto" className="underline">
                Setup guide →
              </a>
            </p>
          </div>
        )}

        {isScaleTrack && (
          <Field label="Corpus label">
            <input
              type="text"
              value={corpus}
              onChange={(e) => setCorpus(e.target.value)}
              placeholder="e.g. msmarco-passages-v1-100k"
              className="input"
            />
            <p className="mt-1 text-xs text-[color:var(--muted)]">
              Must match the label used when loading the corpus.
            </p>
          </Field>
        )}

        {isBrainOnlyTrack && (
          <Field label="Brain mode">
            <select
              value={brainMode}
              onChange={(e) => setBrainMode(e.target.value)}
              className="input"
            >
              {brainModes.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name}
                </option>
              ))}
            </select>
            {brainModes.find((m) => m.name === brainMode) && (
              <p className="mt-1 text-xs text-[color:var(--muted)]">
                {brainModes.find((m) => m.name === brainMode)?.description}
                {brainMode === "compaction" && (
                  <span className="ml-1 text-yellow-400">
                    (requires the Unison compaction endpoint)
                  </span>
                )}
              </p>
            )}
          </Field>
        )}

        <Field label="Limit (questions)">
          <select
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            className="input"
          >
            {LIMIT_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </Field>

        <Field label={BRAIN_TRACKS.has(track) ? "Brain adapters" : "Agent systems"}>
          {!registry ? (
            <p className="text-sm text-[color:var(--muted)]">Loading…</p>
          ) : availableSystems.length === 0 ? (
            <p className="text-sm text-[color:var(--muted)]">
              No {BRAIN_TRACKS.has(track) ? "brain adapters" : "agent adapters"} registered.
            </p>
          ) : (
            <div className="space-y-1">
              {availableSystems.map((a) => (
                <label key={a.name} className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={systems.includes(a.name)}
                    onChange={() => onToggleSystem(a.name)}
                  />
                  <span className="font-mono">{a.name}</span>
                  <span className="text-xs text-[color:var(--muted)]">{a.class}</span>
                </label>
              ))}
            </div>
          )}
        </Field>

        {isAgentTrack && (
          <Field label="Judge model">
            <select value={judge} onChange={(e) => setJudge(e.target.value)} className="input">
              {JUDGE_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </Field>
        )}

        {track === "agent-e2e" && (
          <div className="rounded border px-3 py-2 text-xs text-[color:var(--muted)]" style={{ borderColor: "var(--border)" }}>
            Track 3 — requires a brain-supporting dataset (bitempoqa, longmemeval,
            memoryagentbench, musique). All 8 agent adapters are accepted; adapters whose
            setup() fails are automatically skipped with a [SKIP] log. The run summary
            surfaces brain-efficiency ratios (tokens consumed vs. anthropic-raw baseline).
          </div>
        )}
        <button
          type="submit"
          disabled={submitting}
          className="rounded border px-4 py-2 text-sm hover:opacity-80 disabled:opacity-40"
          style={{ borderColor: "var(--border)" }}
        >
          {submitting ? "Starting…" : "▶ Start run"}
        </button>

        {submitError && (
          <p className="text-sm text-red-400">{submitError}</p>
        )}
      </form>

      <style jsx>{`
        :global(.input) {
          width: 100%;
          padding: 6px 10px;
          border: 1px solid var(--border);
          background: var(--card);
          color: var(--fg);
          border-radius: 4px;
          font-size: 14px;
        }
      `}</style>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="mb-1 block text-sm text-[color:var(--muted)]">{label}</label>
      {children}
    </div>
  );
}
