import Link from "next/link";

export default function ScaleHowtoPage() {
  return (
    <div className="max-w-2xl space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Track 4 (scale) — setup guide</h1>
        <p className="mt-2 text-sm text-[color:var(--muted)]">
          Track 4 evaluates retrieval quality against a large pre-loaded corpus (1M–10M docs).
          Unlike Track 1, there is no per-question ingest — the corpus is loaded once and queried
          many times.
        </p>
      </header>

      <section className="space-y-3">
        <h2 className="text-base font-medium">1. Load the corpus</h2>
        <p className="text-sm text-[color:var(--muted)]">
          Run the loader script for your target corpus. This downloads the data and ingests it into
          the adapters&apos; backing stores (e.g. pgvector). The script takes several minutes for
          large corpora.
        </p>
        <CodeBlock code="bash scripts/load_corpus_msmarco.sh" />
        <p className="text-sm text-[color:var(--muted)]">
          The script sets a corpus label (e.g.{" "}
          <code className="font-mono text-xs">msmarco-passages-v1-100k</code>). Note the label — you
          will need it when starting a run.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-base font-medium">2. Verify the load</h2>
        <p className="text-sm text-[color:var(--muted)]">
          Check that the corpus ingested correctly by querying the adapter directly:
        </p>
        <CodeBlock code={`uv run unison-evals brain-search \\
  --system pgvector-naive \\
  --query "what is machine learning" \\
  --k 5`} />
        <p className="text-sm text-[color:var(--muted)]">
          You should see 5 ranked passages. If you get 0 results, the corpus may not have ingested
          correctly — re-run the loader script.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-base font-medium">3. Start a Track 4 run</h2>
        <p className="text-sm text-[color:var(--muted)]">
          On the{" "}
          <Link href="/runs/new" className="underline">
            new run
          </Link>{" "}
          page, select <strong>Track 4 — Scale corpus</strong>, then enter the corpus label exactly
          as printed by the loader script.
        </p>
        <p className="text-sm text-[color:var(--muted)]">
          Or start from the CLI:
        </p>
        <CodeBlock code={`uv run unison-evals run \\
  --track scale \\
  --dataset msmarco \\
  --systems pgvector-naive \\
  --corpus msmarco-passages-v1-100k \\
  --limit 100`} />
      </section>

      <section className="space-y-3">
        <h2 className="text-base font-medium">Available loaders</h2>
        <table className="w-full text-sm">
          <thead className="text-[color:var(--muted)]">
            <tr>
              <th className="py-1 text-left font-normal">Script</th>
              <th className="py-1 text-left font-normal">Corpus label</th>
              <th className="py-1 text-left font-normal">Size</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-t" style={{ borderColor: "var(--border)" }}>
              <td className="py-1.5 font-mono text-xs">scripts/load_corpus_msmarco.sh</td>
              <td className="py-1.5 font-mono text-xs">msmarco-passages-v1-100k</td>
              <td className="py-1.5 text-[color:var(--muted)]">100k passages (~1 GB)</td>
            </tr>
          </tbody>
        </table>
        <p className="text-xs text-[color:var(--muted)]">
          Add more loaders by creating a new script in <code className="font-mono">scripts/</code>{" "}
          that calls <code className="font-mono">unison-evals load-corpus</code>.
        </p>
      </section>
    </div>
  );
}

function CodeBlock({ code }: { code: string }) {
  return (
    <pre
      className="overflow-x-auto rounded border px-3 py-2.5 text-xs font-mono"
      style={{
        borderColor: "var(--border)",
        background: "var(--card)",
        color: "var(--fg)",
      }}
    >
      {code}
    </pre>
  );
}
