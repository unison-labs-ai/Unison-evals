"""CLI — `unison-evals run --systems X,Y --dataset longmemeval --limit N`."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .config import get_settings
from .memory_evals.adapters import REGISTRY as ADAPTER_REGISTRY
from .memory_evals.datasets import get_dataset
from .memory_evals.metrics.llm_judge import LLMJudge
from .memory_evals.runners.agent_e2e import AgentE2ERunner
from .memory_evals.runners.agent_oracle import AgentOracleRunner
from .results import new_run_id, results_path
from .types import RunSummary, SystemSummary

console = Console()

# The one benchmark list + each one's publishable ("real") canonical judge.
BENCHMARKS = ("longmemeval", "locomo", "memoryagentbench", "context-bench")
CANONICAL_JUDGE = {
    "longmemeval": "gpt-4o-2024-08-06",  # LongMemEval paper judge (>97% human agreement)
    "locomo": "gpt-4o-2024-08-06",  # de-facto LOCOMO judge (Mem0/Zep "J" score parity)
    "memoryagentbench": "gpt-4o-2024-08-06",  # de-facto memory-eval judge
    "context-bench": "gpt-5-mini",  # Letta-leaderboard parity
}


def _resolve_judge(dataset: str, judge: str | None, real_mode: bool) -> str | None:
    """--judge wins; else --real → the benchmark's canonical judge; else --dev →
    the cheap Gemini judge. One rule for all three benchmarks."""
    if judge:
        return judge
    if real_mode:
        return CANONICAL_JUDGE.get(dataset)
    return get_settings().dev_judge_model


@click.group()
def main() -> None:
    """unison-evals — public benchmark harness."""


@main.command(name="systems")
def list_systems() -> None:
    """List registered adapter names."""
    for name in sorted(ADAPTER_REGISTRY):
        console.print(f"  {name}")


@main.command(name="datasets")
def list_datasets() -> None:
    """List benchmark names."""
    for name in BENCHMARKS:
        console.print(f"  {name}")


@main.command()
@click.option(
    "--systems",
    default="unison-agent",
    show_default=True,
    help="Comma-separated adapter names (memory benches only; context-bench always runs unison-agent).",
)
@click.option(
    "--dataset",
    required=True,
    type=click.Choice([*BENCHMARKS, "all"]),
    help="Benchmark to run, or 'all' for the three.",
)
@click.option(
    "--track",
    default="agent-e2e",
    type=click.Choice(["agent-oracle", "agent-e2e"]),
    help="Memory-bench track. agent-e2e=Track 3 (agent + brain). agent-oracle=Track 2 (gold context).",
)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Max questions per benchmark. Omit to run the FULL benchmark.",
)
@click.option("--judge", default=None, help="Explicit judge model id (overrides --dev/--real).")
@click.option(
    "--real/--dev",
    "real_mode",
    default=False,
    show_default=True,
    help="--real: the per-benchmark canonical judge (publishable, leaderboard-comparable). "
    "--dev (default): the cheap Gemini judge (dev_judge_model) for test/tune/research loops.",
)
@click.option(
    "--pass-threshold",
    default=1.0,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Score >= threshold counts as 'passed'. 1.0 = strict, 0.5 = partial credit OK.",
)
@click.option(
    "--no-judge",
    is_flag=True,
    help="Skip the LLM judge — just collect adapter answers (for connectivity smoke).",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the run JSON to this path. Default: results/<run_id>.json",
)
def run(
    systems: str,
    dataset: str,
    track: str,
    limit: int | None,
    judge: str | None,
    real_mode: bool,
    pass_threshold: float,
    no_judge: bool,
    output: Path | None,
) -> None:
    """Run a benchmark (or `--dataset all`) on the unison-agent. One command,
    one --limit, the prod model (no model is sent — the server decides)."""
    sys_list = [s.strip() for s in systems.split(",") if s.strip()]
    if not sys_list:
        raise click.UsageError("--systems cannot be empty")
    for s in sys_list:
        if s not in ADAPTER_REGISTRY:
            raise click.BadParameter(
                f"Unknown system '{s}'. Run `unison-evals systems` to list registered names."
            )

    datasets = list(BENCHMARKS) if dataset == "all" else [dataset]
    if output is not None and len(datasets) > 1:
        # One --output file can't hold multiple benchmarks (each would overwrite
        # the prior, and context-bench writes its own results dir). Every run
        # already persists to results/ automatically.
        raise click.UsageError(
            "--output is only valid for a single --dataset, not 'all'. "
            "Each benchmark already writes to results/ automatically."
        )
    mode = "REAL (canonical, publishable)" if (real_mode or judge) else "DEV (cheap Gemini)"
    headlines: list[dict] = []
    for ds in datasets:
        if len(datasets) > 1:
            click.echo(f"\n=== {ds} ===")

        # Context-Bench has its own per-run runner (fixed corpus, Letta rubric).
        # It always judges; reuse the same judge-resolution rule.
        if ds == "context-bench":
            from .benchmarks.context_bench.run import run_context_bench

            cb_judge = _resolve_judge(ds, judge, real_mode) or CANONICAL_JUDGE["context-bench"]
            click.echo(f"  Judge: {cb_judge}  [{mode}]")
            cb = run_context_bench(
                limit=limit,
                judge_model=cb_judge,
                agent_model=get_settings().unison_agent_model or None,
            )
            headlines.append(
                {
                    "benchmark": ds,
                    "n": cb["n"],
                    "pct": cb["pct"],
                    "cost_usd": cb["total_agent_cost_usd"],
                }
            )
            continue

        # Memory benches (LongMemEval, MemoryAgentBench).
        if track == "agent-e2e" and no_judge:
            raise click.UsageError("--no-judge is not supported with --track agent-e2e")
        resolved_judge = None if no_judge else _resolve_judge(ds, judge, real_mode)
        if not no_judge:
            click.echo(f"  Judge: {resolved_judge or '(env / config default)'}  [{mode}]")
        hl = asyncio.run(
            _run_async(
                systems=sys_list,
                dataset=ds,
                track=track,
                limit=limit,
                judge_model=resolved_judge,
                pass_threshold=pass_threshold,
                no_judge=no_judge,
                output=output,
            )
        )
        if hl:
            headlines.append(hl)

    if len(datasets) > 1 and headlines:
        _print_combined(headlines)


async def _run_async(
    systems: list[str],
    dataset: str,
    track: str,
    limit: int | None,
    judge_model: str | None,
    pass_threshold: float,
    no_judge: bool,
    output: Path | None,
) -> dict | None:
    if track == "agent-e2e":
        return await _run_agent_e2e(
            systems=systems,
            dataset=dataset,
            limit=limit,
            judge_model=judge_model,
            pass_threshold=pass_threshold,
            output=output,
        )
    return await _run_agent_oracle(
        systems=systems,
        dataset=dataset,
        limit=limit,
        judge_model=judge_model,
        pass_threshold=pass_threshold,
        no_judge=no_judge,
        output=output,
    )


async def _run_agent_oracle(
    systems: list[str],
    dataset: str,
    limit: int | None,
    judge_model: str | None,
    pass_threshold: float,
    no_judge: bool,
    output: Path | None,
) -> dict | None:
    settings = get_settings()
    ds = get_dataset(dataset)
    questions = list(ds.load(limit=limit))
    if not questions:
        console.print("[red]Dataset returned 0 questions.[/red]")
        return None

    judge_obj = (
        _NoOpJudge(judge_model or settings.judge_model)
        if no_judge
        else LLMJudge(model=judge_model, pass_threshold=pass_threshold)
    )
    runner = AgentOracleRunner(systems=systems, judge=judge_obj, run_id=new_run_id(dataset))

    console.print(
        f"[bold]Run {runner.run_id}[/bold] · "
        f"track=agent-oracle · dataset={dataset} · n={len(questions)} · systems={','.join(systems)}"
    )
    if no_judge:
        console.print("[yellow]--no-judge: skipping LLM judge, all rows show score=0[/yellow]")

    summary: RunSummary | None = None
    async for ev in runner.run(questions, dataset_name=dataset):
        if ev.type == "question_completed" and ev.result:
            r = ev.result
            mark = "[green]✓[/green]" if r.judge and r.judge.passed else "[red]✗[/red]"
            console.print(
                f"  {mark} {r.system:>14} {r.question_id:>14}  "
                f"${r.adapter.cost_usd:.4f}  {r.adapter.latency_ms / 1000:.2f}s  "
                f"score={r.judge.score if r.judge else 0:.1f}"
                + (f"  [yellow]err: {r.adapter.error[:80]}[/yellow]" if r.adapter.error else "")
            )
        elif ev.type == "run_completed":
            summary = ev.summary
        elif ev.type in ("run_failed", "question_failed"):
            console.print(f"[red]{ev.type}: {ev.error}[/red]")

    if summary is None:
        console.print("[red]Run did not complete.[/red]")
        return None

    return _finalize(dataset, summary, runner.results, output)


async def _run_agent_e2e(
    systems: list[str],
    dataset: str,
    limit: int | None,
    judge_model: str | None,
    pass_threshold: float,
    output: Path | None,
) -> dict | None:
    """Track 3 (agent E2E) — per-question corpus ingest via seed_docs, then judge."""
    ds = get_dataset(dataset)
    try:
        brain_questions = list(ds.load_brain_questions(limit=limit))
    except NotImplementedError as e:
        console.print(f"[red]Track 3 unavailable for dataset={dataset}: {e}[/red]")
        return None

    if not brain_questions:
        console.print(
            "[red]Dataset returned 0 brain questions — cannot run agent-e2e track for "
            f"dataset={dataset}.[/red]"
        )
        return None

    judge_obj = LLMJudge(model=judge_model, pass_threshold=pass_threshold)
    runner = AgentE2ERunner(systems=systems, judge=judge_obj, run_id=new_run_id(dataset))

    console.print(
        f"[bold]Run {runner.run_id}[/bold] · "
        f"track=agent-e2e · dataset={dataset} · n={len(brain_questions)} · systems={','.join(systems)}"
    )

    summary: RunSummary | None = None
    async for ev in runner.run(brain_questions, dataset_name=dataset):
        if ev.type == "question_completed" and ev.result:
            r = ev.result
            mark = "[green]✓[/green]" if r.judge and r.judge.passed else "[red]✗[/red]"
            seed_info = ""
            if "seed_docs_count" in r.adapter.raw:
                seed_info = f"  docs={r.adapter.raw['seed_docs_count']}"
            if "seed_embed_ms" in r.adapter.raw:
                seed_info += f"  embed={r.adapter.raw['seed_embed_ms']:.0f}ms"
            console.print(
                f"  {mark} {r.system:>14} {r.question_id:>14}  "
                f"${r.adapter.cost_usd:.4f}  {r.adapter.latency_ms / 1000:.2f}s  "
                f"score={r.judge.score if r.judge else 0:.1f}"
                + seed_info
                + (f"  [yellow]err: {r.adapter.error[:80]}[/yellow]" if r.adapter.error else "")
            )
        elif ev.type == "run_completed":
            summary = ev.summary
        elif ev.type in ("run_failed", "question_failed"):
            console.print(f"[red]{ev.type}: {ev.error}[/red]")

    if summary is None:
        console.print("[red]Run did not complete.[/red]")
        return None

    return _finalize(dataset, summary, runner.results, output)


def _finalize(dataset: str, summary: RunSummary, results: list, output: Path | None) -> dict:
    """Print the table, write results/<run_id>.json, return the headline."""
    _print_summary_table(summary)
    out_path = output or results_path(summary.run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_to_json(summary, results))
    console.print(f"\n[dim]→ {out_path}[/dim]")
    return _headline(dataset, summary)


def _headline(dataset: str, summary: RunSummary) -> dict:
    """One-line score for the combined --dataset all summary."""
    s0 = summary.summaries[0] if summary.summaries else None
    pct = (s0.pass_rate * 100) if s0 else 0.0
    cost = (s0.cost_per_question_usd * s0.n_questions) if s0 else 0.0
    return {"benchmark": dataset, "n": summary.n_questions, "pct": pct, "cost_usd": cost}


def _print_combined(headlines: list[dict]) -> None:
    """Aggregate table printed once at the end of a `--dataset all` run."""
    table = Table(title="\nAll benchmarks")
    table.add_column("Benchmark")
    table.add_column("N", justify="right")
    table.add_column("Score %", justify="right")
    table.add_column("Agent $", justify="right")
    total = 0.0
    for h in headlines:
        table.add_row(h["benchmark"], str(h["n"]), f"{h['pct']:.1f}%", f"${h['cost_usd']:.3f}")
        total += h["cost_usd"]
    console.print(table)
    console.print(
        f"[dim]Total agent cost: ${total:.3f}  (vs each benchmark's published number)[/dim]"
    )


def _print_summary_table(summary: RunSummary) -> None:
    table = Table(title=f"\n{summary.dataset} · {summary.track.value} · {summary.n_questions} Q")
    table.add_column("System")
    table.add_column("Pass", justify="right")
    table.add_column("Pass %", justify="right")
    table.add_column("$/Q", justify="right")
    table.add_column("$/solved", justify="right")
    table.add_column("p50 ms", justify="right")
    table.add_column("p95 ms", justify="right")
    for s in summary.summaries:
        table.add_row(
            s.system,
            f"{s.n_passed}/{s.n_questions}",
            f"{s.pass_rate * 100:.1f}%",
            f"${s.cost_per_question_usd:.4f}",
            f"${s.cost_per_solved_usd:.4f}" if s.cost_per_solved_usd is not None else "—",
            f"{s.p50_latency_ms:.0f}",
            f"{s.p95_latency_ms:.0f}",
        )
    console.print(table)


def _to_json(summary: RunSummary, results: list) -> str:
    """Serialize the run as a single JSON file (summary + per-question results)."""
    payload = {
        "summary": summary.model_dump(mode="json"),
        "results": [r.model_dump(mode="json") for r in results],
        "exported_at": datetime.now(UTC).isoformat(),
    }
    return json.dumps(payload, indent=2)


class _NoOpJudge:
    """Drop-in for LLMJudge that always returns score=0. Used by --no-judge
    so adapter connectivity can be smoke-tested without burning judge $."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.pass_threshold = 1.0

    async def judge(self, *_: object, **__: object) -> object:
        from .types import JudgeResult

        return JudgeResult(
            score=0.0,
            passed=False,
            confidence=0.0,
            reasoning="--no-judge",
            cost_usd=0.0,
        )


@main.command()
@click.option("--dataset", default="longmemeval", show_default=True, help="Dataset to pre-ingest.")
@click.option("--limit", default=None, type=int, help="Max questions (omit for the full split).")
@click.option(
    "--manifest",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Manifest JSON to write/update (question_id -> persistent tenant_id).",
)
@click.option("--system", default="unison-agent", show_default=True)
def preingest(dataset: str, limit: int | None, manifest: Path, system: str) -> None:
    """Ingest each question's haystack ONCE into a persistent eval tenant and
    record question_id -> tenant_id in the manifest. Selection honours the same
    EVAL_SEED / EVAL_STRATIFIED / EVAL_SPLIT env as `run`. Run the server with
    AGENT_WAIT_GRAPH=1 so the full extract->promote->facts graph is built.

    Then reuse it read-only + fast:
      UNISON_PREINGEST_MANIFEST=<manifest> unison-evals run --dataset <dataset> ...
    Idempotent: re-running skips questions already in the manifest.
    """
    asyncio.run(_preingest(dataset, limit, manifest, system))


async def _preingest(dataset: str, limit: int | None, manifest_path: Path, system: str) -> None:
    import os

    from .memory_evals.datasets import get_dataset
    from .memory_evals.preingest import load_manifest, save_manifest, tenant_for

    if system not in ADAPTER_REGISTRY:
        raise click.BadParameter(f"Unknown system '{system}'.")
    ds = get_dataset(dataset)
    # Track 3 loader: BrainQuestion carries the per-question corpus (.query/.corpus).
    questions = list(ds.load_brain_questions(limit))
    adapter = ADAPTER_REGISTRY[system]()
    await adapter.setup()

    manifest = load_manifest(manifest_path)
    manifest["meta"].update(
        {
            "dataset": dataset,
            "system": system,
            "seed": os.environ.get("EVAL_SEED", "1234"),
            "split": os.environ.get("EVAL_SPLIT", ""),
            "stratified": os.environ.get("EVAL_STRATIFIED", ""),
            "embed_model": os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            "updated": datetime.now(UTC).isoformat(),
        }
    )

    done = skipped = failed = 0
    total = len(questions)
    for i, q in enumerate(questions, 1):
        if tenant_for(manifest, q.id):
            skipped += 1
            continue
        tenant = await adapter.preingest_question(q.query, q.corpus, q.id)
        if tenant:
            manifest["questions"][q.id] = tenant
            save_manifest(manifest_path, manifest)  # incremental — crash-safe
            done += 1
            console.print(f"[green]OK[/] [{i}/{total}] {q.id} -> {tenant}")
        else:
            failed += 1
            console.print(f"[red]FAIL[/] [{i}/{total}] {q.id} ingest failed")
    console.print(
        f"\npre-ingest complete: {done} ingested, {skipped} already present, "
        f"{failed} failed -> {manifest_path}"
    )


# CLI for SystemSummary type-import linting hygiene
_SYSTEM_SUMMARY_TYPE: type = SystemSummary

if __name__ == "__main__":  # pragma: no cover
    main()
