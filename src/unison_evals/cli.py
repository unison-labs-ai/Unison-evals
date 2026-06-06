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
from .memory_evals.adapters import BRAIN_REGISTRY, get_brain_adapter
from .memory_evals.adapters import REGISTRY as ADAPTER_REGISTRY
from .memory_evals.datasets import REGISTRY as DATASET_REGISTRY
from .memory_evals.datasets import get_dataset
from .memory_evals.metrics.llm_judge import LLMJudge
from .memory_evals.runners.agent_e2e import AgentE2ERunner
from .memory_evals.runners.agent_oracle import AgentOracleRunner
from .memory_evals.runners.brain_retrieval import BrainRetrievalRunner
from .memory_evals.runners.scale_retrieval import ScaleRetrievalRunner
from .types import BrainMode, BrainRunSummary, RunSummary, ScaleRunSummary, SystemSummary

console = Console()


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
    """List registered dataset names."""
    for name in sorted(DATASET_REGISTRY):
        console.print(f"  {name}")


@main.command()
@click.option(
    "--systems",
    required=True,
    help="Comma-separated adapter names (e.g. unison-agent,claude-code)",
)
@click.option("--dataset", required=True, type=click.Choice(sorted(DATASET_REGISTRY)))
@click.option(
    "--track",
    default="agent-oracle",
    type=click.Choice(["agent-oracle", "agent-e2e", "brain-only", "scale"]),
    help=(
        "Eval track. agent-oracle=Track 2 (agent given gold context). "
        "agent-e2e=Track 3 (agent + brain, per-Q corpus ingest). "
        "brain-only=Track 1 (retrieval only). "
        "scale=Track 4 (query pre-loaded large corpus, no per-Q ingest)."
    ),
)
@click.option("--limit", default=10, show_default=True, type=int, help="Max questions to run.")
@click.option("--judge", default=None, help="Judge model id. Defaults to JUDGE_MODEL env var.")
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
    "--corpus",
    default=None,
    help=(
        "Human-friendly corpus label for Track 4 (scale) runs. "
        "Required when --track scale. Example: msmarco-passages-v1-100k"
    ),
)
@click.option(
    "--mode",
    default="cold",
    type=click.Choice(["cold", "warm", "bitemporal", "compaction"]),
    show_default=True,
    help=(
        "Sub-mode for --track brain-only. Ignored for other tracks. "
        "cold=per-Q reset+ingest+search (default). "
        "warm=corpus pre-loaded, skip reset+ingest. "
        "bitemporal=as-of temporal correctness scoring. "
        "compaction=LLM-judged wiki synthesis quality (unison-brain only)."
    ),
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
    limit: int,
    judge: str | None,
    pass_threshold: float,
    no_judge: bool,
    corpus: str | None,
    mode: str,
    output: Path | None,
) -> None:
    """Run an evaluation."""
    sys_list = [s.strip() for s in systems.split(",") if s.strip()]
    if not sys_list:
        raise click.UsageError("--systems cannot be empty")

    if track == "scale" and not corpus:
        raise click.UsageError("--corpus is required when --track scale")

    if track != "brain-only" and mode != "cold":
        raise click.UsageError("--mode is only applicable when --track brain-only")

    if track in ("brain-only", "scale"):
        for s in sys_list:
            if s not in BRAIN_REGISTRY:
                raise click.BadParameter(
                    f"Unknown brain system '{s}'. Run `unison-evals systems` to list registered names. "
                    f"Brain-only and scale tracks require a BrainAdapter, e.g. pgvector-naive, mem0, letta."
                )
    else:
        for s in sys_list:
            if s not in ADAPTER_REGISTRY:
                raise click.BadParameter(
                    f"Unknown system '{s}'. Run `unison-evals systems` to list registered names."
                )

    if track == "agent-e2e" and no_judge:
        raise click.UsageError("--no-judge is not supported with --track agent-e2e")

    # Per-benchmark canonical judge → publishable, leaderboard-comparable scores.
    # An explicit --judge (or JUDGE_MODEL env) overrides. Context-Bench is scored
    # by its own runner (gpt-5-mini, Letta-leaderboard parity) and is unaffected.
    CANONICAL_JUDGE = {
        "longmemeval": "gpt-4o-2024-08-06",  # LongMemEval paper judge (>97% human agreement)
        "memoryagentbench": "gpt-4o-2024-08-06",  # de-facto memory-eval judge; documented choice
    }
    resolved_judge = judge or CANONICAL_JUDGE.get(dataset)
    if not no_judge:
        click.echo(f"  Judge: {resolved_judge or '(JUDGE_MODEL env / config default)'}")

    asyncio.run(
        _run_async(
            systems=sys_list,
            dataset=dataset,
            track=track,
            limit=limit,
            judge_model=resolved_judge,
            pass_threshold=pass_threshold,
            no_judge=no_judge,
            corpus=corpus,
            mode=mode,
            output=output,
        )
    )


async def _run_async(
    systems: list[str],
    dataset: str,
    track: str,
    limit: int,
    judge_model: str | None,
    pass_threshold: float,
    no_judge: bool,
    corpus: str | None,
    mode: str,
    output: Path | None,
) -> None:
    if track == "brain-only":
        await _run_brain_only(
            systems=systems,
            dataset=dataset,
            limit=limit,
            mode=BrainMode(mode),
            output=output,
        )
    elif track == "scale":
        await _run_scale(
            systems=systems,
            dataset=dataset,
            limit=limit,
            corpus_label=corpus or "unknown-corpus",
            output=output,
        )
    elif track == "agent-e2e":
        await _run_agent_e2e(
            systems=systems,
            dataset=dataset,
            limit=limit,
            judge_model=judge_model,
            pass_threshold=pass_threshold,
            output=output,
        )
    else:
        await _run_agent_oracle(
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
    limit: int,
    judge_model: str | None,
    pass_threshold: float,
    no_judge: bool,
    output: Path | None,
) -> None:
    settings = get_settings()
    ds = get_dataset(dataset)
    questions = list(ds.load(limit=limit))
    if not questions:
        console.print("[red]Dataset returned 0 questions.[/red]")
        return

    judge_obj = (
        _NoOpJudge(judge_model or settings.judge_model)
        if no_judge
        else LLMJudge(model=judge_model, pass_threshold=pass_threshold)
    )
    runner = AgentOracleRunner(systems=systems, judge=judge_obj)

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
        return

    _print_summary_table(summary)

    out_path = output or settings.results_dir / f"{summary.run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_to_json(summary, runner.results))
    console.print(f"\n[dim]→ {out_path}[/dim]")


async def _run_agent_e2e(
    systems: list[str],
    dataset: str,
    limit: int,
    judge_model: str | None,
    pass_threshold: float,
    output: Path | None,
) -> None:
    """Track 3 (agent E2E) — per-question corpus ingest via seed_docs, then judge."""
    settings = get_settings()
    ds = get_dataset(dataset)
    try:
        brain_questions = list(ds.load_brain_questions(limit=limit))
    except NotImplementedError as e:
        console.print(f"[red]Track 3 unavailable for dataset={dataset}: {e}[/red]")
        return

    if not brain_questions:
        console.print(
            "[red]Dataset returned 0 brain questions — cannot run agent-e2e track for "
            f"dataset={dataset}.[/red]"
        )
        return

    judge_obj = LLMJudge(model=judge_model, pass_threshold=pass_threshold)
    runner = AgentE2ERunner(systems=systems, judge=judge_obj)

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
        return

    _print_summary_table(summary)

    out_path = output or settings.results_dir / f"{runner.run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_to_json(summary, runner.results))
    console.print(f"\n[dim]→ {out_path}[/dim]")


async def _run_brain_only(
    systems: list[str],
    dataset: str,
    limit: int,
    mode: BrainMode,
    output: Path | None,
) -> None:
    """Track 1 (brain-only) runner.

    Calls `dataset.load_brain_questions()` directly — each Dataset subclass
    knows how to convert its native rows into BrainQuestion (corpus + gold
    doc paths). Datasets that don't yet support Track 1 (e.g. FRAMES, which
    needs an external Wikipedia corpus loader) raise NotImplementedError
    with a clear message.

    Compatible (dataset, mode) pairs run; incompatible ones are skipped with
    a [SKIP] log message.
    """
    settings = get_settings()
    ds = get_dataset(dataset)
    try:
        brain_questions = list(ds.load_brain_questions(limit=limit))
    except NotImplementedError as e:
        console.print(f"[red]Track 1 unavailable for dataset={dataset}: {e}[/red]")
        return

    if not brain_questions:
        console.print(
            "[red]Dataset returned 0 brain questions — cannot run brain-only track for "
            f"dataset={dataset}.[/red]"
        )
        return

    # Dataset x mode compatibility guard.
    if mode == BrainMode.WARM and dataset not in ("bitempoqa", "msmarco"):
        console.print(
            f"[SKIP] WARM mode is not meaningful for dataset={dataset} because each "
            "question has its own per-Q corpus (no shared pre-loaded corpus). "
            "Use COLD mode instead. Skipping."
        )
        return

    if mode == BrainMode.BITEMPORAL and dataset != "bitempoqa":
        console.print(
            f"[SKIP] BITEMPORAL mode requires as_of metadata — only bitempoqa supports "
            f"this. dataset={dataset} does not carry temporal metadata. Skipping."
        )
        return

    brain_adapters = {s: get_brain_adapter(s) for s in systems}
    runner = BrainRetrievalRunner(systems=brain_adapters, mode=mode)

    console.print(
        f"[bold]Run {runner.run_id}[/bold] · "
        f"track=brain-only · mode={mode.value} · dataset={dataset} · "
        f"n={len(brain_questions)} · systems={','.join(systems)}"
    )

    summary: BrainRunSummary | None = None
    async for ev in runner.run(brain_questions, dataset_name=dataset):
        if ev.type == "question_completed" and ev.result:
            r = ev.result
            recall = r.metrics.get("recall_at_10", 0.0)
            hit = r.metrics.get("hit_at_1", 0.0)
            mrr_val = r.metrics.get("mrr", 0.0)
            mark = "[green]✓[/green]" if hit > 0 else "[red]✗[/red]"
            extra = ""
            if mode == BrainMode.BITEMPORAL and "temporal_correct_at_1" in r.metrics:
                extra = f"  tc@1={r.metrics['temporal_correct_at_1']:.2f}"
            console.print(
                f"  {mark} {r.system:>14} {r.question_id:>14}  "
                f"recall@10={recall:.2f}  hit@1={hit:.2f}  mrr={mrr_val:.2f}{extra}"
                + (f"  [yellow]err: {r.error[:80]}[/yellow]" if r.error else "")
            )
        elif ev.type == "question_skipped" and ev.result:
            console.print(
                f"  [dim][SKIP] {ev.system} {ev.question_id}: {(ev.skip_reason or '')[:80]}[/dim]"
            )
        elif ev.type == "run_completed":
            summary = ev.summary
        elif ev.type in ("run_failed", "question_failed"):
            console.print(f"[red]{ev.type}: {ev.error}[/red]")

    if summary is None:
        console.print("[red]Run did not complete.[/red]")
        return

    _print_brain_summary_table(summary)

    out_path = output or settings.results_dir / f"{runner.run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_to_brain_json(summary, runner.results))
    console.print(f"\n[dim]→ {out_path}[/dim]")


async def _run_scale(
    systems: list[str],
    dataset: str,
    limit: int,
    corpus_label: str,
    output: Path | None,
) -> None:
    """Track 4 (scale) runner.

    Calls dataset.load_scale_questions() — dataset MUST implement this method.
    The corpus is assumed to be pre-loaded; no reset/ingest is performed.
    """
    settings = get_settings()
    ds = get_dataset(dataset)

    load_fn = getattr(ds, "load_scale_questions", None)
    if load_fn is None:
        raise click.UsageError(
            f"Dataset '{dataset}' does not support --track scale. "
            f"It must implement load_scale_questions(). "
            f"Currently supported: msmarco."
        )

    questions = list(load_fn(limit=limit))
    if not questions:
        console.print("[red]Dataset returned 0 scale questions.[/red]")
        return

    brain_adapters = {s: get_brain_adapter(s) for s in systems}
    runner = ScaleRetrievalRunner(
        systems=brain_adapters,
        corpus_label=corpus_label,
    )

    console.print(
        f"[bold]Run {runner.run_id}[/bold] · "
        f"track=scale · dataset={dataset} · corpus={corpus_label} · "
        f"n={len(questions)} · systems={','.join(systems)}"
    )

    scale_summary: ScaleRunSummary | None = None
    async for ev in runner.run(questions, dataset_name=dataset):
        if ev.type == "corpus_announced":
            console.print(f"  [dim]corpus: {ev.corpus_label}[/dim]")
        elif ev.type == "question_completed" and ev.result:
            r = ev.result
            recall = r.metrics.get("recall_at_10", 0.0)
            hit = r.metrics.get("hit_at_1", 0.0)
            mrr_val = r.metrics.get("mrr", 0.0)
            mark = "[green]✓[/green]" if hit > 0 else "[red]✗[/red]"
            console.print(
                f"  {mark} {r.system:>14} {r.question_id:>20}  "
                f"recall@10={recall:.2f}  hit@1={hit:.2f}  mrr={mrr_val:.2f}"
                + (f"  [yellow]err: {r.error[:80]}[/yellow]" if r.error else "")
            )
        elif ev.type == "run_completed":
            scale_summary = ev.summary
        elif ev.type in ("run_failed", "question_failed"):
            console.print(f"[red]{ev.type}: {ev.error}[/red]")

    if scale_summary is None:
        console.print("[red]Run did not complete.[/red]")
        return

    _print_scale_summary_table(scale_summary)

    out_path = output or settings.results_dir / f"{runner.run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_to_scale_json(scale_summary, runner.results))
    console.print(f"\n[dim]→ {out_path}[/dim]")


def _print_scale_summary_table(summary: ScaleRunSummary) -> None:
    table = Table(
        title=f"\n{summary.dataset} · scale · {summary.n_questions} Q · corpus={summary.corpus_label}"
    )
    table.add_column("System")
    table.add_column("Recall@10", justify="right")
    table.add_column("nDCG@10", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("Hit@1", justify="right")
    table.add_column("$/Q", justify="right")
    table.add_column("p50 ms", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("p99 ms", justify="right")
    for s in summary.summaries:
        table.add_row(
            s.system,
            f"{s.mean_recall_at_10:.3f}",
            f"{s.mean_ndcg_at_10:.3f}",
            f"{s.mean_mrr:.3f}",
            f"{s.mean_hit_at_1:.3f}",
            f"${s.total_cost_usd / max(s.n_questions, 1):.4f}",
            f"{s.p50_latency_ms:.0f}",
            f"{s.p95_latency_ms:.0f}",
            f"{s.p99_latency_ms:.0f}",
        )
    console.print(table)


def _to_scale_json(summary: ScaleRunSummary, results: list) -> str:
    """Serialize a Track 4 run as a single JSON file."""
    payload = {
        "summary": summary.model_dump(mode="json"),
        "results": [r.model_dump(mode="json") for r in results],
        "exported_at": datetime.now(UTC).isoformat(),
    }
    return json.dumps(payload, indent=2)


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


def _print_brain_summary_table(summary: BrainRunSummary) -> None:
    mode_label = summary.mode.value if hasattr(summary, "mode") else "cold"
    table = Table(
        title=f"\n{summary.dataset} · brain-only ({mode_label}) · {summary.n_questions} Q"
    )
    table.add_column("System")
    table.add_column("Recall@10", justify="right")
    table.add_column("nDCG@10", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("Hit@1", justify="right")
    has_temporal = any(s.mean_temporal_correct_at_1 is not None for s in summary.summaries)
    has_compaction = any(s.mean_compaction_quality is not None for s in summary.summaries)
    if has_temporal:
        table.add_column("TC@1", justify="right")
    if has_compaction:
        table.add_column("Compaction", justify="right")
    table.add_column("$/Q", justify="right")
    table.add_column("p50 ms", justify="right")
    for s in summary.summaries:
        row = [
            s.system,
            f"{s.mean_recall_at_10:.3f}",
            f"{s.mean_ndcg_at_10:.3f}",
            f"{s.mean_mrr:.3f}",
            f"{s.mean_hit_at_1:.3f}",
        ]
        if has_temporal:
            row.append(
                f"{s.mean_temporal_correct_at_1:.3f}"
                if s.mean_temporal_correct_at_1 is not None
                else "—"
            )
        if has_compaction:
            row.append(
                f"{s.mean_compaction_quality:.3f}"
                if s.mean_compaction_quality is not None
                else "N/A"
            )
        row.extend(
            [
                f"${s.total_cost_usd / max(s.n_questions, 1):.4f}",
                f"{s.p50_latency_ms:.0f}",
            ]
        )
        table.add_row(*row)
    console.print(table)


def _to_brain_json(summary: BrainRunSummary, results: list) -> str:
    """Serialize a Track 1 run as a single JSON file."""
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


# CLI for SystemSummary type-import linting hygiene
_SYSTEM_SUMMARY_TYPE: type = SystemSummary

if __name__ == "__main__":  # pragma: no cover
    main()
