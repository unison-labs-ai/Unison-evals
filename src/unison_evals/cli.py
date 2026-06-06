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
from .types import RunSummary, SystemSummary

console = Console()

# The one benchmark list + each one's publishable ("real") canonical judge.
BENCHMARKS = ("longmemeval", "memoryagentbench", "context-bench")
CANONICAL_JUDGE = {
    "longmemeval": "gpt-4o-2024-08-06",  # LongMemEval paper judge (>97% human agreement)
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
    mode = "REAL (canonical, publishable)" if (real_mode or judge) else "DEV (cheap Gemini)"
    for ds in datasets:
        if len(datasets) > 1:
            click.echo(f"\n=== {ds} ===")

        # Context-Bench has its own per-run runner (fixed corpus, Letta rubric).
        # It always judges; reuse the same judge-resolution rule.
        if ds == "context-bench":
            from .benchmarks.context_bench.run import run_context_bench

            cb_judge = _resolve_judge(ds, judge, real_mode) or CANONICAL_JUDGE["context-bench"]
            click.echo(f"  Judge: {cb_judge}  [{mode}]")
            run_context_bench(
                limit=limit,
                judge_model=cb_judge,
                agent_model=get_settings().unison_agent_model or None,
            )
            continue

        # Memory benches (LongMemEval, MemoryAgentBench).
        if track == "agent-e2e" and no_judge:
            raise click.UsageError("--no-judge is not supported with --track agent-e2e")
        resolved_judge = None if no_judge else _resolve_judge(ds, judge, real_mode)
        if not no_judge:
            click.echo(f"  Judge: {resolved_judge or '(env / config default)'}  [{mode}]")
        asyncio.run(
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


async def _run_async(
    systems: list[str],
    dataset: str,
    track: str,
    limit: int | None,
    judge_model: str | None,
    pass_threshold: float,
    no_judge: bool,
    output: Path | None,
) -> None:
    if track == "agent-e2e":
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
    limit: int | None,
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
    limit: int | None,
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


# CLI for SystemSummary type-import linting hygiene
_SYSTEM_SUMMARY_TYPE: type = SystemSummary

if __name__ == "__main__":  # pragma: no cover
    main()
