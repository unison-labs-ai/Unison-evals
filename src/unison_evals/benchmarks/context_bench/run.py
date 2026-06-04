"""Context-Bench runner — Unison cell.

Loads Letta's filesystem_cloud.jsonl, runs each question through
`UnisonContextBenchTarget`, grades the answers with the published
rubric + judge model, and persists a paired summary.

Usage:
    .venv/bin/python -m unison_evals.benchmarks.context_bench.run -n 5
    .venv/bin/python -m unison_evals.benchmarks.context_bench.run --task-ids 0,3,7

Comparison cell: Letta's published Sonnet 4.5 + open_files/grep_files
on this exact dataset = 74.0% (leaderboard.letta.com).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[4]
load_dotenv(_REPO_ROOT / ".env")

from . import judge  # noqa: E402
from .target import UnisonContextBenchTarget  # noqa: E402

DATASET_PATH = (
    _REPO_ROOT
    / "vendor"
    / "letta-evals"
    / "letta-leaderboard"
    / "filesystem-agent"
    / "datasets"
    / "filesystem_cloud.jsonl"
)


def _load_rows() -> list[dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Letta dataset not vendored. Expected at {DATASET_PATH}. "
            "Run `git submodule update --init vendor/letta-evals`."
        )
    rows: list[dict] = []
    for line in DATASET_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _git_sha(path: Path) -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _submodule_sha(path: Path, sub: str) -> str | None:
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "-C", str(path), "submodule", "status", sub],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out.split()[0].lstrip("+-U") if out else None
    except Exception:
        return None


def _manifest(model: str, judge_label: str, tenant_id: str | None) -> dict:
    """Reproducibility manifest — pinned to the run so a published number can be
    reproduced and audited (commit SHAs, models, dataset revision, isolation)."""
    from datetime import datetime, timezone

    return {
        "isolation": "per-run ephemeral tenant (ADR-0008)",
        "evals_commit": _git_sha(_REPO_ROOT),
        "letta_evals_submodule": _submodule_sha(_REPO_ROOT, "vendor/letta-evals"),
        "dataset": "vendor/letta-evals/.../filesystem_cloud.jsonl",
        "agent_model": model,
        "judge": judge_label,
        "memory_mode": "fresh",
        "unison_api_url": os.environ.get("UNISON_API_URL", "http://localhost:3001"),
        "ephemeral_tenant_id": tenant_id,
        "run_utc": datetime.now(timezone.utc).isoformat(),
    }


def _resolve_args() -> tuple[list[int] | None, str, str]:
    """Resolve CLI flags. Returns task_ids (None means 'run all rows';
    `[]` is rejected upstream as invalid, never used as a sentinel)."""
    p = argparse.ArgumentParser(
        prog="context_bench.run",
        description="Unison cell on Letta's Context-Bench filesystem suite.",
    )
    p.add_argument(
        "-n",
        "--num-tasks",
        type=int,
        default=None,
        help="Run the first N rows (tasks 0..N-1). Must be >= 1.",
    )
    p.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="Comma-separated row indices (0-based, non-negative), or 'all'.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Agent model. Default: $TAUBENCH_AGENT_MODEL or claude-sonnet-4-5.",
    )
    p.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="LLM-judge model. Default: $JUDGE_MODEL or gpt-5-mini (Letta's leaderboard default).",
    )
    a = p.parse_args()

    model = a.model or os.environ.get("TAUBENCH_AGENT_MODEL") or "claude-sonnet-4-5"
    # Judge model: CLI flag > leaderboard default. We deliberately do NOT
    # honour $JUDGE_MODEL here — it's a holdover from the memory-evals
    # harness and silently breaks the apples-to-apples comparison with
    # Letta's leaderboard cell. Set --judge-model on the CLI if you really
    # want to swap it.
    judge_model = a.judge_model or judge.DEFAULT_JUDGE_MODEL

    # task_ids resolution. Sentinel for "run all rows" is `None` (filled
    # after dataset load), never an empty list. -n must be >= 1; explicit
    # task IDs must be non-negative (Python's reverse-indexing semantics
    # would silently pick the wrong row otherwise).
    task_ids: list[int] | None
    if a.num_tasks is not None:
        if a.num_tasks < 1:
            p.error(f"--num-tasks must be >= 1, got {a.num_tasks}")
        task_ids = list(range(a.num_tasks))
    elif a.task_ids and a.task_ids.strip().lower() == "all":
        task_ids = None  # filled to range(len(rows)) after load
    elif a.task_ids:
        try:
            parsed = [int(t.strip()) for t in a.task_ids.split(",") if t.strip()]
        except ValueError as e:
            p.error(f"--task-ids must be comma-separated integers: {e}")
        if any(t < 0 for t in parsed):
            p.error(f"--task-ids must be non-negative (0-based), got {parsed}")
        if not parsed:
            p.error("--task-ids is empty after parsing")
        task_ids = parsed
    else:
        task_ids = [0, 1, 2]

    return task_ids, model, judge_model


def main() -> int:
    task_ids, model, judge_model = _resolve_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY missing (needed by the judge)", file=sys.stderr)
        return 1
    if model.startswith("claude") and not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY missing (needed by the claude agent inside Unison)",
            file=sys.stderr,
        )
        return 1

    rows = _load_rows()
    # `None` is the "run all" sentinel from _resolve_args; an empty list
    # cannot occur (the parser rejects it).
    if task_ids is None:
        task_ids = list(range(len(rows)))
    if max(task_ids) >= len(rows):
        print(f"ERROR: task id {max(task_ids)} >= dataset size {len(rows)}", file=sys.stderr)
        return 1

    log_dir = _REPO_ROOT / "results" / "context-bench" / "unison" / model.replace("/", "_")
    log_dir.mkdir(parents=True, exist_ok=True)

    target = UnisonContextBenchTarget(model=model)
    judge_provider = judge.derive_provider(judge_model)
    judge_label = f"{judge_provider}/{judge_model}"

    # Pre-flight: confirm the judge model is actually reachable with one
    # tiny call. Better to fail fast than 100 rows in.
    try:
        judge.grade("preflight question", "preflight gt", "preflight gt", model=judge_model)
    except Exception as e:
        print(f"ERROR: judge model {judge_model!r} unreachable: {e}", file=sys.stderr)
        return 1

    print("Context-Bench / filesystem suite — Unison cell")
    print(f"  Agent model: {model}")
    print(f"  Judge model: {judge_label}")
    print(f"  Tasks:       {len(task_ids)} row(s) (idx {task_ids[0]}..{task_ids[-1]})")
    print(f"  Log dir:     {log_dir}")
    print()

    results: list[dict] = []
    total_score = 0.0
    total_cost = 0.0

    # ADR-0008 per-run isolation: provision one ephemeral tenant + seed the
    # fixed corpus once. Guaranteed teardown in `finally` so a crash mid-run
    # never leaks the tenant.
    try:
        target.setup()
    except Exception as e:
        print(f"ERROR: provision/seed failed: {e}", file=sys.stderr)
        target.close()
        return 1
    ran_tenant_id = target.tenant_id  # capture before close() nulls it
    print(f"  Tenant:      {ran_tenant_id} (ephemeral, is_eval) — seeded {target.seeded_pages} pages")
    print()

    try:
        for i in task_ids:
            row = rows[i]
            question = row["input"]
            gt = row["ground_truth"]
            meta = row.get("agent_args", {}).get("extra", {}) or {}
            print(
                f"─── Row {i} ({meta.get('question_type', '?')}/{meta.get('difficulty', '?')}) ───"
            )
            print(f"  Q: {question[:140]}{'…' if len(question) > 140 else ''}")
            print(f"  GT: {gt}")

            try:
                ans = target.ask(question)
            except Exception as e:
                print(f"  TARGET ERROR: {e}")
                results.append({"task_id": i, "score": 0.0, "error": str(e)})
                continue

            print(
                f"  → {ans.elapsed_s:.1f}s, agent-steps={ans.total_steps}, cost=${ans.total_cost_usd:.4f}"
            )
            print(f"  Answer: {ans.answer[:140]}{'…' if len(ans.answer) > 140 else ''}")

            # Charge agent spend as soon as the agent has actually run. If the
            # judge subsequently fails the cost still counts — otherwise
            # total_agent_cost_usd silently undercounts whenever the judge API
            # is flaky.
            total_cost += ans.total_cost_usd

            try:
                score, raw = judge.grade(question, gt, ans.answer, model=judge_model)
            except Exception as e:
                print(f"  JUDGE ERROR: {e}")
                results.append(
                    {
                        "task_id": i,
                        "score": 0.0,
                        "agent_answer": ans.answer,
                        "agent_cost_usd": ans.total_cost_usd,
                        "error": str(e),
                    }
                )
                continue

            print(f"  Score: {score}  (judge raw: {raw[:60]!r})")
            total_score += score
            results.append(
                {
                    "task_id": i,
                    "question": question,
                    "ground_truth": gt,
                    "agent_answer": ans.answer,
                    "score": score,
                    "judge_raw": raw,
                    "elapsed_s": ans.elapsed_s,
                    "agent_steps": ans.total_steps,
                    "agent_cost_usd": ans.total_cost_usd,
                    "question_type": meta.get("question_type"),
                    "difficulty": meta.get("difficulty"),
                }
            )
            # Persist progress per-row so a crash doesn't waste the whole run.
            (log_dir / f"row-{i:03d}.json").write_text(json.dumps(results[-1], indent=2))
            print()
    finally:
        target.close()

    n = len(results)
    pct = (total_score / n * 100) if n else 0.0
    summary = {
        "benchmark": "context-bench-filesystem",
        "cell": "unison-bash-md",
        "agent_model": model,
        "judge_model": judge_model,
        "n": n,
        "score_sum": total_score,
        "mean_score": (total_score / n) if n else 0.0,
        "pct": pct,
        "total_agent_cost_usd": total_cost,
        "task_ids": task_ids,
        "results": results,
        "manifest": _manifest(model, judge_label, ran_tenant_id),
        # leaderboard.letta.com, Filesystem suite, as of 2026-03-13. Same dataset,
        # same gpt-5-mini judge + rubric — only the agent interface differs.
        "comparator_cells": {
            "letta_agent_gpt_5_2_codex": 0.93,
            "letta_agent_gpt_5_4": 0.89,
            "letta_agent_sonnet_4_6": 0.88,
        },
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("─── Summary ──────────────────────────────────────────")
    print(f"  Unison cell ({model}):   {total_score:.1f}/{n} = {pct:.1f}%")
    print("  Letta cell (Sonnet 4.6): 88.0%  (leaderboard.letta.com, 2026-03-13)")
    print(f"  Δ vs Letta Sonnet 4.6:   {pct - 88.0:+.1f}pp  (top model GPT-5.2-codex: 93%)")
    print(f"  Total agent cost:        ${total_cost:.3f}")
    print(f"  Written:                 {log_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
