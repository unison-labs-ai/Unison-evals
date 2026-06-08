"""Context-Bench runner — Unison cell.

Loads Letta's filesystem_cloud.jsonl, runs each question through
`UnisonContextBenchTarget`, grades the answers with the published
rubric + judge model, and persists a paired summary.

Entry point: the unified CLI dispatches here —
    unison-evals run --dataset context-bench --limit N [--dev|--real]
via run_context_bench(). No standalone CLI (one command for all benchmarks).

Comparison cell: Letta's published leaderboard on this exact dataset
(leaderboard.letta.com).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[4]
load_dotenv(_REPO_ROOT / ".env")

from datetime import UTC  # noqa: E402

from ...config import get_settings  # noqa: E402
from ...results import new_run_id, results_path  # noqa: E402
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
    from datetime import datetime

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
        "run_utc": datetime.now(UTC).isoformat(),
    }


def run_context_bench(
    *,
    limit: int | None,
    judge_model: str,
    agent_model: str | None = None,
) -> dict:
    """Run the Context-Bench filesystem suite as the Unison cell.

    limit None = all rows; else the first `limit` rows. agent_model None = the
    server's production model path (auto + escalation), like a live user turn —
    pin one only for an explicit ablation. judge_model is resolved by the caller
    (the unified CLI's --dev/--real). Returns the summary dict.
    """
    rows = _load_rows()
    n_rows = len(rows) if limit is None else max(1, min(limit, len(rows)))
    task_ids = list(range(n_rows))

    model_label = agent_model or "auto-prod"
    run_id = new_run_id("context-bench")
    out_path = results_path(run_id)
    rows_dir = get_settings().results_dir / f"{run_id}.rows"  # per-row crash recovery
    rows_dir.mkdir(parents=True, exist_ok=True)

    target = UnisonContextBenchTarget(model=agent_model)
    judge_provider = judge.derive_provider(judge_model)
    judge_label = f"{judge_provider}/{judge_model}"

    # Pre-flight: confirm the judge model is reachable with one tiny call (also
    # surfaces a missing judge API key). Better to fail fast than 100 rows in.
    try:
        judge.grade("preflight question", "preflight gt", "preflight gt", model=judge_model)
    except Exception as e:
        raise RuntimeError(f"judge model {judge_model!r} unreachable: {e}") from e

    print("Context-Bench / filesystem suite — Unison cell")
    print(f"  Agent model: {model_label}")
    print(f"  Judge model: {judge_label}")
    print(f"  Tasks:       {len(task_ids)} row(s) (idx {task_ids[0]}..{task_ids[-1]})")
    print(f"  Output:      {out_path}")
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
        target.close()
        raise RuntimeError(f"provision/seed failed: {e}") from e
    ran_tenant_id = target.tenant_id  # capture before close() nulls it
    print(
        f"  Tenant:      {ran_tenant_id} (ephemeral, is_eval) — seeded {target.seeded_pages} pages"
    )
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
            (rows_dir / f"row-{i:03d}.json").write_text(json.dumps(results[-1], indent=2))
            print()
    finally:
        target.close()

    n = len(results)
    pct = (total_score / n * 100) if n else 0.0
    summary = {
        "benchmark": "context-bench",
        "run_id": run_id,
        "cell": "unison-bash-md",
        "agent_model": model_label,
        "judge_model": judge_model,
        "n": n,
        "score_sum": total_score,
        "mean_score": (total_score / n) if n else 0.0,
        "pct": pct,
        "total_agent_cost_usd": total_cost,
        "task_ids": task_ids,
        "results": results,
        "manifest": _manifest(model_label, judge_label, ran_tenant_id),
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print("─── Summary ──────────────────────────────────────────")
    print(f"  Score:             {total_score:.1f}/{n} = {pct:.1f}%")
    print(f"  Total agent cost:  ${total_cost:.3f}")
    print(f"  Written:           {out_path}")
    return summary
