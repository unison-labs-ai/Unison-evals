"""Mode A τ-bench runner — control cell for the interface ablation.

Runs τ-bench's stock tool-calling agent against retail tasks. The user
simulator is locked to `gpt-4o` (the upstream-leaderboard standard, so
external comparisons stay comparable). The agent model is configurable
via env vars:

    TAUBENCH_AGENT_MODEL      e.g. claude-sonnet-4-5  (default), gpt-4o
    TAUBENCH_AGENT_PROVIDER   anthropic | openai | … (auto-derived if unset)
    TAUBENCH_TASK_IDS         "0,1,2" (default) or "all" for the full 115

The interface ablation requires running this AND `run_mode_b.py` with the
same `TAUBENCH_AGENT_MODEL`, on the same task set, and comparing pairwise.

Usage:
    # Cell A1 — control: Sonnet 4.5 + native function-calls
    .venv/bin/python -m unison_evals.benchmarks.tau_bench.smoke

    # gpt-4o + native function-calls (legacy / leaderboard anchor)
    TAUBENCH_AGENT_MODEL=gpt-4o \
        .venv/bin/python -m unison_evals.benchmarks.tau_bench.smoke

Required env (set in the repo's .env):
    OPENAI_API_KEY     # user simulator + agent if openai-family
    ANTHROPIC_API_KEY  # agent if claude-family
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env so OPENAI_API_KEY / ANTHROPIC_API_KEY are available to litellm.
_REPO_ROOT = Path(__file__).resolve().parents[4]
load_dotenv(_REPO_ROOT / ".env")

from tau_bench.run import run  # noqa: E402  (after dotenv)
from tau_bench.types import RunConfig  # noqa: E402


def _derive_provider(model: str) -> str:
    """Map a model id to the litellm provider name."""
    m = model.lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if m.startswith("gemini"):
        return "google_ai_studio"
    if m.startswith("mistral"):
        return "mistral"
    raise ValueError(
        f"Cannot derive provider for model {model!r}; set TAUBENCH_AGENT_PROVIDER explicitly"
    )


def _parse_task_ids(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return [0, 1, 2]
    if raw.strip().lower() == "all":
        return None  # tau-bench runs all when task_ids is None
    return [int(t.strip()) for t in raw.split(",") if t.strip()]


USER_SIM_MODEL = "gpt-4o"
USER_SIM_PROVIDER = "openai"


def _build_config(model: str, provider: str | None, task_ids: list[int] | None) -> RunConfig:
    """Construct the RunConfig given CLI/env-resolved knobs."""
    prov = provider or _derive_provider(model)
    log_dir = _REPO_ROOT / "results" / "tau-bench" / "smoke" / model.replace("/", "_")
    return RunConfig(
        model_provider=prov,
        user_model_provider=USER_SIM_PROVIDER,
        model=model,
        user_model=USER_SIM_MODEL,
        num_trials=1,
        env="retail",
        agent_strategy="tool-calling",
        temperature=0.0,
        task_split="test",
        start_index=0,
        end_index=-1,
        task_ids=task_ids,
        log_dir=str(log_dir),
        max_concurrency=1,
        seed=10,
        shuffle=0,
        user_strategy="llm",
        few_shot_displays_path=None,
    )


def _resolve_args() -> tuple[str, str | None, list[int] | None]:
    """CLI flags override env vars override defaults."""
    p = argparse.ArgumentParser(
        prog="smoke", description="Mode A — τ-bench native tool-calling control cell."
    )
    p.add_argument(
        "-n",
        "--num-tasks",
        type=int,
        default=None,
        help="Number of tasks to run, starting at task 0 (e.g. -n 25 → tasks 0..24). "
        "Overrides --task-ids and TAUBENCH_TASK_IDS.",
    )
    p.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="Comma-separated explicit task IDs, e.g. 0,1,5,10. Or 'all' for the full 115. "
        "Overrides TAUBENCH_TASK_IDS.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Agent model id. Default: $TAUBENCH_AGENT_MODEL or claude-sonnet-4-5.",
    )
    p.add_argument(
        "--provider",
        type=str,
        default=None,
        help="litellm provider name. Default: auto-derived from --model.",
    )
    a = p.parse_args()

    model = a.model or os.environ.get("TAUBENCH_AGENT_MODEL") or "claude-sonnet-4-5"
    provider = a.provider or os.environ.get("TAUBENCH_AGENT_PROVIDER")

    if a.num_tasks is not None:
        task_ids: list[int] | None = list(range(a.num_tasks))
    elif a.task_ids is not None:
        task_ids = _parse_task_ids(a.task_ids)
    else:
        task_ids = _parse_task_ids(os.environ.get("TAUBENCH_TASK_IDS"))

    return model, provider, task_ids


def main() -> int:
    model, provider, task_ids = _resolve_args()
    config = _build_config(model, provider, task_ids)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set (needed for the gpt-4o user-sim)", file=sys.stderr)
        return 1
    if config.model_provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY not set (needed because agent is a claude model)",
            file=sys.stderr,
        )
        return 1

    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    task_str = (
        "all 115"
        if config.task_ids is None
        else f"{len(config.task_ids)} tasks "
        + (
            f"[{config.task_ids[0]}..{config.task_ids[-1]}]"
            if len(config.task_ids) > 5
            else str(config.task_ids)
        )
    )
    print(f"τ-bench Mode A (control: native tool-calling) — {task_str}")
    print(f"Agent:    {config.model_provider}/{config.model}")
    print(f"User-sim: {config.user_model_provider}/{config.user_model}")
    print(f"Log dir:  {log_dir}")
    print()

    results = run(config)

    # Summary
    n = len(results)
    rewards = [r.reward for r in results]
    mean_reward = sum(rewards) / n if n else 0.0
    print()
    print("─── Smoke results ─────────────────────────────────")
    for r in results:
        print(f"  task {r.task_id:>3}  trial {r.trial}  reward {r.reward:.2f}  info={r.info}")
    print("───────────────────────────────────────────────────")
    print(f"  n = {n},  mean reward = {mean_reward:.3f}")

    # Persist a flat summary alongside τ-bench's own checkpoint JSON.
    summary = {
        "config": config.model_dump(),
        "n": n,
        "mean_reward": mean_reward,
        "rewards": rewards,
        "results": [json.loads(r.model_dump_json()) for r in results],
    }
    summary_path = log_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
