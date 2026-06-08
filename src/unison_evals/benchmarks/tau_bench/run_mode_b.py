"""Phase-2 entry: run Mode B (Unison via bash + .md) on the same 3 retail
tasks the Phase-1 mode-A smoke covered.

Usage:
    .venv/bin/python -m unison_evals.benchmarks.tau_bench.run_mode_b

Prerequisites:
    - Local Unison API running at $UNISON_API_URL (default http://localhost:3001)
    - Eval tenant provisioned (see your Unison server `.env` for UNISON_LOCAL_EVAL_TENANT_ID/USER_ID)
    - UNISON_LOCAL_EVAL_TENANT_ID + USER_ID set in Unison's .env, API restarted
    - OPENAI_API_KEY (user-sim) + ANTHROPIC_API_KEY (agent inside Unison)
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

from tau_bench.envs import get_env  # noqa: E402

from .mode_b_agent import UnisonModeBAgent  # noqa: E402

TENANT_ID = os.environ.get("UNISON_LOCAL_EVAL_TENANT_ID", "")
USER_ID = os.environ.get("UNISON_LOCAL_EVAL_USER_ID", "")
USER_SIM_MODEL = "gpt-4o"
USER_SIM_PROVIDER = "openai"


def _parse_task_ids(raw: str | None) -> list[int]:
    if raw is None or raw.strip() == "":
        return [0, 1, 2]
    if raw.strip().lower() == "all":
        return list(range(115))
    return [int(t.strip()) for t in raw.split(",") if t.strip()]


def _resolve_args() -> tuple[str, list[int], str]:
    p = argparse.ArgumentParser(
        prog="run_mode_b", description="Mode B — τ-bench treatment cell (Unison bash + .md)."
    )
    p.add_argument(
        "-n",
        "--num-tasks",
        type=int,
        default=None,
        help="Number of tasks starting at task 0 (e.g. -n 25 → tasks 0..24). Overrides --task-ids and TAUBENCH_TASK_IDS.",
    )
    p.add_argument(
        "--task-ids", type=str, default=None, help="Explicit task IDs, comma-separated, or 'all'."
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Agent model id. Default: $TAUBENCH_AGENT_MODEL or claude-sonnet-4-5.",
    )
    p.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="Unison API base URL. Default: $UNISON_API_URL or http://localhost:3001.",
    )
    a = p.parse_args()

    model = (
        a.model
        or os.environ.get("TAUBENCH_AGENT_MODEL")
        or os.environ.get("UNISON_AGENT_MODEL")
        or "claude-sonnet-4-5"
    )
    api_url = a.api_url or os.environ.get("UNISON_API_URL") or "http://localhost:3001"

    if a.num_tasks is not None:
        task_ids = list(range(a.num_tasks))
    elif a.task_ids is not None:
        task_ids = _parse_task_ids(a.task_ids)
    else:
        task_ids = _parse_task_ids(os.environ.get("TAUBENCH_TASK_IDS"))

    return model, task_ids, api_url


def main() -> int:
    agent_model, task_ids, api_url = _resolve_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY missing (needed by user-simulator)", file=sys.stderr)
        return 1

    # Per-model log dir so the paired comparison's artefacts don't clobber.
    log_dir = _REPO_ROOT / "results" / "tau-bench" / "smoke-mode-b" / agent_model.replace("/", "_")
    log_dir.mkdir(parents=True, exist_ok=True)

    env = get_env(
        "retail",
        user_strategy="llm",
        user_model=USER_SIM_MODEL,
        user_provider=USER_SIM_PROVIDER,
        task_split="test",
    )

    agent = UnisonModeBAgent(
        unison_api_url=api_url,
        tenant_id=TENANT_ID,
        user_id=USER_ID,
        model=agent_model,
        trajectory_dir=log_dir / "trajectories",
    )

    print(f"Running τ-bench Mode B (Unison) — tasks {task_ids}")
    print(f"  Unison API: {api_url}")
    print(f"  Agent model: {agent_model}")
    print(f"  User-sim: {USER_SIM_PROVIDER}/{USER_SIM_MODEL}")
    print(f"  Tenant: {TENANT_ID}")
    print()

    results = []
    for task_index in task_ids:
        print(f"─── Task {task_index} ───────────────────────────────────")
        try:
            sr = agent.solve(env, task_index=task_index, max_num_steps=20)
            results.append(
                {
                    "task_id": task_index,
                    "reward": sr.reward,
                    "total_cost": sr.total_cost,
                    "num_messages": len(sr.messages),
                }
            )
            print(
                f"  → reward={sr.reward:.2f}  cost=${sr.total_cost:.4f}  messages={len(sr.messages)}"
            )
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            import traceback

            traceback.print_exc()
            results.append({"task_id": task_index, "reward": 0.0, "error": str(e)})
        print()

    agent.close()

    rewards = [r["reward"] for r in results]
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    print("─── Mode B summary ────────────────────────────────")
    for r in results:
        print(
            f"  task {r['task_id']:>3}  reward {r['reward']:.2f}  cost ${r.get('total_cost', 0):.4f}"
        )
    print(f"  n={len(rewards)}  mean reward = {mean:.3f}")

    summary = {
        "mode": "B",
        "tenant_id": TENANT_ID,
        "agent_model": agent_model,
        "user_sim": f"{USER_SIM_PROVIDER}/{USER_SIM_MODEL}",
        "task_ids": task_ids,
        "mean_reward": mean,
        "results": results,
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWritten to {log_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
