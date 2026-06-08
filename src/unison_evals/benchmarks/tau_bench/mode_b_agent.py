"""Mode B agent — drives the τ-bench multi-turn loop against Unison.

For each task:
  1. Wipe + reseed brain under the eval tenant.
  2. Open a fresh Unison session.
  3. Loop:
       a. POST /api/rest/agents/eval-turn  (question = current user message)
       b. Snapshot the brain.
       c. Diff vs. previous snapshot → translate to τ-bench Actions.
       d. env.step() each translated action (policy enforcement here).
       e. env.step(respond, agent's text) → user-sim's next message.
       f. Break on env_response.done or max_num_steps.

The agent itself is unaware τ-bench exists — it just gets a user message,
reads /private/taubench/, mutates /private/taubench/, replies. All τ-bench-specific glue lives
in the translator, never inside Unison.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import RESPOND_ACTION_NAME, Action, SolveResult

from . import action_translator, brain_client, md_overlay


class UnisonModeBAgent(Agent):
    """τ-bench Agent that uses Unison (bash + .md filesystem) instead of
    typed function-call tools."""

    def __init__(
        self,
        unison_api_url: str,
        tenant_id: str,
        user_id: str,
        model: str = "claude-sonnet-4-5",
        timeout: float = 600.0,
        trajectory_dir: Path | None = None,
    ):
        self.api_url = unison_api_url.rstrip("/")
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.model = model
        self.timeout = timeout
        # If set, every task writes a JSONL trajectory (dispatched actions +
        # full agent_messages reconstruction) into this dir so wiring can
        # be verified post-hoc without modifying Unison.
        self.trajectory_dir = trajectory_dir
        if trajectory_dir is not None:
            trajectory_dir.mkdir(parents=True, exist_ok=True)
        # Local-bypass: no Authorization header.
        self._client = httpx.Client(
            base_url=self.api_url,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )

    def solve(
        self,
        env: Env,
        task_index: int | None = None,
        max_num_steps: int = 30,
    ) -> SolveResult:
        total_cost = 0.0
        reset_resp = env.reset(task_index=task_index)
        initial_user_msg = reset_resp.observation
        info: dict[str, Any] = reset_resp.info.model_dump()

        # 1. Wipe every tenant-scoped table the system touches, then reseed
        # brain. This gives task N a true iid starting state — no leaked facts,
        # messages, or pending jobs from tasks 1..N-1.
        wipe_counts = brain_client.wipe_tenant_sync(self.tenant_id)
        wiped_total = sum(wipe_counts.values())
        pages = md_overlay.build_seed_pages(env.data)
        seeded = brain_client.seed_pages_sync(self.tenant_id, self.user_id, pages)
        print(
            f"  [mode-b] wiped {wiped_total} rows; seeded {seeded} fresh",
            flush=True,
        )

        # 2. Prime session
        session_id = str(uuid.uuid4())
        snapshot_before = brain_client.snapshot_wiki_sync(self.tenant_id)
        parsed_before = md_overlay.parse_snapshot(snapshot_before)

        # The Unison agent has no per-call system-prompt slot, so we inject
        # the customer-service framing AND the τ-bench retail policy verbatim
        # into the first user message. This mirrors Mode A, which gets the
        # same policy text as `messages[0].content` (system role). Putting
        # policy here, not in /private/taubench/policy.md, gives it the same
        # authoritative framing both modes get — without it the agent reads
        # policy as advisory text and over-applies the "verify identity"
        # rules to refuse legitimate authenticated exchanges.
        first_question = (
            "You are a customer service agent for an online retail store. "
            "Your workspace is at /private/taubench/ — start by reading /private/taubench/SCHEMA.md "
            "to learn the layout and the navigation recipes. Help the "
            "customer below by reading and (when needed) editing files "
            "under /private/taubench/orders/, /private/taubench/users/, /private/taubench/products/. Reply "
            "directly to the customer in your final message.\n\n"
            "═══ STORE POLICY (binding — follow exactly) ═══\n"
            f"{env.wiki}\n"
            "═══ END POLICY ═══\n\n"
            f"CUSTOMER:\n{initial_user_msg}"
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": initial_user_msg},
        ]
        env_response = None
        question = first_question
        reward = 0.0
        # Every action our translator dispatches through env.step() — captured
        # for post-hoc verification (e.g., did Sonnet pick the GT new_item_ids?).
        dispatched_actions: list[dict[str, Any]] = []

        for step in range(max_num_steps):
            # 3a. Call Unison
            t0 = time.monotonic()
            try:
                resp = self._client.post(
                    "/api/rest/agents/eval-turn",
                    json={
                        "question": question,
                        "sessionId": session_id,
                        "model": self.model,
                        # Skip Memory-v2 extract.turn jobs — iid evals must
                        # not accumulate facts across tasks. Server-side
                        # support is the Unison V2 ask (gracefully ignored
                        # by older builds; we wipe tenant state anyway).
                        "memoryMode": "fresh",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"  [mode-b] eval-turn ERROR: {e}", flush=True)
                break
            elapsed = time.monotonic() - t0

            agent_answer: str = data.get("answer") or ""
            total_cost += float(data.get("totalCostUsd") or 0.0)
            print(
                f"  [mode-b] step {step}: {elapsed:.1f}s, "
                f"steps={data.get('totalSteps')}, "
                f"cost+=${data.get('totalCostUsd'):.4f}, "
                f"answer={agent_answer[:120]!r}",
                flush=True,
            )

            # 3b. Snapshot brain
            snapshot_after = brain_client.snapshot_wiki_sync(self.tenant_id)
            parsed_after = md_overlay.parse_snapshot(snapshot_after)

            # 3c. Translate brain diff → tau_bench Actions
            translated, unmapped = action_translator.translate(parsed_before, parsed_after)
            for u in unmapped:
                print(f"  [mode-b]   unmapped: {u}", flush=True)

            # 3d. Apply each translated action via env.step
            for ta in translated:
                # Verbose action log — kwargs are the load-bearing fact for
                # wiring verification. Without them we cannot tell whether
                # a reward=0 came from a translator bug, a wrong agent
                # variant pick, or an env.step rejection.
                print(
                    f"  [mode-b]   → DISPATCH {ta.action.name} kwargs={json.dumps(ta.action.kwargs)}",
                    flush=True,
                )
                dispatched_actions.append(
                    {
                        "name": ta.action.name,
                        "kwargs": ta.action.kwargs,
                        "reason": ta.reason,
                        "source_path": ta.source_path,
                    }
                )
                env_response = env.step(ta.action)
                obs = str(env_response.observation)
                ok = "✓" if not obs.lower().startswith("error") else "✗"
                dispatched_actions[-1]["env_response"] = {
                    "ok": ok == "✓",
                    "observation_preview": obs[:300],
                    "reward": env_response.reward,
                    "done": env_response.done,
                }
                print(
                    f"  [mode-b]   {ok} {ta.action.name}({ta.reason}) → {obs[:100]}",
                    flush=True,
                )
                reward = env_response.reward
                if env_response.done:
                    break

            if env_response is not None and env_response.done:
                break

            parsed_before = parsed_after

            # 3e. Forward agent's reply to user-sim
            respond_action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": agent_answer})
            env_response = env.step(respond_action)
            reward = env_response.reward
            messages.append({"role": "assistant", "content": agent_answer})
            user_reply = env_response.observation
            messages.append({"role": "user", "content": user_reply})

            if env_response.done:
                print(f"  [mode-b] user said STOP at step {step}", flush=True)
                break

            # Subsequent turns: just the user message (session retains framing).
            question = user_reply

        if env_response is not None:
            info = {**info, **env_response.info.model_dump()}

        # Persist a wiring-verification trajectory: full agent_messages
        # reconstruction (every bash command + output Sonnet executed) +
        # every translated Action we dispatched. Eval-side only — no
        # Unison modifications. Lets us answer "did the agent actually
        # cat the user file?" and "what new_item_ids did we dispatch?"
        # post-hoc.
        if self.trajectory_dir is not None:
            try:
                trace = brain_client.dump_trajectory_sync(self.tenant_id, session_id)
            except Exception as e:
                print(f"  [mode-b] trajectory dump failed: {e}", flush=True)
                trace = []
            payload = {
                "task_index": task_index,
                "session_id": session_id,
                "reward": reward,
                "total_cost": total_cost,
                "dispatched_actions": dispatched_actions,
                "trace": trace,
            }
            # task_index may be None — Agent.solve's signature allows it
            # (env.reset() picks a random task in that case). Fall back to
            # the session id so the file is still uniquely named.
            task_id_str = (
                f"{task_index:03d}" if task_index is not None else f"unindexed-{session_id[:8]}"
            )
            out_path = self.trajectory_dir / f"task-{task_id_str}.json"
            out_path.write_text(json.dumps(payload, indent=2, default=str))
            print(f"  [mode-b] trajectory → {out_path}", flush=True)

        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )

    def close(self) -> None:
        self._client.close()
