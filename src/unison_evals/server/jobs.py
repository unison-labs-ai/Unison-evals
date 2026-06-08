"""In-process async job queue + per-run event broadcaster.

For v0.0 a single-process server is enough. v0.5 swaps this for a
Postgres-backed queue + worker pool when we deploy.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

from loguru import logger

from ..memory_evals.datasets import get_dataset
from ..memory_evals.metrics.llm_judge import LLMJudge
from ..memory_evals.runners.agent_e2e import AgentE2ERunner
from ..memory_evals.runners.agent_oracle import AgentOracleRunner
from ..types import Track
from .storage import Storage

_SUPPORTED_TRACKS = {Track.AGENT_ORACLE, Track.AGENT_E2E}


class JobManager:
    """Tracks running jobs + per-run event queues for SSE streaming."""

    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        self._tasks: dict[str, asyncio.Task] = {}
        # Per-run subscriber queues. New SSE connections add a queue;
        # the worker fans out events to every subscriber.
        self._subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = defaultdict(list)
        # Buffered history for late subscribers — lets a UI that connects
        # mid-run replay everything to date.
        self._history: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def start_run(
        self,
        *,
        dataset: str,
        track: str,
        systems: list[str],
        limit: int,
        judge_model: str | None,
        pass_threshold: float,
        corpus: str | None = None,
        mode: str | None = None,
    ) -> str:
        try:
            track_enum = Track(track)
        except ValueError:
            supported = ", ".join(t.value for t in _SUPPORTED_TRACKS)
            raise ValueError(f"Unknown track {track!r}. Supported: {supported}") from None

        ds = get_dataset(dataset)

        if track_enum == Track.AGENT_ORACLE:
            questions = list(ds.load(limit=limit))
            judge = LLMJudge(model=judge_model, pass_threshold=pass_threshold)
            runner = AgentOracleRunner(systems=systems, judge=judge)
            run_id = runner.run_id
            self.storage.create_run(
                run_id=run_id,
                dataset=dataset,
                track=track,
                systems=systems,
                n_questions=len(questions),
                judge_model=judge.model,
            )
            task = asyncio.create_task(
                self._drive_run(run_id, runner, questions, dataset),
                name=f"run-{run_id}",
            )

        elif track_enum == Track.AGENT_E2E:
            try:
                e2e_questions = list(ds.load_brain_questions(limit=limit))
            except NotImplementedError as e:
                raise ValueError(
                    f"Dataset {dataset!r} does not support Track 3 (agent-e2e). Reason: {e}"
                ) from e
            judge = LLMJudge(model=judge_model, pass_threshold=pass_threshold)
            e2e_runner = AgentE2ERunner(systems=systems, judge=judge)
            run_id = e2e_runner.run_id
            self.storage.create_run(
                run_id=run_id,
                dataset=dataset,
                track=track,
                systems=systems,
                n_questions=len(e2e_questions),
                judge_model=judge.model,
            )
            task = asyncio.create_task(
                self._drive_e2e_run(run_id, e2e_runner, e2e_questions, dataset),
                name=f"run-{run_id}",
            )

        else:
            raise ValueError(f"Unhandled track {track!r}")

        self._tasks[run_id] = task
        return run_id

    async def _drive_run(
        self,
        run_id: str,
        runner: AgentOracleRunner,
        questions: list,
        dataset_name: str,
    ) -> None:
        self.storage.update_status(run_id, "running")
        try:
            async for ev in runner.run(questions, dataset_name=dataset_name):
                payload = ev.model_dump(mode="json")
                self._broadcast(run_id, payload)
                if ev.type == "run_completed" and ev.summary is not None:
                    self.storage.save_summary(
                        run_id,
                        summary=ev.summary.model_dump(mode="json"),
                        results=[r.model_dump(mode="json") for r in runner.results],
                    )
                    self.storage.update_status(run_id, "completed")
                elif ev.type == "run_failed":
                    self.storage.update_status(run_id, "failed", error=ev.error)
        except Exception as e:
            logger.exception("Run {} crashed", run_id)
            self.storage.update_status(run_id, "failed", error=str(e))
            self._broadcast(
                run_id,
                {"type": "run_failed", "run_id": run_id, "error": str(e)},
            )
        finally:
            # Send a sentinel so SSE consumers can close cleanly.
            self._broadcast(run_id, {"type": "_eof", "run_id": run_id})
            self._tasks.pop(run_id, None)

    async def _drive_e2e_run(
        self,
        run_id: str,
        runner: AgentE2ERunner,
        questions: list,
        dataset_name: str,
    ) -> None:
        self.storage.update_status(run_id, "running")
        try:
            async for ev in runner.run(questions, dataset_name=dataset_name):
                payload = ev.model_dump(mode="json")
                self._broadcast(run_id, payload)
                if ev.type == "run_completed" and ev.summary is not None:
                    self.storage.save_summary(
                        run_id,
                        summary=ev.summary.model_dump(mode="json"),
                        results=[r.model_dump(mode="json") for r in runner.results],
                    )
                    self.storage.update_status(run_id, "completed")
                elif ev.type == "run_failed":
                    self.storage.update_status(run_id, "failed", error=ev.error)
        except asyncio.CancelledError:
            self.storage.update_status(run_id, "cancelled", error="cancelled by user")
            self._broadcast(
                run_id,
                {"type": "run_failed", "run_id": run_id, "error": "cancelled by user"},
            )
            raise
        except Exception as e:
            logger.exception("E2E run {} crashed", run_id)
            self.storage.update_status(run_id, "failed", error=str(e))
            self._broadcast(
                run_id,
                {"type": "run_failed", "run_id": run_id, "error": str(e)},
            )
        finally:
            self._broadcast(run_id, {"type": "_eof", "run_id": run_id})
            self._tasks.pop(run_id, None)

    def _broadcast(self, run_id: str, payload: dict[str, Any]) -> None:
        self._history[run_id].append(payload)
        for q in list(self._subscribers[run_id]):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:  # pragma: no cover
                pass

    async def subscribe(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        """Yield every event for a run — replays history first, then live."""
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1024)
        # Replay buffered history.
        for past in list(self._history.get(run_id, [])):
            await q.put(past)
        self._subscribers[run_id].append(q)
        try:
            while True:
                ev = await q.get()
                if ev.get("type") == "_eof":
                    return
                yield ev
        finally:
            try:
                self._subscribers[run_id].remove(q)
            except ValueError:
                pass

    def is_running(self, run_id: str) -> bool:
        t = self._tasks.get(run_id)
        return t is not None and not t.done()

    def cancel_run(self, run_id: str) -> bool:
        """Cancel an in-flight run. Returns True if it was actually running."""
        task = self._tasks.get(run_id)
        if task is None or task.done():
            return False
        self.storage.update_status(run_id, "cancelled", error="cancelled by user")
        self._broadcast(
            run_id,
            {"type": "run_failed", "run_id": run_id, "error": "cancelled by user"},
        )
        task.cancel()
        return True
