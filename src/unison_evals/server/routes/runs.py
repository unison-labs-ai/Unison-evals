"""HTTP routes for managing eval runs.

POST   /api/runs            — start a new run
GET    /api/runs            — list recent runs
GET    /api/runs/:id        — fetch one run (status + summary + results)
GET    /api/runs/:id/stream — SSE stream of live progress events
GET    /api/registry        — what adapters / datasets are registered
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ...memory_evals.adapters import BRAIN_REGISTRY
from ...memory_evals.adapters import REGISTRY as ADAPTER_REGISTRY
from ...memory_evals.datasets import REGISTRY as DATASET_REGISTRY
from ...types import BrainMode, Track

router = APIRouter(prefix="/api")

_AGENT_TRACKS = {Track.AGENT_ORACLE.value, Track.AGENT_E2E.value}
_BRAIN_TRACKS = {Track.BRAIN_ONLY.value, Track.SCALE.value}
_VALID_BRAIN_MODES = {m.value for m in BrainMode}


class StartRunRequest(BaseModel):
    dataset: str
    track: str = "agent-oracle"
    systems: list[str] = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=10_000)
    judge_model: str | None = None
    pass_threshold: float = Field(default=1.0, ge=0.0, le=1.0)
    corpus: str | None = None
    # Sub-mode selector for track=brain-only. Ignored for other tracks.
    # Defaults to "cold" (original per-Q reset+ingest+search behaviour).
    mode: str | None = None


@router.get("/registry")
def registry() -> dict[str, list[dict[str, Any]]]:
    """What adapters and datasets are available — populates the launcher dropdowns."""
    return {
        "adapters": [
            {"name": name, "class": cls.__name__} for name, cls in sorted(ADAPTER_REGISTRY.items())
        ],
        "brain_adapters": [
            {"name": name, "class": cls.__name__} for name, cls in sorted(BRAIN_REGISTRY.items())
        ],
        "datasets": [
            {
                "name": name,
                "class": cls.__name__,
                "description": getattr(cls, "description", "") or "",
                "total_questions": getattr(cls, "total_questions", None),
                "supported_tracks": sorted(
                    t.value for t in getattr(cls, "supported_tracks", frozenset())
                ),
            }
            for name, cls in sorted(DATASET_REGISTRY.items())
        ],
        "tracks": [
            {"name": Track.BRAIN_ONLY.value, "description": "Track 1 — brain-only retrieval"},
            {"name": Track.AGENT_ORACLE.value, "description": "Track 2 — agent given gold context"},
            {"name": Track.AGENT_E2E.value, "description": "Track 3 — agent + brain E2E"},
            {"name": Track.SCALE.value, "description": "Track 4 — scale corpus retrieval"},
        ],
        "brain_modes": [
            {
                "name": BrainMode.COLD.value,
                "description": "Per-question reset → ingest → search (default)",
            },
            {
                "name": BrainMode.WARM.value,
                "description": "Corpus pre-loaded; skip reset+ingest, iterate search only",
            },
            {
                "name": BrainMode.BITEMPORAL.value,
                "description": "As-of temporal correctness scoring (BitempoQA-style)",
            },
            {
                "name": BrainMode.COMPACTION.value,
                "description": "LLM-judged wiki page synthesis quality (unison-brain only)",
            },
        ],
    }


@router.post("/runs", status_code=201)
async def start_run(req: StartRunRequest, request: Request) -> dict[str, str]:
    jobs = request.app.state.jobs
    if req.dataset not in DATASET_REGISTRY:
        raise HTTPException(400, f"Unknown dataset: {req.dataset}")

    # For agent tracks, validate against agent adapter registry.
    if req.track in _AGENT_TRACKS:
        for s in req.systems:
            if s not in ADAPTER_REGISTRY:
                raise HTTPException(400, f"Unknown agent system: {s}")

    # For brain/scale tracks, validate against brain adapter registry.
    if req.track in _BRAIN_TRACKS:
        for s in req.systems:
            if s not in BRAIN_REGISTRY:
                available = ", ".join(sorted(BRAIN_REGISTRY))
                raise HTTPException(400, f"Unknown brain system: {s}. Available: {available}")

    # scale requires corpus
    if req.track == Track.SCALE.value and not req.corpus:
        raise HTTPException(
            400,
            "track=scale requires a corpus label. "
            "Pass corpus=<label> (e.g. 'msmarco-passages-v1-100k'). "
            "Run scripts/load_corpus_*.sh first to load the corpus.",
        )

    # Validate brain mode if provided.
    if req.mode is not None and req.mode not in _VALID_BRAIN_MODES:
        raise HTTPException(
            400,
            f"Unknown brain mode: {req.mode!r}. "
            f"Valid modes: {', '.join(sorted(_VALID_BRAIN_MODES))}",
        )

    # Validate dataset/track compatibility — catches "FRAMES on Track 1" etc.
    # before the runner is constructed, so the error message lists what the
    # dataset DOES support rather than just NotImplementedError.
    ds_cls = DATASET_REGISTRY[req.dataset]
    ds_supported = {t.value for t in getattr(ds_cls, "supported_tracks", frozenset())}
    if ds_supported and req.track not in ds_supported:
        raise HTTPException(
            400,
            f"Dataset {req.dataset!r} does not support track={req.track!r}. "
            f"It supports: {', '.join(sorted(ds_supported)) or '(none declared)'}.",
        )

    # Warn if limit exceeds the dataset's canonical size (auto-caps in the loader).
    total = getattr(ds_cls, "total_questions", None)
    if total is not None and req.limit > total:
        # Not fatal — loaders gracefully cap. We surface the discrepancy so the
        # client can show a hint, but don't reject.
        pass

    try:
        run_id = jobs.start_run(
            dataset=req.dataset,
            track=req.track,
            systems=req.systems,
            limit=req.limit,
            judge_model=req.judge_model,
            pass_threshold=req.pass_threshold,
            corpus=req.corpus,
            mode=req.mode,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"run_id": run_id}


@router.get("/runs")
def list_runs(request: Request) -> dict[str, list[dict]]:
    storage = request.app.state.storage
    return {"runs": storage.list_runs(limit=50)}


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request) -> dict:
    storage = request.app.state.storage
    row = storage.get_run(run_id)
    if row is None:
        raise HTTPException(404, "Run not found")
    return row


@router.delete("/runs/{run_id}")
def cancel_run(run_id: str, request: Request) -> dict[str, bool]:
    """Cancel a running run. 404 if the run id is unknown; idempotent if already terminal."""
    storage = request.app.state.storage
    jobs = request.app.state.jobs
    if storage.get_run(run_id) is None:
        raise HTTPException(404, "Run not found")
    cancelled = jobs.cancel_run(run_id)
    return {"cancelled": cancelled}


@router.get("/runs/{run_id}/stream")
async def stream_run(run_id: str, request: Request) -> EventSourceResponse:
    """Server-Sent Events stream of progress for one run.

    Replays past events first (so a UI that connects mid-run sees everything),
    then forwards live events until the run ends.
    """
    jobs = request.app.state.jobs
    storage = request.app.state.storage
    if storage.get_run(run_id) is None:
        raise HTTPException(404, "Run not found")

    async def gen():
        # Emit as default ('message') events — the browser's EventSource.onmessage
        # only fires for default-named events. The event type is carried in the
        # JSON payload (`data.type`) so the client can dispatch by reading it.
        async for ev in jobs.subscribe(run_id):
            yield {"data": json.dumps(ev)}

    return EventSourceResponse(gen())
