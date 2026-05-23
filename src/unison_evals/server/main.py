"""FastAPI app entry point. `uv run unison-evals-server`."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import get_settings
from .jobs import JobManager
from .routes.comprehensive import router as comprehensive_router
from .routes.runs import router as runs_router
from .storage import Storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    storage = Storage(db_path=settings.results_dir / "runs.db")
    jobs = JobManager(storage=storage)
    app.state.storage = storage
    app.state.jobs = jobs
    yield


app = FastAPI(
    title="unison-evals",
    version="0.0.1",
    description="Public benchmark harness for Unison and comparable systems.",
    lifespan=lifespan,
)

# Wide-open CORS for v0.0 (localhost only). Locked down in v0.5.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

app.include_router(runs_router)
app.include_router(comprehensive_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "unison-evals"}


def run() -> None:
    """Entry point for `uv run unison-evals-server`."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "unison_evals.server.main:app",
        host="0.0.0.0",
        port=settings.server_port,
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    run()
