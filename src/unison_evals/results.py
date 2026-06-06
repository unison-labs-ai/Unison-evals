"""One run-id + results-file convention for every benchmark.

Every benchmark (LongMemEval, MemoryAgentBench, Context-Bench) writes exactly
one summary file to `results/<benchmark>-<utc>-<hex>.json`. No per-benchmark
layouts, no nested dirs — the filename names the benchmark and the run.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import get_settings


def new_run_id(benchmark: str) -> str:
    """`<benchmark>-<YYYYMMDDThhmmssZ>-<hex6>` — identifies the bench + the run."""
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{benchmark}-{ts}-{uuid.uuid4().hex[:6]}"


def results_path(run_id: str) -> Path:
    return get_settings().results_dir / f"{run_id}.json"


def write_results(run_id: str, payload: dict[str, Any]) -> Path:
    """Write the run summary to results/<run_id>.json and return the path."""
    path = results_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path
