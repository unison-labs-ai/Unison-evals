"""Tests for the comprehensive run orchestrator components.

Covers:
- _estimate_cost.py cost table logic
- /api/comprehensive route (file-scan + summary.json parsing)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# _estimate_cost.py tests
# ---------------------------------------------------------------------------

ESTIMATE_SCRIPT = Path(__file__).parent.parent / "scripts" / "_estimate_cost.py"


def run_estimate(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ESTIMATE_SCRIPT), *args],
        capture_output=True,
        text=True,
    )


def test_estimate_cost_runs():
    result = run_estimate("--limit", "1", "--judge", "claude-haiku-4-5")
    assert result.returncode == 0, result.stderr
    assert "TOTAL" in result.stdout


def test_estimate_cost_shows_systems():
    result = run_estimate("--limit", "1")
    assert result.returncode == 0
    # Should show at least one brain system.
    assert "pgvector-naive" in result.stdout or "unison-brain" in result.stdout


def test_estimate_cost_budget_gate_passes_under_threshold():
    # limit=1 will be well under $20.
    result = run_estimate("--limit", "1", "--check", "--threshold", "20")
    assert result.returncode == 0


def test_estimate_cost_budget_gate_triggers_over_threshold():
    # limit=1000 should exceed $20 threshold.
    result = run_estimate("--limit", "1000", "--check", "--threshold", "0.01")
    assert result.returncode == 1
    assert "Budget gate" in result.stderr or "threshold" in result.stderr.lower()


def test_estimate_cost_only_track_filter():
    result = run_estimate("--limit", "5", "--tracks", "brain")
    assert result.returncode == 0
    # Should not include pure agent systems in output.
    assert "brain" in result.stdout.lower() or "pgvector" in result.stdout


def test_estimate_cost_only_dataset_filter():
    result = run_estimate("--limit", "5", "--datasets", "bitempoqa")
    assert result.returncode == 0
    assert "bitempoqa" in result.stdout


# ---------------------------------------------------------------------------
# /api/comprehensive route tests
# ---------------------------------------------------------------------------


def _make_server(tmp_path: Path) -> TestClient:
    """Spin up a test FastAPI client with a temp results dir."""
    import os

    os.environ.setdefault("UNISON_API_URL", "http://localhost:3001")
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

    from unison_evals.server.jobs import JobManager
    from unison_evals.server.main import app
    from unison_evals.server.storage import Storage

    db_path = tmp_path / "runs.db"
    storage = Storage(db_path=db_path)
    jobs = JobManager(storage=storage)

    # Patch app state directly.
    app.state.storage = storage
    app.state.jobs = jobs

    return TestClient(app)


def test_comprehensive_empty(tmp_path: Path):
    client = _make_server(tmp_path)
    response = client.get("/api/comprehensive")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["runs"] == []


def test_comprehensive_with_summary_json(tmp_path: Path):
    # Create a fake comprehensive run directory with a summary.json.
    results_dir = tmp_path / "results"
    comp_dir = results_dir / "comprehensive-20260510T120000Z"
    comp_dir.mkdir(parents=True)

    summary = {
        "comprehensive_id": "20260510T120000Z",
        "generated_at": "2026-05-10T12:00:00Z",
        "limit": 20,
        "judge": "claude-haiku-4-5",
        "n_combos": 2,
        "n_done": 2,
        "n_failed": 0,
        "combos": [
            {
                "track": "brain",
                "dataset": "bitempoqa",
                "system": "pgvector-naive",
                "status": "done",
                "comprehensive_id": "20260510T120000Z",
                "pass_rate": None,
                "recall_at_10": 0.72,
                "cost_per_solved_usd": None,
                "p50_latency_ms": 42.0,
                "n_questions": 20,
            },
            {
                "track": "agent",
                "dataset": "longmemeval",
                "system": "claude-code",
                "status": "done",
                "comprehensive_id": "20260510T120000Z",
                "pass_rate": 0.55,
                "recall_at_10": None,
                "cost_per_solved_usd": 0.12,
                "p50_latency_ms": 1200.0,
                "n_questions": 20,
            },
        ],
    }
    (comp_dir / "summary.json").write_text(json.dumps(summary))

    # Override the storage db_path so the route resolves results_dir correctly.
    import os

    os.environ.setdefault("UNISON_API_URL", "http://localhost:3001")
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

    from unison_evals.server.jobs import JobManager
    from unison_evals.server.main import app
    from unison_evals.server.storage import Storage

    db_path = results_dir / "runs.db"
    storage = Storage(db_path=db_path)
    jobs = JobManager(storage=storage)
    app.state.storage = storage
    app.state.jobs = jobs

    client = TestClient(app)
    response = client.get("/api/comprehensive")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    group = data["runs"][0]
    assert group["comprehensive_id"] == "20260510T120000Z"
    assert group["n_combos"] == 2
    assert group["n_done"] == 2
    assert len(group["combos"]) == 2


def test_comprehensive_fallback_file_scan(tmp_path: Path):
    """Without a summary.json, the route falls back to scanning combo files."""
    results_dir = tmp_path / "results"
    comp_dir = results_dir / "comprehensive-20260510T130000Z"
    comp_dir.mkdir(parents=True)

    # Write a fake combo file.
    combo_data = {
        "summary": {
            "run_id": "run-abc",
            "dataset": "bitempoqa",
            "track": "brain-only",
            "systems": ["pgvector-naive"],
            "n_questions": 5,
            "started_at": "2026-05-10T13:00:00",
            "finished_at": "2026-05-10T13:01:00",
            "total_cost_usd": 0.001,
            "summaries": [
                {
                    "system": "pgvector-naive",
                    "n_questions": 5,
                    "mean_recall_at_10": 0.80,
                    "mean_ndcg_at_10": 0.75,
                    "mean_mrr": 0.70,
                    "mean_hit_at_1": 0.60,
                    "total_cost_usd": 0.001,
                    "avg_latency_ms": 50.0,
                    "p50_latency_ms": 45.0,
                    "p95_latency_ms": 90.0,
                }
            ],
        },
        "results": [],
        "exported_at": "2026-05-10T13:01:00Z",
    }
    (comp_dir / "brain-bitempoqa-pgvector-naive.json").write_text(json.dumps(combo_data))

    import os

    os.environ.setdefault("UNISON_API_URL", "http://localhost:3001")
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

    from unison_evals.server.jobs import JobManager
    from unison_evals.server.main import app
    from unison_evals.server.storage import Storage

    db_path = results_dir / "runs.db"
    storage = Storage(db_path=db_path)
    jobs = JobManager(storage=storage)
    app.state.storage = storage
    app.state.jobs = jobs

    client = TestClient(app)
    response = client.get("/api/comprehensive")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    group = data["runs"][0]
    assert group["comprehensive_id"] == "20260510T130000Z"
    combos = group["combos"]
    assert len(combos) == 1
    assert combos[0]["system"] == "pgvector-naive"
    assert combos[0]["recall_at_10"] == pytest.approx(0.80)


def test_comprehensive_ignores_non_comprehensive_dirs(tmp_path: Path):
    """Directories not matching comprehensive-<TS> pattern are ignored."""
    results_dir = tmp_path / "results"
    (results_dir / "v1.0-longmemeval-track2-20260510T000000Z.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (results_dir / "v1.0-longmemeval-track2-20260510T000000Z.json").write_text("{}")
    (results_dir / "some-other-dir").mkdir()

    import os

    os.environ.setdefault("UNISON_API_URL", "http://localhost:3001")
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

    from unison_evals.server.jobs import JobManager
    from unison_evals.server.main import app
    from unison_evals.server.storage import Storage

    db_path = results_dir / "runs.db"
    storage = Storage(db_path=db_path)
    jobs = JobManager(storage=storage)
    app.state.storage = storage
    app.state.jobs = jobs

    client = TestClient(app)
    response = client.get("/api/comprehensive")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
