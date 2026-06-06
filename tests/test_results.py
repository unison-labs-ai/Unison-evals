"""Tests for the unified results convention (results.py) + the combined summary."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime

from unison_evals.cli import _headline, _print_combined
from unison_evals.config import get_settings
from unison_evals.results import new_run_id, results_path, write_results
from unison_evals.types import RunSummary, SystemSummary, Track

_RUN_ID_RE = re.compile(r"^[a-z0-9-]+-\d{8}T\d{6}Z-[0-9a-f]{6}$")


def test_new_run_id_names_the_benchmark() -> None:
    rid = new_run_id("longmemeval")
    assert rid.startswith("longmemeval-")
    assert _RUN_ID_RE.match(rid), rid


def test_new_run_id_is_unique() -> None:
    assert new_run_id("context-bench") != new_run_id("context-bench")


def test_results_path_is_flat_under_results_dir() -> None:
    p = results_path("memoryagentbench-20260606T000000Z-abc123")
    assert p == get_settings().results_dir / "memoryagentbench-20260606T000000Z-abc123.json"
    assert p.suffix == ".json"


def test_write_results_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "results_dir", tmp_path)
    rid = new_run_id("longmemeval")
    out = write_results(rid, {"benchmark": "longmemeval", "n": 3})
    assert out == tmp_path / f"{rid}.json"
    assert json.loads(out.read_text()) == {"benchmark": "longmemeval", "n": 3}


def _summary(*, n: int, n_passed: int, cost_per_q: float) -> RunSummary:
    sys_summary = SystemSummary(
        system="unison-agent",
        n_questions=n,
        n_passed=n_passed,
        pass_rate=(n_passed / n) if n else 0.0,
        total_cost_usd=cost_per_q * n,
        cost_per_question_usd=cost_per_q,
        cost_per_solved_usd=None,
        avg_latency_ms=0.0,
        p50_latency_ms=0.0,
        p95_latency_ms=0.0,
    )
    return RunSummary(
        run_id="longmemeval-20260606T000000Z-abc123",
        dataset="longmemeval",
        track=Track.AGENT_E2E,
        systems=["unison-agent"],
        judge_model="gpt-4o-2024-08-06",
        n_questions=n,
        started_at=datetime(2026, 6, 6, tzinfo=UTC),
        summaries=[sys_summary],
    )


def test_headline_math() -> None:
    hl = _headline("longmemeval", _summary(n=10, n_passed=7, cost_per_q=0.01))
    assert hl["benchmark"] == "longmemeval"
    assert hl["n"] == 10
    assert hl["pct"] == 70.0
    assert abs(hl["cost_usd"] - 0.10) < 1e-9


def test_headline_handles_empty_summaries() -> None:
    empty = _summary(n=0, n_passed=0, cost_per_q=0.0)
    empty.summaries = []
    hl = _headline("memoryagentbench", empty)
    assert hl == {"benchmark": "memoryagentbench", "n": 0, "pct": 0.0, "cost_usd": 0.0}


def test_print_combined_smoke(capsys) -> None:
    _print_combined(
        [
            {"benchmark": "longmemeval", "n": 500, "pct": 70.0, "cost_usd": 3.8},
            {"benchmark": "context-bench", "n": 100, "pct": 88.0, "cost_usd": 4.2},
        ]
    )
    out = capsys.readouterr().out
    assert "All benchmarks" in out
    assert "longmemeval" in out and "context-bench" in out
