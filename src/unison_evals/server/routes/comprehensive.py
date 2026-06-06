"""GET /api/comprehensive — return all comprehensive runs grouped by comprehensive_id.

A comprehensive run is any set of result files under results/comprehensive-<TS>/.
Each directory contains one JSON file per (track, dataset, system) combo plus a
summary.json that the run_comprehensive.sh script produces.

This endpoint scans the results/ directory on disk (file-scan strategy, v0.1).
No DB writes are made — the source of truth for comprehensive runs is the
filesystem, not the SQLite runs table.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api")

# Pattern that matches the comprehensive run directory name.
_COMP_DIR_RE = re.compile(r"^comprehensive-(\d{8}T\d{6}Z)$")


def _scan_comprehensive_dirs(results_dir: Path) -> list[Path]:
    """Return all comprehensive-<TS> directories, newest first."""
    dirs = [d for d in results_dir.iterdir() if d.is_dir() and _COMP_DIR_RE.match(d.name)]
    return sorted(dirs, key=lambda d: d.name, reverse=True)


def _load_combo_from_file(filepath: Path, comprehensive_id: str) -> dict[str, Any]:
    """Parse a single (track, dataset, system) result JSON into a combo entry."""
    # File name pattern: <track>-<dataset>-<system>.json
    stem = filepath.stem  # e.g. agent-longmemeval-unison-agent
    parts = stem.split("-", 2)
    track = parts[0] if len(parts) > 0 else "unknown"
    dataset = parts[1] if len(parts) > 1 else "unknown"
    system = parts[2] if len(parts) > 2 else "unknown"

    entry: dict[str, Any] = {
        "comprehensive_id": comprehensive_id,
        "track": track,
        "dataset": dataset,
        "system": system,
        "file": str(filepath),
        "status": "done",
    }

    try:
        data = json.loads(filepath.read_text())
        summary = data.get("summary", {})
        summaries = summary.get("summaries", [])
        sys_summary = next((s for s in summaries if s.get("system") == system), None)
        if sys_summary:
            entry["pass_rate"] = sys_summary.get("pass_rate")
            entry["recall_at_10"] = sys_summary.get("mean_recall_at_10")
            entry["cost_per_solved_usd"] = sys_summary.get("cost_per_solved_usd")
            entry["p50_latency_ms"] = sys_summary.get("p50_latency_ms")
            entry["n_questions"] = sys_summary.get("n_questions")
            entry["total_cost_usd"] = sys_summary.get("total_cost_usd")
    except Exception as exc:
        entry["parse_error"] = str(exc)

    return entry


def _load_comprehensive_group(comp_dir: Path) -> dict[str, Any]:
    """Load all combo files in one comprehensive directory."""
    ts_match = _COMP_DIR_RE.match(comp_dir.name)
    comprehensive_id = ts_match.group(1) if ts_match else comp_dir.name

    # Prefer summary.json if it exists (written by run_comprehensive.sh).
    summary_path = comp_dir / "summary.json"
    if summary_path.exists():
        try:
            summary_data = json.loads(summary_path.read_text())
            summary_data["comprehensive_id"] = comprehensive_id
            return summary_data
        except Exception:
            pass  # fall through to file scan

    # Fall back: scan individual combo files.
    combo_files = [f for f in comp_dir.glob("*.json") if f.name != "summary.json"]
    combos = [_load_combo_from_file(f, comprehensive_id) for f in sorted(combo_files)]

    return {
        "comprehensive_id": comprehensive_id,
        "n_combos": len(combos),
        "n_done": sum(1 for c in combos if c.get("status") == "done"),
        "n_failed": sum(1 for c in combos if c.get("status") == "failed"),
        "combos": combos,
    }


@router.get("/comprehensive")
def list_comprehensive(request: Request) -> dict[str, Any]:
    """Return all comprehensive runs grouped by comprehensive_id (newest first).

    Response shape:
    {
      "runs": [
        {
          "comprehensive_id": "20260510T123456Z",
          "n_combos": 42,
          "n_done": 38,
          "n_failed": 4,
          "combos": [
            {
              "comprehensive_id": "...",
              "track": "agent",
              "dataset": "longmemeval",
              "system": "unison-agent",
              "status": "done",
              "pass_rate": null,        // null for brain track
              "recall_at_10": 0.72,
              "cost_per_solved_usd": null,
              "p50_latency_ms": 42.0,
              "n_questions": 20
            },
            ...
          ]
        }
      ],
      "total": 1
    }
    """
    from ..storage import Storage

    storage: Storage = request.app.state.storage
    # Derive results_dir from the storage db_path (db lives in results/).
    results_dir = Path(storage.engine.url.database).parent  # type: ignore[arg-type]

    comp_dirs = _scan_comprehensive_dirs(results_dir)
    groups = [_load_comprehensive_group(d) for d in comp_dirs]

    return {
        "runs": groups,
        "total": len(groups),
    }
