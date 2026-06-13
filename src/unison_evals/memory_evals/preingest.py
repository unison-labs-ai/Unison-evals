"""Pre-ingestion manifest.

LongMemEval haystacks are per-question, but they don't change between config
runs. Instead of re-seeding + re-extracting every run (~37 extract LLM calls per
question), ingest each haystack ONCE into a persistent eval workspace and record
question_id -> workspace_id here. Subsequent runs reuse the workspace read-only
(memoryMode="fresh", no seedDocs, no teardown) — query-only, seconds/question —
and the brain + agent trajectories persist for post-hoc analysis.

Manifest shape:
  {
    "meta": {"dataset": "longmemeval", "seed": "1234", "split": "...",
             "embed_model": "text-embedding-3-small", "created": "..."},
    "questions": {"<question_id>": "<workspace_id>", ...}
  }

Re-ingest (delete the manifest) only when the embedding model or the extraction
logic changes — otherwise the stored facts/embeddings are stale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_manifest(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"meta": {}, "questions": {}}
    data = json.loads(p.read_text())
    data.setdefault("meta", {})
    data.setdefault("questions", {})
    return data


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def workspace_for(manifest: dict[str, Any], question_id: str) -> str | None:
    return (manifest.get("questions") or {}).get(question_id)
