"""SQLite-backed persistence for runs.

v0.0: simple single-table store. v1.0 migrates to Postgres for the hosted
deployment. Schema is intentionally narrow — adapters/datasets/etc. are
fixed code, not stored data.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlmodel import Field, Session, SQLModel, create_engine, select


class RunRow(SQLModel, table=True):
    """One persisted eval run."""

    __tablename__ = "runs"

    id: str = Field(primary_key=True)
    dataset: str
    track: str
    systems_json: str  # JSON list
    status: str  # queued / running / completed / failed
    n_questions: int
    judge_model: str
    started_at: datetime
    finished_at: datetime | None = None
    summary_json: str | None = None  # full RunSummary serialized
    results_json: str | None = None  # list[QuestionResult] serialized
    error: str | None = None


class Storage:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        SQLModel.metadata.create_all(self.engine)

    def create_run(
        self,
        run_id: str,
        dataset: str,
        track: str,
        systems: list[str],
        n_questions: int,
        judge_model: str,
    ) -> None:
        with Session(self.engine) as s:
            s.add(
                RunRow(
                    id=run_id,
                    dataset=dataset,
                    track=track,
                    systems_json=json.dumps(systems),
                    status="queued",
                    n_questions=n_questions,
                    judge_model=judge_model,
                    started_at=datetime.now(UTC),
                )
            )
            s.commit()

    def update_status(self, run_id: str, status: str, error: str | None = None) -> None:
        with Session(self.engine) as s:
            row = s.get(RunRow, run_id)
            if row is None:
                return
            row.status = status
            if error:
                row.error = error
            if status in {"completed", "failed", "cancelled"}:
                row.finished_at = datetime.now(UTC)
            s.add(row)
            s.commit()

    def save_summary(
        self,
        run_id: str,
        summary: dict[str, Any],
        results: list[dict[str, Any]],
    ) -> None:
        with Session(self.engine) as s:
            row = s.get(RunRow, run_id)
            if row is None:
                return
            row.summary_json = json.dumps(summary, default=str)
            row.results_json = json.dumps(results, default=str)
            s.add(row)
            s.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with Session(self.engine) as s:
            row = s.get(RunRow, run_id)
            if row is None:
                return None
            return _row_to_dict(row)

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with Session(self.engine) as s:
            rows = s.exec(
                select(RunRow).order_by(RunRow.started_at.desc()).limit(limit)  # type: ignore[attr-defined]
            ).all()
            return [_row_to_dict(r) for r in rows]


def _row_to_dict(row: RunRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "dataset": row.dataset,
        "track": row.track,
        "systems": json.loads(row.systems_json),
        "status": row.status,
        "n_questions": row.n_questions,
        "judge_model": row.judge_model,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        "summary": json.loads(row.summary_json) if row.summary_json else None,
        "results": json.loads(row.results_json) if row.results_json else None,
        "error": row.error,
    }
