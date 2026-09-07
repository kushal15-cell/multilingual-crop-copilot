from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from crop_copilot.schemas import (
    AdviceDraft,
    AdviceStatus,
    DiseasePrediction,
    Evidence,
    ReviewRecord,
    RiskAssessment,
)


class SQLiteReviewStore:
    def __init__(self, database_url: str) -> None:
        prefix = "sqlite:///"
        if not database_url.startswith(prefix):
            raise ValueError("This MVP supports sqlite:/// database URLs only")
        self.path = Path(database_url.removeprefix(prefix))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Commit or roll back and always release the SQLite file handle."""
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reviews (
                    id TEXT PRIMARY KEY,
                    request_id TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL,
                    crop TEXT NOT NULL,
                    disease TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    language TEXT NOT NULL,
                    question TEXT NOT NULL,
                    private_draft TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    risk TEXT NOT NULL,
                    reviewer TEXT,
                    reviewer_notes TEXT,
                    approved_text TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def create(
        self,
        request_id: str,
        crop: str,
        language: str,
        question: str,
        prediction: DiseasePrediction,
        draft: AdviceDraft,
        evidence: list[Evidence],
        risk: RiskAssessment,
    ) -> ReviewRecord:
        review_id = str(uuid4())
        now = datetime.now(UTC)
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review_id,
                    request_id,
                    AdviceStatus.PENDING_REVIEW.value,
                    crop,
                    prediction.label,
                    prediction.confidence,
                    language,
                    question,
                    draft.model_dump_json(),
                    json.dumps([item.model_dump(mode="json") for item in evidence]),
                    risk.model_dump_json(),
                    None,
                    None,
                    None,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
        record = self.get(review_id)
        if record is None:
            raise RuntimeError("Failed to create review record")
        return record

    def get(self, review_id: str) -> ReviewRecord | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
        return self._to_record(row) if row else None

    def list_pending(self, limit: int = 100) -> list[ReviewRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM reviews WHERE status = ? ORDER BY created_at ASC LIMIT ?",
                (AdviceStatus.PENDING_REVIEW.value, limit),
            ).fetchall()
        return [self._to_record(row) for row in rows]

    def approve(
        self,
        review_id: str,
        reviewer: str,
        approved_text: AdviceDraft,
        notes: str | None = None,
    ) -> ReviewRecord:
        return self._decide(
            review_id, AdviceStatus.APPROVED, reviewer, notes, approved_text.model_dump_json()
        )

    def reject(self, review_id: str, reviewer: str, notes: str) -> ReviewRecord:
        if not notes.strip():
            raise ValueError("A rejection reason is required")
        return self._decide(review_id, AdviceStatus.REJECTED, reviewer, notes, None)

    def _decide(
        self,
        review_id: str,
        status: AdviceStatus,
        reviewer: str,
        notes: str | None,
        approved_text: str | None,
    ) -> ReviewRecord:
        if not reviewer.strip():
            raise ValueError("Reviewer identity is required")
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE reviews
                SET status = ?, reviewer = ?, reviewer_notes = ?, approved_text = ?, updated_at = ?
                WHERE id = ? AND status = ?
                """,
                (
                    status.value,
                    reviewer,
                    notes,
                    approved_text,
                    datetime.now(UTC).isoformat(),
                    review_id,
                    AdviceStatus.PENDING_REVIEW.value,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Review does not exist or has already been decided")
        record = self.get(review_id)
        if record is None:
            raise RuntimeError("Review disappeared after update")
        return record

    @staticmethod
    def _to_record(row: sqlite3.Row) -> ReviewRecord:
        return ReviewRecord(
            id=row["id"],
            request_id=row["request_id"],
            status=AdviceStatus(row["status"]),
            crop=row["crop"],
            disease=row["disease"],
            confidence=row["confidence"],
            language=row["language"],
            question=row["question"],
            private_draft=AdviceDraft.model_validate_json(row["private_draft"]),
            evidence=[Evidence.model_validate(item) for item in json.loads(row["evidence"])],
            risk=RiskAssessment.model_validate_json(row["risk"]),
            reviewer=row["reviewer"],
            reviewer_notes=row["reviewer_notes"],
            approved_text=(
                AdviceDraft.model_validate_json(row["approved_text"])
                if row["approved_text"]
                else None
            ),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
