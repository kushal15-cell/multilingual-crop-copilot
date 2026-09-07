from crop_copilot.schemas import (
    AdviceDraft,
    AdviceStatus,
    DiseasePrediction,
    Evidence,
    RiskAssessment,
)
from crop_copilot.services.review_store import SQLiteReviewStore


def make_record(store: SQLiteReviewStore):
    return store.create(
        request_id="request-1",
        crop="tomato",
        language="en",
        question="What treatment?",
        prediction=DiseasePrediction(
            label="tomato_early_blight", display_name="Early blight", confidence=0.82
        ),
        draft=AdviceDraft(
            summary="Private draft",
            actions=["Draft action"],
            limitations="Needs review",
        ),
        evidence=[
            Evidence(source_id="source-1", title="Source", text="Evidence", expert_validated=True)
        ],
        risk=RiskAssessment(requires_review=True, categories=["dosage"]),
    )


def test_review_lifecycle(tmp_path) -> None:
    store = SQLiteReviewStore(f"sqlite:///{tmp_path / 'queue.sqlite3'}")
    created = make_record(store)
    assert created.status == AdviceStatus.PENDING_REVIEW
    assert len(store.list_pending()) == 1

    approved = store.approve(
        created.id,
        reviewer="agronomist-7",
        approved_text=AdviceDraft(
            summary="Approved response",
            actions=["Safe action"],
            limitations="Field confirmation still required",
        ),
        notes="Checked against local protocol",
    )
    assert approved.status == AdviceStatus.APPROVED
    assert approved.approved_text.summary == "Approved response"
    assert store.list_pending() == []


def test_review_cannot_be_decided_twice(tmp_path) -> None:
    store = SQLiteReviewStore(f"sqlite:///{tmp_path / 'queue.sqlite3'}")
    created = make_record(store)
    store.reject(created.id, "agronomist-7", "Insufficient evidence")
    try:
        store.reject(created.id, "agronomist-8", "Second decision")
    except ValueError as exc:
        assert "already been decided" in str(exc)
    else:
        raise AssertionError("Second review decision should fail")

