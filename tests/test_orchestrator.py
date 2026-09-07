from pathlib import Path

import pytest

from crop_copilot.agent.orchestrator import CopilotOrchestrator
from crop_copilot.agent.risk import RiskPolicy
from crop_copilot.schemas import (
    AdviceDraft,
    AdviceRequest,
    AdviceStatus,
    DiseasePrediction,
    Evidence,
    Location,
    MarketContext,
    WeatherContext,
)
from crop_copilot.services.review_store import SQLiteReviewStore


class FakeClassifier:
    def __init__(self, confidence: float) -> None:
        self.confidence = confidence

    def predict(self, image_path: str) -> DiseasePrediction:
        return DiseasePrediction(
            label="tomato_early_blight",
            display_name="Tomato early blight",
            confidence=self.confidence,
            model_version="test-model",
        )


class FailingContext:
    def current(self, location):
        raise AssertionError("Weather must not run after abstention")

    def context(self, crop, market):
        raise AssertionError("Market must not run after abstention")


class FakeWeather:
    def current(self, location):
        return WeatherContext(temperature_c=25, source="test")


class FakeMarket:
    def context(self, crop, market):
        return MarketContext(crop=crop, market=market, trend="stable")


class FakeKnowledge:
    def __init__(self, validated: bool) -> None:
        self.validated = validated

    def search(self, crop, disease, language):
        return [
            Evidence(
                source_id="test-source",
                title="Test evidence",
                text="Observe leaf changes",
                expert_validated=self.validated,
            )
        ]


class FakeGenerator:
    def generate(self, question, crop, language, prediction, weather, market, evidence):
        return AdviceDraft(
            summary="Preliminary result",
            actions=["Photograph both sides of the leaf"],
            limitations="Image-only assessment",
            sources=["test-source"],
        )


class MemoryEvents:
    def __init__(self):
        self.items = []

    def log(self, event_type, request_id, **metadata):
        self.items.append((event_type, request_id, metadata))


def request() -> AdviceRequest:
    return AdviceRequest(
        request_id="request-1",
        crop="tomato",
        question="What should I observe?",
        language="en",
        location=Location(latitude=14.4, longitude=75.9, market="Davangere"),
        image_path="unused.jpg",
    )


def test_low_confidence_abstains_before_context_lookup(tmp_path) -> None:
    context = FailingContext()
    orchestrator = CopilotOrchestrator(
        classifier=FakeClassifier(0.40),
        weather=context,
        market=context,
        knowledge=FakeKnowledge(True),
        generator=FakeGenerator(),
        risk_policy=RiskPolicy(Path("configs/risk_policy.yaml")),
        review_store=SQLiteReviewStore(f"sqlite:///{tmp_path / 'reviews.db'}"),
        event_logger=MemoryEvents(),
    )
    result = orchestrator.advise(request())
    assert result.status == AdviceStatus.NEED_MORE_INFORMATION
    assert result.approved_advice is None
    assert result.prediction is None
    assert "tomato_early_blight" not in result.model_dump_json()


def test_unvalidated_evidence_hides_private_draft(tmp_path) -> None:
    store = SQLiteReviewStore(f"sqlite:///{tmp_path / 'reviews.db'}")
    orchestrator = CopilotOrchestrator(
        classifier=FakeClassifier(0.95),
        weather=FakeWeather(),
        market=FakeMarket(),
        knowledge=FakeKnowledge(False),
        generator=FakeGenerator(),
        risk_policy=RiskPolicy(Path("configs/risk_policy.yaml")),
        review_store=store,
        event_logger=MemoryEvents(),
    )
    result = orchestrator.advise(request())
    assert result.status == AdviceStatus.PENDING_REVIEW
    assert result.approved_advice is None
    assert "Preliminary result" not in result.farmer_message
    assert store.get(result.review_id).private_draft.summary == "Preliminary result"


def test_validated_low_risk_response_is_ready(tmp_path) -> None:
    orchestrator = CopilotOrchestrator(
        classifier=FakeClassifier(0.95),
        weather=FakeWeather(),
        market=FakeMarket(),
        knowledge=FakeKnowledge(True),
        generator=FakeGenerator(),
        risk_policy=RiskPolicy(Path("configs/risk_policy.yaml")),
        review_store=SQLiteReviewStore(f"sqlite:///{tmp_path / 'reviews.db'}"),
        event_logger=MemoryEvents(),
    )
    result = orchestrator.advise(request())
    assert result.status == AdviceStatus.READY
    assert result.approved_advice.summary == "Preliminary result"


def test_unsupported_crop_is_rejected_before_inference():
    orchestrator = object.__new__(CopilotOrchestrator)
    with pytest.raises(ValueError, match="tomato only"):
        orchestrator.advise(request().model_copy(update={"crop": "rice"}))


def test_missing_evidence_requires_review():
    policy = RiskPolicy(Path("configs/risk_policy.yaml"))
    draft = AdviceDraft(summary="Observe", limitations="Preliminary")
    assessment = policy.assess("What next?", FakeClassifier(0.99).predict(""), draft, [])
    assert assessment.requires_review
    assert "missing_knowledge" in assessment.categories
