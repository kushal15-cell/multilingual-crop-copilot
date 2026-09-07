from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class AdviceStatus(StrEnum):
    READY = "ready"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEED_MORE_INFORMATION = "need_more_information"


class DiseasePrediction(BaseModel):
    label: str
    display_name: str
    confidence: float = Field(ge=0, le=1)
    alternatives: list[tuple[str, float]] = Field(default_factory=list)
    calibrated: bool = True
    model_version: str = "unknown"


class Location(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    region: str | None = None
    market: str | None = None


class WeatherContext(BaseModel):
    temperature_c: float | None = None
    relative_humidity_pct: float | None = None
    precipitation_mm: float | None = None
    wind_speed_kmh: float | None = None
    observed_at: datetime | None = None
    source: str = "unavailable"
    unavailable_reason: str | None = None


class MarketContext(BaseModel):
    crop: str
    market: str | None = None
    latest_modal_price: float | None = None
    unit: str = "INR/quintal"
    predicted_7d_price: float | None = None
    trend: str = "unavailable"
    as_of: datetime | None = None
    source: str = "unavailable"
    is_synthetic: bool = False
    unavailable_reason: str | None = None


class Evidence(BaseModel):
    source_id: str
    title: str
    text: str
    url: str | None = None
    expert_validated: bool = False


class AdviceDraft(BaseModel):
    summary: str
    actions: list[str] = Field(default_factory=list)
    warning_signs: list[str] = Field(default_factory=list)
    limitations: str
    sources: list[str] = Field(default_factory=list)

    def as_text(self) -> str:
        actions = "\n".join(f"- {item}" for item in self.actions)
        warnings = "\n".join(f"- {item}" for item in self.warning_signs)
        return (
            f"{self.summary}\n\nActions:\n{actions or '- None'}\n\n"
            f"Warning signs:\n{warnings or '- None'}\n\nLimitations: {self.limitations}"
        )


class RiskAssessment(BaseModel):
    requires_review: bool
    categories: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class AdviceRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    crop: str = Field(min_length=2, max_length=80)
    question: str = Field(min_length=2, max_length=2000)
    language: Literal["en", "kn", "hi"] = "en"
    location: Location
    image_path: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AdviceResponse(BaseModel):
    request_id: str
    status: AdviceStatus
    farmer_message: str
    prediction: DiseasePrediction | None = None
    review_id: str | None = None
    approved_advice: AdviceDraft | None = None
    sources: list[Evidence] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReviewRecord(BaseModel):
    id: str
    request_id: str
    status: AdviceStatus
    crop: str
    disease: str
    confidence: float
    language: str
    question: str
    private_draft: AdviceDraft
    evidence: list[Evidence]
    risk: RiskAssessment
    reviewer: str | None = None
    reviewer_notes: str | None = None
    approved_text: AdviceDraft | None = None
    created_at: datetime
    updated_at: datetime
