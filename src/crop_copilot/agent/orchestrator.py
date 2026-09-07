from __future__ import annotations

import time
from typing import Protocol

from crop_copilot.agent.risk import RiskPolicy
from crop_copilot.monitoring.events import EventLogger
from crop_copilot.schemas import (
    AdviceDraft,
    AdviceRequest,
    AdviceResponse,
    AdviceStatus,
    DiseasePrediction,
    Evidence,
    Location,
    MarketContext,
    WeatherContext,
)
from crop_copilot.services.knowledge import KnowledgeBase
from crop_copilot.services.review_store import SQLiteReviewStore


class Classifier(Protocol):
    def predict(self, image_path: str) -> DiseasePrediction: ...


class WeatherProvider(Protocol):
    def current(self, location: Location) -> WeatherContext: ...


class MarketProvider(Protocol):
    def context(self, crop: str, market: str | None) -> MarketContext: ...


class AdviceGenerator(Protocol):
    def generate(
        self,
        question: str,
        crop: str,
        language: str,
        prediction: DiseasePrediction,
        weather: WeatherContext,
        market: MarketContext,
        evidence: list[Evidence],
    ) -> AdviceDraft: ...


MESSAGES = {
    "en": {
        "uncertain": (
            "I cannot identify this safely from the current image. Please send a clear close-up "
            "of both sides of an affected leaf in daylight and include a wider photo of the plant."
        ),
        "pending": (
            "Your case has been sent to an agronomist for review. High-risk or uncertain advice "
            "will not be released until it is approved."
        ),
        "rejected": "The draft was not approved. Please contact a qualified local agronomist.",
    },
    "kn": {
        "uncertain": (
            "ಈ ಚಿತ್ರದ ಆಧಾರದಲ್ಲಿ ಸುರಕ್ಷಿತವಾಗಿ ಗುರುತಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ಹಗಲು ಬೆಳಕಿನಲ್ಲಿ ಬಾಧಿತ "
            "ಎಲೆಯ ಎರಡೂ ಬದಿಗಳ ಸ್ಪಷ್ಟ ಹತ್ತಿರದ ಚಿತ್ರ ಮತ್ತು ಸಂಪೂರ್ಣ ಗಿಡದ ಚಿತ್ರ ಕಳುಹಿಸಿ."
        ),
        "pending": (
            "ನಿಮ್ಮ ಪ್ರಕರಣವನ್ನು ಕೃಷಿ ತಜ್ಞರ ಪರಿಶೀಲನೆಗೆ ಕಳುಹಿಸಲಾಗಿದೆ. ಅನುಮೋದನೆಗೂ ಮೊದಲು "
            "ಅಪಾಯದ ಅಥವಾ ಅನಿಶ್ಚಿತ ಸಲಹೆಯನ್ನು ನೀಡಲಾಗುವುದಿಲ್ಲ."
        ),
        "rejected": "ಕರಡು ಅನುಮೋದಿಸಲಿಲ್ಲ. ಅರ್ಹ ಸ್ಥಳೀಯ ಕೃಷಿ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
    },
    "hi": {
        "uncertain": (
            "इस चित्र से सुरक्षित पहचान संभव नहीं है। दिन के उजाले में प्रभावित पत्ती के दोनों "
            "तरफ की साफ नज़दीकी तस्वीर और पूरे पौधे की तस्वीर भेजें।"
        ),
        "pending": (
            "आपका मामला कृषि विशेषज्ञ की समीक्षा के लिए भेजा गया है। स्वीकृति से पहले "
            "उच्च-जोखिम या अनिश्चित सलाह जारी नहीं होगी।"
        ),
        "rejected": "मसौदा स्वीकृत नहीं हुआ। किसी योग्य स्थानीय कृषि विशेषज्ञ से संपर्क करें।",
    },
}


class CopilotOrchestrator:
    def __init__(
        self,
        classifier: Classifier,
        weather: WeatherProvider,
        market: MarketProvider,
        knowledge: KnowledgeBase,
        generator: AdviceGenerator,
        risk_policy: RiskPolicy,
        review_store: SQLiteReviewStore,
        event_logger: EventLogger,
    ) -> None:
        self.classifier = classifier
        self.weather = weather
        self.market = market
        self.knowledge = knowledge
        self.generator = generator
        self.risk_policy = risk_policy
        self.review_store = review_store
        self.events = event_logger

    def advise(self, request: AdviceRequest) -> AdviceResponse:
        started = time.perf_counter()
        if request.crop.strip().casefold() != "tomato":
            raise ValueError("This model supports tomato only. Please submit a tomato leaf.")
        request = request.model_copy(update={"crop": "tomato"})
        prediction = self.classifier.predict(request.image_path)
        language_messages = MESSAGES.get(request.language, MESSAGES["en"])
        if prediction.confidence < self.risk_policy.low_confidence:
            response = AdviceResponse(
                request_id=request.request_id,
                status=AdviceStatus.NEED_MORE_INFORMATION,
                farmer_message=language_messages["uncertain"],
                metadata={
                    "reason": "vision_confidence_below_abstention_threshold",
                    "confidence": prediction.confidence,
                },
            )
            self._log(response, started)
            return response

        weather = self.weather.current(request.location)
        market = self.market.context(request.crop, request.location.market)
        evidence = self.knowledge.search(
            request.crop, prediction.label, request.language
        )
        draft = self.generator.generate(
            request.question,
            request.crop,
            request.language,
            prediction,
            weather,
            market,
            evidence,
        )
        risk = self.risk_policy.assess(request.question, prediction, draft, evidence)
        context_metadata = {
            "weather_source": weather.source,
            "market_available": market.unavailable_reason is None,
            "market_source": market.source,
            "market_is_synthetic": market.is_synthetic,
            "market_name": market.market,
            "market_latest_modal_price": market.latest_modal_price,
            "market_predicted_7d_price": market.predicted_7d_price,
            "market_trend": market.trend,
            "risk_categories": risk.categories,
        }
        if risk.requires_review:
            record = self.review_store.create(
                request_id=request.request_id,
                crop=request.crop,
                language=request.language,
                question=request.question,
                prediction=prediction,
                draft=draft,
                evidence=evidence,
                risk=risk,
            )
            response = AdviceResponse(
                request_id=request.request_id,
                status=AdviceStatus.PENDING_REVIEW,
                farmer_message=language_messages["pending"],
                prediction=prediction,
                review_id=record.id,
                metadata=context_metadata,
            )
        else:
            response = AdviceResponse(
                request_id=request.request_id,
                status=AdviceStatus.READY,
                farmer_message=draft.as_text(),
                prediction=prediction,
                approved_advice=draft,
                sources=evidence,
                metadata=context_metadata,
            )
        self._log(response, started)
        return response

    def public_review_status(self, review_id: str) -> AdviceResponse:
        record = self.review_store.get(review_id)
        if record is None:
            raise KeyError(review_id)
        messages = MESSAGES.get(record.language, MESSAGES["en"])
        if record.status == AdviceStatus.APPROVED and record.approved_text:
            return AdviceResponse(
                request_id=record.request_id,
                status=AdviceStatus.APPROVED,
                farmer_message=record.approved_text.as_text(),
                review_id=record.id,
                approved_advice=record.approved_text,
                sources=record.evidence,
            )
        if record.status == AdviceStatus.REJECTED:
            return AdviceResponse(
                request_id=record.request_id,
                status=AdviceStatus.REJECTED,
                farmer_message=messages["rejected"],
                review_id=record.id,
            )
        return AdviceResponse(
            request_id=record.request_id,
            status=AdviceStatus.PENDING_REVIEW,
            farmer_message=messages["pending"],
            review_id=record.id,
        )

    def _log(self, response: AdviceResponse, started: float) -> None:
        self.events.log(
            "advice_completed",
            response.request_id,
            status=response.status.value,
            disease=response.prediction.label if response.prediction else None,
            confidence=response.prediction.confidence if response.prediction else None,
            model_version=response.prediction.model_version if response.prediction else None,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
