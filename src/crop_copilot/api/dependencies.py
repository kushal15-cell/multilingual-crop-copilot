from __future__ import annotations

from functools import lru_cache

from crop_copilot.agent.orchestrator import CopilotOrchestrator
from crop_copilot.agent.risk import RiskPolicy
from crop_copilot.config import get_settings
from crop_copilot.cv.demo import DemoClassifier
from crop_copilot.monitoring.events import EventLogger
from crop_copilot.services.knowledge import KnowledgeBase
from crop_copilot.services.llm import OpenAICompatibleGenerator, TemplateGenerator
from crop_copilot.services.review_store import SQLiteReviewStore
from crop_copilot.services.speech import LocalWhisperSTT
from crop_copilot.services.weather import OpenMeteoWeatherService
from crop_copilot.tabular.inference import PriceForecaster


@lru_cache
def get_event_logger() -> EventLogger:
    return EventLogger(get_settings().events_path)


@lru_cache
def get_review_store() -> SQLiteReviewStore:
    return SQLiteReviewStore(get_settings().database_url)


@lru_cache
def get_stt_service() -> LocalWhisperSTT:
    """Load Whisper once; repeated model construction is too expensive on CPU."""
    return LocalWhisperSTT(get_settings().whisper_model_size)


@lru_cache
def get_orchestrator() -> CopilotOrchestrator:
    settings = get_settings()
    if settings.allow_demo_classifier:
        classifier = DemoClassifier()
    else:
        from crop_copilot.cv.inference import VisionClassifier

        classifier = VisionClassifier(settings.cv_checkpoint)
    if settings.llm_api_base and settings.llm_api_key and settings.llm_model:
        generator = OpenAICompatibleGenerator(
            settings.llm_api_base,
            settings.llm_api_key,
            settings.llm_model,
            settings.prompts_path,
            settings.llm_timeout_seconds,
        )
    else:
        generator = TemplateGenerator()
    return CopilotOrchestrator(
        classifier=classifier,
        weather=OpenMeteoWeatherService(settings.weather_base_url),
        market=PriceForecaster(settings.price_model_path, settings.market_data_path),
        knowledge=KnowledgeBase(settings.knowledge_path),
        generator=generator,
        risk_policy=RiskPolicy(settings.risk_policy_path),
        review_store=get_review_store(),
        event_logger=get_event_logger(),
    )
