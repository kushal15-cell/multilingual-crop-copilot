from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = "development"
    log_level: str = "INFO"
    database_url: str = "sqlite:///./artifacts/review_queue.sqlite3"
    reviewer_api_key: str = "dev-review-key"
    api_base_url: str = "http://localhost:8000"
    cv_checkpoint: Path = Path("artifacts/cv/best_model.pt")
    cv_class_names: Path = Path("artifacts/cv/class_names.json")
    cv_abstain_threshold: float = Field(0.65, ge=0, le=1)
    cv_review_threshold: float = Field(0.80, ge=0, le=1)
    price_model_path: Path = Path("artifacts/price/price_forecaster.joblib")
    market_data_path: Path = Path("data/processed/market_prices.csv")
    weather_base_url: str = "https://api.open-meteo.com/v1/forecast"
    llm_api_base: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_timeout_seconds: float = 30
    whisper_model_size: str = "base"
    tts_provider: str = "gtts"
    allow_demo_classifier: bool = False
    knowledge_path: Path = Path("data/knowledge/advisories.yaml")
    risk_policy_path: Path = Path("configs/risk_policy.yaml")
    prompts_path: Path = Path("configs/prompts.yaml")
    events_path: Path = Path("artifacts/events.jsonl")

    @model_validator(mode="after")
    def block_demo_in_production(self) -> Settings:
        if self.environment.casefold() == "production" and self.allow_demo_classifier:
            raise ValueError("ALLOW_DEMO_CLASSIFIER cannot be enabled in production")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
