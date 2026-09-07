from __future__ import annotations

import tempfile
from pathlib import Path

from crop_copilot.agent.orchestrator import CopilotOrchestrator
from crop_copilot.agent.risk import RiskPolicy
from crop_copilot.cv.demo import DemoClassifier
from crop_copilot.monitoring.events import EventLogger
from crop_copilot.schemas import (
    AdviceRequest,
    AdviceStatus,
    Location,
    MarketContext,
    WeatherContext,
)
from crop_copilot.services.knowledge import KnowledgeBase
from crop_copilot.services.llm import TemplateGenerator
from crop_copilot.services.review_store import SQLiteReviewStore
from crop_copilot.tabular.inference import PriceForecaster

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class OfflineWeather:
    def current(self, location: Location) -> WeatherContext:
        return WeatherContext(
            temperature_c=27.0,
            relative_humidity_pct=70.0,
            precipitation_mm=0.0,
            wind_speed_kmh=8.0,
            source="offline-smoke-test",
        )


class OfflineMarket:
    def context(self, crop: str, market: str | None) -> MarketContext:
        forecaster = PriceForecaster(
            PROJECT_ROOT / "artifacts/price/price_forecaster.joblib",
            PROJECT_ROOT / "data/processed/market_prices.csv",
        )
        return forecaster.context(crop, market)


def main() -> None:
    image_path = PROJECT_ROOT / "data/sample/demo_leaf.jpg"
    if not image_path.exists():
        raise FileNotFoundError("Run python scripts/setup_demo.py first")

    with tempfile.TemporaryDirectory(prefix="crop-copilot-smoke-") as temp_dir:
        temp = Path(temp_dir)
        store = SQLiteReviewStore(f"sqlite:///{temp / 'reviews.sqlite3'}")
        orchestrator = CopilotOrchestrator(
            classifier=DemoClassifier(),
            weather=OfflineWeather(),
            market=OfflineMarket(),
            knowledge=KnowledgeBase(PROJECT_ROOT / "data/knowledge/advisories.yaml"),
            generator=TemplateGenerator(),
            risk_policy=RiskPolicy(PROJECT_ROOT / "configs/risk_policy.yaml"),
            review_store=store,
            event_logger=EventLogger(temp / "events.jsonl"),
        )
        request = AdviceRequest(
            crop="tomato",
            question="What should I do next?",
            language="en",
            location=Location(
                latitude=14.4644,
                longitude=75.9218,
                region="Davangere",
                market="Davangere",
            ),
            image_path=str(image_path),
        )
        pending = orchestrator.advise(request)
        assert pending.status == AdviceStatus.PENDING_REVIEW
        assert pending.review_id
        assert pending.approved_advice is None

        private_record = store.get(pending.review_id)
        assert private_record is not None
        approved = store.approve(
            pending.review_id,
            reviewer="smoke-test-agronomist",
            approved_text=private_record.private_draft,
            notes="Automated local smoke test only",
        )
        assert approved.status == AdviceStatus.APPROVED
        public = orchestrator.public_review_status(pending.review_id)
        assert public.status == AdviceStatus.APPROVED
        assert public.approved_advice is not None

    print("PASS: image -> context -> private draft -> review -> approval -> farmer response")
    print("Reminder: DemoClassifier uses a fixed synthetic result and is not diagnostic.")


if __name__ == "__main__":
    main()
