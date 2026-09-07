"""Exercise bundled models and review routing without network access or live records."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from crop_copilot.agent.orchestrator import CopilotOrchestrator
from crop_copilot.agent.risk import RiskPolicy
from crop_copilot.cv.inference import VisionClassifier
from crop_copilot.monitoring.events import EventLogger
from crop_copilot.schemas import AdviceRequest, AdviceStatus, Location, WeatherContext
from crop_copilot.services.knowledge import KnowledgeBase
from crop_copilot.services.llm import TemplateGenerator
from crop_copilot.services.review_store import SQLiteReviewStore
from crop_copilot.tabular.inference import PriceForecaster


class OfflineWeather:
    def current(self, location):
        return WeatherContext(source="unavailable", unavailable_reason="Offline verification")


def main():
    assets = Path("deploy_assets")
    manifest = json.loads((assets / "deployment_manifest.json").read_text())
    digest = hashlib.sha256((assets / "best_model.pt").read_bytes()).hexdigest()
    assert digest == manifest["checkpoint_sha256"].lower(), "Checkpoint checksum mismatch"
    classifier = VisionClassifier(assets / "best_model.pt")
    market = PriceForecaster(assets / "price_forecaster.joblib", assets / "market_prices.csv")
    cases = json.loads((assets / "demo_cases/manifest.json").read_text())["cases"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        store = SQLiteReviewStore(f"sqlite:///{root / 'reviews.sqlite3'}")
        orchestrator = CopilotOrchestrator(
            classifier, OfflineWeather(), market,
            KnowledgeBase(Path("data/knowledge/advisories.yaml")), TemplateGenerator(),
            RiskPolicy(Path("configs/risk_policy.yaml")), store,
            EventLogger(root / "events.jsonl"),
        )
        for case in cases:
            for language in ("en", "kn", "hi"):
                result = orchestrator.advise(AdviceRequest(
                    crop="tomato", question=case["recommended_question"], language=language,
                    location=Location(latitude=14.46, longitude=75.92, market="Davangere"),
                    image_path=str(assets / "demo_cases" / case["filename"]),
                ))
                if case["id"] == "safe_abstention":
                    assert result.status == AdviceStatus.NEED_MORE_INFORMATION
                    assert result.prediction is None
                else:
                    assert result.status == AdviceStatus.PENDING_REVIEW
                    assert result.approved_advice is None
                    assert result.metadata["market_is_synthetic"]
                    store.reject(result.review_id, "verification", "Verification only")
                    assert orchestrator.public_review_status(
                        result.review_id
                    ).status == AdviceStatus.REJECTED
                print(f"PASS {case['id']} ({language}): {result.status.value}")
    print("PASS checkpoint integrity, real inference, simulated prices, and review lifecycle")


if __name__ == "__main__":
    main()
