from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import yaml

from crop_copilot.schemas import (
    AdviceDraft,
    DiseasePrediction,
    Evidence,
    MarketContext,
    WeatherContext,
)

LANGUAGE_NAMES = {"en": "English", "kn": "Kannada", "hi": "Hindi"}


def _weather_text(weather: WeatherContext) -> str:
    if weather.unavailable_reason:
        return "Unavailable"
    return (
        f"temperature {weather.temperature_c} C, humidity {weather.relative_humidity_pct}%, "
        f"precipitation {weather.precipitation_mm} mm, wind {weather.wind_speed_kmh} km/h; "
        f"source {weather.source}"
    )


def _market_text(market: MarketContext) -> str:
    if market.unavailable_reason:
        return "Unavailable"
    warning = "SIMULATED DEMO DATA; " if market.is_synthetic else ""
    return (
        warning + f"latest modal price {market.latest_modal_price} {market.unit}, "
        f"model trend {market.trend}, predicted value {market.predicted_7d_price}; "
        f"source {market.source}; this is contextual, not a guarantee"
    )


class TemplateGenerator:
    """Conservative fallback used when no configured LLM is available."""

    def generate(
        self,
        question: str,
        crop: str,
        language: str,
        prediction: DiseasePrediction,
        weather: WeatherContext,
        market: MarketContext,
        evidence: list[Evidence],
    ) -> AdviceDraft:
        source_ids = [item.source_id for item in evidence]
        if language == "kn":
            return AdviceDraft(
                summary=(
                    f"ಚಿತ್ರ ಮಾದರಿಯ ಪ್ರಾಥಮಿಕ ಫಲಿತಾಂಶ: {prediction.display_name} "
                    f"(ವಿಶ್ವಾಸ {prediction.confidence:.0%}). ಇದು ಅಂತಿಮ ರೋಗನಿರ್ಣಯವಲ್ಲ."
                ),
                actions=[
                    "ಬಾಧಿತ ಎಲೆಗಳ ಎರಡೂ ಬದಿಯ ಸ್ಪಷ್ಟ ಚಿತ್ರಗಳನ್ನು ತೆಗೆದುಕೊಳ್ಳಿ.",
                    "ಹತ್ತಿರದ ಗಿಡಗಳಲ್ಲಿ ಲಕ್ಷಣ ಹರಡುತ್ತಿದೆಯೇ ಎಂದು ಗಮನಿಸಿ.",
                    "ಯಾವುದೇ ರಾಸಾಯನಿಕ ಬಳಕೆಗೆ ಮೊದಲು ಸ್ಥಳೀಯ ಕೃಷಿ ತಜ್ಞರ ಪರಿಶೀಲನೆ ಪಡೆಯಿರಿ.",
                ],
                warning_signs=["ಲಕ್ಷಣಗಳು ವೇಗವಾಗಿ ಹರಡುವುದು", "ಗಿಡ ಒಣಗುವುದು ಅಥವಾ ಹಣ್ಣು ಹಾನಿಯಾಗುವುದು"],
                limitations="ಚಿತ್ರ ಮತ್ತು ಲಭ್ಯ ಸಂದರ್ಭ ಮಾತ್ರ ಬಳಸಲಾಗಿದೆ; ಬೀಜ, ಮಣ್ಣು ಅಥವಾ ಪ್ರಯೋಗಾಲಯ ಪರೀಕ್ಷೆ ಮಾಡಿಲ್ಲ.",
                sources=source_ids,
            )
        if language == "hi":
            return AdviceDraft(
                summary=(
                    f"चित्र मॉडल का प्रारंभिक परिणाम: {prediction.display_name} "
                    f"(विश्वास {prediction.confidence:.0%})। यह अंतिम निदान नहीं है।"
                ),
                actions=[
                    "प्रभावित पत्तियों के दोनों ओर की साफ तस्वीरें लें।",
                    "पास के पौधों में लक्षण फैल रहे हैं या नहीं, इसकी निगरानी करें।",
                    "किसी भी रसायन के उपयोग से पहले स्थानीय कृषि विशेषज्ञ से समीक्षा कराएँ।",
                ],
                warning_signs=["लक्षणों का तेजी से फैलना", "मुरझाना या फल को नुकसान"],
                limitations="केवल चित्र और उपलब्ध संदर्भ का उपयोग हुआ; मिट्टी या प्रयोगशाला जाँच नहीं हुई।",
                sources=source_ids,
            )
        actions = [item.text for item in evidence[:2]] or [
            "Capture clearer photographs of both leaf surfaces and consult a local agronomist."
        ]
        return AdviceDraft(
            summary=(
                f"The image model's preliminary result is {prediction.display_name} "
                f"at {prediction.confidence:.0%} confidence. This is not a confirmed diagnosis."
            ),
            actions=actions,
            warning_signs=["rapid symptom spread", "wilting, stem lesions, or fruit damage"],
            limitations=(
                "This uses the submitted image and available context only; no field, soil, or "
                "laboratory examination was performed. Any simulated market values are a "
                "software demonstration, not real mandi evidence or a guarantee."
            ),
            sources=source_ids,
        )


class OpenAICompatibleGenerator(TemplateGenerator):
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        prompts_path: Path,
        timeout_seconds: float = 30,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.prompts = yaml.safe_load(prompts_path.read_text(encoding="utf-8"))

    def generate(
        self,
        question: str,
        crop: str,
        language: str,
        prediction: DiseasePrediction,
        weather: WeatherContext,
        market: MarketContext,
        evidence: list[Evidence],
    ) -> AdviceDraft:
        allowed_sources = {item.source_id for item in evidence}
        evidence_text = "\n".join(
            f"[{item.source_id}] {item.text} (expert_validated={item.expert_validated})"
            for item in evidence
        ) or "No disease-specific evidence retrieved."
        user_prompt = self.prompts["user"].format(
            language_name=LANGUAGE_NAMES.get(language, language),
            question=question,
            crop=crop,
            disease=prediction.display_name,
            confidence=prediction.confidence,
            weather=_weather_text(weather),
            market=_market_text(market),
            evidence=evidence_text,
        )
        try:
            response = httpx.post(
                f"{self.api_base}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": self.prompts["system"]},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            content = re.sub(r"^```(?:json)?|```$", "", content.strip()).strip()
            draft = AdviceDraft.model_validate(json.loads(content))
            draft.sources = [source for source in draft.sources if source in allowed_sources]
            return draft
        except (httpx.HTTPError, KeyError, IndexError, AttributeError, TypeError, ValueError):
            return super().generate(
                question, crop, language, prediction, weather, market, evidence
            )
