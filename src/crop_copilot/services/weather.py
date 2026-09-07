from __future__ import annotations

from datetime import datetime

import httpx

from crop_copilot.schemas import Location, WeatherContext


class OpenMeteoWeatherService:
    def __init__(self, base_url: str, timeout_seconds: float = 8.0) -> None:
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def current(self, location: Location) -> WeatherContext:
        params = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
            ),
            "timezone": "auto",
        }
        try:
            response = httpx.get(self.base_url, params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
            current = response.json()["current"]
            observed_at = datetime.fromisoformat(current["time"])
            return WeatherContext(
                temperature_c=current.get("temperature_2m"),
                relative_humidity_pct=current.get("relative_humidity_2m"),
                precipitation_mm=current.get("precipitation"),
                wind_speed_kmh=current.get("wind_speed_10m"),
                observed_at=observed_at,
                source="Open-Meteo",
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            return WeatherContext(
                source="unavailable",
                unavailable_reason=f"Weather lookup failed: {type(exc).__name__}",
            )

