from __future__ import annotations

from typing import TypedDict


class LocationOption(TypedDict):
    latitude: float
    longitude: float
    market: str


KARNATAKA_LOCATIONS: dict[str, LocationOption] = {
    "Bengaluru": {"latitude": 12.9716, "longitude": 77.5946, "market": "Bengaluru"},
    "Belagavi": {"latitude": 15.8497, "longitude": 74.4977, "market": "Belagavi"},
    "Davangere": {"latitude": 14.4644, "longitude": 75.9218, "market": "Davangere"},
    "Hubballi": {"latitude": 15.3647, "longitude": 75.1240, "market": "Hubballi"},
    "Mangaluru": {"latitude": 12.9141, "longitude": 74.8560, "market": "Mangaluru"},
    "Mysuru": {"latitude": 12.2958, "longitude": 76.6394, "market": "Mysuru"},
    "Shivamogga": {"latitude": 13.9299, "longitude": 75.5681, "market": "Shivamogga"},
}


def resolve_location(
    name: str,
    precise_latitude: float,
    precise_longitude: float,
) -> dict[str, str | float]:
    selected = KARNATAKA_LOCATIONS[name]
    return {
        "name": name,
        "latitude": precise_latitude,
        "longitude": precise_longitude,
        "market": selected["market"],
    }
