from crop_copilot.ui.locations import KARNATAKA_LOCATIONS, resolve_location


def test_karnataka_location_choices_map_to_market_and_coordinates() -> None:
    resolved = resolve_location("Mysuru", 12.3000, 76.6500)

    assert len(KARNATAKA_LOCATIONS) == 7
    assert resolved == {
        "name": "Mysuru",
        "latitude": 12.3,
        "longitude": 76.65,
        "market": "Mysuru",
    }
