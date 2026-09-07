import pandas as pd

from crop_copilot.tabular.features import build_features


def test_lag_features_use_only_past_values() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=20),
            "market": ["A"] * 20,
            "crop": ["tomato"] * 20,
            "modal_price": list(range(100, 120)),
            "min_price": list(range(90, 110)),
            "max_price": list(range(110, 130)),
            "rainfall_mm": [0] * 20,
            "temp_c": [25] * 20,
            "humidity": [60] * 20,
        }
    )
    featured = build_features(frame)
    first = featured.iloc[0]
    original = frame.loc[frame["date"] == first["date"]].iloc[0]
    assert first["lag_1"] == original["modal_price"] - 1
    assert first["rolling_mean_7"] < original["modal_price"]

